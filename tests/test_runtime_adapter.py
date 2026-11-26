from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from cole.core.feedback import (
    ATTENTION_PROXY_ASYNC_FEATURE_KEYS_KEY,
    ATTENTION_PROXY_ASYNC_VALUES_KEY,
    ExecFeedback,
)
from cole.core.types import MaskPlan
from cole.runtime.wan_adapter import (
    StepCallbackUpdate,
    _decode_vae_latents_with_fallback,
    _freeze_inference_modules,
    _restore_offloaded_modules,
    consume_runtime_feedback,
    publish_mask_plan,
    record_runtime_feedback,
    run_pipe_with_step_callback,
)


class _Dummy:
    pass


class _DummyTransformer:
    dtype = torch.float32
    config = SimpleNamespace(in_channels=2)

    def __init__(self) -> None:
        self.grad_enabled_flags: list[bool] = []
        self.call_count = 0

    def __call__(self, *, hidden_states, timestep, encoder_hidden_states, attention_kwargs=None, return_dict=False):
        del timestep, encoder_hidden_states, attention_kwargs, return_dict
        self.call_count += 1
        self.grad_enabled_flags.append(bool(torch.is_grad_enabled()))
        return (hidden_states + 1.0,)


class _DummyScheduler:
    order = 1

    def __init__(self) -> None:
        self.timesteps = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
        self.model_output_means: list[float] = []

    def set_timesteps(self, num_inference_steps: int, device=None) -> None:
        values = [float(idx) for idx in range(int(num_inference_steps), 0, -1)]
        self.timesteps = torch.tensor(values, dtype=torch.float32, device=device)

    def step(self, model_output, timestep, sample, return_dict=False):
        del timestep, return_dict
        self.model_output_means.append(float(model_output.detach().float().mean().item()))
        return (sample - 0.5 * model_output,)


class _DummyVAE:
    dtype = torch.float32
    config = SimpleNamespace(latents_mean=[0.0, 0.0], latents_std=[1.0, 1.0], z_dim=2)

    def decode(self, latents, return_dict=False):
        del return_dict
        return (latents,)


class _DummyVideoProcessor:
    def postprocess_video(self, video, output_type="pt"):
        del output_type
        return video


class _DummyPipe:
    def __init__(self) -> None:
        self.transformer = _DummyTransformer()
        self.scheduler = _DummyScheduler()
        self.vae = _DummyVAE()
        self.video_processor = _DummyVideoProcessor()
        self.vae_scale_factor_temporal = 4
        self._execution_device = torch.device("cpu")
        self._guidance_scale = 1.0
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False

    @property
    def do_classifier_free_guidance(self) -> bool:
        return bool(float(self._guidance_scale) > 1.0)

    @property
    def interrupt(self) -> bool:
        return bool(self._interrupt)

    def maybe_free_model_hooks(self) -> None:
        return None

    def progress_bar(self, total: int):
        del total

        @contextmanager
        def _ctx():
            yield SimpleNamespace(update=lambda n=1: None)

        return _ctx()


class _FrozenPipe:
    def __init__(self) -> None:
        shared = torch.nn.Linear(2, 2)
        self.components = {
            "transformer": shared,
            "vae": torch.nn.Linear(2, 2),
            "text_encoder": shared,
            "non_module": object(),
        }


class _FakeDeviceRef:
    def __init__(self, device_type: str) -> None:
        self.device = torch.device(device_type)


class _RetryThenSuccessVAE:
    def __init__(self) -> None:
        self.current_device = "cuda"
        self.decode_calls = 0

    def decode(self, latents, return_dict=False):
        del return_dict
        self.decode_calls += 1
        if self.current_device == "cuda" and self.decode_calls == 1:
            raise torch.OutOfMemoryError("simulated decode OOM")
        return (latents + 0.5,)

    def parameters(self):
        return iter([_FakeDeviceRef(self.current_device)])

    def to(self, device):
        self.current_device = torch.device(device).type if not isinstance(device, str) else device
        return self


class _TiledFallbackVAE:
    cole_supports_spatial_tiling = True
    cole_tile_latent_hw_candidates = ((2, 2),)
    cole_tile_overlap_factor = 0.0

    def __init__(self) -> None:
        self.current_device = "cuda"
        self.decode_shapes: list[tuple[int, int]] = []

    def decode(self, latents, return_dict=False):
        del return_dict
        shape = (int(latents.shape[-2]), int(latents.shape[-1]))
        self.decode_shapes.append(shape)
        if self.current_device == "cuda" and shape == (4, 4):
            raise torch.OutOfMemoryError("simulated full decode OOM")
        return (latents + 0.25,)

    def parameters(self):
        return iter([_FakeDeviceRef(self.current_device)])

    def to(self, device):
        self.current_device = torch.device(device).type if not isinstance(device, str) else device
        return self


class _CpuFallbackVAE:
    def __init__(self) -> None:
        self.current_device = "cuda"
        self.decode_calls = 0

    def decode(self, latents, return_dict=False):
        del return_dict
        self.decode_calls += 1
        if self.current_device == "cuda":
            raise torch.OutOfMemoryError("simulated decode OOM")
        return (latents + 1.0,)

    def parameters(self):
        return iter([_FakeDeviceRef(self.current_device)])

    def to(self, device):
        self.current_device = torch.device(device).type if not isinstance(device, str) else device
        return self


def test_publish_mask_plan_sets_current_execution_plan() -> None:
    obj = _Dummy()
    plan = MaskPlan(
        backend_name="sparse",
        profile_name="sparse_mid",
        sparse_mode="topk",
        sparse_value=0.5,
        estimated_density=0.5,
        selected_units=16,
        plan_signature=("sparse_mid",),
    )
    publish_mask_plan(obj, plan)
    assert getattr(obj, "cole_mask_plan").profile_name == "sparse_mid"


def test_runtime_feedback_accumulates_and_clears() -> None:
    obj = _Dummy()
    first = ExecFeedback(
        backend_prepare_ms=1.0,
        backend_run_ms=2.0,
        total_attn_ms=3.0,
        step_wall_ms=5.0,
        actual_density=0.2,
        selected_units=12,
        backend_name="sparse",
        executed_plan_signature="('dense_full',)",
        executed_plan_density=0.2,
        backend_is_sparse=True,
        backend_first_call=True,
    )
    record_runtime_feedback(
        obj,
        first,
    )
    record_runtime_feedback(
        obj,
        ExecFeedback(
            backend_prepare_ms=4.0,
            backend_run_ms=5.0,
            total_attn_ms=6.0,
            step_wall_ms=7.0,
            actual_density=0.6,
            selected_units=18,
            backend_name="sparse",
            executed_plan_signature="('sparse_mid',)",
            executed_plan_density=0.6,
            backend_is_sparse=True,
            backend_first_call=False,
        ),
    )
    feedback = consume_runtime_feedback(obj)
    assert isinstance(feedback, ExecFeedback)
    assert feedback is first
    assert feedback.backend_prepare_ms == 5.0
    assert feedback.backend_run_ms == 7.0
    assert feedback.total_attn_ms == 9.0
    assert feedback.step_wall_ms == 7.0
    assert abs(feedback.actual_density - 0.4) < 1e-6
    assert feedback.selected_units == 18
    assert feedback.backend_name == "sparse"
    assert feedback.executed_plan_signature == "('sparse_mid',)"
    assert abs(feedback.executed_plan_density - 0.4) < 1e-6
    assert feedback.backend_is_sparse is True
    assert feedback.backend_first_call is True
    assert consume_runtime_feedback(obj) is None


def test_consume_runtime_feedback_resolves_pending_attention_proxy() -> None:
    obj = _Dummy()
    record_runtime_feedback(
        obj,
        ExecFeedback(
            attention_proxy={
                ATTENTION_PROXY_ASYNC_VALUES_KEY: torch.tensor([0.25, 0.75], dtype=torch.float32),
                ATTENTION_PROXY_ASYNC_FEATURE_KEYS_KEY: ("attn_eff_support_ratio", "attn_top1_prob"),
                "attn_proxy_samples": 8.0,
            },
        ),
    )

    feedback = consume_runtime_feedback(obj)

    assert isinstance(feedback, ExecFeedback)
    assert feedback.attention_proxy == {
        "attn_eff_support_ratio": 0.25,
        "attn_top1_prob": 0.75,
        "attn_proxy_samples": 8.0,
    }
    payload = feedback.to_dict()
    assert payload["attention_proxy"] == feedback.attention_proxy


def test_runtime_feedback_merges_attention_proxy_in_place() -> None:
    obj = _Dummy()
    first = ExecFeedback(attention_proxy={"attn_top1_prob": 0.25, "attn_proxy_updates_samples": 1.0})
    record_runtime_feedback(obj, first)
    record_runtime_feedback(
        obj,
        ExecFeedback(attention_proxy={"attn_top1_prob": 0.75, "attn_proxy_updates_samples": 1.0}),
    )

    feedback = consume_runtime_feedback(obj)

    assert feedback is first
    assert isinstance(feedback, ExecFeedback)
    assert feedback.attention_proxy == {
        "attn_top1_prob": 0.5,
        "attn_proxy_updates_samples": 2.0,
    }


def test_runtime_feedback_merges_attention_prob_patches_without_serializing() -> None:
    obj = _Dummy()
    first_patch = torch.full((1, 1, 1, 2, 3), 0.25, dtype=torch.float16)
    second_patch = torch.full((1, 1, 1, 2, 3), 0.75, dtype=torch.float16)
    first = ExecFeedback(attention_prob_patches={0: first_patch})
    record_runtime_feedback(obj, first)
    record_runtime_feedback(obj, ExecFeedback(attention_prob_patches={5: second_patch}))

    feedback = consume_runtime_feedback(obj)

    assert feedback is first
    assert isinstance(feedback.attention_prob_patches, dict)
    assert sorted(feedback.attention_prob_patches) == [0, 5]
    assert torch.equal(feedback.attention_prob_patches[0], first_patch)
    assert torch.equal(feedback.attention_prob_patches[5], second_patch)
    assert feedback.to_dict()["attention_prob_patches"] is None


def test_runtime_feedback_merges_attention_sampling_overhead() -> None:
    obj = _Dummy()
    first = ExecFeedback(attention_sampling_elapsed_sec=0.25, attention_sampling_updates=1)
    record_runtime_feedback(obj, first)
    record_runtime_feedback(obj, ExecFeedback(attention_sampling_elapsed_sec=0.5, attention_sampling_updates=1))

    feedback = consume_runtime_feedback(obj)

    assert feedback is first
    assert feedback.attention_sampling_elapsed_sec == pytest.approx(0.75)
    assert feedback.attention_sampling_updates == 2


def test_run_pipe_with_step_callback_exposes_pre_step_latents_and_skip() -> None:
    pipe = _DummyPipe()
    latents = torch.zeros(1, 2, 2, 4, 4)
    prompt_embeds = torch.zeros(1, 4, 4)
    observed_steps: list[int] = []
    observed_noise_shapes: list[tuple[int, ...]] = []
    observed_first_input = None

    def _on_step_end(step_idx, timestep, callback_kwargs):
        del timestep
        nonlocal observed_first_input
        observed_steps.append(int(step_idx))
        observed_noise_shapes.append(tuple(callback_kwargs["noise_pred"].shape))
        if observed_first_input is None:
            observed_first_input = callback_kwargs["latents_before_step"].clone()
        if int(step_idx) == 0:
            callback_kwargs["skip_step_count"] = 1
        return callback_kwargs

    output = run_pipe_with_step_callback(
        pipe=pipe,
        call_kwargs={
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "guidance_scale": 1.0,
            "num_inference_steps": 3,
            "output_type": "latent",
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": None,
        },
        on_step_end=_on_step_end,
        tensor_inputs=("latents", "latents_before_step", "noise_pred"),
        require_step_callback=False,
    )

    frames = output.frames if hasattr(output, "frames") else output
    assert observed_steps == [0, 2]
    assert observed_noise_shapes == [tuple(latents.shape), tuple(latents.shape)]
    assert torch.allclose(observed_first_input, latents)
    assert tuple(frames.shape) == tuple(latents.shape)
    assert float(frames.abs().sum()) > 0.0


def test_run_pipe_with_step_callback_resyncs_scheduler_timesteps_after_callback_mutation() -> None:
    pipe = _DummyPipe()
    latents = torch.zeros(1, 2, 2, 4, 4)
    prompt_embeds = torch.zeros(1, 4, 4)
    observed_timesteps: list[int] = []

    def _on_step_end(step_idx, timestep, callback_kwargs):
        del callback_kwargs
        observed_timesteps.append(int(float(timestep)))
        if int(step_idx) == 0:
            pipe.scheduler.timesteps = torch.tensor([3.0, 9.0, 8.0], dtype=torch.float32)
        return StepCallbackUpdate()

    _ = run_pipe_with_step_callback(
        pipe=pipe,
        call_kwargs={
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "guidance_scale": 1.0,
            "num_inference_steps": 3,
            "output_type": "latent",
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": None,
        },
        on_step_end=_on_step_end,
        tensor_inputs=("latents", "latents_before_step", "noise_pred"),
        require_step_callback=False,
    )

    assert observed_timesteps == [3, 9, 8]


def test_run_pipe_with_step_callback_disables_grad_tracking() -> None:
    pipe = _DummyPipe()
    latents = torch.zeros(1, 2, 2, 4, 4)
    prompt_embeds = torch.zeros(1, 4, 4)

    output = run_pipe_with_step_callback(
        pipe=pipe,
        call_kwargs={
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "guidance_scale": 1.0,
            "num_inference_steps": 3,
            "output_type": "latent",
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": None,
        },
        on_step_end=lambda step_idx, timestep, callback_kwargs: StepCallbackUpdate(),
        tensor_inputs=("latents",),
        require_step_callback=False,
    )

    assert pipe.transformer.grad_enabled_flags
    assert all(flag is False for flag in pipe.transformer.grad_enabled_flags)
    assert output.decode_backend == "latent_passthrough"


def test_run_pipe_with_step_callback_prediction_reuse_off_preserves_transformer_calls() -> None:
    pipe = _DummyPipe()
    latents = torch.zeros(1, 2, 2, 4, 4)
    prompt_embeds = torch.zeros(1, 4, 4)

    output = run_pipe_with_step_callback(
        pipe=pipe,
        call_kwargs={
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "guidance_scale": 1.0,
            "num_inference_steps": 5,
            "output_type": "latent",
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": None,
        },
        on_step_end=lambda step_idx, timestep, callback_kwargs: StepCallbackUpdate(),
        tensor_inputs=("latents",),
        require_step_callback=False,
    )

    assert pipe.transformer.call_count == 5
    assert len(pipe.scheduler.model_output_means) == 5
    assert output.prediction_reuse is None


def test_run_pipe_with_step_callback_applies_deferred_prediction_reuse_policy_update() -> None:
    pipe = _DummyPipe()

    def on_step_end(step_idx, timestep, callback_kwargs):
        del timestep, callback_kwargs
        if int(step_idx) == 1:
            return StepCallbackUpdate(
                prediction_reuse_policy_update={
                    "mode": "adaptive_output",
                    "adaptive_delta_threshold": 1e9,
                    "max_consecutive_cached_steps": 4,
                    "full_steps": [0, 1],
                    "cache_steps": [2, 3, 4],
                },
            )
        return StepCallbackUpdate()

    output = run_pipe_with_step_callback(
        pipe=pipe,
        call_kwargs={
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "guidance_scale": 1.0,
            "num_inference_steps": 5,
            "output_type": "latent",
            "latents": torch.zeros(1, 2, 2, 4, 4),
            "prompt_embeds": torch.zeros(1, 4, 4),
            "negative_prompt_embeds": None,
            "cole_prediction_reuse_policy_enabled": True,
        },
        on_step_end=on_step_end,
        tensor_inputs=("latents",),
        require_step_callback=False,
    )

    assert pipe.transformer.call_count == 2
    assert output.prediction_reuse["stats"]["compute_step_indices"] == [0, 1]
    assert output.prediction_reuse["stats"]["cache_step_indices"] == [2, 3, 4]
    assert output.prediction_reuse["stats"]["policy_update_count"] == 1


def test_run_pipe_with_step_callback_adaptive_prediction_reuse_uses_latent_delta() -> None:
    pipe = _DummyPipe()
    latents = torch.zeros(1, 2, 2, 4, 4)
    prompt_embeds = torch.zeros(1, 4, 4)

    output = run_pipe_with_step_callback(
        pipe=pipe,
        call_kwargs={
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "guidance_scale": 1.0,
            "num_inference_steps": 5,
            "output_type": "latent",
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": None,
            "cole_prediction_reuse": {
                "mode": "adaptive_output",
                "cache_steps": [1, 2, 3],
                "full_steps": [0, 4],
                "max_consecutive_cached_steps": 3,
                "adaptive_delta_threshold": 10.0,
            },
        },
        on_step_end=lambda step_idx, timestep, callback_kwargs: StepCallbackUpdate(),
        tensor_inputs=("latents",),
        require_step_callback=False,
    )

    assert pipe.transformer.call_count == 3
    assert output.prediction_reuse["config"]["mode"] == "adaptive_output"
    assert output.prediction_reuse["stats"]["cache_step_indices"] == [1, 2]
    assert output.prediction_reuse["stats"]["cache_reasons"]["adaptive_output"] == 2
    assert output.prediction_reuse["stats"]["compute_reasons"]["adaptive_delta"] == 1
    assert output.prediction_reuse["stats"]["adaptive_delta_max"] is not None


def test_run_pipe_with_step_callback_adaptive_prediction_reuse_computes_on_large_delta() -> None:
    pipe = _DummyPipe()
    latents = torch.zeros(1, 2, 2, 4, 4)
    prompt_embeds = torch.zeros(1, 4, 4)

    output = run_pipe_with_step_callback(
        pipe=pipe,
        call_kwargs={
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "guidance_scale": 1.0,
            "num_inference_steps": 5,
            "output_type": "latent",
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": None,
            "cole_prediction_reuse": {
                "mode": "adaptive_output",
                "cache_steps": [1, 2, 3],
                "full_steps": [0, 4],
                "max_consecutive_cached_steps": 3,
                "adaptive_delta_threshold": 0.0,
            },
        },
        on_step_end=lambda step_idx, timestep, callback_kwargs: StepCallbackUpdate(),
        tensor_inputs=("latents",),
        require_step_callback=False,
    )

    assert pipe.transformer.call_count == 5
    assert output.prediction_reuse["stats"]["cache_step_indices"] == []
    assert output.prediction_reuse["stats"]["compute_reasons"]["adaptive_delta"] == 3


def test_full_gpu_decode_keeps_other_models_resident(monkeypatch) -> None:
    pipe = SimpleNamespace(vae=_RetryThenSuccessVAE())
    pipe.vae.decode_calls = 1
    latents = torch.zeros(1, 2, 3, 4, 4)

    def unexpected_offload(*args, **kwargs):
        pytest.fail("Full GPU decode must keep other modules resident")

    monkeypatch.setattr("cole.runtime.wan_adapter._offload_modules_for_decode", unexpected_offload)

    video, backend = _decode_vae_latents_with_fallback(pipe, latents, [])

    assert backend == "gpu_full"
    assert torch.equal(video, torch.full_like(latents, 0.5))


def test_decode_fallback_retries_gpu_full_once_after_offload(monkeypatch) -> None:
    pipe = SimpleNamespace(vae=_RetryThenSuccessVAE())
    latents = torch.zeros(1, 2, 3, 4, 4)
    other_module = torch.nn.Linear(1, 1)
    released = [(other_module, torch.device("cuda"))]

    def offload(pipe_to_offload, *, extra_modules):
        assert pipe_to_offload.vae.decode_calls == 1
        assert extra_modules == (other_module,)
        return released

    monkeypatch.setattr("cole.runtime.wan_adapter._offload_modules_for_decode", offload)
    released_modules = []
    video, backend = _decode_vae_latents_with_fallback(
        pipe, latents, released_modules, extra_modules=(other_module,),
    )

    assert backend == "gpu_full_retry"
    assert torch.allclose(video, torch.full_like(latents, 0.5))
    assert pipe.vae.decode_calls == 2
    assert pipe.vae.current_device == "cuda"
    assert released_modules == released


def test_decode_fallback_retries_when_gpu_postprocessing_runs_out_of_memory() -> None:
    calls = []

    def postprocess(video, *, output_type):
        calls.append(output_type)
        if len(calls) == 1:
            raise torch.OutOfMemoryError("simulated postprocess OOM")
        return video * 2

    pipe = SimpleNamespace(
        vae=_RetryThenSuccessVAE(),
        video_processor=SimpleNamespace(postprocess_video=postprocess),
    )
    pipe.vae.decode_calls = 1
    latents = torch.zeros(1, 2, 3, 4, 4)

    video, backend = _decode_vae_latents_with_fallback(pipe, latents, [], output_type="pt")

    assert calls == ["pt", "pt"]
    assert backend == "gpu_full_retry"
    assert torch.equal(video, torch.ones_like(latents))


def test_decode_fallback_uses_spatial_tiled_gpu_decode_before_cpu() -> None:
    pipe = SimpleNamespace(vae=_TiledFallbackVAE())
    latents = torch.zeros(1, 2, 3, 4, 4)

    video, backend = _decode_vae_latents_with_fallback(pipe, latents, [])

    assert backend == "gpu_tiled"
    assert torch.allclose(video, torch.full((1, 2, 3, 4, 4), 0.25))
    assert pipe.vae.current_device == "cuda"
    assert pipe.vae.decode_shapes[:2] == [(4, 4), (4, 4)]
    assert any(shape == (2, 2) for shape in pipe.vae.decode_shapes[2:])


def test_decode_fallback_blends_on_cpu_before_moving_vae_off_gpu(monkeypatch) -> None:
    pipe = SimpleNamespace(vae=_TiledFallbackVAE())
    latents = torch.zeros(1, 2, 3, 4, 4)
    destinations = []

    def decode_tiles(vae, tile_latents, *, tile_latent_hw, overlap_factor, output_device):
        assert vae.current_device == "cuda"
        assert tile_latent_hw == (2, 2)
        assert overlap_factor == 0.0
        destinations.append(output_device)
        if len(destinations) == 1:
            raise torch.OutOfMemoryError("simulated GPU tile blend OOM")
        return tile_latents + 0.25

    monkeypatch.setattr("cole.runtime.wan_adapter._decode_wan_vae_spatially_tiled", decode_tiles)
    video, backend = _decode_vae_latents_with_fallback(pipe, latents, [])

    assert destinations == [latents.device, "cpu"]
    assert backend == "gpu_tiled_cpu_blend"
    assert torch.equal(video, torch.full_like(latents, 0.25))


def test_decode_fallback_moves_to_cpu_when_tiled_decode_is_unavailable() -> None:
    pipe = SimpleNamespace(vae=_CpuFallbackVAE())
    latents = torch.zeros(1, 2, 3, 4, 4)
    released_modules: list[tuple[object, torch.device]] = []

    video, backend = _decode_vae_latents_with_fallback(pipe, latents, released_modules)

    assert backend == "cpu_fallback"
    assert torch.allclose(video, torch.ones_like(latents))
    assert pipe.vae.current_device == "cpu"
    assert pipe.vae.decode_calls == 3
    assert len(released_modules) == 1


def test_decode_reports_failure_to_restore_modules() -> None:
    class OffloadedModule(torch.nn.Module):
        def to(self, device):
            raise torch.OutOfMemoryError("simulated restore OOM")

    with pytest.raises(torch.OutOfMemoryError, match="Could not restore OffloadedModule"):
        _restore_offloaded_modules([(OffloadedModule(), torch.device("cuda:0"))])


def test_freeze_inference_modules_disables_grad_for_all_pipe_components() -> None:
    pipe = _FrozenPipe()

    _freeze_inference_modules(pipe)

    for module in {value for value in pipe.components.values() if isinstance(value, torch.nn.Module)}:
        assert module.training is False
        assert all(param.requires_grad is False for param in module.parameters())


