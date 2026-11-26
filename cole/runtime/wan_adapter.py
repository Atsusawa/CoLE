from __future__ import annotations

import gc
import inspect
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Optional

import torch

from cole.core.feedback import ExecFeedback, clone_attention_prob_patches, resolve_attention_proxy_payload
from cole.runtime.prediction_reuse import PredictionReuseConfig, PredictionReuseRuntime


StepCallbackFn = Callable[[int, Any, Dict[str, Any]], Any]


@dataclass
class StepCallbackUpdate:
    """Mutable runtime updates emitted by a custom Wan step callback."""

    latents: torch.Tensor | None = None
    prompt_embeds: torch.Tensor | None = None
    negative_prompt_embeds: torch.Tensor | None = None
    skip_step_count: int = 0
    prediction_reuse_policy_update: dict[str, Any] | None = None


def resolve_pretrained_kwargs(from_pretrained_fn, *, dtype, local_files_only: bool = True) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"local_files_only": bool(local_files_only)}
    try:
        signature = inspect.signature(from_pretrained_fn)
        if "dtype" in signature.parameters:
            kwargs["dtype"] = dtype
        else:
            kwargs["torch_dtype"] = dtype
    except (TypeError, ValueError):
        kwargs["torch_dtype"] = dtype
    return kwargs


def _freeze_inference_modules(pipe) -> None:
    components = getattr(pipe, "components", None)
    if isinstance(components, dict):
        candidates = components.values()
    else:
        candidates = (
            getattr(pipe, name, None)
            for name in ("transformer", "vae", "text_encoder", "transformer_2", "text_encoder_2")
        )
    seen: set[int] = set()
    for module in candidates:
        if not isinstance(module, torch.nn.Module):
            continue
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        module.eval()
        module.requires_grad_(False)



def load_wan_pipeline(model_dir: str, *, dtype, local_files_only: bool = True):
    from diffusers import WanPipeline

    kwargs = resolve_pretrained_kwargs(WanPipeline.from_pretrained, dtype=dtype, local_files_only=local_files_only)
    pipe = WanPipeline.from_pretrained(model_dir, **kwargs)
    _freeze_inference_modules(pipe)
    setattr(pipe, "_cole_model_backend", "wan")
    return pipe


def build_scheduler_timesteps(
    scheduler,
    *,
    num_inference_steps: int,
    device,
    flow_shift: float | None = None,
    disable_corrector: list[int] | None = None,
) -> list[int]:
    configure_scheduler_runtime(
        scheduler,
        flow_shift=flow_shift,
        disable_corrector=disable_corrector,
    )
    scheduler.set_timesteps(int(num_inference_steps), device=device)
    return snapshot_scheduler_timesteps(scheduler)


def snapshot_scheduler_timesteps(scheduler) -> list[int]:
    schedule = getattr(scheduler, "timesteps", None)
    if schedule is None:
        raise ValueError("Scheduler does not expose timesteps.")
    if isinstance(schedule, torch.Tensor):
        values = schedule.detach().reshape(-1).to(device="cpu").tolist()
        return [int(value) for value in values]
    return [int(x) for x in schedule]


def get_scheduler_flow_shift(scheduler) -> float | None:
    config = getattr(scheduler, "config", None)
    if config is None:
        return None
    value = getattr(config, "flow_shift", None)
    if value is None:
        return None
    return float(value)


def configure_scheduler_runtime(
    scheduler,
    *,
    flow_shift: float | None = None,
    disable_corrector: list[int] | None = None,
) -> None:
    register = getattr(scheduler, "register_to_config", None)
    config_updates: dict[str, Any] = {}
    if flow_shift is not None:
        config_updates["flow_shift"] = float(flow_shift)
    if disable_corrector is not None:
        normalized = [int(step) for step in disable_corrector]
        config_updates["disable_corrector"] = normalized
        setattr(scheduler, "disable_corrector", normalized)
    if callable(register) and config_updates:
        register(**config_updates)


def extract_timestep_int(timestep: Any) -> int | None:
    if isinstance(timestep, torch.Tensor):
        if timestep.numel() == 0:
            return None
        return int(timestep.reshape(-1)[0].item())
    if timestep is None:
        return None
    return int(timestep)


def _timestep_to_device(timestep: Any, *, device: torch.device, batch_size: int | None = None):
    if isinstance(timestep, torch.Tensor):
        value = timestep.reshape(-1)[0].to(device=device)
    else:
        value = torch.tensor(timestep, device=device)
    if batch_size is not None:
        value = value.expand(int(batch_size))
    return value


def _attention_kwargs_with_timestep_int(attention_kwargs: Any, timestep_int: int) -> dict[str, Any]:
    kwargs = dict(attention_kwargs) if isinstance(attention_kwargs, dict) else {}
    kwargs["cole_timestep_int"] = int(timestep_int)
    return kwargs


def run_pipe_with_step_callback(
    *,
    pipe,
    call_kwargs: Dict[str, Any],
    on_step_end: Optional[StepCallbackFn],
    tensor_inputs: Iterable[str] = ("latents",),
    require_step_callback: bool = True,
):
    if on_step_end is None:
        return pipe(**call_kwargs)

    if _supports_custom_wan_loop(pipe):
        return _run_wan_pipe_with_step_callback(
            pipe=pipe,
            call_kwargs=call_kwargs,
            on_step_end=on_step_end,
            tensor_inputs=tensor_inputs,
        )

    call_sig = inspect.signature(pipe.__call__)
    if "callback_on_step_end" in call_sig.parameters:
        kwargs = dict(call_kwargs)

        def _wrapped_on_step_end(pipe_ref, step, timestep, callback_kwargs):
            del pipe_ref
            payload = callback_kwargs if isinstance(callback_kwargs, dict) else {}
            out = on_step_end(int(step), timestep, payload)
            normalized = _normalize_step_callback_update(out)
            if normalized.latents is not None:
                payload["latents"] = normalized.latents
            if normalized.prompt_embeds is not None:
                payload["prompt_embeds"] = normalized.prompt_embeds
            if normalized.negative_prompt_embeds is not None:
                payload["negative_prompt_embeds"] = normalized.negative_prompt_embeds
            if int(normalized.skip_step_count) > 0:
                payload["skip_step_count"] = int(normalized.skip_step_count)
            return payload

        kwargs["callback_on_step_end"] = _wrapped_on_step_end
        if "callback_on_step_end_tensor_inputs" in call_sig.parameters:
            kwargs["callback_on_step_end_tensor_inputs"] = list(tensor_inputs)
        return pipe(**kwargs)

    if "callback" in call_sig.parameters:
        kwargs = dict(call_kwargs)

        def _wrapped_callback(step, timestep, latents):
            payload = {"latents": latents}
            _ = on_step_end(int(step), timestep, payload)
            return None

        kwargs["callback"] = _wrapped_callback
        if "callback_steps" in call_sig.parameters:
            kwargs["callback_steps"] = 1
        return pipe(**kwargs)

    if require_step_callback:
        raise RuntimeError("Pipeline does not support step callbacks.")
    return pipe(**call_kwargs)


def _supports_custom_wan_loop(pipe) -> bool:
    if str(getattr(pipe, "_cole_model_backend", "wan")) != "wan":
        return False
    return all(hasattr(pipe, attr) for attr in ("transformer", "scheduler", "vae", "video_processor"))


def _run_wan_pipe_with_step_callback(
    *,
    pipe,
    call_kwargs: Dict[str, Any],
    on_step_end: StepCallbackFn,
    tensor_inputs: Iterable[str],
):
    height = int(call_kwargs.get("height", 480))
    width = int(call_kwargs.get("width", 832))
    num_frames = int(call_kwargs.get("num_frames", 81))
    num_inference_steps = int(call_kwargs.get("num_inference_steps", 50))
    guidance_scale = float(call_kwargs.get("guidance_scale", 5.0))
    num_videos_per_prompt = int(call_kwargs.get("num_videos_per_prompt", 1) or 1)
    decode_offload_modules = tuple(call_kwargs.get("cole_decode_offload_modules") or ())
    generator = call_kwargs.get("generator")
    prompt_embeds = call_kwargs.get("prompt_embeds")
    negative_prompt_embeds = call_kwargs.get("negative_prompt_embeds")
    output_type = call_kwargs.get("output_type", "np")
    return_dict = bool(call_kwargs.get("return_dict", True))
    attention_kwargs = call_kwargs.get("attention_kwargs")
    prompt = call_kwargs.get("prompt")
    negative_prompt = call_kwargs.get("negative_prompt")
    max_sequence_length = int(call_kwargs.get("max_sequence_length", 512))
    prediction_reuse_config = PredictionReuseConfig.from_payload(call_kwargs.get("cole_prediction_reuse"))
    prediction_reuse_policy_enabled = bool(call_kwargs.get("cole_prediction_reuse_policy_enabled", False))
    prediction_reuse = (
        PredictionReuseRuntime(prediction_reuse_config)
        if prediction_reuse_config.enabled or prediction_reuse_policy_enabled
        else None
    )
    if num_frames % int(pipe.vae_scale_factor_temporal) != 1:
        num_frames = num_frames // int(pipe.vae_scale_factor_temporal) * int(pipe.vae_scale_factor_temporal) + 1
    num_frames = max(num_frames, 1)

    pipe._guidance_scale = guidance_scale
    pipe._attention_kwargs = attention_kwargs
    pipe._current_timestep = None
    pipe._cole_current_timestep_int = None
    pipe._interrupt = False

    device = getattr(pipe, "_execution_device", None)
    if device is None:
        device = torch.device("cpu")

    transformer_dtype = pipe.transformer.dtype
    if prompt_embeds is None:
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=bool(float(guidance_scale) > 1.0),
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=transformer_dtype,
            )
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = getattr(pipe.scheduler, "timesteps")
    timesteps_int = snapshot_scheduler_timesteps(pipe.scheduler)

    latents = call_kwargs.get("latents")
    if latents is None:
        latents = pipe.prepare_latents(
            num_videos_per_prompt,
            pipe.transformer.config.in_channels,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            None,
        )
    step_index = 0
    denoise_loop_start = time.perf_counter()
    with torch.no_grad():
        with pipe.progress_bar(total=max(0, len(timesteps) - int(step_index))) as progress_bar:
            while step_index < len(timesteps):
                if bool(getattr(pipe, "interrupt", False)):
                    step_index += 1
                    progress_bar.update()
                    continue

                t = timesteps[step_index]
                timestep_int = int(timesteps_int[step_index]) if int(step_index) < len(timesteps_int) else extract_timestep_int(t)
                if timestep_int is None:
                    raise ValueError(f"Could not resolve timestep for step_index={step_index}.")
                pipe._current_timestep = t
                pipe._cole_current_timestep_int = int(timestep_int)
                latents_before_step = latents
                latent_model_input = latents.to(transformer_dtype)
                timestep = _timestep_to_device(t, device=latent_model_input.device, batch_size=latents.shape[0])
                transformer_attention_kwargs = _attention_kwargs_with_timestep_int(attention_kwargs, int(timestep_int))
                prediction_reuse_decision = None
                if prediction_reuse is not None:
                    prediction_reuse_decision = prediction_reuse.decide(
                        step_index=int(step_index),
                        expected_shape=tuple(int(dim) for dim in latent_model_input.shape),
                        target_device=latents.device,
                        latent_model_input=latent_model_input,
                    )

                if (
                    prediction_reuse_decision is not None
                    and prediction_reuse_decision.use_cache
                    and prediction_reuse_decision.noise_pred is not None
                ):
                    noise_pred = prediction_reuse_decision.noise_pred
                    record_runtime_feedback(
                        pipe,
                        ExecFeedback(
                            backend_prepare_ms=0.0,
                            backend_run_ms=0.0,
                            total_attn_ms=0.0,
                            actual_density=0.0,
                            selected_units=0,
                            backend_name="prediction_reuse",
                            executed_plan_signature=f"prediction_reuse_{prediction_reuse_decision.reason}",
                            executed_plan_density=0.0,
                            backend_is_sparse=False,
                            backend_first_call=False,
                        ),
                    )
                else:
                    noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=dict(transformer_attention_kwargs),
                        return_dict=False,
                    )[0]
                    if bool(getattr(pipe, "do_classifier_free_guidance", False)):
                        noise_uncond = pipe.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=dict(transformer_attention_kwargs),
                            return_dict=False,
                        )[0]
                        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                    if prediction_reuse is not None:
                        reason = prediction_reuse_decision.reason if prediction_reuse_decision is not None else "compute"
                        prediction_reuse.record_compute(
                            noise_pred=noise_pred,
                            reason=reason,
                            step_index=int(step_index),
                            latent_model_input=latent_model_input,
                        )

                latents = pipe.scheduler.step(
                    noise_pred,
                    _timestep_to_device(t, device=latents.device),
                    latents,
                    return_dict=False,
                )[0]
                payload = _build_callback_payload(
                    tensor_inputs=tensor_inputs,
                    latents=latents,
                    latents_before_step=latents_before_step,
                    noise_pred=noise_pred,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                )
                callback_outputs = on_step_end(int(step_index), int(timestep_int), payload)
                callback_update = _normalize_step_callback_update(callback_outputs)
                if callback_update.prediction_reuse_policy_update is not None:
                    if prediction_reuse is None:
                        raise RuntimeError(
                            "Step callback emitted a prediction reuse update while the "
                            "prediction reuse runtime is disabled."
                        )
                    prediction_reuse.apply_policy_update(
                        callback_update.prediction_reuse_policy_update
                    )
                if callback_update.latents is not None:
                    latents = callback_update.latents
                if callback_update.prompt_embeds is not None:
                    prompt_embeds = callback_update.prompt_embeds
                if callback_update.negative_prompt_embeds is not None:
                    negative_prompt_embeds = callback_update.negative_prompt_embeds
                skip_step_count = max(0, int(callback_update.skip_step_count))
                updated_timesteps = getattr(pipe.scheduler, "timesteps", timesteps)
                if updated_timesteps is not timesteps or len(updated_timesteps) != len(timesteps) or int(skip_step_count) > 0:
                    timesteps = updated_timesteps
                    timesteps_int = snapshot_scheduler_timesteps(pipe.scheduler)

                progress_bar.update(min(1 + skip_step_count, len(timesteps) - step_index))
                step_index += 1 + skip_step_count

    pipe._current_timestep = None
    pipe._cole_current_timestep_int = None
    if output_type != "latent" and device.type == "cuda":
        torch.cuda.synchronize(device)
    denoise_loop_sec = time.perf_counter() - denoise_loop_start

    decode_post_start = time.perf_counter()
    released_modules: list[tuple[torch.nn.Module, torch.device]] = []
    decode_backend = "latent_passthrough"
    try:
        with torch.no_grad():
            if output_type != "latent":
                try:
                    vae_device = next(pipe.vae.parameters()).device
                except StopIteration:
                    vae_device = latents.device
                latents = latents.to(device=vae_device, dtype=pipe.vae.dtype)
                latents_mean = (
                    torch.tensor(pipe.vae.config.latents_mean)
                    .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
                    latents.device, latents.dtype
                )
                latents = latents / latents_std + latents_mean
                video, decode_backend = _decode_vae_latents_with_fallback(
                    pipe,
                    latents,
                    released_modules,
                    extra_modules=decode_offload_modules,
                    output_type=output_type,
                )
                # The exporter needs CPU frames. Release GPU frames before reloading offloaded weights.
                if released_modules and isinstance(video, torch.Tensor) and video.is_cuda:
                    video = video.cpu()
                if vae_device.type == "cuda":
                    torch.cuda.synchronize(vae_device)
            else:
                video = latents
    finally:
        _restore_offloaded_modules(released_modules)

    decode_post_sec = time.perf_counter() - decode_post_start
    end_to_end_sec = float(denoise_loop_sec + decode_post_sec)
    pipe.maybe_free_model_hooks()

    if not return_dict:
        return (video,)
    return SimpleNamespace(
        frames=video,
        denoise_loop_sec=float(denoise_loop_sec),
        decode_post_sec=float(decode_post_sec),
        end_to_end_sec=end_to_end_sec,
        decode_backend=str(decode_backend),
        prediction_reuse=(prediction_reuse.to_dict() if prediction_reuse is not None else None),
    )


def _build_callback_payload(
    *,
    tensor_inputs: Iterable[str],
    latents: torch.Tensor,
    latents_before_step: torch.Tensor,
    noise_pred: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
) -> Dict[str, Any]:
    available = {
        "latents": latents,
        "latents_before_step": latents_before_step,
        "noise_pred": noise_pred,
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
    }
    payload: Dict[str, Any] = {}
    for name in tensor_inputs:
        if name not in available:
            raise ValueError(f"Unsupported callback tensor input for Wan runtime: {name}")
        payload[str(name)] = available[name]
    return payload


def _normalize_step_callback_update(callback_outputs: Any) -> StepCallbackUpdate:
    if callback_outputs is None:
        return StepCallbackUpdate()
    if isinstance(callback_outputs, StepCallbackUpdate):
        return callback_outputs
    if isinstance(callback_outputs, dict):
        return StepCallbackUpdate(
            latents=callback_outputs.get("latents"),
            prompt_embeds=callback_outputs.get("prompt_embeds"),
            negative_prompt_embeds=callback_outputs.get("negative_prompt_embeds"),
            skip_step_count=int(callback_outputs.get("skip_step_count", 0)),
            prediction_reuse_policy_update=(
                dict(callback_outputs["prediction_reuse_policy_update"])
                if isinstance(callback_outputs.get("prediction_reuse_policy_update"), dict)
                else None
            ),
        )
    raise TypeError(f"Unsupported callback update type: {type(callback_outputs)}")


def publish_mask_plan(model_or_pipe, mask_plan) -> None:
    setattr(model_or_pipe, "cole_mask_plan", mask_plan)


def stamp_step_wall_ms(model_or_pipe, *, anchor_sec: float, now_sec: float | None = None) -> float:
    resolved_now = time.perf_counter() if now_sec is None else float(now_sec)
    feedback = getattr(model_or_pipe, "cole_exec_feedback", None)
    if isinstance(feedback, ExecFeedback):
        feedback.step_wall_ms = max(0.0, (resolved_now - float(anchor_sec)) * 1000.0)
    return resolved_now


def record_runtime_feedback(model_or_pipe, feedback: ExecFeedback) -> None:
    current = getattr(model_or_pipe, "cole_exec_feedback", None)
    count = int(getattr(model_or_pipe, "cole_exec_feedback_count", 0))
    if not isinstance(current, ExecFeedback):
        setattr(model_or_pipe, "cole_exec_feedback", feedback)
        setattr(model_or_pipe, "cole_exec_feedback_count", 1)
        return
    current_count = max(count, 1)
    current_attention_proxy = current.attention_proxy if isinstance(current.attention_proxy, dict) else None
    feedback_attention_proxy = feedback.attention_proxy if isinstance(feedback.attention_proxy, dict) else None
    if current_attention_proxy and feedback_attention_proxy:
        current.attention_proxy = _merge_attention_proxy_feedback(
            resolve_attention_proxy_payload(current_attention_proxy),
            resolve_attention_proxy_payload(feedback_attention_proxy),
            current_count=current_count,
        )
    elif current_attention_proxy:
        current.attention_proxy = current_attention_proxy
    elif feedback_attention_proxy:
        current.attention_proxy = feedback_attention_proxy
    else:
        current.attention_proxy = None
    if isinstance(feedback.attention_prob_patches, dict):
        if not isinstance(current.attention_prob_patches, dict):
            current.attention_prob_patches = {}
        current.attention_prob_patches.update(clone_attention_prob_patches(feedback.attention_prob_patches) or {})
    current.attention_sampling_elapsed_sec = float(
        current.attention_sampling_elapsed_sec + feedback.attention_sampling_elapsed_sec
    )
    current.attention_sampling_updates = int(
        current.attention_sampling_updates + feedback.attention_sampling_updates
    )
    current.summary_build_ms = float(current.summary_build_ms + feedback.summary_build_ms)
    current.backend_prepare_ms = float(current.backend_prepare_ms + feedback.backend_prepare_ms)
    current.backend_run_ms = float(current.backend_run_ms + feedback.backend_run_ms)
    current.total_attn_ms = float(current.total_attn_ms + feedback.total_attn_ms)
    current.step_wall_ms = max(float(current.step_wall_ms), float(feedback.step_wall_ms))
    current.actual_density = float(
        (current.actual_density * current_count + feedback.actual_density) / (current_count + 1)
    )
    current.executed_plan_density = float(
        (
            current.executed_plan_density * current_count
            + feedback.executed_plan_density
        )
        / (current_count + 1)
    )
    current.selected_units = max(int(current.selected_units), int(feedback.selected_units))
    current.latency_ema = float(feedback.latency_ema)
    current.density_ema = float(feedback.density_ema)
    current.backend_name = str(feedback.backend_name or current.backend_name)
    current.executed_plan_signature = str(feedback.executed_plan_signature or current.executed_plan_signature)
    current.backend_is_sparse = bool(feedback.backend_is_sparse or current.backend_is_sparse)
    current.decision_reused = bool(feedback.decision_reused or current.decision_reused)
    current.backend_first_call = bool(feedback.backend_first_call or current.backend_first_call)
    setattr(model_or_pipe, "cole_exec_feedback_count", count + 1)


def _merge_attention_proxy_feedback(
    current: dict[str, object] | None,
    incoming: dict[str, object] | None,
    *,
    current_count: int,
) -> dict[str, float] | None:
    if not current and not incoming:
        return None
    if not current:
        return dict(incoming or {})
    if not incoming:
        return dict(current)
    out = dict(current)
    count = max(1, int(current_count))
    for key, value in incoming.items():
        try:
            incoming_value = float(value)
        except (TypeError, ValueError):
            continue
        if str(key).endswith("_samples"):
            out[str(key)] = float(out.get(str(key), 0.0)) + incoming_value
            continue
        previous_value = out.get(str(key))
        if previous_value is None:
            out[str(key)] = incoming_value
        else:
            out[str(key)] = float((float(previous_value) * count + incoming_value) / float(count + 1))
    return out


def consume_runtime_feedback(model_or_pipe):
    value = getattr(model_or_pipe, "cole_exec_feedback", None)
    setattr(model_or_pipe, "cole_exec_feedback", None)
    setattr(model_or_pipe, "cole_exec_feedback_count", 0)
    if isinstance(value, ExecFeedback):
        value.attention_proxy = resolve_attention_proxy_payload(value.attention_proxy)
        value.attention_prob_patches = clone_attention_prob_patches(value.attention_prob_patches)
    return value


def reset_cole_runtime_state(model_or_pipe) -> None:
    setattr(model_or_pipe, "cole_mask_plan", None)
    setattr(model_or_pipe, "cole_exec_feedback", None)
    setattr(model_or_pipe, "cole_exec_feedback_count", 0)
    for attr_name in ("transformer", "transformer_2", "vae", "text_encoder", "text_encoder_2"):
        module = getattr(model_or_pipe, attr_name, None)
        clear_cache = getattr(module, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()
    pipe_clear_cache = getattr(model_or_pipe, "clear_cache", None)
    if callable(pipe_clear_cache):
        pipe_clear_cache()


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cleanup_decode_memory() -> None:
    gc.collect()
    _empty_cuda_cache()


def _iter_decode_offload_modules(pipe, extra_modules: Iterable[torch.nn.Module] | None = None):
    for attr_name in ("transformer", "text_encoder", "transformer_2", "text_encoder_2"):
        module = getattr(pipe, attr_name, None)
        if isinstance(module, torch.nn.Module):
            yield module
    for module in extra_modules or ():
        if isinstance(module, torch.nn.Module):
            yield module


def _offload_modules_for_decode(
    pipe,
    *,
    extra_modules: Iterable[torch.nn.Module] | None = None,
) -> list[tuple[torch.nn.Module, torch.device]]:
    released: list[tuple[torch.nn.Module, torch.device]] = []
    seen: set[int] = set()
    for module in _iter_decode_offload_modules(pipe, extra_modules):
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        try:
            original_device = next(module.parameters()).device
        except StopIteration:
            continue
        if original_device.type != "cuda":
            continue
        module.to("cpu")
        released.append((module, original_device))
    _cleanup_decode_memory()
    return released


def _restore_offloaded_modules(released_modules: list[tuple[torch.nn.Module, torch.device]]) -> None:
    for module, original_device in released_modules:
        try:
            module.to(original_device)
        except torch.OutOfMemoryError as exc:
            raise torch.OutOfMemoryError(
                f"Could not restore {type(module).__name__} to {original_device} after VAE decode."
            ) from exc


def _supports_wan_spatial_tiled_decode(vae) -> bool:
    return callable(getattr(vae, "decode", None)) and bool(
        getattr(vae, "cole_supports_spatial_tiling", hasattr(vae, "clear_cache"))
    )


def _tile_starts(size: int, tile_size: int, overlap_factor: float) -> list[int]:
    if tile_size >= size:
        return [0]
    stride = max(1, int(round(tile_size * max(0.0, 1.0 - float(overlap_factor)))))
    stride = min(stride, tile_size)
    starts: list[int] = []
    start = 0
    while True:
        starts.append(int(start))
        if start + tile_size >= size:
            break
        next_start = min(start + stride, size - tile_size)
        if next_start == start:
            break
        start = next_start
    return starts


def _blend_weights_1d(
    length: int,
    *,
    start: int,
    total_length: int,
    blend_extent: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights = torch.ones(length, device=device, dtype=dtype)
    if blend_extent <= 0:
        return weights
    if start > 0:
        extent = min(blend_extent, length)
        weights[:extent] = torch.linspace(0.0, 1.0, extent, device=device, dtype=dtype)
    if start + length < total_length:
        extent = min(blend_extent, length)
        tail = torch.linspace(1.0, 0.0, extent, device=device, dtype=dtype)
        weights[-extent:] = torch.minimum(weights[-extent:], tail)
    return weights


def _spatial_tile_candidates(vae, latents: torch.Tensor) -> list[tuple[int, int]]:
    custom = getattr(vae, "cole_tile_latent_hw_candidates", None)
    if custom is None:
        custom = ((32, 32), (24, 24), (16, 16))
    candidates: list[tuple[int, int]] = []
    full_hw = (int(latents.shape[-2]), int(latents.shape[-1]))
    for raw_h, raw_w in custom:
        tile_h = min(max(1, int(raw_h)), full_hw[0])
        tile_w = min(max(1, int(raw_w)), full_hw[1])
        tile_hw = (tile_h, tile_w)
        if tile_hw == full_hw or tile_hw in candidates:
            continue
        candidates.append(tile_hw)
    return candidates


def _decode_wan_vae_spatially_tiled(
    vae,
    latents: torch.Tensor,
    *,
    tile_latent_hw: tuple[int, int],
    overlap_factor: float = 0.25,
    output_device: str | torch.device,
) -> torch.Tensor:
    if latents.ndim != 5:
        raise ValueError(f"Expected 5D video latents, got shape={tuple(latents.shape)}")

    latent_h = int(latents.shape[-2])
    latent_w = int(latents.shape[-1])
    tile_h = min(max(1, int(tile_latent_hw[0])), latent_h)
    tile_w = min(max(1, int(tile_latent_hw[1])), latent_w)
    if tile_h >= latent_h and tile_w >= latent_w:
        raise torch.OutOfMemoryError("Spatial tiling does not reduce the decode footprint for this latent shape.")

    y_starts = _tile_starts(latent_h, tile_h, overlap_factor)
    x_starts = _tile_starts(latent_w, tile_w, overlap_factor)
    output_device = torch.device(output_device)
    acc: torch.Tensor | None = None
    weight_acc: torch.Tensor | None = None
    scale_h: int | None = None
    scale_w: int | None = None
    blend_h_px = 0
    blend_w_px = 0

    for y0 in y_starts:
        for x0 in x_starts:
            tile_latents = latents[:, :, :, y0 : y0 + tile_h, x0 : x0 + tile_w]
            decoded_tile = vae.decode(tile_latents, return_dict=False)[0]
            tile = decoded_tile.detach().to(device=output_device, dtype=torch.float32)
            if acc is None or weight_acc is None or scale_h is None or scale_w is None:
                scale_h = max(1, int(round(tile.shape[-2] / max(1, tile_latents.shape[-2]))))
                scale_w = max(1, int(round(tile.shape[-1] / max(1, tile_latents.shape[-1]))))
                full_h = latent_h * scale_h
                full_w = latent_w * scale_w
                acc = torch.zeros(
                    (*tile.shape[:-2], full_h, full_w),
                    device=output_device,
                    dtype=torch.float32,
                )
                weight_acc = torch.zeros((1, 1, 1, full_h, full_w), device=output_device, dtype=torch.float32)
                blend_h_px = int(round(tile_h * float(overlap_factor))) * scale_h
                blend_w_px = int(round(tile_w * float(overlap_factor))) * scale_w
            assert acc is not None and weight_acc is not None and scale_h is not None and scale_w is not None
            out_y0 = int(y0 * scale_h)
            out_x0 = int(x0 * scale_w)
            out_y1 = out_y0 + int(tile.shape[-2])
            out_x1 = out_x0 + int(tile.shape[-1])
            weights_h = _blend_weights_1d(
                int(tile.shape[-2]),
                start=out_y0,
                total_length=int(acc.shape[-2]),
                blend_extent=blend_h_px,
                device=output_device,
                dtype=torch.float32,
            )
            weights_w = _blend_weights_1d(
                int(tile.shape[-1]),
                start=out_x0,
                total_length=int(acc.shape[-1]),
                blend_extent=blend_w_px,
                device=output_device,
                dtype=torch.float32,
            )
            tile_weights = weights_h.view(1, 1, 1, -1, 1) * weights_w.view(1, 1, 1, 1, -1)
            acc[..., out_y0:out_y1, out_x0:out_x1] += tile * tile_weights
            weight_acc[..., out_y0:out_y1, out_x0:out_x1] += tile_weights
            del decoded_tile, tile, tile_weights

    if acc is None or weight_acc is None:
        raise RuntimeError("Spatial tiled decode did not produce any tiles.")
    return acc.div_(weight_acc.clamp_min_(1e-6))


def _decode_vae_latents_with_fallback(
    pipe,
    latents: torch.Tensor,
    released_modules: list[tuple[torch.nn.Module, torch.device]],
    *,
    extra_modules: Iterable[torch.nn.Module] = (),
    output_type: str | None = None,
) -> tuple[Any, str]:
    vae = pipe.vae
    try:
        vae_device = next(vae.parameters()).device
    except StopIteration:
        vae_device = latents.device

    def postprocess(video: torch.Tensor) -> Any:
        if output_type is None:
            return video
        return pipe.video_processor.postprocess_video(video, output_type=output_type)

    if vae_device.type != "cuda":
        return postprocess(vae.decode(latents, return_dict=False)[0]), "cpu_full"

    try:
        return postprocess(vae.decode(latents, return_dict=False)[0]), "gpu_full"
    except torch.OutOfMemoryError:
        pass

    # Release cached VAE activations and other GPU weights only when decode needs the space.
    clear_vae_cache = getattr(vae, "clear_cache", None)
    if callable(clear_vae_cache):
        clear_vae_cache()
    released_modules.extend(_offload_modules_for_decode(pipe, extra_modules=extra_modules))
    try:
        return postprocess(vae.decode(latents, return_dict=False)[0]), "gpu_full_retry"
    except torch.OutOfMemoryError:
        pass

    if callable(clear_vae_cache):
        clear_vae_cache()
    _cleanup_decode_memory()

    if _supports_wan_spatial_tiled_decode(vae):
        tile_candidates = _spatial_tile_candidates(vae, latents)
        overlap = float(getattr(vae, "cole_tile_overlap_factor", 0.25))
        for output_device, backend in ((latents.device, "gpu_tiled"), ("cpu", "gpu_tiled_cpu_blend")):
            for tile_hw in tile_candidates:
                try:
                    return postprocess(
                        _decode_wan_vae_spatially_tiled(
                            vae,
                            latents,
                            tile_latent_hw=tile_hw,
                            overlap_factor=overlap,
                            output_device=output_device,
                        )
                    ), backend
                except torch.OutOfMemoryError:
                    pass
                if callable(clear_vae_cache):
                    clear_vae_cache()
                _cleanup_decode_memory()

    vae.to("cpu")
    released_modules.append((vae, vae_device))
    _cleanup_decode_memory()
    return postprocess(vae.decode(latents.to("cpu"), return_dict=False)[0]), "cpu_fallback"
