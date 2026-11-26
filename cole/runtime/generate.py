from __future__ import annotations

import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from cole.common.io import write_json
from cole.common.seed import configure_torch_runtime
from cole.common.video_utils import save_video_tensor
from cole.core.program import StrategyProgram
from cole.core.reuse_presets import build_reuse_config
from cole.runtime.backends.sparse_backend import load_sparse_kernel
from cole.runtime.controller_policy import ControllerPolicy
from cole.runtime.inference_utils import (
    build_generator,
    encode_prompt_once,
    infer_highres_latent_hw,
    prepare_initial_latents,
    warmup_backend_profiles,
)
from cole.runtime.online_controller import OnlineController
from cole.runtime.prompt_embedding import mean_pool_prompt_embedding
from cole.runtime.resolution_transition import (
    ResolutionTransition,
    ResolutionTransitionConfig,
    ResolutionTransitionRuntime,
    maybe_apply_resolution_transition,
)
from cole.runtime.wan_adapter import (
    StepCallbackUpdate,
    build_scheduler_timesteps,
    load_wan_pipeline,
    publish_mask_plan,
    reset_cole_runtime_state,
    run_pipe_with_step_callback,
    snapshot_scheduler_timesteps,
    stamp_step_wall_ms,
)

HEIGHT, WIDTH, FRAMES, FPS = 480, 832, 81, 16
TOTAL_STEPS = 50
GUIDANCE_SCALE = 5.0
LOW_RES_SCALE = 0.75
INITIAL_SWITCH_STEP = 21
WARMUP_PROFILES = ("dense_full", "sparse_safe", "sparse_mid", "sparse_fast")
TRANSITION_CONFIG = ResolutionTransitionConfig(
    scheduler_flow_shift=7.0, post_switch_flow_shift=9.0,
    corrector_disable_pre_steps=0, corrector_disable_post_steps=0,
)


class WanGenerator:
    """Keep the pipeline and frozen controller resident across prompts."""

    def __init__(
        self, model_dir: Path, *, program: StrategyProgram, checkpoint_path: Path, device: torch.device,
        variant: str = "cole", reuse_checkpoint: Path | None = None,
    ) -> None:
        from cole.runtime.install_wan_processor import install_cole_wan_processor

        if variant not in ("cole", "cole-l", "cole-l-no-reuse"):
            raise ValueError(f"Unknown CoLE variant {variant!r}.")
        if (variant == "cole-l") != (reuse_checkpoint is not None):
            raise ValueError("The CoLE-L variant requires a six-action reuse checkpoint.")
        self.variant = variant
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        load_sparse_kernel(device)
        self.device = device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        self.dtype = torch.bfloat16
        self.policy = ControllerPolicy(
            program, checkpoint_path=checkpoint_path, device=device,
            total_steps=TOTAL_STEPS, switch_step=INITIAL_SWITCH_STEP,
            use_program=variant == "cole", reuse_checkpoint=reuse_checkpoint,
        )
        self.pipe = load_wan_pipeline(str(model_dir), dtype=self.dtype)
        self.pipe.to(device)
        install_cole_wan_processor(self.pipe, height=HEIGHT, width=WIDTH, num_frames=FRAMES)

    def generate(self, *, prompt: str, seed: int, output_dir: Path) -> dict:
        from cole.runtime.wan_processor import refresh_runtime_env_cache

        configure_torch_runtime(seed=seed, device_str=str(self.device), use_tf32=True)
        pipe = self.pipe
        reset_cole_runtime_state(pipe)
        os.environ.update({
            "COLE_ATTENTION_PROXY_ENABLE": "0",
            "COLE_ATTENTION_PROB_PATCHES_ENABLE": "1",
            "COLE_ATTENTION_PROXY_STEP_INTERVAL": "1",
            "COLE_ATTENTION_PROXY_LAYERS": "0,5,10,15,20,25",
            "COLE_ATTENTION_PROXY_HEADS": "2",
            "COLE_ATTENTION_PROXY_QUERIES": "64",
            "COLE_ATTENTION_PROXY_KEYS": "512",
        })

        def set_attention_sample_timesteps(schedule: list[int]) -> list[int]:
            sampled = list(dict.fromkeys(schedule[step] for step in self.policy.attention_sample_steps))
            os.environ["COLE_ATTENTION_SAMPLE_TIMESTEPS"] = ",".join(map(str, sampled))
            refresh_runtime_env_cache()
            return sampled

        encode_start = time.perf_counter()
        prompt_embeds, negative_embeds = encode_prompt_once(
            pipe, prompt=prompt, negative_prompt=None, device=self.device, guidance_scale=GUIDANCE_SCALE,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        prompt_encoding_sec = time.perf_counter() - encode_start
        prompt_embedding = mean_pool_prompt_embedding(prompt_embeds)
        highres_hw = infer_highres_latent_hw(pipe, height=HEIGHT, width=WIDTH)
        controller = OnlineController(
            policy=self.policy, prompt_embedding=prompt_embedding, highres_hw=highres_hw,
            low_res_scale=LOW_RES_SCALE,
        )
        generator = build_generator(str(self.device), seed)
        runtime_schedule = build_scheduler_timesteps(
            pipe.scheduler, num_inference_steps=TOTAL_STEPS, device=self.device,
            flow_shift=TRANSITION_CONFIG.scheduler_flow_shift, disable_corrector=[],
        )
        attention_sample_timesteps = set_attention_sample_timesteps(runtime_schedule)
        latents = prepare_initial_latents(
            pipe, height=HEIGHT, width=WIDTH, num_frames=FRAMES, dtype=self.dtype,
            device=self.device, generator=generator, low_res_scale=LOW_RES_SCALE, direct_low_res_noise=True,
        )
        prewarm_start = time.perf_counter()
        decision = controller.prewarm(pipe, timestep=runtime_schedule[0], latent=latents)
        prewarm_sec = time.perf_counter() - prewarm_start
        first_latents = decision.latents
        warmups = warmup_backend_profiles(
            pipe, latents=first_latents, timestep=runtime_schedule[0], prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds, guidance_scale=GUIDANCE_SCALE,
            profile_names=list(WARMUP_PROFILES), reference_hw=highres_hw,
        )
        publish_mask_plan(pipe, decision.mask_plan)
        transition = ResolutionTransition(TRANSITION_CONFIG)
        trace = []

        def on_step_end(step: int, timestep: int, payload: dict) -> StepCallbackUpdate:
            nonlocal decision, runtime_schedule, wall_anchor, attention_sample_timesteps
            stamp_step_wall_ms(pipe, anchor_sec=wall_anchor)
            executed = decision
            decision = controller.step(pipe, timestep=timestep, latent=payload["latents"])
            application = maybe_apply_resolution_transition(
                resolution_event=executed.resolution_event,
                transition=transition,
                runtime=ResolutionTransitionRuntime(
                    latents_before_step=payload["latents_before_step"], latents_after_step=decision.latents,
                    noise_pred=payload["noise_pred"], timestep=timestep, step_index=step,
                    timesteps=runtime_schedule, scheduler=pipe.scheduler,
                    target_highres_shape=highres_hw, generator=generator,
                ),
            )
            transition_meta = application.metadata
            if transition_meta is not None:
                runtime_schedule = snapshot_scheduler_timesteps(pipe.scheduler)
                attention_sample_timesteps = set_attention_sample_timesteps(runtime_schedule)
            row = executed.trace_row(step=step, timestep=timestep)
            row["latent_hw_after_step"] = list(application.latents.shape[-2:])
            if decision.executed_feedback is not None:
                row["executed_feedback"] = decision.executed_feedback.to_dict()
            if transition_meta is not None:
                row["transition"] = transition_meta.to_dict()
            trace.append(row)
            wall_anchor = time.perf_counter()
            return StepCallbackUpdate(
                latents=application.latents,
                skip_step_count=application.skip_step_count,
                prediction_reuse_policy_update=self.policy.consume_reuse_update(),
            )

        wall_anchor = time.perf_counter()
        output = run_pipe_with_step_callback(
            pipe=pipe, on_step_end=on_step_end,
            tensor_inputs=("latents", "latents_before_step", "noise_pred"),
            call_kwargs={
                "height": HEIGHT, "width": WIDTH, "num_frames": FRAMES,
                "num_inference_steps": TOTAL_STEPS, "guidance_scale": GUIDANCE_SCALE,
                "output_type": "pt", "generator": generator, "latents": first_latents,
                "prompt_embeds": prompt_embeds, "negative_prompt_embeds": negative_embeds,
                "cole_prediction_reuse_policy_enabled": self.policy.reuse_readout is not None,
                "cole_prediction_reuse": build_reuse_config(
                    self.policy.initial_reuse_action, total_steps=TOTAL_STEPS, switch_step=INITIAL_SWITCH_STEP,
                    force_full_steps=self.policy.attention_sample_steps,
                ),
            },
        )
        metadata = {
            "variant": self.variant, "prompt": prompt, "seed": seed,
            "height": HEIGHT, "width": WIDTH, "num_frames": FRAMES, "output_fps": FPS,
            "num_inference_steps": TOTAL_STEPS, "guidance_scale": GUIDANCE_SCALE,
            "low_res_scale": LOW_RES_SCALE, "transition_config": TRANSITION_CONFIG.to_dict(),
            "switch_step": controller.resolution.state.switch_step,
            "switch_probabilities": self.policy.switch_probabilities,
            "program_execution": self.policy.strategy.to_dict() if self.policy.strategy is not None else None,
            "prediction_reuse_action": self.policy.reuse_action,
            "prediction_reuse_probabilities": self.policy.reuse_probabilities,
            "prediction_reuse": output.prediction_reuse,
            "attention_sample_steps": self.policy.attention_sample_steps,
            "attention_sample_timesteps": attention_sample_timesteps,
            "prompt_encoding_sec": prompt_encoding_sec, "prewarm_sec": prewarm_sec,
            "profile_warmup_rows": warmups, "warmup_sec": sum(row["elapsed_sec"] for row in warmups),
            "denoise_loop_sec": output.denoise_loop_sec, "decode_post_sec": output.decode_post_sec,
            "decode_backend": output.decode_backend, "end_to_end_sec": output.end_to_end_sec,
            "prompt_inclusive_generation_sec": prompt_encoding_sec + output.end_to_end_sec,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output_dir) as temp_dir:
            staged = Path(temp_dir)
            export_start = time.perf_counter()
            save_video_tensor(output.frames, staged / "output.mp4", fps=FPS)
            metadata["video_write_sec"] = time.perf_counter() - export_start
            write_json(staged / "decision_trace.json", trace)
            write_json(staged / "run_metadata.json", metadata)
            # Publish the completion marker only after both artifacts are in place.
            (output_dir / "run_metadata.json").unlink(missing_ok=True)
            for name in ("output.mp4", "decision_trace.json", "run_metadata.json"):
                (staged / name).replace(output_dir / name)
        return metadata
