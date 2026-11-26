from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from cole.runtime.wan_adapter import build_scheduler_timesteps, get_scheduler_flow_shift

RESOLUTION_TRANSITION_NAME = "resolution_switch"
TRANSITION_NOISE_FRESH = "fresh"


@dataclass(frozen=True)
class ResolutionTransitionConfig:
    resolution_transition: str = RESOLUTION_TRANSITION_NAME
    timestep_skip: int = 0
    clean_rebuild_strength: float = 1.0
    scheduler_flow_shift: float = 8.0
    post_switch_flow_shift: float = 10.0
    corrector_disable_pre_steps: int = 1
    corrector_disable_post_steps: int = 12

    def __post_init__(self) -> None:
        if str(self.resolution_transition) != RESOLUTION_TRANSITION_NAME:
            raise ValueError(
                f"Unsupported resolution_transition={self.resolution_transition!r}; "
                f"expected {RESOLUTION_TRANSITION_NAME!r}."
            )
        object.__setattr__(self, "timestep_skip", max(0, int(self.timestep_skip)))
        strength = min(1.0, max(0.0, float(self.clean_rebuild_strength)))
        object.__setattr__(self, "clean_rebuild_strength", strength)
        object.__setattr__(self, "scheduler_flow_shift", float(self.scheduler_flow_shift))
        object.__setattr__(self, "post_switch_flow_shift", float(self.post_switch_flow_shift))
        object.__setattr__(self, "corrector_disable_pre_steps", max(0, int(self.corrector_disable_pre_steps)))
        object.__setattr__(self, "corrector_disable_post_steps", max(0, int(self.corrector_disable_post_steps)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_aligned(self) -> bool:
        return True


@dataclass(frozen=True)
class ResolutionTransitionMetadata:
    resolution_transition: str = RESOLUTION_TRANSITION_NAME
    transition_applied: bool = False
    switch_step: int | None = None
    applies_to_step: int | None = None
    input_latent_hw: tuple[int, int] | None = None
    output_latent_hw: tuple[int, int] | None = None
    input_latent_shape: tuple[int, ...] | None = None
    output_latent_shape: tuple[int, ...] | None = None
    target_highres_hw: tuple[int, int] | None = None
    current_timestep: int | None = None
    handoff_timestep: int | None = None
    restart_timestep: int | None = None
    timestep_skip: int = 0
    clean_rebuild_strength: float = 1.0
    transition_noise_mode: str = TRANSITION_NOISE_FRESH
    scheduler_reset_applied: bool = False
    scheduler_flow_shift_before: float | None = None
    scheduler_flow_shift_after: float | None = None
    scheduler_snr_shift_scale_before: float | None = None
    scheduler_snr_shift_scale_after: float | None = None
    disable_corrector_steps: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolutionTransitionApplication:
    latents: torch.Tensor
    metadata: ResolutionTransitionMetadata | None = None
    skip_step_count: int = 0
    scheduler: Any | None = None


@dataclass(frozen=True)
class ResolutionTransitionRuntime:
    latents_before_step: torch.Tensor
    latents_after_step: torch.Tensor
    noise_pred: torch.Tensor
    timestep: int
    step_index: int
    timesteps: Sequence[int]
    scheduler: Any
    target_highres_shape: tuple[int, int] | tuple[int, int, int] | torch.Size
    generator: torch.Generator | None = None


class ResolutionTransition:
    """Scheduler-aware low-to-high resolution handoff for Wan runtimes."""

    def __init__(self, config: ResolutionTransitionConfig | None = None) -> None:
        self.config = config or ResolutionTransitionConfig()

    def apply(self, runtime: ResolutionTransitionRuntime) -> ResolutionTransitionApplication:
        target_hw, next_step_index, _, actual_skip, restart_timestep = self._resolve_switch_indices(runtime)
        handoff = self._build_handoff_latents(
            runtime,
            target_hw=target_hw,
            restart_timestep=restart_timestep,
        )
        flow_shift_before = get_scheduler_flow_shift(runtime.scheduler)
        disable_corrector = _build_disable_corrector_window(
            switch_step=int(runtime.step_index),
            total_steps=len(runtime.timesteps),
            pre_steps=int(self.config.corrector_disable_pre_steps),
            post_steps=int(self.config.corrector_disable_post_steps),
        )
        _ = build_scheduler_timesteps(
            runtime.scheduler,
            num_inference_steps=len(runtime.timesteps),
            device=runtime.latents_before_step.device,
            flow_shift=float(self.config.post_switch_flow_shift),
            disable_corrector=disable_corrector,
        )
        metadata = ResolutionTransitionMetadata(
            resolution_transition=self.config.resolution_transition,
            transition_applied=True,
            switch_step=int(runtime.step_index),
            applies_to_step=int(next_step_index + actual_skip),
            input_latent_hw=_latent_hw(runtime.latents_before_step),
            output_latent_hw=_latent_hw(handoff),
            input_latent_shape=tuple(int(dim) for dim in runtime.latents_before_step.shape),
            output_latent_shape=tuple(int(dim) for dim in handoff.shape),
            target_highres_hw=target_hw,
            current_timestep=int(runtime.timestep),
            handoff_timestep=restart_timestep,
            restart_timestep=restart_timestep,
            timestep_skip=actual_skip,
            clean_rebuild_strength=float(self.config.clean_rebuild_strength),
            scheduler_reset_applied=True,
            scheduler_flow_shift_before=flow_shift_before,
            scheduler_flow_shift_after=float(self.config.post_switch_flow_shift),
            disable_corrector_steps=tuple(int(step) for step in disable_corrector),
        )
        return ResolutionTransitionApplication(
            latents=handoff,
            metadata=metadata,
            skip_step_count=actual_skip,
        )

    def _resolve_switch_indices(
        self,
        runtime: ResolutionTransitionRuntime,
    ) -> tuple[tuple[int, int], int, int, int, int | None]:
        target_hw = _normalize_target_hw(runtime.target_highres_shape)
        next_step_index = min(int(runtime.step_index) + 1, len(runtime.timesteps) - 1)
        restart_step_index = min(
            next_step_index + int(self.config.timestep_skip),
            len(runtime.timesteps) - 1,
        )
        actual_skip = max(0, int(restart_step_index - next_step_index))
        restart_timestep = int(runtime.timesteps[restart_step_index]) if runtime.timesteps else None
        return target_hw, next_step_index, restart_step_index, actual_skip, restart_timestep

    def _build_handoff_latents(
        self,
        runtime: ResolutionTransitionRuntime,
        *,
        target_hw: tuple[int, int],
        restart_timestep: int | None,
    ) -> torch.Tensor:
        naive_highres = _resize_latent(runtime.latents_after_step, target_hw=target_hw)
        strength = float(self.config.clean_rebuild_strength)
        if strength <= 0.0 or restart_timestep is None:
            return naive_highres

        clean_lowres = rebuild_clean_latent(
            scheduler=runtime.scheduler,
            sample=runtime.latents_before_step,
            model_output=runtime.noise_pred,
            timestep=int(runtime.timestep),
        )
        clean_highres = _resize_latent(clean_lowres, target_hw=target_hw)
        highres_noise = _sample_noise_like(clean_highres, generator=runtime.generator)
        rebuilt_handoff = re_noise_latent_for_timestep(
            scheduler=runtime.scheduler,
            clean_latent=clean_highres,
            noise=highres_noise,
            timestep=restart_timestep,
        )
        return _blend_transition_handoff(
            naive_handoff=naive_highres,
            rebuilt_handoff=rebuilt_handoff,
            strength=strength,
        )


def maybe_apply_resolution_transition(
    *,
    resolution_event: str | None,
    transition: ResolutionTransition,
    runtime: ResolutionTransitionRuntime,
) -> ResolutionTransitionApplication:
    if str(resolution_event) != "switch_to_highres":
        return ResolutionTransitionApplication(latents=runtime.latents_after_step)
    return transition.apply(runtime)


def rebuild_clean_latent(
    *,
    scheduler,
    sample: torch.Tensor,
    model_output: torch.Tensor,
    timestep: int,
) -> torch.Tensor:
    prediction_type = str(getattr(getattr(scheduler, "config", None), "prediction_type", "epsilon"))
    if prediction_type == "flow_prediction":
        sigma_t = resolve_sigma_for_timestep(scheduler, timestep=int(timestep), sample=sample)
        return sample - sigma_t * model_output.to(device=sample.device, dtype=sample.dtype)
    if getattr(scheduler, "alphas_cumprod", None) is not None:
        alpha_prod_t = _resolve_alpha_prod_for_timestep(
            scheduler,
            timestep=int(timestep),
            sample=sample,
        )
        beta_prod_t = (1.0 - alpha_prod_t).clamp(min=0.0)
        model_output = model_output.to(device=sample.device, dtype=sample.dtype)
        if prediction_type == "epsilon":
            return (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt().clamp(min=1e-8)
        if prediction_type == "sample":
            return model_output
        if prediction_type == "v_prediction":
            return alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        raise ValueError(f"Unsupported scheduler prediction_type={prediction_type!r} for resolution transition.")
    sigma_t = resolve_sigma_for_timestep(scheduler, timestep=int(timestep), sample=sample)
    return sample - sigma_t * model_output.to(dtype=sample.dtype)




def re_noise_latent_for_timestep(
    *,
    scheduler,
    clean_latent: torch.Tensor,
    noise: torch.Tensor,
    timestep: int,
) -> torch.Tensor:
    timestep_tensor = torch.tensor([int(timestep)], device=clean_latent.device, dtype=_timestep_dtype(scheduler))
    scale_noise = getattr(scheduler, "scale_noise", None)
    if callable(scale_noise):
        return scale_noise(clean_latent, timestep_tensor, noise=noise)
    add_noise = getattr(scheduler, "add_noise", None)
    if callable(add_noise):
        return add_noise(clean_latent, noise, timestep_tensor)
    raise ValueError("Scheduler does not provide scale_noise() or add_noise() required for resolution transition.")


def resolve_sigma_for_timestep(
    scheduler,
    *,
    timestep: int,
    sample: torch.Tensor,
) -> torch.Tensor:
    sigmas = getattr(scheduler, "sigmas", None)
    schedule = getattr(scheduler, "timesteps", None)
    if sigmas is None:
        if getattr(scheduler, "alphas_cumprod", None) is None:
            raise ValueError(
                "Scheduler must expose sigmas or alphas_cumprod for resolution transition."
            )
        alpha_prod_t = _resolve_alpha_prod_for_timestep(
            scheduler,
            timestep=int(timestep),
            sample=sample,
        )
        return (1.0 - alpha_prod_t).clamp(min=0.0).sqrt()
    if schedule is None:
        raise ValueError("Scheduler must expose timesteps for resolution transition.")
    if isinstance(schedule, torch.Tensor):
        schedule_tensor = schedule.to(device=sample.device)
    else:
        schedule_tensor = torch.tensor(list(schedule), device=sample.device)
    if hasattr(scheduler, "index_for_timestep"):
        sigma_index = int(scheduler.index_for_timestep(schedule_tensor.new_tensor(int(timestep)), schedule_tensor))
    else:
        matches = (schedule_tensor == int(timestep)).nonzero(as_tuple=False)
        if matches.numel() == 0:
            raise ValueError(f"Could not resolve timestep {timestep} from scheduler schedule.")
        sigma_index = int(matches.reshape(-1)[0].item())
    sigma_value = sigmas[sigma_index]
    sigma_value = torch.as_tensor(sigma_value, device=sample.device, dtype=sample.dtype)
    while sigma_value.dim() < sample.dim():
        sigma_value = sigma_value.unsqueeze(-1)
    return sigma_value


def _resolve_alpha_prod_for_timestep(
    scheduler,
    *,
    timestep: int,
    sample: torch.Tensor,
) -> torch.Tensor:
    alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
    if alphas_cumprod is None:
        raise ValueError("Scheduler does not expose alphas_cumprod.")
    timestep_index = int(timestep)
    alpha_values = torch.as_tensor(alphas_cumprod)
    if timestep_index < 0 or timestep_index >= int(alpha_values.numel()):
        raise ValueError(
            f"Timestep {timestep_index} is outside alphas_cumprod with length {int(alpha_values.numel())}."
        )
    alpha_prod_t = alpha_values.reshape(-1)[timestep_index].to(
        device=sample.device,
        dtype=sample.dtype,
    )
    while alpha_prod_t.dim() < sample.dim():
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
    return alpha_prod_t


def _blend_transition_handoff(
    *,
    naive_handoff: torch.Tensor,
    rebuilt_handoff: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    if float(strength) >= 1.0:
        return rebuilt_handoff
    if float(strength) <= 0.0:
        return naive_handoff
    blended = torch.lerp(naive_handoff.float(), rebuilt_handoff.float(), float(strength))
    return blended.to(dtype=rebuilt_handoff.dtype)


def _build_disable_corrector_window(
    *,
    switch_step: int,
    total_steps: int,
    pre_steps: int,
    post_steps: int,
) -> list[int]:
    if pre_steps <= 0 and post_steps <= 0:
        return []
    start = max(0, int(switch_step) - int(pre_steps))
    end = min(int(total_steps), int(switch_step) + int(post_steps) + 1)
    return list(range(start, end))


def _normalize_target_hw(target_highres_shape: tuple[int, int] | tuple[int, int, int] | torch.Size) -> tuple[int, int]:
    if len(target_highres_shape) < 2:
        raise ValueError(f"Expected target shape with at least two dimensions, got {target_highres_shape}.")
    return int(target_highres_shape[-2]), int(target_highres_shape[-1])


def _resize_latent(latent: torch.Tensor, *, target_hw: tuple[int, int]) -> torch.Tensor:
    video, squeezed = _as_video(latent)
    target_size = (int(video.shape[-3]), int(target_hw[0]), int(target_hw[1]))
    resized = F.interpolate(video.float(), size=target_size, mode="trilinear", align_corners=False)
    resized = resized.to(dtype=latent.dtype)
    if squeezed:
        return resized.squeeze(2)
    return resized


def _sample_noise_like(latent: torch.Tensor, *, generator: torch.Generator | None) -> torch.Tensor:
    return torch.randn(
        tuple(latent.shape),
        generator=generator,
        device=latent.device,
        dtype=latent.dtype,
    )




def _latent_hw(latent: torch.Tensor) -> tuple[int, int]:
    return int(latent.shape[-2]), int(latent.shape[-1])


def _as_video(latent: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if latent.dim() == 5:
        return latent, False
    if latent.dim() == 4:
        return latent.unsqueeze(2), True
    raise ValueError(f"Unsupported latent shape for resolution transition: {tuple(latent.shape)}")


def _timestep_dtype(scheduler) -> torch.dtype:
    schedule = getattr(scheduler, "timesteps", None)
    if isinstance(schedule, torch.Tensor):
        return schedule.dtype
    return torch.float32
