from __future__ import annotations

from dataclasses import dataclass

from .types import BaseSparsePlan


@dataclass
class BaseSchedule:
    """Simple phase-aware base schedule."""

    lowres_dense_last_step: int = 9
    switch_step: int = 10
    highres_safe_last_step: int = 31
    lowres_profile: str = "dense_full"
    post_switch_profile: str = "sparse_safe"
    late_profile: str = "sparse_mid"
    # Kept for backward compatibility in callers that still pass scheduler timesteps.
    early_progress: float = 0.20
    mid_progress: float = 0.65
    highres_switch_progress: float = 0.55

    def resolve(
        self,
        *,
        timestep: int,
        total_steps: int,
        resolution_phase: str,
        step_index: int | None = None,
    ) -> BaseSparsePlan:
        resolved_step = self._resolve_step_index(
            timestep=int(timestep),
            total_steps=int(total_steps),
            step_index=step_index,
        )
        if resolved_step is not None:
            if resolved_step <= int(self.lowres_dense_last_step):
                default_sparse_profile = str(self.lowres_profile)
            elif resolved_step <= int(self.highres_safe_last_step):
                default_sparse_profile = str(self.post_switch_profile)
            else:
                default_sparse_profile = str(self.late_profile)
            promote_resolution = resolved_step >= int(self.switch_step) and str(resolution_phase).lower() != "high"
            return BaseSparsePlan(
                default_sparse_profile=default_sparse_profile,
                promote_resolution=bool(promote_resolution),
            )

        progress = 0.0
        if total_steps > 1:
            progress = float(timestep) / float(total_steps - 1)

        if progress <= self.early_progress:
            default_sparse_profile = "dense_full"
        elif progress <= self.mid_progress:
            default_sparse_profile = "sparse_safe"
        else:
            default_sparse_profile = "sparse_mid"

        promote_resolution = progress >= self.highres_switch_progress and str(resolution_phase).lower() != "high"
        return BaseSparsePlan(
            default_sparse_profile=default_sparse_profile,
            promote_resolution=bool(promote_resolution),
        )

    def _resolve_step_index(
        self,
        *,
        timestep: int,
        total_steps: int,
        step_index: int | None,
    ) -> int | None:
        if step_index is not None:
            upper = max(0, int(total_steps) - 1)
            return max(0, min(int(step_index), upper))
        if 0 <= int(timestep) < int(total_steps):
            return int(timestep)
        return None
