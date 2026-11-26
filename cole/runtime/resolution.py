from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class ResolutionState:
    current_phase: str = "low"
    highres_hw: tuple[int, int] | None = None
    last_event: str | None = None
    initial_lowres_applied: bool = False
    switch_step: int | None = None


class ResolutionManager:
    """One-way low-to-high resolution phase controller."""

    def __init__(
        self,
        low_res_scale: float = 0.5,
        highres_hw: tuple[int, int] | None = None,
        switch_min_step: int = 0,
    ) -> None:
        self.low_res_scale = float(low_res_scale)
        self.switch_min_step = max(0, int(switch_min_step))
        self.state = ResolutionState(highres_hw=highres_hw)

    def get_phase(self) -> str:
        return self.state.current_phase

    def set_highres_hw(self, highres_hw: tuple[int, int]) -> None:
        self.state.highres_hw = (int(highres_hw[0]), int(highres_hw[1]))

    def can_switch(self, *, step_index: int) -> bool:
        if self.low_res_scale >= 1.0:
            return False
        if str(self.state.current_phase).lower() == "high":
            return False
        return int(step_index) >= int(self.switch_min_step)

    def prepare_initial_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, str | None]:
        if latent.dim() not in (4, 5):
            return latent, None
        if self.state.initial_lowres_applied:
            return latent, None
        self._remember_highres(latent)
        updated = latent
        event: str | None = None
        if self.state.current_phase == "low":
            if self.low_res_scale < 1.0:
                current_hw = (int(latent.shape[-2]), int(latent.shape[-1]))
                if self.state.highres_hw is not None and current_hw != tuple(self.state.highres_hw):
                    event = "sampled_lowres_noise"
                else:
                    updated = self._resize(updated, target_hw=self._resolve_lowres_hw(current_hw))
                    event = "force_lowres_downsample"
            else:
                self.state.current_phase = "high"
                event = "start_highres"
        self.state.initial_lowres_applied = True
        self.state.last_event = event
        return updated, event

    def apply(self, latent: torch.Tensor, *, a_res: int, step_index: int, total_steps: int) -> tuple[torch.Tensor, str | None]:
        del total_steps
        if latent.dim() not in (4, 5):
            return latent, None
        event: str | None = None
        updated = latent
        if step_index == 0 and not self.state.initial_lowres_applied:
            updated, event = self.prepare_initial_latent(updated)
        else:
            self._remember_highres(updated)
        if int(a_res) >= 1 and self.can_switch(step_index=int(step_index)):
            self.state.current_phase = "high"
            self.state.switch_step = int(step_index)
            event = "switch_to_highres"
        self.state.last_event = event
        return updated, event

    def _resolve_lowres_hw(self, current_hw: tuple[int, int]) -> tuple[int, int]:
        reference_hw = self.state.highres_hw or current_hw
        target_h = max(1, int(reference_hw[0] * self.low_res_scale))
        target_w = max(1, int(reference_hw[1] * self.low_res_scale))
        return target_h, target_w

    def _remember_highres(self, latent: torch.Tensor) -> None:
        if self.state.highres_hw is None:
            self.state.highres_hw = (int(latent.shape[-2]), int(latent.shape[-1]))

    def _resize(
        self,
        latent: torch.Tensor,
        *,
        target_hw: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        if target_hw is None:
            return latent
        video, squeezed = self._as_video(latent)
        batch, channels, frames, _, _ = video.shape
        flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, video.shape[-2], video.shape[-1])
        resized = F.interpolate(flat.float(), size=target_hw, mode="bilinear", align_corners=False)
        resized = resized.to(dtype=latent.dtype)
        resized = resized.reshape(batch, frames, channels, int(target_hw[0]), int(target_hw[1])).permute(0, 2, 1, 3, 4)
        if squeezed:
            return resized.squeeze(2)
        return resized

    def _as_video(self, latent: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if latent.dim() == 5:
            return latent, False
        if latent.dim() == 4:
            return latent.unsqueeze(2), True
        raise ValueError(f"Unsupported latent shape for resolution operations: {tuple(latent.shape)}")
