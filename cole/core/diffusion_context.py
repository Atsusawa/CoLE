from __future__ import annotations

import torch


def build_diffusion_context(
    *,
    step_index: int | torch.Tensor,
    total_steps: int,
    noise_sigma: float | torch.Tensor,
    batch_size: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build the [B,2] diffusion context for the controller.

    noise_sigma should be the true scheduler sigma for the current timestep,
    resolved by the caller from scheduler.timesteps and scheduler.sigmas.
    """
    dtype = dtype or torch.float32
    batch = int(batch_size)
    if batch <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    denom = float(max(1, int(total_steps) - 1))
    step = torch.as_tensor(step_index, device=device, dtype=dtype).reshape(-1)
    sigma = torch.as_tensor(noise_sigma, device=device, dtype=dtype).reshape(-1)
    if int(step.numel()) == 1:
        step = step.expand(batch)
    elif int(step.numel()) != batch:
        raise ValueError(f"step_index must be scalar or have batch_size values, got shape={tuple(step.shape)}.")
    if int(sigma.numel()) == 1:
        sigma = sigma.expand(batch)
    elif int(sigma.numel()) != batch:
        raise ValueError(f"noise_sigma must be scalar or have batch_size values, got shape={tuple(sigma.shape)}.")
    progress = torch.clamp(step / denom, min=0.0, max=1.0)
    return torch.nan_to_num(torch.stack([progress, sigma], dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
