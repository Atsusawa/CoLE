from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from cole.core.types import SummaryBundle


@dataclass
class SummaryBuilderConfig:
    block_grid_h: int = 4
    block_grid_w: int = 4
    topk_stats: int = 3
    resolution_aware: bool = True


class SummaryBuilder:
    """Builds framewise and blockwise pooled summaries."""

    def __init__(self, config: SummaryBuilderConfig | None = None) -> None:
        self.config = config or SummaryBuilderConfig()

    def build(self, latent: torch.Tensor, *, reference_hw: tuple[int, int] | None = None) -> SummaryBundle:
        video = self._as_video(latent).float()
        latent_hw = (int(video.shape[-2]), int(video.shape[-1]))
        resolved_reference_hw = self._resolve_reference_hw(latent_hw, reference_hw=reference_hw)
        if bool(self.config.resolution_aware) and reference_hw is not None:
            block_grid_hw = self._resolve_block_grid_hw(latent_hw, reference_hw=resolved_reference_hw)
        else:
            block_grid_hw = (
                max(1, min(int(latent_hw[0]), int(self.config.block_grid_h))),
                max(1, min(int(latent_hw[1]), int(self.config.block_grid_w))),
            )
        approx_block_hw = (
            max(1, int(math.ceil(latent_hw[0] / max(1, block_grid_hw[0])))),
            max(1, int(math.ceil(latent_hw[1] / max(1, block_grid_hw[1])))),
        )
        frame_embeddings = self._frame_embeddings(video)
        block_embeddings = self._block_embeddings(video, block_grid_hw=block_grid_hw)
        query = frame_embeddings.mean(dim=0, keepdim=True)
        frame_scores = F.cosine_similarity(frame_embeddings, query, dim=-1)
        block_scores = F.cosine_similarity(block_embeddings, query.unsqueeze(1), dim=-1)
        stats = self._stats(
            frame_scores,
            block_scores,
            latent_hw=latent_hw,
            reference_hw=resolved_reference_hw,
            block_grid_hw=block_grid_hw,
            approx_block_hw=approx_block_hw,
        )
        return SummaryBundle(
            frame_embeddings=frame_embeddings,
            block_embeddings=block_embeddings,
            frame_scores=frame_scores,
            block_scores=block_scores,
            latent_hw=latent_hw,
            reference_hw=resolved_reference_hw,
            block_grid_hw=block_grid_hw,
            approx_block_hw=approx_block_hw,
            stats=stats,
        )

    def _as_video(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() == 5:
            return latent
        if latent.dim() == 4:
            return latent.unsqueeze(2)
        if latent.dim() == 3:
            return latent.unsqueeze(0).unsqueeze(2)
        raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")

    def _frame_embeddings(self, video: torch.Tensor) -> torch.Tensor:
        return video.mean(dim=(0, 3, 4)).transpose(0, 1).contiguous()

    def _resolve_reference_hw(
        self,
        latent_hw: tuple[int, int],
        *,
        reference_hw: tuple[int, int] | None,
    ) -> tuple[int, int]:
        if reference_hw is None:
            return latent_hw
        return (
            max(int(latent_hw[0]), int(reference_hw[0])),
            max(int(latent_hw[1]), int(reference_hw[1])),
        )

    def _resolve_block_grid_hw(
        self,
        latent_hw: tuple[int, int],
        *,
        reference_hw: tuple[int, int],
    ) -> tuple[int, int]:
        base_grid_h = max(1, int(self.config.block_grid_h))
        base_grid_w = max(1, int(self.config.block_grid_w))
        ref_block_h = max(1, int(math.ceil(reference_hw[0] / base_grid_h)))
        ref_block_w = max(1, int(math.ceil(reference_hw[1] / base_grid_w)))
        grid_h = max(1, min(int(latent_hw[0]), int(round(latent_hw[0] / ref_block_h))))
        grid_w = max(1, min(int(latent_hw[1]), int(round(latent_hw[1] / ref_block_w))))
        return (grid_h, grid_w)

    def _block_embeddings(self, video: torch.Tensor, *, block_grid_hw: tuple[int, int]) -> torch.Tensor:
        batch, channels, frames, _, _ = video.shape
        blocks: list[torch.Tensor] = []
        for frame_idx in range(frames):
            pooled = F.adaptive_avg_pool2d(video[:, :, frame_idx], block_grid_hw)
            pooled = pooled.mean(dim=0).permute(1, 2, 0).reshape(block_grid_hw[0] * block_grid_hw[1], channels)
            blocks.append(pooled)
        return torch.stack(blocks, dim=0)

    def _stats(
        self,
        frame_scores: torch.Tensor,
        block_scores: torch.Tensor,
        *,
        latent_hw: tuple[int, int],
        reference_hw: tuple[int, int],
        block_grid_hw: tuple[int, int],
        approx_block_hw: tuple[int, int],
    ) -> dict[str, float]:
        num_frames = int(frame_scores.numel())
        center = num_frames // 2
        near_mask = torch.zeros_like(frame_scores, dtype=torch.bool)
        if num_frames > 0:
            left = max(0, center - 1)
            right = min(num_frames, center + 2)
            near_mask[left:right] = True
        far_mask = ~near_mask
        near_vals = frame_scores[near_mask]
        far_vals = frame_scores[far_mask]
        dtype = frame_scores.dtype if torch.is_floating_point(frame_scores) else torch.float32
        zero = torch.zeros((), device=frame_scores.device, dtype=dtype)
        one = torch.ones((), device=frame_scores.device, dtype=dtype)
        near_top1_t = near_vals.max() if near_vals.numel() > 0 else zero
        far_top1_t = far_vals.max() if far_vals.numel() > 0 else zero
        near_topk = (
            near_vals.topk(min(self.config.topk_stats, near_vals.numel())).values
            if near_vals.numel() > 0
            else zero.reshape(1)
        )
        far_topk = (
            far_vals.topk(min(self.config.topk_stats, far_vals.numel())).values
            if far_vals.numel() > 0
            else zero.reshape(1)
        )
        flat_block = block_scores.reshape(-1)
        block_topk = flat_block.topk(min(self.config.topk_stats, flat_block.numel())).values
        probs = torch.softmax(frame_scores, dim=0) if frame_scores.numel() > 0 else one.reshape(1)
        entropy_t = -(probs * torch.log(probs.clamp(min=1e-8))).sum()
        block_top1_t = block_topk[0] if block_topk.numel() > 0 else zero
        block_topk_mean_t = block_topk.mean() if block_topk.numel() > 0 else zero
        estimated_density_proxy_t = torch.sigmoid(flat_block.mean()) if flat_block.numel() > 0 else one
        stat_values = torch.stack(
            [
                near_top1_t,
                near_topk.mean(),
                far_top1_t,
                far_topk.mean(),
                near_top1_t - far_top1_t,
                entropy_t,
                block_top1_t,
                block_topk_mean_t,
                block_top1_t - block_topk_mean_t,
                estimated_density_proxy_t,
            ]
        ).detach().cpu().float().tolist()
        (
            near_top1,
            near_topk_mean,
            far_top1,
            far_topk_mean,
            near_far_gap,
            entropy,
            block_top1,
            block_topk_mean,
            block_gap,
            estimated_density_proxy,
        ) = [float(value) for value in stat_values]
        return {
            "near_top1": near_top1,
            "near_topk_mean": near_topk_mean,
            "far_top1": far_top1,
            "far_topk_mean": far_topk_mean,
            "near_far_gap": near_far_gap,
            "frame_entropy": entropy,
            "block_top1": block_top1,
            "block_topk_mean": block_topk_mean,
            "block_gap": block_gap,
            "estimated_density_proxy": estimated_density_proxy,
            "latent_h": float(latent_hw[0]),
            "latent_w": float(latent_hw[1]),
            "reference_h": float(reference_hw[0]),
            "reference_w": float(reference_hw[1]),
            "block_grid_h": float(block_grid_hw[0]),
            "block_grid_w": float(block_grid_hw[1]),
            "approx_block_h": float(approx_block_hw[0]),
            "approx_block_w": float(approx_block_hw[1]),
            "spatial_area_ratio": float((latent_hw[0] * latent_hw[1]) / max(1, reference_hw[0] * reference_hw[1])),
        }
