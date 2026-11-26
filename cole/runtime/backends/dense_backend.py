from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from cole.core.feedback import ExecFeedback
from cole.core.types import MaskPlan

from .base import AttentionBackend


class DenseAttentionBackend(AttentionBackend):
    """PyTorch attention for dense profiles."""

    def supports(self, mask_plan: MaskPlan, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
        return str(mask_plan.backend_name) == "dense"

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask_plan: MaskPlan) -> tuple[torch.Tensor, ExecFeedback]:
        start = time.perf_counter()
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        elapsed = (time.perf_counter() - start) * 1000.0
        feedback = ExecFeedback(
            backend_prepare_ms=0.0,
            backend_run_ms=elapsed,
            total_attn_ms=elapsed,
            actual_density=float(mask_plan.estimated_density),
            selected_units=int(mask_plan.selected_units),
            backend_name="dense",
            executed_plan_signature=str(tuple(mask_plan.plan_signature)),
            executed_plan_density=float(mask_plan.estimated_density),
            backend_is_sparse=False,
            backend_first_call=False,
        )
        return out, feedback
