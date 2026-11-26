from __future__ import annotations

import torch

from cole.core.types import MaskPlan

from .backends.base import AttentionBackend
from .backends.dense_backend import DenseAttentionBackend
from .backends.sparse_backend import SparseAttentionBackend


class AttentionBridge:
    """Registry over available attention backends."""

    def __init__(self) -> None:
        self.backends: tuple[AttentionBackend, ...] = (SparseAttentionBackend(), DenseAttentionBackend())

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask_plan: MaskPlan):
        for backend in self.backends:
            if backend.supports(mask_plan, q, k, v):
                return backend.run(q, k, v, mask_plan)
        raise RuntimeError(
            "No backend supports the provided inputs. "
            f"q_shape={tuple(q.shape)}, k_shape={tuple(k.shape)}, "
            f"backend_name={mask_plan.backend_name}, profile_name={mask_plan.profile_name}, "
            f"mask_shape={None if mask_plan.block_mask is None else tuple(mask_plan.block_mask.shape)}"
        )
