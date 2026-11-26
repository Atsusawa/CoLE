from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from cole.core.feedback import ExecFeedback
from cole.core.types import MaskPlan


class AttentionBackend(ABC):
    """Sparse attention backend interface."""

    @abstractmethod
    def supports(self, mask_plan: MaskPlan, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
        raise NotImplementedError

    @abstractmethod
    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask_plan: MaskPlan) -> tuple[torch.Tensor, ExecFeedback]:
        raise NotImplementedError
