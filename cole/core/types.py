from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from .feedback import ExecFeedback


@dataclass
class BaseSparsePlan:
    """Base sparse schedule output for a step."""

    default_sparse_profile: str
    promote_resolution: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectedAction:
    """Legally projected action values."""

    a_res: int = 0
    a_sparse_profile: str = "dense_full"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SummaryBundle:
    """Lightweight pooled summaries shared by state and planner."""

    frame_embeddings: torch.Tensor
    block_embeddings: torch.Tensor
    frame_scores: torch.Tensor
    block_scores: torch.Tensor
    latent_hw: tuple[int, int] = (0, 0)
    reference_hw: tuple[int, int] = (0, 0)
    block_grid_hw: tuple[int, int] = (0, 0)
    approx_block_hw: tuple[int, int] = (0, 0)
    stats: dict[str, float] = field(default_factory=dict)


@dataclass
class MaskPlan:
    """Execution-ready attention plan metadata."""

    backend_name: str
    profile_name: str
    sparse_mode: str
    sparse_value: float | None
    estimated_density: float
    selected_units: int
    plan_signature: tuple[Any, ...]
    block_mask: torch.Tensor | None = None
    query_block_size: int = 1
    key_block_size: int = 1
    latent_hw: tuple[int, int] = (0, 0)
    reference_hw: tuple[int, int] = (0, 0)
    planner_block_grid_hw: tuple[int, int] = (0, 0)
    planner_block_hw: tuple[int, int] = (0, 0)
    sparse_return_sparsity: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if isinstance(payload.get("block_mask"), torch.Tensor):
            payload["block_mask"] = payload["block_mask"].detach().cpu().int().tolist()
        return payload


@dataclass
class StateBuildOutput:
    """Named and flat state outputs."""

    vector: torch.Tensor
    named: dict[str, list[float]]


@dataclass
class StepState:
    """Context used to build a state."""

    timestep: int
    total_steps: int
    resolution_phase: str
    latent: torch.Tensor
    prev_action: ProjectedAction | None = None
    exec_feedback: ExecFeedback | None = None
    steps_since_sparse_change: int = 0
    steps_since_resolution_change: int = 0
    lowres_streak: int = 0
    decision_reused: bool = False
