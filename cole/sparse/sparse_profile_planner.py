from __future__ import annotations

from cole.core.sparse_profiles import get_sparse_profile_spec
from cole.core.types import BaseSparsePlan, MaskPlan, ProjectedAction, SummaryBundle


class SparseProfilePlanner:
    """Turns a sparse profile action into an execution-ready attention plan."""

    def build(self, summary: SummaryBundle, base_plan: BaseSparsePlan, action: ProjectedAction) -> MaskPlan:
        spec = get_sparse_profile_spec(action.a_sparse_profile or base_plan.default_sparse_profile)
        total_units = int(summary.block_scores.shape[0] * summary.block_scores.shape[1])
        selected_units = int(round(float(spec.estimated_density) * float(total_units)))
        signature = (
            spec.name,
            spec.backend_name,
            spec.sparse_mode,
            spec.sparse_value,
            base_plan.default_sparse_profile,
            tuple(summary.latent_hw),
            tuple(summary.block_grid_hw),
        )
        return MaskPlan(
            backend_name=spec.backend_name,
            profile_name=spec.name,
            sparse_mode=spec.sparse_mode,
            sparse_value=spec.sparse_value,
            estimated_density=float(spec.estimated_density),
            selected_units=selected_units,
            plan_signature=signature,
            block_mask=None,
            query_block_size=1,
            key_block_size=1,
            latent_hw=tuple(summary.latent_hw),
            reference_hw=tuple(summary.reference_hw),
            planner_block_grid_hw=tuple(summary.block_grid_hw),
            planner_block_hw=tuple(summary.approx_block_hw),
        )
