from __future__ import annotations

from dataclasses import asdict, dataclass
from cole.core.sparse_profiles import get_sparse_profile_name_for_topk


SPARSE_BASELINE_TOPK = 0.6
SPARSE_POST_BODY_MIN_START_STEP = 26


@dataclass(frozen=True)
class SparseWindow:
    name: str
    start_step: int
    end_step: int

    @property
    def length(self) -> int:
        return max(0, int(self.end_step) - int(self.start_step))

    def contains(self, step_index: int) -> bool:
        step = int(step_index)
        return int(self.start_step) <= step < int(self.end_step)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_sparse_windows(
    *,
    total_steps: int,
    switch_step: int,
    first_policy_step: int = 3,
    transition_len: int = 5,
    late_len: int = 10,
    post_body_min_start_step: int = SPARSE_POST_BODY_MIN_START_STEP,
) -> list[SparseWindow]:
    """Build four causal windows using switch-relative and denoising-time guards."""

    total = max(0, int(total_steps))
    switch = max(0, int(switch_step))
    first = max(0, int(first_policy_step))
    transition = max(0, int(transition_len))
    late = max(0, int(late_len))
    late_start = max(first, total - late)
    post_body_floor = max(first, int(post_body_min_start_step))
    transition_end = min(max(switch, switch + transition, post_body_floor), late_start)

    candidates = [
        SparseWindow("pre_switch", first, min(switch, late_start)),
        SparseWindow("switch_transition", max(first, switch), transition_end),
        SparseWindow("post_switch_body", max(first, transition_end), late_start),
        SparseWindow("late", late_start, total),
    ]
    return [window for window in candidates if window.length > 0]


def shift_sparse_topk(
    baseline_topk: float,
    action: str,
    *,
    topk_step: float = 0.1,
    min_topk: float = 0.1,
) -> float:
    action_name = str(action).strip().lower()
    current = float(baseline_topk)
    if action_name == "keep":
        shifted = current
    elif action_name == "safer1":
        shifted = min(1.0, current + float(topk_step))
    elif action_name == "aggr1":
        shifted = max(float(min_topk), current - float(topk_step))
    elif action_name == "aggr2":
        shifted = max(float(min_topk), current - 2.0 * float(topk_step))
    else:
        raise ValueError(f"Unsupported sparsity action: {action!r}")
    return round(float(shifted), 10)


def sparse_profile_for_action(
    *,
    baseline_topk: float,
    action: str,
    topk_step: float = 0.1,
    min_topk: float = 0.1,
) -> str:
    topk = shift_sparse_topk(
        baseline_topk,
        action,
        topk_step=topk_step,
        min_topk=min_topk,
    )
    if topk >= 1.0:
        return "dense_full"
    profile = get_sparse_profile_name_for_topk(topk)
    if profile is None:
        raise ValueError(f"No sparse profile is registered for topk={topk}.")
    return str(profile)


