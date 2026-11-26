from __future__ import annotations

from typing import Any

from cole.core.feedback import ExecFeedback


def _feedback_value(feedback: ExecFeedback | dict[str, Any] | None, key: str) -> Any:
    if isinstance(feedback, ExecFeedback):
        return getattr(feedback, key, None)
    if isinstance(feedback, dict):
        return feedback.get(key)
    return None


def resolve_feedback_step_cost_ms(feedback: ExecFeedback | dict[str, Any] | None) -> float:
    step_wall_ms = _feedback_value(feedback, "step_wall_ms")
    if step_wall_ms is not None:
        resolved = max(0.0, float(step_wall_ms))
        if resolved > 0.0:
            return float(resolved)
    total_attn_ms = _feedback_value(feedback, "total_attn_ms")
    if total_attn_ms is None:
        return 0.0
    return max(0.0, float(total_attn_ms))


def resolve_feedback_latency_ema_ms(feedback: ExecFeedback | dict[str, Any] | None) -> float:
    latency_ema = _feedback_value(feedback, "latency_ema")
    if latency_ema is not None:
        resolved = max(0.0, float(latency_ema))
        if resolved > 0.0:
            return float(resolved)
    return resolve_feedback_step_cost_ms(feedback)


def update_feedback_ema(prev: ExecFeedback | None, current: ExecFeedback, momentum: float = 0.9) -> ExecFeedback:
    current_cost_ms = resolve_feedback_step_cost_ms(current)
    if prev is None:
        current.latency_ema = current_cost_ms
        current.density_ema = current.actual_density
        return current
    current.latency_ema = momentum * float(prev.latency_ema) + (1.0 - momentum) * float(current_cost_ms)
    current.density_ema = momentum * float(prev.density_ema) + (1.0 - momentum) * float(current.actual_density)
    return current
