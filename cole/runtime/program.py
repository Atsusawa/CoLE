from __future__ import annotations

import math
import time
from typing import Any, Mapping

from cole.core.program import ACTIONS, STATE_FEATURES, WINDOWS, StrategyProgram, validate_program


OBSERVATION_PANEL = (
    "motion.temporal_peak", "motion.temporal_diff_mean", "structure.energy_concentration",
    "attention.prev_attn_eff_support_ratio", "attention.prev_attn_top1_prob", "efficiency.realized_density",
)


class StrategyAdapter:
    """Compiled first-match rules. Only preexisting Python scalars cross this API."""

    def __init__(self, program: StrategyProgram) -> None:
        self.program = validate_program(program)
        self.by_head = {
            (head, window): tuple(rule for rule in program.rules if rule.head == head and rule.window == window)
            for head in ACTIONS for window in ((None,) if head != "sparse" else WINDOWS)
        }
        self.features = frozenset(OBSERVATION_PANEL).union(c.feature for r in program.rules for c in r.conditions)
        self.reset()

    def reset(self) -> None:
        self.trace: list[dict[str, Any]] = []
        self.overhead_sec = 0.0

    def observations(
        self, *, step_index: int, state_named: Mapping[str, Any] | None,
    ) -> dict[str, float]:
        start = time.perf_counter()
        result = {"step_index": float(step_index)}
        state = state_named or {}
        for feature in self.features.intersection(STATE_FEATURES):
            group, index = STATE_FEATURES[feature]
            values = state.get(group)
            if isinstance(values, (list, tuple)) and index < len(values):
                value = values[index]
                if type(value) in (int, float) and math.isfinite(value):
                    result[feature] = float(value)
        self.overhead_sec += time.perf_counter() - start
        return result

    def choose(
        self, head: str, incumbent: str | int, *, observations: Mapping[str, float],
        window: str | None = None, selected_switch_step: int | None = None,
    ) -> str | int:
        start = time.perf_counter()
        if head not in ACTIONS or incumbent not in ACTIONS[head]:
            raise ValueError(f"Invalid incumbent {head} action: {incumbent!r}.")
        scoped = dict(observations)
        if window is not None:
            scoped["window_index"] = float(WINDOWS.index(window))
        if selected_switch_step is not None:
            scoped["selected_switch_step"] = float(selected_switch_step)
        action = incumbent
        matched = None
        rules = self.by_head[(head, window)]
        required = {condition.feature for rule in rules for condition in rule.conditions}
        for rule in rules:
            if all(condition.matches(scoped) for condition in rule.conditions):
                action, matched = rule.action, rule.rule_id
                break
        self.trace.append({
            "head": head, "window": window, "step_index": int(observations["step_index"]),
            "incumbent_action": incumbent, "action": action, "rule_id": matched,
            "observations": {k: v for k, v in scoped.items() if k in self.features or k == "step_index"},
            "missing_features": sorted(required - scoped.keys()),
        })
        self.overhead_sec += time.perf_counter() - start
        return action

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "cole_program_execution_v1",
            "candidate_id": self.program.candidate_id, "fingerprint": self.program.fingerprint,
            "program": self.program.to_dict(), "decisions": list(self.trace),
            "overhead_sec": self.overhead_sec,
        }
