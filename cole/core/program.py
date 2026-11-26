from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "cole_program_v1"
_LEGACY_SCHEMA = "resolution_sparse_strategy_program_v1"
WINDOWS = ("pre_switch", "switch_transition", "post_switch_body", "late")
ACTIONS = {
    "switch": (18, 21, 24),
    "sparse": ("keep", "safer1", "aggr1", "aggr2"),
    "reuse": ("off", "safe", "balanced", "fast", "turbo", "ultra"),
}
# Explicit, versioned whitelist; never accept arbitrary state keys or expressions.
# These indices match StateBuilder.GROUP_FEATURES. No tensor reduction is added.
STATE_FEATURES = {
    "phase.progress": ("phase", 0),
    "phase.is_highres": ("phase", 4),
    "history.prev_sparse_level": ("history", 0),
    "history.steps_since_sparse_change": ("history", 1),
    "history.steps_since_resolution_change": ("history", 2),
    "structure.energy_concentration": ("structure", 1),
    "structure.coarse_grad": ("structure", 7),
    "motion.temporal_diff_mean": ("motion", 0),
    "motion.temporal_peak": ("motion", 3),
    "efficiency.realized_density": ("efficiency", 0),
    "attention.prev_attn_eff_support_ratio": ("attention", 0),
    "attention.prev_attn_top1_prob": ("attention", 1),
}
FEATURES = tuple(STATE_FEATURES) + (
    "step_index", "window_index", "selected_switch_step",
)
OPS = ("lt", "le", "gt", "ge", "eq", "ne")
MAX_RULES = 24
MAX_CONDITIONS = 4
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _keys(value: Any, required: set[str], optional: set[str] | None = None) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("Expected a JSON object.")
    missing = required - value.keys()
    extra = value.keys() - required - (optional or set())
    if missing or extra:
        raise ValueError(f"Invalid object keys: missing={sorted(missing)}, extra={sorted(extra)}.")


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid identifier: {value!r}.")
    return value


def finite_number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Expected a finite JSON number, got {value!r}.")
    return float(value)


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    value: float

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "op": self.op, "value": self.value}

    def matches(self, observations: Mapping[str, float]) -> bool:
        value = observations.get(self.feature)
        # Unknown/nonfinite observations never satisfy even `ne` conditions.
        if type(value) not in (int, float) or not math.isfinite(value):
            return False
        if self.op == "lt":
            return value < self.value
        if self.op == "le":
            return value <= self.value
        if self.op == "gt":
            return value > self.value
        if self.op == "ge":
            return value >= self.value
        if self.op == "eq":
            return value == self.value
        return value != self.value


@dataclass(frozen=True)
class Rule:
    rule_id: str
    head: str
    action: str | int
    conditions: tuple[Condition, ...]
    window: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id, "head": self.head, "action": self.action,
            "conditions": [item.to_dict() for item in self.conditions], "window": self.window,
        }


@dataclass(frozen=True)
class StrategyProgram:
    candidate_id: str
    parent_id: str | None
    checkpoint_path: str
    checkpoint_sha256: str
    rules: tuple[Rule, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "candidate_id": self.candidate_id, "parent_id": self.parent_id,
            "incumbent": {
                "checkpoint_path": self.checkpoint_path,
                "checkpoint_sha256": self.checkpoint_sha256,
            },
            "rules": [rule.to_dict() for rule in self.rules],
        }

    @property
    def fingerprint(self) -> str:
        """Semantic identity ignores candidate lineage and names, but preserves priority."""
        payload = self.to_dict()
        payload.pop("candidate_id")
        payload.pop("parent_id")
        for rule in payload["rules"]:
            rule.pop("rule_id")
        return digest(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StrategyProgram":
        return validate_program(payload)

    @classmethod
    def load(cls, path: str | Path) -> "StrategyProgram":
        raw = Path(path).read_text(encoding="utf-8")
        if len(raw.encode("utf-8")) > 65536:
            raise ValueError("StrategyProgram exceeds 64 KiB.")
        def unique_keys(pairs):
            out = {}
            for key, value in pairs:
                if key in out:
                    raise ValueError(f"Duplicate JSON key: {key}.")
                out[key] = value
            return out
        return validate_program(json.loads(raw, object_pairs_hook=unique_keys))


def validate_program(payload: Mapping[str, Any] | StrategyProgram) -> StrategyProgram:
    if isinstance(payload, StrategyProgram):
        payload = payload.to_dict()
    _keys(payload, {"schema", "candidate_id", "parent_id", "incumbent", "rules"})
    legacy = payload["schema"] == _LEGACY_SCHEMA
    if payload["schema"] != SCHEMA and not legacy:
        raise ValueError(f"Expected schema={SCHEMA}.")
    candidate_id = identifier(payload["candidate_id"])
    parent_id = None if payload["parent_id"] is None else identifier(payload["parent_id"])
    if parent_id == candidate_id:
        raise ValueError("A candidate cannot parent itself.")
    incumbent = payload["incumbent"]
    legacy_fields = {"velocity_cache_selector"} if legacy else set()
    _keys(incumbent, {"checkpoint_path", "checkpoint_sha256"} | legacy_fields)
    if legacy and incumbent["velocity_cache_selector"] is not False:
        raise ValueError("The legacy program must use rule-based prediction reuse.")
    if not isinstance(incumbent["checkpoint_path"], str) or not incumbent["checkpoint_path"].strip():
        raise ValueError("incumbent.checkpoint_path is required.")
    if not isinstance(incumbent["checkpoint_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", incumbent["checkpoint_sha256"]):
        raise ValueError("incumbent.checkpoint_sha256 must be a lowercase SHA-256.")
    raw_rules = payload["rules"]
    if not isinstance(raw_rules, list) or len(raw_rules) > MAX_RULES:
        raise ValueError(f"rules must be a list of at most {MAX_RULES} rules.")
    rules = []
    ids = set()
    for raw in raw_rules:
        _keys(raw, {"rule_id", "head", "action", "conditions", "window"})
        rule_id = identifier(raw["rule_id"])
        if rule_id in ids:
            raise ValueError(f"Duplicate rule_id={rule_id}.")
        ids.add(rule_id)
        head = "reuse" if legacy and raw["head"] == "cache" else raw["head"]
        if not isinstance(head, str) or head not in ACTIONS:
            raise ValueError(f"Unknown decision head: {head!r}.")
        action = raw["action"]
        if action not in ACTIONS[head] or type(action) is not type(ACTIONS[head][0]):
            raise ValueError(f"Illegal {head} action={action!r}.")
        window = raw["window"]
        if head == "sparse":
            if window not in WINDOWS:
                raise ValueError("Sparse rules require one of the four canonical windows.")
        elif window is not None:
            raise ValueError("Only sparse rules may specify a window.")
        if not isinstance(raw["conditions"], list) or len(raw["conditions"]) > MAX_CONDITIONS:
            raise ValueError(f"At most {MAX_CONDITIONS} conjunctive conditions are allowed.")
        conditions = []
        for condition in raw["conditions"]:
            _keys(condition, {"feature", "op", "value"})
            feature = condition["feature"]
            if feature not in FEATURES or condition["op"] not in OPS:
                raise ValueError(f"Illegal feature/operator: {condition!r}.")
            # Tau is unknown before the switch head; windows can move with tau.
            if head == "switch" and feature in ("selected_switch_step", "window_index"):
                raise ValueError(f"Noncausal {head} feature: {feature}.")
            if head == "reuse" and feature == "window_index":
                raise ValueError("Prediction reuse is selected once at the switch probe, outside window routing.")
            if head == "sparse" and window == "pre_switch" and feature == "selected_switch_step":
                raise ValueError("Tau is not selected at the pre-switch window boundary.")
            conditions.append(Condition(feature, condition["op"], finite_number(condition["value"])))
        rules.append(Rule(rule_id, head, action, tuple(conditions), window))
    return StrategyProgram(
        candidate_id, parent_id, incumbent["checkpoint_path"], incumbent["checkpoint_sha256"],
        tuple(rules),
    )


def program_json_schema() -> dict[str, Any]:
    """JSON Schema plus stricter causal checks in validate_program()."""
    def obj(properties):
        return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}
    number = {"type": "number"}
    condition = obj({"feature": {"enum": list(FEATURES)}, "op": {"enum": list(OPS)}, "value": number})
    variants = []
    for head, actions in ACTIONS.items():
        allowed_features = list(FEATURES)
        if head == "switch":
            allowed_features = [name for name in allowed_features if name not in ("selected_switch_step", "window_index")]
        elif head == "reuse":
            allowed_features.remove("window_index")
        scoped_condition = obj({**condition["properties"], "feature": {"enum": allowed_features}})
        variant = obj({
            "rule_id": {"type": "string", "pattern": _IDENTIFIER.pattern.replace("\\Z", "$")},
            "head": {"const": head}, "action": {"enum": list(actions)},
            "conditions": {"type": "array", "maxItems": MAX_CONDITIONS, "items": scoped_condition},
            "window": {"enum": list(WINDOWS)} if head == "sparse" else {"type": "null"},
        })
        if head == "sparse":
            variant["allOf"] = [{
                "if": {"properties": {"window": {"const": "pre_switch"}}},
                "then": {"properties": {"conditions": {"items": {"properties": {
                    "feature": {"not": {"const": "selected_switch_step"}},
                }}}}},
            }]
        variants.append(variant)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **obj({
            "schema": {"const": SCHEMA}, "candidate_id": {"type": "string", "pattern": _IDENTIFIER.pattern.replace("\\Z", "$")},
            "parent_id": {"anyOf": [{"type": "string", "pattern": _IDENTIFIER.pattern.replace("\\Z", "$")}, {"type": "null"}]},
            "incumbent": obj({
                "checkpoint_path": {"type": "string", "minLength": 1},
                "checkpoint_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            }),
            "rules": {"type": "array", "maxItems": MAX_RULES, "items": {"oneOf": variants}},
        }),
    }
