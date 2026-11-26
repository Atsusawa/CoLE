import json
from dataclasses import replace

import pytest

from cole.core.program import (
    SCHEMA, Condition, StrategyProgram, program_json_schema, validate_program,
)
from cole.runtime.program import StrategyAdapter


def root_payload():
    return {
        "schema": SCHEMA, "candidate_id": "incumbent", "parent_id": None,
        "incumbent": {"checkpoint_path": "/models/shared.pt", "checkpoint_sha256": "a" * 64},
        "rules": [],
    }


def rule(head="sparse", action="aggr1", window="late", feature=None):
    return {
        "rule_id": "override", "head": head, "action": action, "window": window,
        "conditions": [] if feature is None else [{"feature": feature, "op": "ge", "value": 0.5}],
    }


@pytest.mark.parametrize("feature", ["prompt_id", "prompt_hash_00", "future.quality", "next_latent", "unevaluated_action_gain", "__import__('os')"])
def test_rejects_identity_future_and_expressions(feature):
    payload = root_payload()
    payload["rules"] = [rule(feature=feature)]
    with pytest.raises(ValueError, match="Illegal feature"):
        validate_program(payload)


@pytest.mark.parametrize("bad_rule", [
    rule("switch", 17, None), rule("switch", True, None), rule("switch", 18.0, None),
    rule("reuse", "invalid", None), rule(window="future_window"),
    rule("switch", 18, None, "selected_switch_step"), rule("switch", 18, None, "window_index"),
    rule("reuse", "safe", None, "window_index"), rule(window="pre_switch", feature="selected_switch_step"),
])
def test_rejects_illegal_actions_windows_and_noncausal_conditions(bad_rule):
    payload = root_payload()
    payload["rules"] = [bad_rule]
    with pytest.raises(ValueError):
        validate_program(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.5", True])
def test_rejects_nonfinite_or_coerced_values(value):
    payload = root_payload()
    payload["rules"] = [rule(feature="motion.temporal_peak")]
    payload["rules"][0]["conditions"][0]["value"] = value
    with pytest.raises(ValueError):
        validate_program(payload)


def test_strict_schema_bounds_and_json_duplicates(tmp_path):
    payload = root_payload()
    payload["python"] = "pass"
    with pytest.raises(ValueError):
        validate_program(payload)
    payload = root_payload()
    payload["rules"] = [rule()] * 25
    with pytest.raises(ValueError):
        validate_program(payload)
    payload["rules"] = [rule(), rule()]
    with pytest.raises(ValueError, match="Duplicate rule"):
        validate_program(payload)
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"bad","schema":"also_bad"}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        StrategyProgram.load(path)


def test_schema_and_compilation_are_deterministic_first_match():
    payload = root_payload()
    payload["rules"] = [rule(feature="motion.temporal_peak"), {**rule(action="safer1"), "rule_id": "second"}]
    program = validate_program(payload)
    assert validate_program(json.loads(json.dumps(program.to_dict()))) == program
    assert program.fingerprint == replace(program, candidate_id="new", parent_id="incumbent").fingerprint
    left, right = StrategyAdapter(program), StrategyAdapter(program)
    for adapter in (left, right):
        obs = adapter.observations(step_index=40, state_named={"motion": [0.0, 0.0, 0.0, 0.8]})
        assert adapter.choose("sparse", "keep", window="late", observations=obs) == "aggr1"
    assert left.trace == right.trace
    assert left.trace[0]["rule_id"] == "override"
    schema = program_json_schema()
    assert schema["additionalProperties"] is False
    assert schema == program_json_schema()


def test_missing_nonfinite_and_tensor_observations_do_not_match():
    condition = Condition("motion.temporal_peak", "ne", 0.0)
    for value in (None, float("nan"), float("inf"), "1", True):
        assert not condition.matches({"motion.temporal_peak": value})
    assert not condition.matches({})
    payload = root_payload()
    payload["rules"] = [rule(feature="motion.temporal_peak")]
    adapter = StrategyAdapter(validate_program(payload))
    observations = adapter.observations(
        step_index=40, state_named={"motion": [0, 0, 0, 0.75], "prompt_id": [123]},
    )
    assert "prompt_id" not in observations
    assert adapter.choose("sparse", "keep", window="late", observations=observations) == "aggr1"
    adapter.reset()
    assert adapter.trace == [] and adapter.overhead_sec == 0


def test_published_schema_matches_strict_causal_validator():
    jsonschema = pytest.importorskip("jsonschema")
    schema = program_json_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    payload = root_payload()
    jsonschema.validate(payload, schema)
    payload["rules"] = [rule(feature="motion.temporal_peak")]
    jsonschema.validate(payload, schema)
    for bad_rule in (rule("switch", 18, None, "selected_switch_step"), rule(feature="prompt_id"),
                     rule(window="pre_switch", feature="selected_switch_step")):
        payload["rules"] = [bad_rule]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, schema)


def test_feature_registry_matches_existing_state_builders():
    from cole.core.state_builder import GROUP_FEATURES as state_groups
    from cole.core.program import STATE_FEATURES

    for name, (group, index) in STATE_FEATURES.items():
        assert state_groups[group][index] == name.split(".", 1)[1]


def test_program_trace_records_scalar_observations():
    adapter = StrategyAdapter(validate_program(root_payload()))
    observations = adapter.observations(step_index=14, state_named={"motion": [0.2, 0.0, 0.0, 0.8]})
    assert observations["motion.temporal_peak"] == 0.8
    assert adapter.choose("switch", 21, observations=observations) == 21
    assert adapter.trace[0]["observations"]["motion.temporal_diff_mean"] == 0.2
