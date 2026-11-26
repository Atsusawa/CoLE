from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


ATTENTION_PROXY_ASYNC_VALUES_KEY = "_cole_async_attention_proxy_values"
ATTENTION_PROXY_ASYNC_FEATURE_KEYS_KEY = "_cole_async_attention_proxy_feature_keys"
ATTENTION_PROXY_ASYNC_EVENT_KEY = "_cole_async_attention_proxy_event"
ATTENTION_PROXY_ASYNC_SOURCE_KEY = "_cole_async_attention_proxy_source"
_ATTENTION_PROXY_INTERNAL_KEYS = {
    ATTENTION_PROXY_ASYNC_VALUES_KEY,
    ATTENTION_PROXY_ASYNC_FEATURE_KEYS_KEY,
    ATTENTION_PROXY_ASYNC_EVENT_KEY,
    ATTENTION_PROXY_ASYNC_SOURCE_KEY,
}


def attention_proxy_has_pending(payload: dict[str, Any] | None) -> bool:
    return isinstance(payload, dict) and ATTENTION_PROXY_ASYNC_VALUES_KEY in payload


def _float_value(value: Any) -> float:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        if int(value.numel()) == 0:
            raise ValueError("empty tensor cannot be converted to float")
        return float(value.detach().reshape(-1)[0].cpu().item())
    return float(value)


def resolve_attention_proxy_payload(payload: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(payload, dict):
        return None
    out: dict[str, float] = {}
    if attention_proxy_has_pending(payload):
        event = payload.get(ATTENTION_PROXY_ASYNC_EVENT_KEY)
        synchronize = getattr(event, "synchronize", None)
        if callable(synchronize):
            synchronize()
        feature_keys = [str(key) for key in (payload.get(ATTENTION_PROXY_ASYNC_FEATURE_KEYS_KEY) or [])]
        values = payload.get(ATTENTION_PROXY_ASYNC_VALUES_KEY)
        if values is not None and feature_keys:
            try:
                import torch
            except Exception:
                torch = None
            if torch is not None and isinstance(values, torch.Tensor):
                values_list = values.detach().reshape(-1).float().tolist()
            else:
                values_list = list(values)
            for key, value in zip(feature_keys, values_list):
                out[str(key)] = float(value)
    for key, value in payload.items():
        key_str = str(key)
        if key_str in _ATTENTION_PROXY_INTERNAL_KEYS:
            continue
        try:
            out[key_str] = _float_value(value)
        except (TypeError, ValueError):
            continue
    return out or None


@dataclass
class ExecFeedback:
    """Execution feedback passed to the next step."""

    summary_build_ms: float = 0.0
    backend_prepare_ms: float = 0.0
    backend_run_ms: float = 0.0
    total_attn_ms: float = 0.0
    step_wall_ms: float = 0.0
    actual_density: float = 1.0
    selected_units: int = 0
    latency_ema: float = 0.0
    density_ema: float = 1.0
    backend_name: str = ""
    executed_plan_signature: str = ""
    executed_plan_density: float = 1.0
    backend_is_sparse: bool = False
    decision_reused: bool = False
    backend_first_call: bool = False
    attention_proxy: dict[str, Any] | None = None
    attention_prob_patches: dict[int, Any] | None = None
    attention_sampling_elapsed_sec: float = 0.0
    attention_sampling_updates: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "attention_prob_patches"
        }
        payload["attention_proxy"] = resolve_attention_proxy_payload(self.attention_proxy)
        payload["attention_prob_patches"] = None
        return payload


def clone_attention_prob_patches(patches: dict[int, Any] | None) -> dict[int, Any] | None:
    if not isinstance(patches, dict):
        return None
    try:
        import torch
    except Exception:
        torch = None
    out: dict[int, Any] = {}
    for key, value in patches.items():
        try:
            layer = int(key)
        except (TypeError, ValueError):
            continue
        if torch is not None and isinstance(value, torch.Tensor):
            out[layer] = value.detach()
    return out or None
