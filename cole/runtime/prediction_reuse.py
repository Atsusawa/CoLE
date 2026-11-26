from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def _normalize_steps(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
        items: list[Any] = [item for item in raw_items if item]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    out: set[int] = set()
    for item in items:
        step = int(item)
        if step >= 0:
            out.add(step)
    return tuple(sorted(out))


@dataclass(frozen=True)
class PredictionReuseConfig:
    mode: str = "off"
    max_consecutive_cached_steps: int = 2
    adaptive_delta_threshold: float = 0.08
    switch_step: int | None = None
    full_steps: tuple[int, ...] = ()
    cache_steps: tuple[int, ...] = ()

    @property
    def enabled(self) -> bool:
        return self.mode == "adaptive_output"

    @classmethod
    def from_payload(cls, payload: Any) -> "PredictionReuseConfig":
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            return cls()
        mode = str(payload.get("mode", "off")).strip().lower()
        if mode not in ("adaptive_output", "off"):
            raise ValueError(f"Unsupported prediction reuse mode: {mode!r}.")
        switch_step_raw = payload.get("switch_step")
        switch_step = None
        if switch_step_raw is not None:
            try:
                parsed = int(switch_step_raw)
                if parsed >= 0:
                    switch_step = parsed
            except (TypeError, ValueError):
                switch_step = None
        return cls(
            mode=mode,
            max_consecutive_cached_steps=max(
                0,
                int(payload.get("max_consecutive_cached_steps", 2) or 0),
            ),
            adaptive_delta_threshold=max(0.0, float(payload.get("adaptive_delta_threshold", 0.08) or 0.0)),
            switch_step=switch_step,
            full_steps=_normalize_steps(payload.get("full_steps")),
            cache_steps=_normalize_steps(payload.get("cache_steps")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": str(self.mode),
            "max_consecutive_cached_steps": int(self.max_consecutive_cached_steps),
            "adaptive_delta_threshold": float(self.adaptive_delta_threshold),
            "switch_step": (None if self.switch_step is None else int(self.switch_step)),
            "full_steps": [int(step) for step in self.full_steps],
            "cache_steps": [int(step) for step in self.cache_steps],
        }


@dataclass(frozen=True)
class PredictionReuseDecision:
    use_cache: bool
    reason: str
    noise_pred: torch.Tensor | None = None


@dataclass
class PredictionReuseStats:
    compute_steps: int = 0
    cache_steps: int = 0
    fallback_compute_steps: int = 0
    cache_updates: int = 0
    reset_count: int = 0
    policy_update_count: int = 0
    compute_reasons: dict[str, int] = field(default_factory=dict)
    cache_reasons: dict[str, int] = field(default_factory=dict)
    compute_step_indices: list[int] = field(default_factory=list)
    cache_step_indices: list[int] = field(default_factory=list)
    adaptive_delta_values: list[float] = field(default_factory=list)

    def inc_compute(self, reason: str, *, fallback: bool = False, step_index: int | None = None) -> None:
        self.compute_steps += 1
        if bool(fallback):
            self.fallback_compute_steps += 1
        if step_index is not None:
            self.compute_step_indices.append(int(step_index))
        key = str(reason or "compute")
        self.compute_reasons[key] = int(self.compute_reasons.get(key, 0)) + 1

    def inc_cache(self, reason: str, *, step_index: int | None = None) -> None:
        self.cache_steps += 1
        if step_index is not None:
            self.cache_step_indices.append(int(step_index))
        key = str(reason or "cache")
        self.cache_reasons[key] = int(self.cache_reasons.get(key, 0)) + 1

    def add_adaptive_delta(self, value: float) -> None:
        self.adaptive_delta_values.append(float(value))

    def to_dict(self) -> dict[str, object]:
        total_steps = int(self.compute_steps) + int(self.cache_steps)
        return {
            "compute_steps": int(self.compute_steps),
            "cache_steps": int(self.cache_steps),
            "fallback_compute_steps": int(self.fallback_compute_steps),
            "cache_updates": int(self.cache_updates),
            "reset_count": int(self.reset_count),
            "policy_update_count": int(self.policy_update_count),
            "cache_fraction": (0.0 if total_steps <= 0 else float(self.cache_steps) / float(total_steps)),
            "compute_step_indices": [int(step) for step in self.compute_step_indices],
            "cache_step_indices": [int(step) for step in self.cache_step_indices],
            "compute_reasons": {str(key): int(value) for key, value in sorted(self.compute_reasons.items())},
            "cache_reasons": {str(key): int(value) for key, value in sorted(self.cache_reasons.items())},
            "adaptive_delta_mean": (
                None
                if not self.adaptive_delta_values
                else float(sum(self.adaptive_delta_values) / float(len(self.adaptive_delta_values)))
            ),
            "adaptive_delta_max": (
                None
                if not self.adaptive_delta_values
                else float(max(self.adaptive_delta_values))
            ),
        }


class PredictionReuseRuntime:
    """Reuse the last full guided Wan prediction when the latent delta permits it."""

    def __init__(self, config: PredictionReuseConfig) -> None:
        self.config = config
        self.stats = PredictionReuseStats()
        self.cached_noise_pred: torch.Tensor | None = None
        self.cached_shape: tuple[int, ...] | None = None
        self.cached_latent_ref: torch.Tensor | None = None
        self.consecutive_cached_steps = 0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def reset(self) -> None:
        self.cached_noise_pred = None
        self.cached_shape = None
        self.cached_latent_ref = None
        self.consecutive_cached_steps = 0
        self.stats.reset_count += 1

    def apply_policy_update(self, config: dict[str, Any]) -> None:
        self.config = PredictionReuseConfig.from_payload(config)
        self.consecutive_cached_steps = 0
        self.stats.policy_update_count += 1

    def decide(
        self,
        *,
        step_index: int,
        expected_shape: tuple[int, ...],
        target_device: torch.device,
        latent_model_input: torch.Tensor | None = None,
    ) -> PredictionReuseDecision:
        step = int(step_index)
        if not self.enabled:
            return PredictionReuseDecision(use_cache=False, reason="disabled")
        reason = self._forced_compute_reason(step)
        if reason is not None:
            return PredictionReuseDecision(use_cache=False, reason=reason)
        if self.cached_noise_pred is None or self.cached_shape is None:
            return PredictionReuseDecision(use_cache=False, reason="cache_empty")
        if tuple(expected_shape) != tuple(self.cached_shape):
            self.reset()
            return PredictionReuseDecision(use_cache=False, reason="shape_mismatch")
        cached = self.cached_noise_pred
        if cached.device != target_device:
            cached = cached.to(device=target_device)
        delta = self._adaptive_delta(latent_model_input, reference_latent=self.cached_latent_ref)
        if delta is None:
            return PredictionReuseDecision(use_cache=False, reason="adaptive_missing_latent")
        self.stats.add_adaptive_delta(delta)
        if delta > self.config.adaptive_delta_threshold:
            return PredictionReuseDecision(use_cache=False, reason="adaptive_delta")
        self.consecutive_cached_steps += 1
        self.stats.inc_cache("adaptive_output", step_index=step)
        return PredictionReuseDecision(use_cache=True, reason="adaptive_output", noise_pred=cached)

    def record_compute(
        self,
        *,
        noise_pred: torch.Tensor,
        reason: str,
        step_index: int | None = None,
        latent_model_input: torch.Tensor | None = None,
    ) -> None:
        reason_name = str(reason or "compute")
        self.stats.inc_compute(
            reason_name,
            fallback=reason_name in {"cache_empty", "shape_mismatch"},
            step_index=step_index,
        )
        self.cached_noise_pred = noise_pred.detach()
        self.cached_shape = tuple(int(dim) for dim in noise_pred.shape)
        if latent_model_input is not None:
            self.cached_latent_ref = latent_model_input.detach()
        self.consecutive_cached_steps = 0
        self.stats.cache_updates += 1

    def _adaptive_delta(
        self,
        latent_model_input: torch.Tensor | None,
        *,
        reference_latent: torch.Tensor | None,
    ) -> float | None:
        if latent_model_input is None or reference_latent is None:
            return None
        if tuple(int(dim) for dim in latent_model_input.shape) != tuple(int(dim) for dim in reference_latent.shape):
            self.reset()
            return None
        current = latent_model_input.detach()
        reference = reference_latent
        if current.device != reference.device:
            reference = reference.to(device=current.device)
        current_f = current.float()
        reference_f = reference.float()
        numerator = torch.linalg.vector_norm(current_f - reference_f).item()
        denominator = torch.linalg.vector_norm(reference_f).item()
        if denominator <= 1e-12:
            denominator = 1.0
        return float(numerator / denominator)

    def _forced_compute_reason(self, step: int) -> str | None:
        if step in self.config.full_steps:
            return "explicit_full_step"
        if step not in self.config.cache_steps:
            return "outside_explicit_cache_schedule"
        max_cached = self.config.max_consecutive_cached_steps
        if max_cached <= 0:
            return "max_consecutive_zero"
        if self.consecutive_cached_steps >= max_cached:
            return "max_consecutive"
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "config": self.config.to_dict(),
            "stats": self.stats.to_dict(),
        }
