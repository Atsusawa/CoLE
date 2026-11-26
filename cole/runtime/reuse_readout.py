from __future__ import annotations

import math
from pathlib import Path
from pickle import UnpicklingError

import torch
from torch import nn

from cole.core.program import ACTIONS
from cole.core.reuse_presets import PRESETS, build_reuse_config
from cole.runtime.controller_model import (
    CandidateScoringHead,
    SharedResolutionSparsePolicy,
    SharedTensorFeatures,
    TaskAdapter,
)

REUSE_ACTIONS = ACTIONS["reuse"]


class ReuseReadout(nn.Module):
    """Score the six reuse candidates using the frozen controller's probe features."""

    def __init__(self, *, shared_width: int, width: int, dropout: float) -> None:
        super().__init__()
        self.reuse_decision_proj = nn.Sequential(nn.LayerNorm(9), nn.Linear(9, width), nn.GELU())
        self.reuse_adapter = TaskAdapter(input_dim=shared_width + width, width=width, dropout=dropout)
        self.reuse_head = CandidateScoringHead(
            hidden_dim=width, candidate_feature_dim=8, width=width, dropout=dropout,
        )

    @classmethod
    def load(cls, path: Path, model: SharedResolutionSparsePolicy) -> ReuseReadout:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (EOFError, OSError, RuntimeError, UnpicklingError) as exc:
            raise ValueError(f"Could not load CoLE-L checkpoint at {path}.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Expected a shared controller checkpoint for CoLE-L.")
        # Accept the historical field names when reading a saved training checkpoint.
        legacy = "prediction_reuse" not in payload
        policy = payload.get("velocity_cache_policy" if legacy else "prediction_reuse")
        candidates = policy.get("candidates") if isinstance(policy, dict) else None
        if (
            not isinstance(candidates, (list, tuple))
            or any(not isinstance(candidate, dict) for candidate in candidates)
            or tuple(candidate.get("action") for candidate in candidates) != REUSE_ACTIONS
        ):
            raise ValueError(f"CoLE-L requires the six ordered reuse actions {REUSE_ACTIONS}.")
        config = payload.get("model_config")
        feature_prefix = "cache" if legacy else "reuse"
        if (
            not isinstance(config, dict)
            or config.get(f"{feature_prefix}_decision_feature_dim") != 9
            or config.get(f"{feature_prefix}_candidate_feature_dim") != 8
        ):
            raise ValueError("CoLE-L requires a 9-feature probe and 8-feature reuse candidates.")
        if policy.get("probe_step", 14) != 14:
            raise ValueError("CoLE-L requires the step-14 probe.")

        reference = model.state_dict()
        state = payload.get("model_state_dict")
        if not isinstance(state, dict) or any(not isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError("CoLE-L checkpoint must contain a model_state_dict with tensor weights.")
        if any(
            key not in state or reference[key].dtype != state[key].dtype
            or not torch.equal(reference[key], state[key]) for key in reference
        ):
            raise ValueError("CoLE-L checkpoint must share the bundled frozen Resolution and Sparsity weights.")
        readout = cls(
            shared_width=model.backbone.output_dim,
            width=model.config.adapter_width, dropout=model.config.dropout,
        )
        readout_state = {key: value for key, value in state.items() if key not in reference}
        if legacy:
            if any(not isinstance(key, str) or not key.startswith("cache_") for key in readout_state):
                raise ValueError("CoLE-L checkpoint has incompatible prediction reuse readout weights.")
            readout_state = {
                "reuse_" + key.removeprefix("cache_"): value for key, value in readout_state.items()
            }
        try:
            readout.load_state_dict(readout_state, strict=True)
        except RuntimeError as exc:
            raise ValueError("CoLE-L checkpoint has incompatible prediction reuse readout weights.") from exc
        stats = []
        for candidate in candidates:
            reuse = candidate.get(f"expected_{feature_prefix}_fraction")
            time_reduction = candidate.get("expected_time_reduction_fraction")
            if not all(isinstance(value, (int, float)) and 0.0 <= value <= 1.0 for value in (reuse, time_reduction)):
                raise ValueError(f"CoLE-L candidate {candidate['action']} needs reuse and time reduction statistics in [0, 1].")
            stats.append((float(reuse), float(time_reduction)))
        readout.candidate_stats = tuple(stats)
        return readout

    def forward(
        self, probe: SharedTensorFeatures, *, context: torch.Tensor, switch_step: int,
        total_steps: int, force_full_steps: set[int],
    ) -> torch.Tensor:
        context = torch.nan_to_num(context.float(), nan=0.0, posinf=0.0, neginf=0.0)
        progress = context[:, 0].clamp(0, 1)
        sigma = context[:, 1].clamp_min(0)
        switch_progress = switch_step / (total_steps - 1)
        decision = torch.stack((
            progress, sigma, torch.log1p(sigma), torch.sin(math.pi * progress), torch.cos(math.pi * progress),
            torch.full_like(progress, switch_progress), (switch_progress - progress).clamp(-1, 1),
            torch.full_like(progress, switch_step / total_steps),
            torch.full_like(progress, (total_steps - switch_step) / total_steps),
        ), dim=-1)
        rows = []
        max_threshold = max(preset[0] for preset in PRESETS.values())
        max_reuse = max(preset[1] for preset in PRESETS.values())
        for action, empirical in zip(REUSE_ACTIONS, self.candidate_stats):
            config = build_reuse_config(
                action, total_steps=total_steps, switch_step=switch_step, force_full_steps=force_full_steps,
            )
            threshold = config.get("adaptive_delta_threshold", 0.0)
            cap = config.get("max_consecutive_cached_steps", 0)
            eligible = len(config["cache_steps"]) / total_steps
            rows.append((float(action == "off"), threshold, threshold / max_threshold, cap / max_reuse,
                         1.0 - eligible, eligible, *empirical))
        features = torch.tensor(rows, device=context.device, dtype=context.dtype).unsqueeze(0)
        hidden = self.reuse_adapter(torch.cat((probe.concatenate(), self.reuse_decision_proj(decision)), dim=-1))
        return self.reuse_head(hidden, features)
