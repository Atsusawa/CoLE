from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from cole.common.finite_guard import finite_guard_enabled
from cole.common.timing import resolve_feedback_latency_ema_ms, resolve_feedback_step_cost_ms

from .feedback import ExecFeedback
from .sparse_profiles import SPARSE_PROFILE_ORDER, get_sparse_profile_index
from .state_schema import StateSchema
from .types import StateBuildOutput, StepState, SummaryBundle


# Scalar observations consumed by the deployment program.
PHASE_EARLY_MAX = 0.20
PHASE_MID_MAX = 0.65
EPS = 1e-6
STATE_VERSION = "s2_attention_proxy_30d"
GROUP_FEATURES: dict[str, tuple[str, ...]] = {
    "phase": (
        "progress",
        "phase_early",
        "phase_mid",
        "phase_late",
        "is_highres",
    ),
    "history": (
        "prev_sparse_level",
        "steps_since_sparse_change",
        "steps_since_resolution_change",
        "lowres_streak",
        "prev_decision_reused",
    ),
    "structure": (
        "spatial_area_ratio",
        "energy_concentration",
        "energy_dispersion",
        "com_x",
        "com_y",
        "lr_imbalance",
        "tb_imbalance",
        "coarse_grad",
    ),
    "motion": (
        "temporal_diff_mean",
        "temporal_diff_std",
        "frame_energy_std",
        "temporal_peak",
    ),
    "efficiency": (
        "realized_density",
        "density_ema",
        "step_cost_log_ms",
        "cost_ema_log_ms",
    ),
    "attention": (
        "prev_attn_eff_support_ratio",
        "prev_attn_top1_prob",
        "prev_attn_topk_mass_10pct",
        "prev_attn_logit_std",
    ),
}


def _as_video_latent(latent: torch.Tensor) -> torch.Tensor:
    if latent.dim() == 5:
        return latent
    if latent.dim() == 4:
        return latent.unsqueeze(2)
    if latent.dim() == 3:
        return latent.unsqueeze(0).unsqueeze(2)
    raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")


class StateBuilder:
    """Build scalar observations from latents and execution feedback."""

    def __init__(self, schema: Optional[StateSchema] = None) -> None:
        self.schema = schema or StateSchema()
        self._validate_groups()

    @property
    def version(self) -> str:
        return STATE_VERSION

    @property
    def group_features(self) -> dict[str, list[str]]:
        return {name: list(features) for name, features in GROUP_FEATURES.items()}

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for group_name in self.schema.groups.keys():
            names.extend(GROUP_FEATURES[group_name])
        return names

    def export_metadata(self) -> dict[str, object]:
        return {
            "version": self.version,
            "dim": int(self.schema.dim),
            "groups": {name: int(size) for name, size in self.schema.groups.items()},
            "group_features": self.group_features,
            "feature_names": self.feature_names,
        }

    def build(
        self,
        state: StepState,
        summary: SummaryBundle,
    ) -> StateBuildOutput:
        latent = _as_video_latent(state.latent).float()
        self._assert_finite_tensor(latent, name="latent", state=state)
        phase = self._build_phase(state)
        history = self._build_history(state)
        structure = self._build_structure(latent, summary)
        motion = self._build_motion(latent)
        efficiency = self._build_efficiency(state.exec_feedback)
        attention = self._build_attention(state.exec_feedback)
        named = {
            "phase": phase.tolist(),
            "history": history.tolist(),
            "structure": structure.tolist(),
            "motion": motion.tolist(),
            "efficiency": efficiency.tolist(),
            "attention": attention.tolist(),
        }
        vector = torch.cat([phase, history, structure, motion, efficiency, attention], dim=0)
        self._assert_finite_tensor(
            vector,
            name="state_vector",
            state=state,
            feature_names=self.feature_names,
        )
        return StateBuildOutput(vector=vector, named=named)

    def _build_phase(self, state: StepState) -> torch.Tensor:
        progress = self._progress(state.timestep, state.total_steps)
        phase_early = 1.0 if progress <= PHASE_EARLY_MAX else 0.0
        phase_mid = 1.0 if PHASE_EARLY_MAX < progress <= PHASE_MID_MAX else 0.0
        phase_late = 1.0 if progress > PHASE_MID_MAX else 0.0
        resolution_is_high = 1.0 if str(state.resolution_phase).lower() == "high" else 0.0
        return torch.tensor(
            [
                progress,
                phase_early,
                phase_mid,
                phase_late,
                resolution_is_high,
            ],
            dtype=torch.float32,
        )

    def _build_history(self, state: StepState) -> torch.Tensor:
        prev_sparse_level = 0.0
        if state.prev_action is not None:
            prev_sparse_level = self._normalize_sparse_level(state.prev_action.a_sparse_profile)
        values = [
            prev_sparse_level,
            self._normalize_count(state.steps_since_sparse_change, state.total_steps),
            self._normalize_count(state.steps_since_resolution_change, state.total_steps),
            self._normalize_count(state.lowres_streak, state.total_steps),
            1.0 if state.decision_reused else 0.0,
        ]
        return torch.tensor(values, dtype=torch.float32)

    def _build_structure(self, latent: torch.Tensor, summary: SummaryBundle) -> torch.Tensor:
        coarse = self._coarse_energy_map(latent)
        mean_energy = coarse.mean()
        total_energy = coarse.sum()
        concentration_t = coarse.max() / (mean_energy + EPS)
        dispersion_t = coarse.std(unbiased=False) / (mean_energy + EPS)
        com_x_t, com_y_t = self._center_of_mass_tensors(coarse, total_energy=total_energy)
        lr_imbalance_t = self._half_plane_imbalance_tensor(
            coarse[:, :2].sum(),
            coarse[:, 2:].sum(),
            total_energy=total_energy,
        )
        tb_imbalance_t = self._half_plane_imbalance_tensor(
            coarse[:2, :].sum(),
            coarse[2:, :].sum(),
            total_energy=total_energy,
        )
        coarse_grad_t = self._coarse_gradient_tensor(coarse) / (mean_energy + EPS)
        spatial_area_ratio = self._spatial_area_ratio(summary)
        tensor_values = torch.stack(
            [
                concentration_t,
                dispersion_t,
                com_x_t,
                com_y_t,
                lr_imbalance_t,
                tb_imbalance_t,
                coarse_grad_t,
            ]
        ).detach().cpu().float().tolist()
        concentration, dispersion, com_x, com_y, lr_imbalance, tb_imbalance, coarse_grad = [
            float(value) for value in tensor_values
        ]
        values = [
            spatial_area_ratio,
            concentration,
            dispersion,
            com_x,
            com_y,
            lr_imbalance,
            tb_imbalance,
            coarse_grad,
        ]
        return torch.tensor(values, dtype=torch.float32)

    def _build_motion(self, latent: torch.Tensor) -> torch.Tensor:
        frames = latent.shape[2]
        if frames < 2:
            return torch.zeros(self.schema.groups["motion"], dtype=torch.float32)
        diffs = (latent[:, :, 1:] - latent[:, :, :-1]).abs()
        base_scale_t = latent.abs().mean()
        diff_mean_t = diffs.mean()
        diff_std_t = diffs.std(unbiased=False)
        frame_energy = latent.square().mean(dim=(0, 1, 3, 4))
        frame_energy_mean_t = frame_energy.mean()
        frame_energy_std_t = frame_energy.std(unbiased=False)
        peak_t = diffs.max()
        base_scale, diff_mean, diff_std, frame_energy_mean, frame_energy_std, peak = [
            float(value)
            for value in torch.stack(
                [base_scale_t, diff_mean_t, diff_std_t, frame_energy_mean_t, frame_energy_std_t, peak_t]
            )
            .detach()
            .cpu()
            .float()
            .tolist()
        ]
        values = [
            self._safe_div(diff_mean, base_scale),
            self._safe_div(diff_std, base_scale),
            self._safe_div(frame_energy_std, frame_energy_mean),
            self._safe_div(peak, base_scale),
        ]
        return torch.tensor(values, dtype=torch.float32)

    def _build_efficiency(self, feedback: ExecFeedback | None) -> torch.Tensor:
        ref = feedback or ExecFeedback()
        values = [
            float(max(0.0, ref.actual_density)),
            float(max(0.0, ref.density_ema)),
            float(math.log1p(resolve_feedback_step_cost_ms(ref))),
            float(math.log1p(resolve_feedback_latency_ema_ms(ref))),
        ]
        return torch.tensor(values, dtype=torch.float32)

    def _build_attention(self, feedback: ExecFeedback | None) -> torch.Tensor:
        payload = feedback.attention_proxy if isinstance(feedback, ExecFeedback) and isinstance(feedback.attention_proxy, dict) else {}
        values = [
            self._clamp01(payload.get("attn_eff_support_ratio")),
            self._clamp01(payload.get("attn_top1_prob")),
            self._clamp01(payload.get("attn_topk_mass_10pct")),
            self._clamp_nonnegative(payload.get("attn_logit_std")),
        ]
        return torch.tensor(values, dtype=torch.float32)

    def _progress(self, timestep: int, total_steps: int) -> float:
        if total_steps <= 1:
            return 0.0
        return float(max(0.0, min(1.0, float(timestep) / float(total_steps - 1))))

    def _normalize_count(self, value: int, total_steps: int) -> float:
        if total_steps <= 0:
            return 0.0
        return float(max(0.0, min(1.0, float(value) / float(total_steps))))

    def _normalize_sparse_level(self, profile_name: str) -> float:
        max_index = max(1, len(SPARSE_PROFILE_ORDER) - 1)
        return float(get_sparse_profile_index(profile_name) / max_index)

    def _coarse_energy_map(self, latent: torch.Tensor) -> torch.Tensor:
        energy_map = latent.square().mean(dim=(0, 1, 2), keepdim=False)
        coarse = F.adaptive_avg_pool2d(energy_map.unsqueeze(0).unsqueeze(0), (4, 4))
        return coarse.reshape(4, 4)

    def _center_of_mass_tensors(self, coarse: torch.Tensor, *, total_energy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_coords = torch.linspace(-1.0, 1.0, steps=coarse.shape[0], device=coarse.device, dtype=coarse.dtype)
        x_coords = torch.linspace(-1.0, 1.0, steps=coarse.shape[1], device=coarse.device, dtype=coarse.dtype)
        row_mass = coarse.sum(dim=1)
        col_mass = coarse.sum(dim=0)
        denom = total_energy.clamp(min=EPS)
        com_y = torch.where(total_energy <= EPS, torch.zeros_like(total_energy), (row_mass * y_coords).sum() / denom)
        com_x = torch.where(total_energy <= EPS, torch.zeros_like(total_energy), (col_mass * x_coords).sum() / denom)
        return com_x, com_y

    def _half_plane_imbalance_tensor(self, lhs: torch.Tensor, rhs: torch.Tensor, *, total_energy: torch.Tensor) -> torch.Tensor:
        return torch.abs(lhs - rhs) / total_energy.clamp(min=EPS)

    def _coarse_gradient_tensor(self, coarse: torch.Tensor) -> torch.Tensor:
        gradients: list[torch.Tensor] = []
        if coarse.shape[1] > 1:
            gradients.append(torch.abs(coarse[:, 1:] - coarse[:, :-1]).mean())
        if coarse.shape[0] > 1:
            gradients.append(torch.abs(coarse[1:, :] - coarse[:-1, :]).mean())
        if not gradients:
            return torch.zeros((), device=coarse.device, dtype=coarse.dtype)
        return torch.stack(gradients).mean()

    def _spatial_area_ratio(self, summary: SummaryBundle) -> float:
        latent_h, latent_w = summary.latent_hw
        ref_h, ref_w = summary.reference_hw
        return self._safe_div(float(latent_h * latent_w), float(max(1, ref_h * ref_w)))

    def _safe_div(self, numer: float, denom: float) -> float:
        return float(numer / max(float(denom), EPS))

    def _clamp01(self, value: object) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return float(max(0.0, min(1.0, numeric)))

    def _clamp_nonnegative(self, value: object) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return float(max(0.0, numeric))

    def _assert_finite_tensor(
        self,
        tensor: torch.Tensor,
        *,
        name: str,
        state: StepState,
        feature_names: list[str] | None = None,
        max_examples: int = 8,
    ) -> None:
        if not finite_guard_enabled():
            return
        work = tensor.detach().cpu().float()
        finite_mask = torch.isfinite(work)
        if bool(finite_mask.all().item()):
            return
        bad_indices = torch.nonzero(~finite_mask, as_tuple=False)
        examples: list[str] = []
        for raw_idx in bad_indices[: int(max_examples)]:
            idx = tuple(int(value) for value in raw_idx.tolist())
            if feature_names is not None and len(idx) == 1 and 0 <= idx[0] < len(feature_names):
                label = str(feature_names[idx[0]])
            else:
                label = str(idx)
            value = float(work[idx].item())
            examples.append(f"{label}={value!r}")
        bad_count = int(bad_indices.shape[0])
        total = int(work.numel())
        raise RuntimeError(
            f"Non-finite {name} in StateBuilder.build "
            f"timestep={int(state.timestep)} total_steps={int(state.total_steps)} "
            f"resolution_phase={state.resolution_phase!r} bad={bad_count}/{total} "
            f"examples={examples}"
        )

    def _validate_groups(self) -> None:
        if tuple(self.schema.groups.keys()) != tuple(GROUP_FEATURES.keys()):
            raise ValueError(
                f"Schema groups {tuple(self.schema.groups.keys())} do not match metadata groups {tuple(GROUP_FEATURES.keys())}."
            )
        for group_name, expected_size in self.schema.groups.items():
            actual_size = len(GROUP_FEATURES[group_name])
            if int(expected_size) != int(actual_size):
                raise ValueError(f"Group {group_name} expects {expected_size} dims but metadata defines {actual_size}.")
