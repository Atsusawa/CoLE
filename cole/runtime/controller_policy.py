from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from cole.core.program import ACTIONS, WINDOWS, StrategyProgram, file_sha256
from cole.core.reuse_presets import build_reuse_config
from cole.core.sparse_windows import (
    SparseWindow,
    SPARSE_BASELINE_TOPK,
    build_sparse_windows,
    sparse_profile_for_action,
)
from cole.runtime.controller_model import SharedResolutionSparsePolicy, SharedResolutionSparsePolicyConfig
from cole.runtime.program import StrategyAdapter
from cole.runtime.reuse_readout import REUSE_ACTIONS, ReuseReadout

PROBE_STEP = 14
DEFAULT_REUSE = "balanced"
_CHECKPOINT_SCHEMAS = ("cole_controller_v1", "resolution_sparse_shared_tensor_policy_checkpoint_v1")


@dataclass(frozen=True)
class WindowDecision:
    window: SparseWindow
    action: str
    action_probabilities: dict[str, float]
    profile: str
    baseline_topk: float


class ControllerPolicy:
    """Frozen Resolution and Sparsity readouts with program or learned reuse selection."""

    def __init__(
        self, program: StrategyProgram, *, checkpoint_path: Path, device: torch.device,
        total_steps: int, switch_step: int, use_program: bool = True,
        reuse_checkpoint: Path | None = None,
    ) -> None:
        if file_sha256(checkpoint_path) != program.checkpoint_sha256:
            raise ValueError("Controller checkpoint SHA-256 differs from the bundled program.")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            payload.get("schema") not in _CHECKPOINT_SCHEMAS
            or payload.get("model_type") != SharedResolutionSparsePolicy.model_type
            or tuple(payload["switch_steps"]) != ACTIONS["switch"]
        ):
            raise ValueError("Invalid frozen Resolution and Sparsity checkpoint.")
        self.device = device
        self.total_steps = total_steps
        self.initial_switch_step = switch_step
        config = SharedResolutionSparsePolicyConfig.from_dict(payload["model_config"])
        self.model = SharedResolutionSparsePolicy(config)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.reuse_readout = None
        if reuse_checkpoint is not None:
            if use_program:
                raise ValueError("The learned reuse readout requires the CoLE-L variant.")
            self.reuse_readout = ReuseReadout.load(reuse_checkpoint, self.model).to(device).eval().requires_grad_(False)
        self.model.to(device).eval().requires_grad_(False)
        self.strategy = StrategyAdapter(program) if use_program else None
        self.initial_reuse_action = DEFAULT_REUSE if use_program else "off"
        self.attention_sample_steps = sorted({
            PROBE_STEP - 1,
            *(
                window.start_step - 1
                for step in ACTIONS["switch"]
                for window in build_sparse_windows(total_steps=total_steps, switch_step=step)
            ),
        })
        self.reset()

    def reset(self) -> None:
        if self.strategy is not None:
            self.strategy.reset()
        self.switch_step = self.initial_switch_step
        self.windows = build_sparse_windows(total_steps=self.total_steps, switch_step=self.switch_step)
        self.decisions: dict[str, WindowDecision] = {}
        self.switch_probabilities: dict[int, float] | None = None
        self.reuse_action = self.initial_reuse_action
        self.reuse_probabilities: dict[str, float] | None = None
        self.pending_reuse_update: dict | None = None

    def consume_reuse_update(self) -> dict | None:
        update = self.pending_reuse_update
        self.pending_reuse_update = None
        return update

    def _switch_candidates(self) -> torch.Tensor:
        rows = []
        denom = self.total_steps - 1
        for step in ACTIONS["switch"]:
            progress = step / denom
            rows.append((
                progress, step / self.total_steps, (self.total_steps - step) / self.total_steps,
                (step - PROBE_STEP) / denom, math.sin(math.pi * progress), math.cos(math.pi * progress),
            ))
        return torch.tensor([rows], device=self.device, dtype=torch.float32)

    def _sparse_candidates(self) -> torch.Tensor:
        signed = (0.0, -0.5, 0.5, 1.0)
        rows = [(value, abs(value), float(value < 0.0), float(value > 0.0)) for value in signed]
        return torch.tensor([rows], device=self.device, dtype=torch.float32)

    def _attention_tensor(self, patches: dict[int, torch.Tensor] | None) -> torch.Tensor:
        tensors = []
        shape = None
        for _, patch in sorted((patches or {}).items(), key=lambda item: int(item[0])):
            if not isinstance(patch, torch.Tensor) or patch.ndim != 5:
                continue
            if shape is None:
                shape = patch.shape
            if patch.shape == shape:
                tensors.append(patch.detach().to(device=self.device))
        if not tensors:
            raise ValueError("The controller requires sampled attention probability patches at decision steps.")
        return torch.stack(tensors, dim=1)

    @staticmethod
    def _sparse_decision_features(context: torch.Tensor) -> torch.Tensor:
        context = torch.nan_to_num(context.float(), nan=0.0, posinf=0.0, neginf=0.0)
        progress = context[:, 0].clamp(0.0, 1.0)
        sigma = context[:, 1].clamp_min(0.0)
        return torch.stack(
            [progress, sigma, torch.log1p(sigma), torch.sin(math.pi * progress), torch.cos(math.pi * progress)],
            dim=-1,
        )

    def _select_switch_and_reuse(
        self, *, observations: dict[str, float], latent: torch.Tensor,
        patches: dict[int, torch.Tensor] | None, context: torch.Tensor, prompt: list[float],
    ) -> None:
        context = context.detach().to(device=self.device, dtype=torch.float32)
        probe = self.model.encode(
            latent.detach().to(device=self.device),
            self._attention_tensor(patches),
            context_features=context,
            prompt_embedding=torch.tensor([prompt], device=self.device, dtype=torch.float32),
        )
        logits = self.model.forward_switch_from_features(
            probe, candidate_features=self._switch_candidates(),
        ).squeeze(0)
        switches = ACTIONS["switch"]
        switch = switches[int(torch.argmax(logits).item())]
        if self.strategy is not None:
            switch = int(self.strategy.choose("switch", switch, observations=observations))
        self.switch_step = switch
        probabilities = torch.softmax(logits.float(), dim=-1).detach().cpu().tolist()
        self.switch_probabilities = dict(zip(switches, probabilities))
        self.windows = build_sparse_windows(total_steps=self.total_steps, switch_step=switch)

        force_full = {*range(PROBE_STEP + 1), *self.attention_sample_steps}
        if self.strategy is not None and self.strategy.by_head[("reuse", None)]:
            self.reuse_action = str(self.strategy.choose(
                "reuse", DEFAULT_REUSE, observations=observations, selected_switch_step=switch,
            ))
        elif self.reuse_readout is not None:
            force_full.update((switch - 1, switch, switch + 1, self.total_steps - 1))
            force_full.update(window.start_step for window in self.windows)
            reuse_logits = self.reuse_readout(
                probe, context=context, switch_step=switch, total_steps=self.total_steps,
                force_full_steps=force_full,
            ).squeeze(0)
            self.reuse_action = REUSE_ACTIONS[int(torch.argmax(reuse_logits).item())]
            self.reuse_probabilities = dict(zip(REUSE_ACTIONS, reuse_logits.softmax(dim=-1).cpu().tolist()))
        if self.reuse_action != self.initial_reuse_action:
            # Retain the latest full prediction as the reference for the selected preset.
            self.pending_reuse_update = build_reuse_config(
                self.reuse_action, total_steps=self.total_steps, switch_step=switch, force_full_steps=force_full,
            )

    @torch.no_grad()
    def decide(
        self, *, step_index: int, state_named: dict[str, list[float]], latent: torch.Tensor,
        patches: dict[int, torch.Tensor] | None, context: torch.Tensor, prompt: list[float],
    ) -> WindowDecision | None:
        observations = (
            self.strategy.observations(step_index=step_index, state_named=state_named)
            if self.strategy is not None else {}
        )
        if step_index == PROBE_STEP:
            self._select_switch_and_reuse(
                observations=observations, latent=latent,
                patches=patches, context=context, prompt=prompt,
            )
        window = next((item for item in self.windows if item.contains(step_index)), None)
        if window is None:
            return None
        decision = self.decisions.get(window.name)
        if decision is None:
            context_input = context.detach().to(device=self.device, dtype=torch.float32)
            raw_logits = self.model.forward_sparse(
                latent.detach().to(device=self.device),
                self._attention_tensor(patches),
                context_features=context_input,
                prompt_embedding=torch.tensor([prompt], device=self.device, dtype=torch.float32),
                window_indices=torch.tensor([WINDOWS.index(window.name)], device=self.device),
                decision_features=self._sparse_decision_features(context_input),
                candidate_features=self._sparse_candidates(),
            ).squeeze(0)
            probabilities = torch.softmax(raw_logits, dim=-1).detach().cpu().float().tolist()
            actions = ACTIONS["sparse"]
            incumbent_index = max(range(len(actions)), key=probabilities.__getitem__)
            action = actions[incumbent_index]
            if self.strategy is not None:
                action = str(self.strategy.choose(
                    "sparse", action, observations=observations, window=window.name,
                    selected_switch_step=self.switch_step if self.switch_probabilities is not None else None,
                ))
            baseline = SPARSE_BASELINE_TOPK
            decision = WindowDecision(
                window, action, dict(zip(actions, probabilities)),
                sparse_profile_for_action(baseline_topk=baseline, action=action), baseline,
            )
            self.decisions[window.name] = decision
        if decision.window != window:
            decision = WindowDecision(
                window, decision.action, decision.action_probabilities, decision.profile, decision.baseline_topk,
            )
            self.decisions[window.name] = decision
        return decision
