from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from cole.common.timing import update_feedback_ema
from cole.core.base_schedule import BaseSchedule
from cole.core.diffusion_context import build_diffusion_context
from cole.core.feedback import ExecFeedback, clone_attention_prob_patches
from cole.core.state_builder import StateBuilder
from cole.core.types import MaskPlan, ProjectedAction, StepState
from cole.runtime.controller_policy import ControllerPolicy, WindowDecision
from cole.runtime.resolution import ResolutionManager
from cole.runtime.resolution_transition import resolve_sigma_for_timestep
from cole.runtime.wan_adapter import consume_runtime_feedback, publish_mask_plan
from cole.sparse.sparse_profile_planner import SparseProfilePlanner
from cole.sparse.summary_builder import SummaryBuilder


@dataclass
class StepDecision:
    projected_action: ProjectedAction
    mask_plan: MaskPlan
    latents: torch.Tensor
    resolution_event: str | None
    window: WindowDecision | None
    executed_feedback: ExecFeedback | None

    def trace_row(self, *, step: int, timestep: int) -> dict:
        row = {
            "step_index": step,
            "timestep": timestep,
            "projected_action": self.projected_action.to_dict(),
            "mask_plan": self.mask_plan.to_dict(),
            "resolution_event": self.resolution_event,
        }
        if self.window is not None:
            row["sparsity"] = {
                **self.window.window.to_dict(),
                "action": self.window.action,
                "action_probabilities": self.window.action_probabilities,
                "baseline_topk": self.window.baseline_topk,
            }
        return row


class OnlineController:
    def __init__(
        self, *, policy: ControllerPolicy, prompt_embedding: list[float],
        highres_hw: tuple[int, int], low_res_scale: float,
    ) -> None:
        self.policy = policy
        self.policy.reset()
        self.prompt_embedding = prompt_embedding
        self.resolution = ResolutionManager(low_res_scale=low_res_scale, highres_hw=highres_hw)
        self.state_builder = StateBuilder()
        self.summary_builder = SummaryBuilder()
        self.schedule = BaseSchedule()
        self.mask_builder = SparseProfilePlanner()
        self.prev_action: ProjectedAction | None = None
        self.prev_feedback: ExecFeedback | None = None
        self.step_index = 0
        self.last_sparse_change_step = 0
        self.last_resolution_change_step = 0
        self.lowres_streak = 0

    @torch.no_grad()
    def prewarm(self, pipe, *, timestep: int, latent: torch.Tensor) -> StepDecision:
        latent, event = self.resolution.prepare_initial_latent(latent)
        decision = self._decide(pipe, timestep=timestep, latent=latent, prewarm=True)
        decision.resolution_event = event or "prewarm_plan_published"
        self.step_index = 1
        return decision

    @torch.no_grad()
    def step(self, pipe, *, timestep: int, latent: torch.Tensor) -> StepDecision:
        result = self._decide(pipe, timestep=timestep, latent=latent, prewarm=False)
        self.step_index += 1
        return result

    def _decide(self, pipe, *, timestep: int, latent: torch.Tensor, prewarm: bool) -> StepDecision:
        model_feedback = None if prewarm else consume_runtime_feedback(pipe)
        if isinstance(model_feedback, ExecFeedback):
            self.prev_feedback = model_feedback

        step = self.step_index
        total_steps = self.policy.total_steps
        schedule_step = min(step, total_steps - 1)
        if self.prev_action is None and 0 <= timestep < total_steps:
            schedule_step = timestep
        summary_start = time.perf_counter()
        summary = self.summary_builder.build(latent, reference_hw=self.resolution.state.highres_hw)
        summary_ms = (time.perf_counter() - summary_start) * 1000.0
        phase = self.resolution.get_phase()
        state = StepState(
            timestep=schedule_step,
            total_steps=total_steps,
            resolution_phase=phase,
            latent=latent,
            prev_action=self.prev_action,
            exec_feedback=self.prev_feedback,
            steps_since_sparse_change=0 if self.prev_action is None else step - self.last_sparse_change_step,
            steps_since_resolution_change=step - self.last_resolution_change_step,
            lowres_streak=self.lowres_streak,
            decision_reused=False,
        )
        named = self.state_builder.build(state, summary).named
        base_plan = self.schedule.resolve(
            timestep=timestep, total_steps=total_steps, resolution_phase=phase, step_index=schedule_step,
        )
        sigma = resolve_sigma_for_timestep(pipe.scheduler, timestep=timestep, sample=latent)
        context = build_diffusion_context(
            step_index=step, total_steps=total_steps, noise_sigma=sigma.detach().reshape(-1)[0],
            batch_size=latent.shape[0], device=latent.device, dtype=torch.float32,
        )
        decision = self.policy.decide(
            step_index=step, state_named=named, latent=latent,
            patches=None if self.prev_feedback is None else self.prev_feedback.attention_prob_patches,
            context=context, prompt=self.prompt_embedding,
        )
        switch = base_plan.promote_resolution if decision is None else (step >= self.policy.switch_step)
        projected = ProjectedAction(
            a_res=int(switch and self.resolution.can_switch(step_index=step)),
            a_sparse_profile=base_plan.default_sparse_profile if decision is None else decision.profile,
        )
        mask_plan = self.mask_builder.build(summary, base_plan, projected)
        event = None
        if not prewarm:
            latent, event = self.resolution.apply(latent, a_res=projected.a_res, step_index=step, total_steps=total_steps)
        self.prev_feedback = self._build_controller_feedback(
            mask_plan=mask_plan, model_feedback=model_feedback, summary_build_ms=summary_ms,
        )
        if self.prev_action is None or self.prev_action.a_sparse_profile != projected.a_sparse_profile:
            self.last_sparse_change_step = step
        self.prev_action = projected
        if event == "switch_to_highres":
            self.last_resolution_change_step = step
        self.lowres_streak = 0 if self.resolution.get_phase() == "high" else self.lowres_streak + 1
        publish_mask_plan(pipe, mask_plan)
        return StepDecision(projected, mask_plan, latent, event, decision, model_feedback)

    def _build_controller_feedback(
        self,
        *,
        mask_plan: MaskPlan,
        model_feedback: ExecFeedback | None,
        summary_build_ms: float,
    ) -> ExecFeedback:
        feedback = ExecFeedback(
            summary_build_ms=float(summary_build_ms),
            backend_prepare_ms=0.0,
            backend_run_ms=0.0,
            total_attn_ms=float(summary_build_ms),
            step_wall_ms=0.0,
            actual_density=float(mask_plan.estimated_density),
            selected_units=int(mask_plan.selected_units),
            backend_name=str(mask_plan.backend_name),
            executed_plan_signature=str(tuple(mask_plan.plan_signature)),
            executed_plan_density=float(mask_plan.estimated_density),
            backend_is_sparse=bool(mask_plan.backend_name != "dense"),
            decision_reused=False,
        )
        if isinstance(model_feedback, ExecFeedback):
            feedback.backend_prepare_ms = float(model_feedback.backend_prepare_ms)
            feedback.backend_run_ms = float(model_feedback.backend_run_ms)
            feedback.total_attn_ms = float(summary_build_ms + model_feedback.total_attn_ms)
            feedback.step_wall_ms = float(model_feedback.step_wall_ms)
            feedback.actual_density = float(model_feedback.actual_density)
            feedback.selected_units = int(model_feedback.selected_units)
            feedback.backend_name = str(model_feedback.backend_name)
            feedback.executed_plan_signature = str(model_feedback.executed_plan_signature)
            feedback.executed_plan_density = float(model_feedback.executed_plan_density)
            feedback.backend_is_sparse = bool(model_feedback.backend_is_sparse)
            feedback.backend_first_call = bool(model_feedback.backend_first_call)
            feedback.attention_proxy = (
                None
                if not isinstance(model_feedback.attention_proxy, dict)
                else dict(model_feedback.attention_proxy)
            )
            feedback.attention_prob_patches = clone_attention_prob_patches(model_feedback.attention_prob_patches)
            feedback.attention_sampling_elapsed_sec = float(model_feedback.attention_sampling_elapsed_sec)
            feedback.attention_sampling_updates = int(model_feedback.attention_sampling_updates)
        return update_feedback_ema(self.prev_feedback, feedback)

