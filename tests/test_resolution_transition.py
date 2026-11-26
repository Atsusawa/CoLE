import torch
import torch.nn.functional as F

from cole.runtime.resolution_transition import (
    RESOLUTION_TRANSITION_NAME,
    ResolutionTransition,
    ResolutionTransitionConfig,
    ResolutionTransitionRuntime,
    rebuild_clean_latent,
)


class _DummyConfig:
    solver_order = 2
    flow_shift = 8.0


class _DummyScheduler:
    def __init__(self) -> None:
        self.config = _DummyConfig()
        self.timesteps = torch.tensor([90.0, 80.0, 70.0, 60.0], dtype=torch.float32)
        self.sigmas = torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float32)
        self.model_outputs = [1, 2]
        self.timestep_list = [3, 4]
        self.lower_order_nums = 1
        self.last_sample = torch.ones(1)
        self.this_order = 1
        self._step_index = 2
        self._begin_index = 6
        self.solver_p = None
        self.disable_corrector: list[int] = []

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        schedule = self.timesteps if schedule_timesteps is None else schedule_timesteps
        if isinstance(timestep, torch.Tensor):
            timestep = float(timestep.reshape(-1)[0].item())
        matches = (schedule == float(timestep)).nonzero(as_tuple=False)
        return int(matches.reshape(-1)[0].item())

    def scale_noise(self, sample, timestep, noise=None):
        idx = self.index_for_timestep(timestep, self.timesteps.to(sample.device))
        sigma = self.sigmas[idx].to(device=sample.device, dtype=sample.dtype)
        while sigma.dim() < sample.dim():
            sigma = sigma.unsqueeze(-1)
        return (1.0 - sigma) * sample + sigma * noise

    def register_to_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.config, key, value)

    def set_timesteps(self, num_inference_steps: int, device=None):
        del num_inference_steps
        shift = float(self.config.flow_shift)
        base = torch.tensor([10.0, 9.0, 8.0, 7.0], dtype=torch.float32, device=device)
        self.timesteps = base * shift
        self.model_outputs = [None, None]
        self.timestep_list = [None, None]
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = 0
        self._step_index = None
        self._begin_index = None


class _FlowSchedulerWithAlphas(_DummyScheduler):
    def __init__(self) -> None:
        super().__init__()
        self.config.prediction_type = "flow_prediction"
        self.alphas_cumprod = torch.linspace(1.0, 0.1, steps=100)


def test_clean_rebuild_uses_sigma_for_flow_prediction_with_alpha_table() -> None:
    scheduler = _FlowSchedulerWithAlphas()
    sample = torch.full((1, 2, 3), 2.0)
    model_output = torch.full_like(sample, 0.5)

    clean = rebuild_clean_latent(
        scheduler=scheduler,
        sample=sample,
        model_output=model_output,
        timestep=80,
    )

    assert torch.allclose(clean, sample - 0.8 * model_output)




def _resize_like_transition(latent: torch.Tensor, *, target_hw: tuple[int, int]) -> torch.Tensor:
    resized = F.interpolate(
        latent.float(),
        size=(int(latent.shape[-3]), int(target_hw[0]), int(target_hw[1])),
        mode="trilinear",
        align_corners=False,
    )
    return resized.to(dtype=latent.dtype)




def test_resolution_transition_rebuilds_and_re_noises_to_skipped_timestep() -> None:
    scheduler = _DummyScheduler()
    generator = torch.Generator().manual_seed(123)
    transition = ResolutionTransition(
        ResolutionTransitionConfig(
            timestep_skip=1,
            clean_rebuild_strength=1.0,
        )
    )
    latents_before_step = torch.ones(1, 4, 2, 8, 8)
    latents_after_step = torch.full((1, 4, 2, 8, 8), 2.0)
    noise_pred = torch.full_like(latents_before_step, 0.5)

    application = transition.apply(
        ResolutionTransitionRuntime(
            latents_before_step=latents_before_step,
            latents_after_step=latents_after_step,
            noise_pred=noise_pred,
            timestep=80,
            step_index=1,
            timesteps=[90, 80, 70, 60],
            scheduler=scheduler,
            target_highres_shape=(16, 16),
            generator=generator,
        )
    )

    transitioned = application.latents
    meta = application.metadata
    assert meta is not None
    assert tuple(transitioned.shape[-2:]) == (16, 16)
    assert meta.transition_applied is True
    assert meta.resolution_transition == RESOLUTION_TRANSITION_NAME
    assert meta.switch_step == 1
    assert meta.applies_to_step == 3
    assert meta.current_timestep == 80
    assert meta.restart_timestep == 60
    assert meta.timestep_skip == 1
    assert meta.clean_rebuild_strength == 1.0
    assert meta.input_latent_shape == (1, 4, 2, 8, 8)
    assert meta.output_latent_shape == (1, 4, 2, 16, 16)
    assert meta.scheduler_reset_applied is True
    assert application.skip_step_count == 1
    assert scheduler.model_outputs == [None, None]
    assert scheduler.timestep_list == [None, None]


def test_resolution_transition_strength_zero_falls_back_to_upsampled_next_latent() -> None:
    scheduler = _DummyScheduler()
    transition = ResolutionTransition(
        ResolutionTransitionConfig(
            timestep_skip=0,
            clean_rebuild_strength=0.0,
        )
    )
    latents_before_step = torch.ones(1, 4, 2, 8, 8)
    latents_after_step = torch.arange(1, 1 + (1 * 4 * 2 * 8 * 8), dtype=torch.float32).reshape(1, 4, 2, 8, 8)
    noise_pred = torch.full_like(latents_before_step, 0.5)

    application = transition.apply(
        ResolutionTransitionRuntime(
            latents_before_step=latents_before_step,
            latents_after_step=latents_after_step,
            noise_pred=noise_pred,
            timestep=80,
            step_index=1,
            timesteps=[90, 80, 70, 60],
            scheduler=scheduler,
            target_highres_shape=(16, 16),
            generator=None,
        )
    )

    expected = _resize_like_transition(latents_after_step, target_hw=(16, 16))
    assert torch.allclose(application.latents, expected)
    assert application.metadata is not None
    assert application.metadata.restart_timestep == 70
    assert application.metadata.clean_rebuild_strength == 0.0
    assert application.skip_step_count == 0




def test_resolution_transition_retimes_scheduler_and_disables_corrector_window() -> None:
    scheduler = _DummyScheduler()
    generator = torch.Generator().manual_seed(7)
    transition = ResolutionTransition(
        ResolutionTransitionConfig(
            resolution_transition=RESOLUTION_TRANSITION_NAME,
            timestep_skip=0,
            clean_rebuild_strength=1.0,
            scheduler_flow_shift=8.0,
            post_switch_flow_shift=10.0,
            corrector_disable_pre_steps=1,
            corrector_disable_post_steps=2,
        )
    )
    application = transition.apply(
        ResolutionTransitionRuntime(
            latents_before_step=torch.ones(1, 4, 2, 8, 8),
            latents_after_step=torch.full((1, 4, 2, 8, 8), 2.0),
            noise_pred=torch.full((1, 4, 2, 8, 8), 0.5),
            timestep=80,
            step_index=1,
            timesteps=[90, 80, 70, 60],
            scheduler=scheduler,
            target_highres_shape=(16, 16),
            generator=generator,
        )
    )

    assert application.metadata is not None
    assert application.metadata.resolution_transition == RESOLUTION_TRANSITION_NAME
    assert application.metadata.scheduler_flow_shift_before == 8.0
    assert application.metadata.scheduler_flow_shift_after == 10.0
    assert application.metadata.disable_corrector_steps == (0, 1, 2, 3)
    assert scheduler.disable_corrector == [0, 1, 2, 3]
    assert tuple(int(x.item()) for x in scheduler.timesteps) == (100, 90, 80, 70)
    assert application.metadata.applies_to_step == 2
    assert application.skip_step_count == 0
