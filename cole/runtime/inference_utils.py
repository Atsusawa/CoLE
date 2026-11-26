from __future__ import annotations

from contextlib import nullcontext
import torch
from cole.core.sparse_profiles import get_sparse_profile_spec
from cole.core.types import MaskPlan, SummaryBundle
from cole.runtime.wan_adapter import consume_runtime_feedback, publish_mask_plan


def encode_prompt_once(
    pipe,
    *,
    prompt: str,
    negative_prompt: str | None,
    device: torch.device,
    guidance_scale: float,
    max_sequence_length: int = 512,
):
    transformer_dtype = pipe.transformer.dtype if pipe.transformer is not None else pipe.transformer_2.dtype
    configured_max_length = getattr(getattr(pipe.transformer, "config", None), "max_text_seq_length", None)
    if configured_max_length is not None:
        max_sequence_length = min(int(max_sequence_length), int(configured_max_length))
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=bool(float(guidance_scale) > 1.0),
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=transformer_dtype,
        )
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
    return prompt_embeds, negative_prompt_embeds


def build_generator(device_str: str, seed: int) -> torch.Generator:
    generator_device = "cpu" if str(device_str) == "cpu" else torch.device(device_str)
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


def infer_num_latent_frames(pipe, *, num_frames: int) -> int:
    return (int(num_frames) - 1) // int(pipe.vae_scale_factor_temporal) + 1


def infer_highres_latent_hw(pipe, *, height: int, width: int) -> tuple[int, int]:
    return (
        int(height) // int(pipe.vae_scale_factor_spatial),
        int(width) // int(pipe.vae_scale_factor_spatial),
    )


def infer_latent_spatial_multiple(pipe) -> int:
    patch_size = getattr(pipe.transformer.config, "patch_size", (1, 1))
    if isinstance(patch_size, int):
        return max(1, int(patch_size))
    if len(patch_size) < 2:
        return 1
    return max(1, int(patch_size[1]))


def scale_latent_hw(
    highres_hw: tuple[int, int],
    *,
    low_res_scale: float,
    spatial_multiple: int = 1,
) -> tuple[int, int]:
    target_h = max(1, int(highres_hw[0] * float(low_res_scale)))
    target_w = max(1, int(highres_hw[1] * float(low_res_scale)))
    if int(spatial_multiple) > 1:
        target_h = max(int(spatial_multiple), target_h - (target_h % int(spatial_multiple)))
        target_w = max(int(spatial_multiple), target_w - (target_w % int(spatial_multiple)))
    return int(target_h), int(target_w)


def sample_noise_latents(
    *,
    num_channels_latents: int,
    num_latent_frames: int,
    latent_hw: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randn(
        (1, int(num_channels_latents), int(num_latent_frames), int(latent_hw[0]), int(latent_hw[1])),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def prepare_initial_latents(
    pipe,
    *,
    height: int,
    width: int,
    num_frames: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    low_res_scale: float = 1.0,
    direct_low_res_noise: bool = False,
) -> torch.Tensor:
    num_channels_latents = int(getattr(pipe.transformer.config, "in_channels", 16))
    if bool(direct_low_res_noise) and float(low_res_scale) < 1.0:
        highres_hw = infer_highres_latent_hw(pipe, height=int(height), width=int(width))
        latent_hw = scale_latent_hw(
            highres_hw,
            low_res_scale=float(low_res_scale),
            spatial_multiple=infer_latent_spatial_multiple(pipe),
        )
        return sample_noise_latents(
            num_channels_latents=num_channels_latents,
            num_latent_frames=infer_num_latent_frames(pipe, num_frames=int(num_frames)),
            latent_hw=latent_hw,
            dtype=dtype,
            device=device,
            generator=generator,
        )
    native_latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=num_channels_latents,
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        dtype=dtype,
        device=device,
        generator=generator,
        latents=None,
    )
    return native_latents


def warmup_backend_once(
    pipe,
    *,
    latents: torch.Tensor,
    timestep: int,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    guidance_scale: float,
) -> None:
    if pipe.transformer is None:
        return
    do_cfg = bool(float(guidance_scale) > 1.0)
    transformer_dtype = pipe.transformer.dtype
    latent_model_input = latents.to(transformer_dtype)
    t = torch.tensor([int(timestep)], device=latent_model_input.device, dtype=torch.long).expand(
        latent_model_input.shape[0]
    )
    cond_ctx = pipe.transformer.cache_context("cond") if hasattr(pipe.transformer, "cache_context") else nullcontext()
    uncond_ctx = pipe.transformer.cache_context("uncond") if hasattr(pipe.transformer, "cache_context") else nullcontext()
    with torch.no_grad():
        with cond_ctx:
            _ = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]
        if do_cfg and negative_prompt_embeds is not None:
            with uncond_ctx:
                _ = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=t,
                encoder_hidden_states=negative_prompt_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]
    _ = consume_runtime_feedback(pipe)


def parse_warmup_profiles(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    seen: set[str] = set()
    profiles: list[str] = []
    for item in str(raw_value).split(","):
        name = item.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        profiles.append(name)
    return profiles


def build_profile_mask_plan(
    profile_name: str,
    *,
    total_units: int,
    summary: SummaryBundle | None = None,
) -> MaskPlan:
    spec = get_sparse_profile_spec(str(profile_name))
    selected_units = int(round(float(spec.estimated_density) * float(total_units)))
    latent_hw = tuple(summary.latent_hw) if summary is not None else (0, 0)
    reference_hw = tuple(summary.reference_hw) if summary is not None else (0, 0)
    block_grid_hw = tuple(summary.block_grid_hw) if summary is not None else (0, 0)
    approx_block_hw = tuple(summary.approx_block_hw) if summary is not None else (0, 0)
    return MaskPlan(
        backend_name=spec.backend_name,
        profile_name=spec.name,
        sparse_mode=spec.sparse_mode,
        sparse_value=spec.sparse_value,
        estimated_density=float(spec.estimated_density),
        selected_units=selected_units,
        plan_signature=(spec.name, spec.backend_name, spec.sparse_mode, spec.sparse_value, spec.name, latent_hw, block_grid_hw),
        block_mask=None,
        query_block_size=1,
        key_block_size=1,
        latent_hw=latent_hw,
        reference_hw=reference_hw,
        planner_block_grid_hw=block_grid_hw,
        planner_block_hw=approx_block_hw,
    )


def warmup_backend_profiles(
    pipe,
    *,
    latents: torch.Tensor,
    timestep: int,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    guidance_scale: float,
    profile_names: list[str],
    reference_hw: tuple[int, int] | None = None,
    resolution_aware_summary: bool = True,
) -> list[dict[str, object]]:
    if pipe.transformer is None:
        return []
    profile_names = parse_warmup_profiles(",".join(profile_names))
    if not profile_names:
        return []

    from cole.sparse.summary_builder import SummaryBuilder, SummaryBuilderConfig

    summary = SummaryBuilder(
        SummaryBuilderConfig(resolution_aware=bool(resolution_aware_summary))
    ).build(latents, reference_hw=reference_hw)
    total_units = int(summary.block_scores.shape[0] * summary.block_scores.shape[1])
    rows: list[dict[str, object]] = []
    for profile_name in profile_names:
        mask_plan = build_profile_mask_plan(profile_name, total_units=total_units, summary=summary)
        publish_mask_plan(pipe, mask_plan)
        import time

        perf_start = time.perf_counter()
        warmup_backend_once(
            pipe,
            latents=latents,
            timestep=timestep,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
        )
        elapsed_sec = time.perf_counter() - perf_start
        rows.append(
            {
                "profile_name": profile_name,
                "elapsed_sec": float(elapsed_sec),
                "mask_plan": mask_plan.to_dict(),
            }
        )
    return rows


