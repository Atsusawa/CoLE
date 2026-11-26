from __future__ import annotations

from cole.core.types import MaskPlan
from cole.runtime.attention_bridge import AttentionBridge
from cole.runtime.wan_adapter import publish_mask_plan, record_runtime_feedback
from .wan_fast_patch import replace_sparse_forward
from .wan_processor import CoLEWanAttnProcessor


def build_bootstrap_mask_plan() -> MaskPlan:
    return MaskPlan(
        backend_name="dense",
        profile_name="dense_full",
        sparse_mode="full",
        sparse_value=None,
        estimated_density=1.0,
        selected_units=0,
        plan_signature=("bootstrap_dense_full",),
        block_mask=None,
        query_block_size=1,
        key_block_size=1,
    )


def install_cole_wan_processor(
    pipe,
    *,
    height: int,
    width: int,
    num_frames: int,
) -> AttentionBridge:
    replace_sparse_forward()

    context_length = 0
    num_frame_patches = 1 + int(num_frames) // (
        pipe.vae_scale_factor_temporal * pipe.transformer.config.patch_size[0]
    )
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    frame_patches_one_frame = int(height // mod_value) * int(width // mod_value)
    num_layers = len(pipe.transformer.blocks)
    bridge = AttentionBridge()

    def _plan_provider():
        return getattr(pipe, "cole_mask_plan", None)

    def _feedback_sink(feedback):
        record_runtime_feedback(pipe, feedback)

    for layer_idx, block in enumerate(pipe.transformer.blocks):
        if hasattr(block, "attn1") and hasattr(block.attn1, "set_processor"):
            processor = CoLEWanAttnProcessor(
                layer_idx=layer_idx,
                context_length=context_length,
                num_frame=num_frame_patches,
                frame_size=frame_patches_one_frame,
                num_layers=num_layers,
                attention_bridge=bridge,
                plan_provider=_plan_provider,
                feedback_sink=_feedback_sink,
            )
            block.attn1.set_processor(processor)

    publish_mask_plan(
        pipe,
        build_bootstrap_mask_plan(),
    )
    return bridge
