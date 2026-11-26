"""Vendored Triton kernel wrappers required by cole fast-kernel path."""

from .layernorm import triton_layernorm_forward
from .modulate import triton_modulate_gate_residual_forward, triton_modulate_shift_forward
from .rmsnorm import triton_rmsnorm_forward
from .rope import triton_apply_qk_rope_inplace

__all__ = [
    "triton_layernorm_forward",
    "triton_modulate_gate_residual_forward",
    "triton_modulate_shift_forward",
    "triton_apply_qk_rope_inplace",
    "triton_rmsnorm_forward",
]
