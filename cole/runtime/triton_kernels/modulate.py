import torch
import triton
import triton.language as tl

from .utils import flatten_if_batched


@triton.jit
def _modulate_shift_fwd_fused(X, Y, SCALE, SHIFT, x_stride, y_stride, M, N: tl.constexpr, N2: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]
    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * y_stride + cols[None, :]
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(SCALE + cols, mask=col_mask, other=0.0)
    shift = tl.load(SHIFT + cols, mask=col_mask, other=0.0)
    y = x * (1 + scale) + shift
    tl.store(y_ptr, y.to(Y.type.element_ty), mask=mask)


def triton_modulate_shift_forward(x, scale, shift, output_dtype=torch.float32):
    [x], batched, batch_size = flatten_if_batched(x)
    M, N = x.shape
    y = torch.empty_like(x, dtype=output_dtype)
    N2 = triton.next_power_of_2(N)
    block_m = 32 if N <= 512 else 1
    _modulate_shift_fwd_fused[(triton.cdiv(M, block_m),)](
        x,
        y,
        scale,
        shift,
        x.stride(0),
        y.stride(0),
        M,
        N,
        N2,
        num_warps=8,
        BLOCK_M=block_m,
    )
    if batched:
        y = y.reshape(batch_size, -1, y.shape[-1])
    return y


@triton.jit
def _modulate_gate_residual_fwd_fused(R, X, Y, GATE, r_stride, x_stride, y_stride, M, N: tl.constexpr, N2: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]
    r_ptr = R + rows[:, None] * r_stride + cols[None, :]
    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * y_stride + cols[None, :]
    r = tl.load(r_ptr, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GATE + cols, mask=col_mask, other=0.0)
    y = r + x * gate
    tl.store(y_ptr, y.to(Y.type.element_ty), mask=mask)


def triton_modulate_gate_residual_forward(residual, x, gate, output_dtype=torch.float32):
    residuals, batched, batch_size = flatten_if_batched(residual, x)
    residual, x = residuals
    M, N = residual.shape
    y = torch.empty_like(residual, dtype=output_dtype)
    N2 = triton.next_power_of_2(N)
    block_m = 32 if N <= 512 else 1
    _modulate_gate_residual_fwd_fused[(triton.cdiv(M, block_m),)](
        residual,
        x,
        y,
        gate,
        residual.stride(0),
        x.stride(0),
        y.stride(0),
        M,
        N,
        N2,
        num_warps=8,
        BLOCK_M=block_m,
    )
    if batched:
        y = y.reshape(batch_size, -1, y.shape[-1])
    return y

