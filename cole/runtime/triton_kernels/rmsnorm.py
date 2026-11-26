import torch
import triton
import triton.language as tl

from .utils import flatten_if_batched


@triton.jit
def _rms_norm_fwd_fused(X, Y, W, x_stride, y_stride, M, N: tl.constexpr, N2: tl.constexpr, eps, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]
    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * y_stride + cols[None, :]
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=1) / N
    rstd = 1 / tl.sqrt(var + eps)
    rstd = tl.reshape(rstd, (BLOCK_M, 1))
    w = tl.load(W + cols, mask=col_mask, other=0.0)
    y = (x * rstd) * w
    tl.store(y_ptr, y.to(Y.type.element_ty), mask=mask)


def triton_rmsnorm_forward(x, w, eps):
    [x], batched, batch_size = flatten_if_batched(x)
    M, N = x.shape
    y = torch.empty_like(x, dtype=x.dtype)
    N2 = triton.next_power_of_2(N)
    block_m = 32 if N <= 512 else 1
    _rms_norm_fwd_fused[(triton.cdiv(M, block_m),)](
        x,
        y,
        w,
        x.stride(0),
        y.stride(0),
        M,
        N,
        N2,
        eps,
        num_warps=8,
        BLOCK_M=block_m,
    )
    if batched:
        y = y.reshape(batch_size, -1, y.shape[-1])
    return y

