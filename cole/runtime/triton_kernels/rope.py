import torch
import triton
import triton.language as tl


@triton.jit
def _qk_rope_cossin_inplace_kernel(
    Q,
    K,
    COS,
    SIN,
    TOTAL: tl.constexpr,
    Q_STRIDE_B: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    Q_STRIDE_S: tl.constexpr,
    Q_STRIDE_D: tl.constexpr,
    K_STRIDE_B: tl.constexpr,
    K_STRIDE_H: tl.constexpr,
    K_STRIDE_S: tl.constexpr,
    K_STRIDE_D: tl.constexpr,
    COS_STRIDE_S: tl.constexpr,
    COS_STRIDE_D: tl.constexpr,
    SIN_STRIDE_S: tl.constexpr,
    SIN_STRIDE_D: tl.constexpr,
    HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    pair_d = offsets % HALF_DIM
    seq = (offsets // HALF_DIM) % SEQ_LEN
    head = (offsets // (HALF_DIM * SEQ_LEN)) % HEADS
    batch = offsets // (HALF_DIM * SEQ_LEN * HEADS)

    cos = tl.load(COS + seq * COS_STRIDE_S + pair_d * COS_STRIDE_D, mask=mask, other=1.0).to(tl.float32)
    sin = tl.load(SIN + seq * SIN_STRIDE_S + pair_d * SIN_STRIDE_D, mask=mask, other=0.0).to(tl.float32)

    q_base = batch * Q_STRIDE_B + head * Q_STRIDE_H + seq * Q_STRIDE_S + (pair_d * 2) * Q_STRIDE_D
    k_base = batch * K_STRIDE_B + head * K_STRIDE_H + seq * K_STRIDE_S + (pair_d * 2) * K_STRIDE_D

    q_real = tl.load(Q + q_base, mask=mask, other=0.0).to(tl.float32)
    q_imag = tl.load(Q + q_base + Q_STRIDE_D, mask=mask, other=0.0).to(tl.float32)
    k_real = tl.load(K + k_base, mask=mask, other=0.0).to(tl.float32)
    k_imag = tl.load(K + k_base + K_STRIDE_D, mask=mask, other=0.0).to(tl.float32)

    tl.store(Q + q_base, q_real * cos - q_imag * sin, mask=mask)
    tl.store(Q + q_base + Q_STRIDE_D, q_imag * cos + q_real * sin, mask=mask)
    tl.store(K + k_base, k_real * cos - k_imag * sin, mask=mask)
    tl.store(K + k_base + K_STRIDE_D, k_imag * cos + k_real * sin, mask=mask)


def triton_apply_qk_rope_inplace(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.shape != key.shape:
        raise ValueError(f"query/key shape mismatch for RoPE: {tuple(query.shape)} vs {tuple(key.shape)}")
    if query.dim() != 4:
        raise ValueError(f"query/key must be 4D for RoPE, got {query.dim()}D")
    if query.shape[-1] % 2 != 0:
        raise ValueError(f"query/key head dim must be even for RoPE, got {query.shape[-1]}")
    if not query.is_cuda or not key.is_cuda:
        raise ValueError("Triton RoPE requires CUDA query/key tensors")
    if cos.shape != sin.shape:
        raise ValueError(f"cos/sin shape mismatch for RoPE: {tuple(cos.shape)} vs {tuple(sin.shape)}")

    batch, heads, seq_len, head_dim = query.shape
    half_dim = int(head_dim) // 2
    if cos.dim() != 2 or int(cos.shape[0]) != int(seq_len) or int(cos.shape[1]) != half_dim:
        raise ValueError(
            "Triton RoPE expects cos/sin shaped [seq_len, head_dim // 2], "
            f"got {tuple(cos.shape)} for query {tuple(query.shape)}"
        )

    if cos.device != query.device or cos.dtype != torch.float32:
        cos = cos.to(device=query.device, dtype=torch.float32)
    if sin.device != query.device or sin.dtype != torch.float32:
        sin = sin.to(device=query.device, dtype=torch.float32)
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()

    total = int(batch) * int(heads) * int(seq_len) * int(half_dim)
    block = 256
    _qk_rope_cossin_inplace_kernel[(triton.cdiv(total, block),)](
        query,
        key,
        cos,
        sin,
        total,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        int(heads),
        int(seq_len),
        int(half_dim),
        BLOCK=block,
        num_warps=4,
    )
    return query, key
