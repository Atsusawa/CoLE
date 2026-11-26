from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import warnings

import torch
import torch.nn.functional as F

from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm

try:
    from .triton_kernels import (
        triton_apply_qk_rope_inplace,
        triton_layernorm_forward,
        triton_modulate_gate_residual_forward,
        triton_modulate_shift_forward,
        triton_rmsnorm_forward,
    )
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    TRITON_AVAILABLE = False
else:
    TRITON_AVAILABLE = True

ENABLE_TRITON_KERNELS = TRITON_AVAILABLE and os.environ.get("WAN_DISABLE_TRITON_KERNELS", "0") != "1"
ENABLE_ROPE_KERNEL = os.environ.get("WAN_DISABLE_ROPE_KERNEL", "0") != "1"
ENABLE_TRITON_RMSNORM_KERNEL = ENABLE_TRITON_KERNELS and os.environ.get("WAN_DISABLE_RMSNORM_KERNEL", "0") != "1"
ENABLE_TRITON_BLOCK_KERNELS = ENABLE_TRITON_KERNELS and os.environ.get("WAN_DISABLE_BLOCK_KERNELS", "0") != "1"
ENABLE_FAST_KERNEL = ENABLE_TRITON_BLOCK_KERNELS
ROPE_KERNEL_IMPORT_OK = False
ROPE_KERNEL_SEARCH_DIRS: tuple[str, ...] = ()


def _candidate_local_kernel_dirs(repo_root: Path | None = None) -> list[Path]:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    for env_key in ("WAN_LOCAL_KERNELS_DIR", "COLE_LOCAL_KERNELS_DIR"):
        env_value = os.environ.get(env_key, "").strip()
        if env_value:
            candidates.append(Path(env_value).expanduser())
    candidates.append(root / "kernels" / "build")
    candidates.append(root / "kernels")
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        unique.append(path)
    return unique


def _append_local_kernel_paths(search_dirs: list[Path] | None = None) -> list[str]:
    search_dirs = _candidate_local_kernel_dirs() if search_dirs is None else list(search_dirs)
    added: list[str] = []
    for path in search_dirs:
        if not path.is_dir():
            continue
        path_str = str(path)
        if path_str in sys.path:
            continue
        sys.path.insert(0, path_str)
        added.append(path_str)
    return added


def _apply_rotary_emb_torch(hidden_states: torch.Tensor, freqs):
    if isinstance(freqs, (tuple, list)) and len(freqs) == 2:
        cos, sin = freqs
        cos = cos.squeeze()
        sin = sin.squeeze()
        if cos.dim() == 2 and sin.dim() == 2:
            seq_len = hidden_states.shape[2]
            head_dim = hidden_states.shape[3]
            if cos.shape[0] == head_dim and cos.shape[1] == seq_len:
                cos = cos.t()
                sin = sin.t()
        if cos.dim() == 2 and sin.dim() == 2 and cos.shape[-1] == hidden_states.shape[-1]:
            cos = cos.to(hidden_states.device)
            sin = sin.to(hidden_states.device)
            while cos.dim() < hidden_states.dim():
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
            x_real, x_imag = hidden_states.reshape(*hidden_states.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
            out = (hidden_states.float() * cos + x_rotated.float() * sin).to(hidden_states.dtype)
            return out
        freqs = torch.complex(cos, sin)
    if isinstance(freqs, torch.Tensor):
        if torch.is_complex(freqs):
            x_complex = torch.view_as_complex(hidden_states.to(torch.float64).reshape(*hidden_states.shape[:-1], -1, 2))
            freqs = freqs.to(hidden_states.device)
            while freqs.dim() < x_complex.dim():
                freqs = freqs.unsqueeze(0)
            out = torch.view_as_real(x_complex * freqs).flatten(-2)
            return out.type_as(hidden_states)
        if freqs.dim() >= 2 and freqs.shape[-1] == 2:
            x_complex = torch.view_as_complex(hidden_states.to(torch.float64).reshape(*hidden_states.shape[:-1], -1, 2))
            freqs_complex = torch.view_as_complex(freqs.float().reshape(*freqs.shape[:-1], -1, 2)).to(hidden_states.device)
            while freqs_complex.dim() < x_complex.dim():
                freqs_complex = freqs_complex.unsqueeze(0)
            out = torch.view_as_real(x_complex * freqs_complex).flatten(-2)
            return out.type_as(hidden_states)
    raise ValueError(f"Unsupported freqs type/shape for RoPE: {type(freqs)} {getattr(freqs, 'shape', None)}")


def _normalize_cossin_rope_for_triton(hidden_states: torch.Tensor, freqs) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not ENABLE_TRITON_KERNELS or not ENABLE_ROPE_KERNEL:
        return None
    if not hidden_states.is_cuda:
        return None
    if not (isinstance(freqs, (tuple, list)) and len(freqs) == 2):
        return None
    cos, sin = freqs
    if not isinstance(cos, torch.Tensor) or not isinstance(sin, torch.Tensor):
        return None
    cos = cos.squeeze()
    sin = sin.squeeze()
    if torch.is_complex(cos) or torch.is_complex(sin):
        return None
    if cos.dim() != 2 or sin.dim() != 2:
        return None
    seq_len = int(hidden_states.shape[2])
    half_dim = int(hidden_states.shape[3]) // 2
    if int(hidden_states.shape[3]) % 2 != 0:
        return None
    if cos.shape == (half_dim, seq_len) and sin.shape == (half_dim, seq_len):
        cos = cos.t()
        sin = sin.t()
    if cos.shape != (seq_len, half_dim) or sin.shape != (seq_len, half_dim):
        return None
    return cos, sin


def apply_rotary_emb(query: torch.Tensor, key: torch.Tensor, freqs):
    cossin = _normalize_cossin_rope_for_triton(query, freqs)
    if cossin is not None:
        try:
            return triton_apply_qk_rope_inplace(query, key, cossin[0], cossin[1])
        except Exception:
            pass
    return _apply_rotary_emb_torch(query, freqs), _apply_rotary_emb_torch(key, freqs)


def _rmsnorm_forward_torch(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp32 = x.float()
    w_fp32 = w.float()
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    out = x_fp32 * torch.rsqrt(var + float(eps))
    out = out * w_fp32
    return out.to(dtype=x.dtype)


def rmsnorm_forward(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    if ENABLE_TRITON_RMSNORM_KERNEL and x.is_cuda:
        return triton_rmsnorm_forward(x, w, eps)
    return _rmsnorm_forward_torch(x, w, eps)


def rmsnorm_forward_torch(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return _rmsnorm_forward_torch(x, w, eps)


def apply_rotary_emb_torch(query: torch.Tensor, key: torch.Tensor, freqs):
    return _apply_rotary_emb_torch(query, freqs), _apply_rotary_emb_torch(key, freqs)


def layernorm_forward(
    x: torch.Tensor,
    w: torch.Tensor | None,
    b: torch.Tensor | None,
    eps: float,
    elementwise_affine: bool = True,
) -> torch.Tensor:
    if ENABLE_TRITON_BLOCK_KERNELS and x.is_cuda:
        return triton_layernorm_forward(x, w, b, eps, elementwise_affine)
    weight = w if elementwise_affine else None
    bias = b if elementwise_affine else None
    return F.layer_norm(x.float(), (x.shape[-1],), weight=weight, bias=bias, eps=eps)


def modulate_shift_forward(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if ENABLE_TRITON_BLOCK_KERNELS and x.is_cuda:
        return triton_modulate_shift_forward(x, scale, shift, output_dtype=output_dtype)
    return (x.float() * (1 + scale.float()) + shift.float()).to(dtype=output_dtype)


def modulate_gate_residual_forward(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if ENABLE_TRITON_BLOCK_KERNELS and x.is_cuda:
        return triton_modulate_gate_residual_forward(residual, x, gate, output_dtype=output_dtype)
    return (residual.float() + x.float() * gate.float()).to(dtype=output_dtype)


try:
    if ENABLE_ROPE_KERNEL:
        _append_local_kernel_paths()
        import _kernels  # type: ignore

        ROPE_KERNEL_IMPORT_OK = True
        ROPE_KERNEL_SEARCH_DIRS = tuple(
            str(path)
            for path in _candidate_local_kernel_dirs()
            if path.is_dir()
        )

        def apply_rotary_emb(query: torch.Tensor, key: torch.Tensor, freqs):
            if not query.is_cuda:
                return _apply_rotary_emb_torch(query, freqs), _apply_rotary_emb_torch(key, freqs)
            freqs_real, freqs_imag = freqs
            if freqs_real.dim() != 2 or freqs_imag.dim() != 2:
                freqs_real = freqs_real.squeeze()
                freqs_imag = freqs_imag.squeeze()
            if freqs_real.dim() != 2 or freqs_imag.dim() != 2:
                return _apply_rotary_emb_torch(query, freqs), _apply_rotary_emb_torch(key, freqs)
            try:
                _kernels.apply_qk_rope_inplace_cossin_complex(query, key, freqs_real, freqs_imag, 0)
            except Exception:
                cossin = _normalize_cossin_rope_for_triton(query, (freqs_real, freqs_imag))
                if cossin is not None:
                    try:
                        return triton_apply_qk_rope_inplace(query, key, cossin[0], cossin[1])
                    except Exception:
                        pass
                return _apply_rotary_emb_torch(query, freqs), _apply_rotary_emb_torch(key, freqs)
            return query, key

except Exception:
    fallback_name = "Triton/PyTorch" if ENABLE_TRITON_KERNELS and ENABLE_ROPE_KERNEL else "PyTorch"
    warnings.warn(
        "Local cole RoPE extension is unavailable; "
        f"using {fallback_name} rotary embedding fallback.",
        RuntimeWarning,
        stacklevel=2,
    )
    ROPE_KERNEL_SEARCH_DIRS = tuple(
        str(path)
        for path in _candidate_local_kernel_dirs()
        if path.is_dir()
    )


__all__ = [
    "DiffusersRMSNorm",
    "ENABLE_FAST_KERNEL",
    "ENABLE_ROPE_KERNEL",
    "ENABLE_TRITON_BLOCK_KERNELS",
    "ENABLE_TRITON_KERNELS",
    "ENABLE_TRITON_RMSNORM_KERNEL",
    "ROPE_KERNEL_SEARCH_DIRS",
    "ROPE_KERNEL_IMPORT_OK",
    "apply_rotary_emb",
    "apply_rotary_emb_torch",
    "layernorm_forward",
    "modulate_gate_residual_forward",
    "modulate_shift_forward",
    "rmsnorm_forward",
    "rmsnorm_forward_torch",
    "triton_layernorm_forward",
    "triton_modulate_gate_residual_forward",
    "triton_modulate_shift_forward",
    "triton_rmsnorm_forward",
]
