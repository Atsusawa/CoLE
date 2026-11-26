# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# CoLE adapts Diffusers' Wan attention processor for sparse attention.

from __future__ import annotations

import os
import math
import time
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

from cole.core.feedback import (
    ATTENTION_PROXY_ASYNC_EVENT_KEY,
    ATTENTION_PROXY_ASYNC_FEATURE_KEYS_KEY,
    ATTENTION_PROXY_ASYNC_SOURCE_KEY,
    ATTENTION_PROXY_ASYNC_VALUES_KEY,
    ExecFeedback,
)
from cole.core.types import MaskPlan
from cole.runtime.attention_bridge import AttentionBridge
from cole.runtime.wan_fast_support import (
    DiffusersRMSNorm,
    apply_rotary_emb,
    apply_rotary_emb_torch,
    rmsnorm_forward,
    rmsnorm_forward_torch,
)

_WAN_TEXT_CONTEXT_LENGTH = 512
_ATTENTION_PROXY_INDEX_CACHE: dict[tuple[object, ...], torch.Tensor] = {}
_ATTENTION_PROXY_RANK_CACHE: dict[tuple[object, ...], torch.Tensor] = {}
_ATTENTION_PROXY_FEATURE_KEYS = (
    "attn_eff_support_ratio",
    "attn_entropy_norm",
    "attn_entropy_eff_support_ratio",
    "attn_top1_prob",
    "attn_top1_top2_margin",
    "attn_topk_mass_10pct",
    "attn_gini",
    "attn_logit_std",
    "attn_entropy_norm_head_std",
    "attn_entropy_norm_head_max",
    "attn_topk_mass_10pct_head_std",
    "attn_topk_mass_10pct_head_min",
    "key_sim_top1_mean",
    "key_sim_top1_std",
    "key_sim_top1_p90",
    "key_sim_above_0p85_frac",
    "key_sim_above_0p90_frac",
    "key_sim_samples",
)
_RUNTIME_ENV_KEYS = (
    "COLE_TRACE_SPARSE_RUNTIME",
    "COLE_TRACE_SPARSE_RUNTIME_LAYER",
    "COLE_SPARSE_DISABLE_FAST_PREPROC",
    "COLE_SPARSE_DISABLE_FAST_RMSNORM",
    "COLE_SPARSE_DISABLE_FAST_ROPE",
    "COLE_SPARSE_DEBUG_SYNC",
    "COLE_SPARSE_DEBUG_INPUT_STATS",
    "COLE_ATTENTION_PROXY_ENABLE",
    "COLE_ATTENTION_PROXY_STEP_INTERVAL",
    "COLE_ATTENTION_PROXY_LAYERS",
    "COLE_ATTENTION_PROXY_HEADS",
    "COLE_ATTENTION_PROXY_QUERIES",
    "COLE_ATTENTION_PROXY_KEYS",
    "COLE_ATTENTION_PROB_PATCHES_ENABLE",
    "COLE_ATTENTION_SAMPLE_TIMESTEPS",
    "COLE_ATTENTION_PROXY_KEY_SIM_ENABLE",
    "COLE_ATTENTION_PROXY_KEY_SIM_TOKENS",
)
_RUNTIME_ENV_CACHE: dict[str, object] = {}


def _cached_even_index(*, length: int, count: int, device: torch.device) -> torch.Tensor:
    resolved_length = max(1, int(length))
    resolved_count = max(1, min(resolved_length, int(count)))
    key = (str(device), int(resolved_length), int(resolved_count))
    cached = _ATTENTION_PROXY_INDEX_CACHE.get(key)
    if cached is not None and cached.device == device:
        return cached
    if resolved_count >= resolved_length:
        index = torch.arange(resolved_length, device=device, dtype=torch.long)
    else:
        index = torch.linspace(0, resolved_length - 1, steps=resolved_count, device=device).round().long()
        index = torch.clamp(index, 0, resolved_length - 1)
    _ATTENTION_PROXY_INDEX_CACHE[key] = index
    return index


def _cached_rank(*, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    resolved_length = max(1, int(length))
    key = (str(device), str(dtype), int(resolved_length))
    cached = _ATTENTION_PROXY_RANK_CACHE.get(key)
    if cached is not None and cached.device == device and cached.dtype == dtype:
        return cached
    rank = torch.arange(1, resolved_length + 1, device=device, dtype=dtype)
    _ATTENTION_PROXY_RANK_CACHE[key] = rank
    return rank


def _bool_from_raw(value: str | None, default: str = "0") -> bool:
    return str(default if value is None else value) == "1"


def refresh_runtime_env_cache() -> dict[str, object]:
    global _RUNTIME_ENV_CACHE
    raw = {key: os.environ.get(key) for key in _RUNTIME_ENV_KEYS}
    layer_filter_raw = str(raw.get("COLE_TRACE_SPARSE_RUNTIME_LAYER") or "").strip()
    trace_layer = 0
    trace_layer_valid = True
    if layer_filter_raw:
        try:
            trace_layer = int(layer_filter_raw)
        except ValueError:
            trace_layer_valid = False
    config = {
        "flags": {
            "COLE_TRACE_SPARSE_RUNTIME": _bool_from_raw(raw.get("COLE_TRACE_SPARSE_RUNTIME")),
            "COLE_SPARSE_DISABLE_FAST_PREPROC": _bool_from_raw(raw.get("COLE_SPARSE_DISABLE_FAST_PREPROC")),
            "COLE_SPARSE_DISABLE_FAST_RMSNORM": _bool_from_raw(raw.get("COLE_SPARSE_DISABLE_FAST_RMSNORM")),
            "COLE_SPARSE_DISABLE_FAST_ROPE": _bool_from_raw(raw.get("COLE_SPARSE_DISABLE_FAST_ROPE")),
            "COLE_SPARSE_DEBUG_SYNC": _bool_from_raw(raw.get("COLE_SPARSE_DEBUG_SYNC")),
            "COLE_SPARSE_DEBUG_INPUT_STATS": _bool_from_raw(raw.get("COLE_SPARSE_DEBUG_INPUT_STATS")),
            "COLE_ATTENTION_PROXY_ENABLE": _bool_from_raw(raw.get("COLE_ATTENTION_PROXY_ENABLE")),
            "COLE_ATTENTION_PROB_PATCHES_ENABLE": _bool_from_raw(raw.get("COLE_ATTENTION_PROB_PATCHES_ENABLE")),
        },
        "trace_sparse_runtime": _bool_from_raw(raw.get("COLE_TRACE_SPARSE_RUNTIME")),
        "trace_sparse_runtime_layer_active": bool(layer_filter_raw),
        "trace_sparse_runtime_layer_valid": bool(trace_layer_valid),
        "trace_sparse_runtime_layer": int(trace_layer),
        "attention_proxy_interval": max(1, int(raw.get("COLE_ATTENTION_PROXY_STEP_INTERVAL") or 1)),
        "attention_proxy_layers": _parse_int_set(str(raw.get("COLE_ATTENTION_PROXY_LAYERS") or "")),
        "attention_proxy_heads": max(1, int(raw.get("COLE_ATTENTION_PROXY_HEADS") or 2)),
        "attention_proxy_queries": max(1, int(raw.get("COLE_ATTENTION_PROXY_QUERIES") or 64)),
        "attention_proxy_keys": max(1, int(raw.get("COLE_ATTENTION_PROXY_KEYS") or 512)),
        "attention_sample_timesteps": _parse_int_set(str(raw.get("COLE_ATTENTION_SAMPLE_TIMESTEPS") or "")),
        "attention_proxy_key_sim_enable": _bool_from_raw(raw.get("COLE_ATTENTION_PROXY_KEY_SIM_ENABLE")),
        "attention_proxy_key_sim_tokens": max(2, int(raw.get("COLE_ATTENTION_PROXY_KEY_SIM_TOKENS") or 256)),
    }
    _RUNTIME_ENV_CACHE = config
    return _RUNTIME_ENV_CACHE


def _runtime_env_config() -> dict[str, object]:
    if not _RUNTIME_ENV_CACHE:
        return refresh_runtime_env_cache()
    return _RUNTIME_ENV_CACHE


def _trace_sparse_runtime_enabled() -> bool:
    return bool(_runtime_env_config()["trace_sparse_runtime"])


def _trace_sparse_runtime_layer_enabled(layer_idx: int) -> bool:
    config = _runtime_env_config()
    if not bool(config["trace_sparse_runtime_layer_active"]):
        return True
    if not bool(config["trace_sparse_runtime_layer_valid"]):
        return False
    return int(config["trace_sparse_runtime_layer"]) == int(layer_idx)


def _env_flag(name: str, default: str = "0") -> bool:
    flags = _runtime_env_config().get("flags")
    if isinstance(flags, dict) and name in flags:
        return bool(flags[name])
    return os.environ.get(name, default) == "1"


def _sparse_plan_active(plan: MaskPlan | None) -> bool:
    return isinstance(plan, MaskPlan) and str(plan.backend_name) == "sparse"


def _disable_fast_rmsnorm_for_sparse(plan: MaskPlan | None) -> bool:
    return _sparse_plan_active(plan) and (
        _env_flag("COLE_SPARSE_DISABLE_FAST_PREPROC")
        or _env_flag("COLE_SPARSE_DISABLE_FAST_RMSNORM")
    )


def _disable_fast_rope_for_sparse(plan: MaskPlan | None) -> bool:
    return _sparse_plan_active(plan) and (
        _env_flag("COLE_SPARSE_DISABLE_FAST_PREPROC")
        or _env_flag("COLE_SPARSE_DISABLE_FAST_ROPE")
    )


def _debug_sync_sparse() -> bool:
    return _env_flag("COLE_SPARSE_DEBUG_SYNC")


def _debug_input_stats_sparse() -> bool:
    return _env_flag("COLE_SPARSE_DEBUG_INPUT_STATS")


def _attention_proxy_enabled() -> bool:
    return _env_flag("COLE_ATTENTION_PROXY_ENABLE")


def _attention_prob_patches_enabled() -> bool:
    return _env_flag("COLE_ATTENTION_PROB_PATCHES_ENABLE")


def _parse_int_set(raw: str) -> set[int]:
    values: set[int] = set()
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.add(int(token))
        except ValueError:
            continue
    return values


def _attention_proxy_should_sample(layer_idx: int, timestep: int) -> bool:
    if not _attention_proxy_enabled():
        return False
    return _attention_sampling_should_sample(layer_idx, timestep)


def _attention_sampling_should_sample(layer_idx: int, timestep: int) -> bool:
    if not (_attention_proxy_enabled() or _attention_prob_patches_enabled()):
        return False
    config = _runtime_env_config()
    interval = max(1, int(config["attention_proxy_interval"]))
    if int(timestep) % interval != 0:
        return False
    layer_filter = config["attention_proxy_layers"]
    if layer_filter and int(layer_idx) not in layer_filter:
        return False
    timestep_filter = config["attention_sample_timesteps"]
    if timestep_filter and int(timestep) not in timestep_filter:
        return False
    return True


def _attention_qk_sample(query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if query.dim() != 4 or key.dim() != 4:
        return None
    config = _runtime_env_config()
    heads = max(1, min(int(query.shape[1]), int(config["attention_proxy_heads"])))
    query_count = max(1, min(int(query.shape[-2]), int(config["attention_proxy_queries"])))
    key_count = max(1, min(int(key.shape[-2]), int(config["attention_proxy_keys"])))
    head_idx = _cached_even_index(length=int(query.shape[1]), count=heads, device=query.device)
    query_idx = _cached_even_index(length=int(query.shape[-2]), count=query_count, device=query.device)
    key_idx = _cached_even_index(length=int(key.shape[-2]), count=key_count, device=key.device)
    q_sample = query[:, head_idx][:, :, query_idx, :].float()
    k_sample = key[:, head_idx][:, :, key_idx, :].float()
    scale = float(q_sample.shape[-1]) ** -0.5
    scores = torch.matmul(q_sample, k_sample.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return q_sample, k_sample, scores, probs


def _attention_prob_patch_from_sample(sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    probs = sample[3]
    patch = torch.nan_to_num(probs.detach(), nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(2)
    return patch.to(dtype=torch.float16)


def _key_similarity_proxy_stats(k_sample: torch.Tensor, *, key_count: int) -> torch.Tensor:
    if k_sample.dim() != 4:
        return k_sample.new_zeros((6,), dtype=torch.float32)
    token_count = max(2, min(int(k_sample.shape[-2]), int(key_count)))
    if int(k_sample.shape[-2]) > token_count:
        token_idx = _cached_even_index(length=int(k_sample.shape[-2]), count=token_count, device=k_sample.device)
        k_work = k_sample[:, :, token_idx, :]
    else:
        k_work = k_sample
    metric = F.normalize(k_work.float().mean(dim=1), dim=-1)
    if int(metric.shape[-2]) < 2:
        return k_sample.new_zeros((6,), dtype=torch.float32)
    src = metric[:, ::2, :]
    dst = metric[:, 1::2, :]
    if int(src.shape[-2]) <= 0 or int(dst.shape[-2]) <= 0:
        return k_sample.new_zeros((6,), dtype=torch.float32)
    scores = torch.matmul(src, dst.transpose(-2, -1))
    top1 = scores.max(dim=-1).values.reshape(-1)
    if int(top1.numel()) <= 0:
        return k_sample.new_zeros((6,), dtype=torch.float32)
    p90 = torch.quantile(top1, 0.90)
    values = torch.stack(
        [
            top1.mean(),
            top1.std(unbiased=False),
            p90,
            (top1 > 0.85).float().mean(),
            (top1 > 0.90).float().mean(),
            top1.new_tensor(float(top1.numel())),
        ]
    )
    return torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _attention_proxy_stats_from_sample(
    sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, object]:
    config = _runtime_env_config()
    q_sample, k_sample, scores, probs = sample
    del q_sample
    support = max(1.0, float(probs.shape[-1]))
    prob_sq_sum = probs.square().sum(dim=-1).clamp(min=1e-8)
    eff_support_ratio = (1.0 / prob_sq_sum) / support
    entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)
    entropy_norm = entropy / max(1e-8, math.log(float(support)))
    entropy_eff_support_ratio = torch.exp(entropy) / support
    topk_count = max(1, int(round(0.1 * float(probs.shape[-1]))))
    top_values = probs.topk(max(min(2, int(probs.shape[-1])), topk_count), dim=-1).values
    top1 = top_values[..., 0]
    top2 = top_values[..., 1] if top_values.shape[-1] > 1 else torch.zeros_like(top1)
    topk_mass = top_values[..., :topk_count].sum(dim=-1)
    sorted_probs = probs.sort(dim=-1).values
    rank = _cached_rank(length=int(sorted_probs.shape[-1]), device=probs.device, dtype=sorted_probs.dtype)
    gini = ((2.0 * rank - float(sorted_probs.shape[-1]) - 1.0) * sorted_probs).sum(dim=-1) / support
    logit_std = scores.std(dim=-1, unbiased=False)
    entropy_head_mean = entropy_norm.mean(dim=(0, 2))
    topk_mass_head_mean = topk_mass.mean(dim=(0, 2))
    if bool(config.get("attention_proxy_key_sim_enable")):
        key_sim_values = _key_similarity_proxy_stats(
            k_sample,
            key_count=int(config.get("attention_proxy_key_sim_tokens") or int(k_sample.shape[-2])),
        )
    else:
        key_sim_values = scores.new_zeros((6,), dtype=torch.float32)
    values = torch.stack(
        [
            eff_support_ratio.mean(),
            entropy_norm.mean(),
            entropy_eff_support_ratio.mean(),
            top1.mean(),
            (top1 - top2).mean(),
            topk_mass.mean(),
            gini.mean(),
            logit_std.mean(),
            entropy_head_mean.std(unbiased=False),
            entropy_head_mean.max(),
            topk_mass_head_mean.std(unbiased=False),
            topk_mass_head_mean.min(),
            key_sim_values[0],
            key_sim_values[1],
            key_sim_values[2],
            key_sim_values[3],
            key_sim_values[4],
            key_sim_values[5],
        ]
    ).detach()
    if values.is_cuda:
        values_float = values.float()
        cpu_values = torch.empty((len(_ATTENTION_PROXY_FEATURE_KEYS),), device="cpu", dtype=torch.float32, pin_memory=True)
        current_stream = torch.cuda.current_stream(values.device)
        stream = torch.cuda.Stream(device=values.device)
        with torch.cuda.stream(stream):
            stream.wait_stream(current_stream)
            cpu_values.copy_(values_float, non_blocking=True)
        event = torch.cuda.Event(blocking=False)
        stream.record_event(event)
        values_float.record_stream(stream)
        return {
            ATTENTION_PROXY_ASYNC_VALUES_KEY: cpu_values,
            ATTENTION_PROXY_ASYNC_FEATURE_KEYS_KEY: _ATTENTION_PROXY_FEATURE_KEYS,
            ATTENTION_PROXY_ASYNC_EVENT_KEY: event,
            ATTENTION_PROXY_ASYNC_SOURCE_KEY: "attention_proxy_stats_v1",
            "attn_proxy_samples": float(eff_support_ratio.numel()),
        }
    values_list = values.cpu().float().tolist()
    return {
        "attn_eff_support_ratio": float(values_list[0]),
        "attn_entropy_norm": float(values_list[1]),
        "attn_entropy_eff_support_ratio": float(values_list[2]),
        "attn_top1_prob": float(values_list[3]),
        "attn_top1_top2_margin": float(values_list[4]),
        "attn_topk_mass_10pct": float(values_list[5]),
        "attn_gini": float(values_list[6]),
        "attn_logit_std": float(values_list[7]),
        "attn_entropy_norm_head_std": float(values_list[8]),
        "attn_entropy_norm_head_max": float(values_list[9]),
        "attn_topk_mass_10pct_head_std": float(values_list[10]),
        "attn_topk_mass_10pct_head_min": float(values_list[11]),
        "key_sim_top1_mean": float(values_list[12]),
        "key_sim_top1_std": float(values_list[13]),
        "key_sim_top1_p90": float(values_list[14]),
        "key_sim_above_0p85_frac": float(values_list[15]),
        "key_sim_above_0p90_frac": float(values_list[16]),
        "key_sim_samples": float(values_list[17]),
        "attn_proxy_samples": float(eff_support_ratio.numel()),
    }


def _attention_proxy_stats(query: torch.Tensor, key: torch.Tensor) -> dict[str, object] | None:
    with torch.no_grad():
        sample = _attention_qk_sample(query, key)
        if sample is None:
            return None
        return _attention_proxy_stats_from_sample(sample)


def _tensor_debug_stats(name: str, tensor: torch.Tensor) -> str:
    tensor_detached = tensor.detach()
    finite = bool(torch.isfinite(tensor_detached).all().item())
    work = torch.nan_to_num(tensor_detached.float(), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(work.min().item())
    max_value = float(work.max().item())
    mean_value = float(work.mean().item())
    return (
        f"{name}_finite={finite}"
        f" {name}_min={min_value:.6f}"
        f" {name}_max={max_value:.6f}"
        f" {name}_mean={mean_value:.6f}"
    )


class CoLEWanAttnProcessor:
    """CoLE-local Wan attention processor with optional fast-kernel support."""

    def __init__(
        self,
        layer_idx: int,
        *,
        context_length: int,
        num_frame: int,
        frame_size: int,
        num_layers: int,
        attention_bridge: AttentionBridge,
        plan_provider: Callable[[], MaskPlan | None],
        feedback_sink: Callable[[ExecFeedback], None],
    ) -> None:
        self.layer_idx = int(layer_idx)
        self.context_length = int(context_length)
        self.num_frame = int(num_frame)
        self.frame_size = int(frame_size)
        self.num_layers = int(num_layers)
        self.attention_bridge = attention_bridge
        self.plan_provider = plan_provider
        self.feedback_sink = feedback_sink

    def get_qkv(self, attn, hidden_states, encoder_hidden_states):
        return attn.to_q(hidden_states), attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)

    def get_qk_norm(self, attn, query, key, *, execution_plan: MaskPlan | None = None):
        if attn.norm_q is not None:
            if isinstance(attn.norm_q, (torch.nn.RMSNorm, DiffusersRMSNorm)):
                if _disable_fast_rmsnorm_for_sparse(execution_plan):
                    query = rmsnorm_forward_torch(query, attn.norm_q.weight, attn.norm_q.eps)
                else:
                    query = rmsnorm_forward(query, attn.norm_q.weight, attn.norm_q.eps)
            else:
                raise ValueError(f"Unsupported norm_q type: {type(attn.norm_q)}")
        if attn.norm_k is not None:
            if isinstance(attn.norm_k, (torch.nn.RMSNorm, DiffusersRMSNorm)):
                if _disable_fast_rmsnorm_for_sparse(execution_plan):
                    key = rmsnorm_forward_torch(key, attn.norm_k.weight, attn.norm_k.eps)
                else:
                    key = rmsnorm_forward(key, attn.norm_k.weight, attn.norm_k.eps)
            else:
                raise ValueError(f"Unsupported norm_k type: {type(attn.norm_k)}")
        return query, key

    def get_transpose_qkv(self, attn, query, key, value):
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        return query, key, value

    def get_rotary_emb(self, query, key, rotary_emb, *, execution_plan: MaskPlan | None = None):
        if rotary_emb is not None:
            if _disable_fast_rope_for_sparse(execution_plan):
                query, key = apply_rotary_emb_torch(query, key, rotary_emb)
            else:
                query, key = apply_rotary_emb(query, key, rotary_emb)
        return query, key

    def get_o(self, attn, query, hidden_states, hidden_states_img):
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    def split_encoder_hidden_states(self, attn, encoder_hidden_states: Optional[torch.Tensor]):
        encoder_hidden_states_img = None
        if attn.add_k_proj is None or encoder_hidden_states is None:
            return encoder_hidden_states, encoder_hidden_states_img

        total_context_length = int(encoder_hidden_states.shape[1])
        text_context_length = min(_WAN_TEXT_CONTEXT_LENGTH, total_context_length)
        image_context_length = max(0, total_context_length - text_context_length)
        if image_context_length == 0:
            return encoder_hidden_states, encoder_hidden_states_img

        encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
        encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        return encoder_hidden_states, encoder_hidden_states_img

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        execution_plan = self.plan_provider() if timestep is not None else None
        encoder_hidden_states, encoder_hidden_states_img = self.split_encoder_hidden_states(attn, encoder_hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query, key, value = self.get_qkv(attn, hidden_states, encoder_hidden_states)
        query, key = self.get_qk_norm(attn, query, key, execution_plan=execution_plan)
        query, key, value = self.get_transpose_qkv(attn, query, key, value)
        query, key = self.get_rotary_emb(query, key, rotary_emb, execution_plan=execution_plan)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)
            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        if timestep is None:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            hidden_states = self.attention_core_logic(query, key, value, timestep, execution_plan=execution_plan)

        hidden_states = self.get_o(attn, query, hidden_states, hidden_states_img)
        return hidden_states

    def attention_core_logic(self, query, key, value, timestep, *, execution_plan: MaskPlan | None = None):
        cfg, num_heads, seq_len, dim = query.size()
        if cfg != 1:
            raise ValueError(f"CoLEWanAttnProcessor currently expects batch size 1, got cfg={cfg}")
        if self.context_length != 0:
            raise NotImplementedError(
                "The CoLE processor supports only context_length=0 video self-attention."
            )

        plan = execution_plan if execution_plan is not None else self.plan_provider()
        if plan is None:
            raise RuntimeError("cole_mask_plan is not published yet for sparse execution.")
        trace_runtime = _trace_sparse_runtime_enabled() and _trace_sparse_runtime_layer_enabled(self.layer_idx)
        if trace_runtime:
            print(
                "[cole-sparse-runtime-start]"
                f" layer={self.layer_idx}"
                f" timestep={int(timestep)}"
                f" profile={plan.profile_name}"
                f" backend={plan.backend_name}"
                f" mode={plan.sparse_mode}"
                f" value={plan.sparse_value}"
                f" shape={tuple(query.shape)}"
            )
            if _debug_input_stats_sparse():
                print(
                    "[cole-sparse-input-stats]"
                    f" layer={self.layer_idx}"
                    f" timestep={int(timestep)}"
                    f" profile={plan.profile_name}"
                    f" {_tensor_debug_stats('q', query)}"
                    f" {_tensor_debug_stats('k', key)}"
                    f" {_tensor_debug_stats('v', value)}"
                )
        if _debug_sync_sparse() and query.is_cuda:
            torch.cuda.synchronize(query.device)
        output_video, feedback = self.attention_bridge.run(query, key, value, plan)
        if _debug_sync_sparse() and query.is_cuda:
            torch.cuda.synchronize(query.device)
        if _attention_sampling_should_sample(self.layer_idx, int(timestep)):
            proxy_start = time.perf_counter()
            sample = _attention_qk_sample(query, key)
            feedback.attention_proxy = (
                None
                if sample is None or not _attention_proxy_enabled()
                else _attention_proxy_stats_from_sample(sample)
            )
            if sample is not None and _attention_prob_patches_enabled():
                feedback.attention_prob_patches = {int(self.layer_idx): _attention_prob_patch_from_sample(sample)}
            proxy_elapsed_sec = time.perf_counter() - proxy_start
            feedback.attention_sampling_elapsed_sec = float(proxy_elapsed_sec)
            feedback.attention_sampling_updates = 1
            if isinstance(feedback.attention_proxy, dict):
                feedback.attention_proxy["attn_proxy_elapsed_sec"] = float(proxy_elapsed_sec)
                feedback.attention_proxy["attn_proxy_elapsed_sec_samples"] = float(proxy_elapsed_sec)
                feedback.attention_proxy["attn_proxy_updates"] = 1.0
                feedback.attention_proxy["attn_proxy_updates_samples"] = 1.0
        if trace_runtime:
            print(
                "[cole-sparse-runtime-end]"
                f" layer={self.layer_idx}"
                f" timestep={int(timestep)}"
                f" profile={plan.profile_name}"
                f" backend={feedback.backend_name}"
                f" run_ms={feedback.backend_run_ms:.3f}"
                f" prepare_ms={feedback.backend_prepare_ms:.3f}"
            )
        self.feedback_sink(feedback)
        return output_video.reshape(cfg, num_heads, seq_len, dim)

    def _resolve_runtime_video_geometry(self, seq_len: int) -> tuple[int, int]:
        expected_video_length = int(self.num_frame * self.frame_size)
        if seq_len == expected_video_length:
            return int(self.num_frame), int(self.frame_size)
        if int(self.num_frame) > 0 and seq_len % int(self.num_frame) == 0:
            return int(self.num_frame), seq_len // int(self.num_frame)
        raise ValueError(
            "Video sequence length mismatch and cannot infer runtime geometry: "
            f"seq_len={seq_len}, expected={expected_video_length}, "
            f"context_length={self.context_length}, num_frame={self.num_frame}, frame_size={self.frame_size}"
        )
