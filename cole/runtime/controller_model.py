from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

def _safe_tensor(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _parameter_dtype(module: nn.Module) -> torch.dtype:
    for parameter in module.parameters():
        return parameter.dtype
    return torch.float32


def _coordinate_axis(length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if int(length) <= 1:
        return torch.zeros((max(1, int(length)),), device=device, dtype=dtype)
    return torch.linspace(-1.0, 1.0, steps=int(length), device=device, dtype=dtype)


def _coordinate_grid(
    *axis_lengths: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    axes = [_coordinate_axis(length, device=device, dtype=dtype) for length in axis_lengths]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, len(axis_lengths))


class CrossAttentionPoolLayer(nn.Module):
    def __init__(self, *, width: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(int(width))
        self.source_norm = nn.LayerNorm(int(width))
        self.cross = nn.MultiheadAttention(
            embed_dim=int(width),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(int(width)),
            nn.Linear(int(width), int(width) * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(width) * 4, int(width)),
        )

    def forward(self, queries: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        delta, _ = self.cross(
            self.query_norm(queries),
            self.source_norm(source),
            self.source_norm(source),
            need_weights=False,
        )
        hidden = queries + delta
        return hidden + self.ff(hidden)


class CandidateScoringHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        candidate_feature_dim: int,
        width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.hidden_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, int(width)),
            nn.GELU(),
        )
        self.candidate_proj = nn.Sequential(
            nn.LayerNorm(self.candidate_feature_dim),
            nn.Linear(self.candidate_feature_dim, int(width)),
            nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(int(width) * 4),
            nn.Linear(int(width) * 4, int(width)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(width), 1),
        )

    def forward(self, hidden: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError(f"hidden must have shape [B,D], got {tuple(hidden.shape)}.")
        if candidate_features.ndim != 3:
            raise ValueError(
                "candidate_features must have shape [B,K,D], "
                f"got {tuple(candidate_features.shape)}."
            )
        if int(candidate_features.shape[0]) != int(hidden.shape[0]):
            raise ValueError("candidate_features batch size does not match hidden batch size.")
        if int(candidate_features.shape[2]) != self.candidate_feature_dim:
            raise ValueError(
                f"candidate feature dim mismatch: expected {self.candidate_feature_dim}, "
                f"got {int(candidate_features.shape[2])}."
            )
        state = self.hidden_proj(hidden).unsqueeze(1).expand(-1, int(candidate_features.shape[1]), -1)
        candidate = self.candidate_proj(candidate_features)
        joint = torch.cat(
            [state, candidate, state * candidate, torch.abs(state - candidate)],
            dim=-1,
        )
        return self.scorer(joint).squeeze(-1)


@dataclass(frozen=True)
class SharedTensorBackboneConfig:
    latent_channels: int = 16
    latent_patch_frames: int = 1
    latent_patch_height: int = 4
    latent_patch_width: int = 13
    attention_input_channels: int = 1
    attention_key_count: int = 512
    text_embedding_dim: int = 4096
    context_dim: int = 2
    width: int = 128
    latent_query_count: int = 8
    attention_query_count: int = 8
    cross_depth: int = 1
    fusion_depth: int = 2
    num_heads: int = 4
    dropout: float = 0.1


@dataclass(frozen=True)
class SharedResolutionSparsePolicyConfig:
    backbone: SharedTensorBackboneConfig
    switch_candidate_feature_dim: int = 6
    sparse_candidate_feature_dim: int = 4
    sparse_decision_feature_dim: int = 5
    sparse_window_count: int = 4
    adapter_width: int = 128
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SharedResolutionSparsePolicyConfig":
        raw = dict(payload)
        raw["backbone"] = SharedTensorBackboneConfig(**dict(raw.get("backbone") or {}))
        return cls(**raw)


@dataclass(frozen=True)
class SharedTensorFeatures:
    global_hidden: torch.Tensor
    latent_hidden: torch.Tensor
    attention_hidden: torch.Tensor
    text_hidden: torch.Tensor
    context_hidden: torch.Tensor

    def concatenate(self) -> torch.Tensor:
        return torch.cat(
            [
                self.global_hidden,
                self.latent_hidden,
                self.attention_hidden,
                self.text_hidden,
                self.context_hidden,
            ],
            dim=-1,
        )


class SharedPerceiverTensorBackbone(nn.Module):
    """Small coordinate-aware tensor/text encoder shared by policy tasks."""

    def __init__(self, config: SharedTensorBackboneConfig) -> None:
        super().__init__()
        self.config = config
        width = int(config.width)
        if width < 1 or width % int(config.num_heads) != 0:
            raise ValueError("width must be positive and divisible by num_heads.")
        if int(config.cross_depth) < 1 or int(config.fusion_depth) < 1:
            raise ValueError("cross_depth and fusion_depth must be positive.")

        latent_patch_dim = (
            int(config.latent_channels)
            * int(config.latent_patch_frames)
            * int(config.latent_patch_height)
            * int(config.latent_patch_width)
        )
        attention_token_dim = int(config.attention_input_channels) * int(config.attention_key_count)
        self.latent_proj = nn.Sequential(
            nn.LayerNorm(latent_patch_dim),
            nn.Linear(latent_patch_dim, width),
            nn.GELU(),
        )
        self.attention_proj = nn.Sequential(
            nn.LayerNorm(attention_token_dim),
            nn.Linear(attention_token_dim, width),
            nn.GELU(),
        )
        self.latent_coord_proj = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.attention_coord_proj = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.text_proj = nn.Sequential(
            nn.LayerNorm(int(config.text_embedding_dim)),
            nn.Linear(int(config.text_embedding_dim), width),
            nn.GELU(),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(int(config.context_dim), width),
            nn.GELU(),
            nn.Linear(width, width),
        )

        self.latent_queries = nn.Parameter(torch.zeros(1, int(config.latent_query_count), width))
        self.attention_queries = nn.Parameter(torch.zeros(1, int(config.attention_query_count), width))
        self.global_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.latent_queries, std=0.02)
        nn.init.normal_(self.attention_queries, std=0.02)
        nn.init.normal_(self.global_token, std=0.02)
        self.type_embedding = nn.Embedding(5, width)
        self.latent_cross = nn.ModuleList(
            [
                CrossAttentionPoolLayer(width=width, num_heads=int(config.num_heads), dropout=float(config.dropout))
                for _ in range(int(config.cross_depth))
            ]
        )
        self.attention_cross = nn.ModuleList(
            [
                CrossAttentionPoolLayer(width=width, num_heads=int(config.num_heads), dropout=float(config.dropout))
                for _ in range(int(config.cross_depth))
            ]
        )
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=int(config.num_heads),
            dim_feedforward=width * 4,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            fusion_layer,
            num_layers=int(config.fusion_depth),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(width)

    @property
    def output_dim(self) -> int:
        return int(self.config.width) * 5

    def _type(self, index: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        value = self.type_embedding(torch.tensor(int(index), device=device, dtype=torch.long))
        return value.reshape(1, 1, -1).to(dtype=dtype)

    def _latent_tokens(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5:
            raise ValueError(f"latent must have shape [B,C,T,H,W], got {tuple(latent.shape)}.")
        batch, channels, frames, height, width = [int(value) for value in latent.shape]
        config = self.config
        if channels != int(config.latent_channels):
            raise ValueError(f"latent channel mismatch: expected {config.latent_channels}, got {channels}.")
        pf = int(config.latent_patch_frames)
        ph = int(config.latent_patch_height)
        pw = int(config.latent_patch_width)
        if frames % pf or height % ph or width % pw:
            raise ValueError(
                f"latent shape {(frames, height, width)} is not divisible by patch {(pf, ph, pw)}."
            )
        ft, ht, wt = frames // pf, height // ph, width // pw
        raw = latent.reshape(batch, channels, ft, pf, ht, ph, wt, pw)
        raw = raw.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(batch, ft * ht * wt, -1)
        raw = _safe_tensor(raw).to(dtype=_parameter_dtype(self.latent_proj))
        tokens = self.latent_proj(raw)
        coords = _coordinate_grid(ft, ht, wt, device=tokens.device, dtype=tokens.dtype)
        coords = self.latent_coord_proj(coords.to(dtype=_parameter_dtype(self.latent_coord_proj)))
        tokens = tokens + coords.unsqueeze(0).to(dtype=tokens.dtype)
        return tokens + self._type(0, device=tokens.device, dtype=tokens.dtype)

    def _attention_tokens(self, attention_probs: torch.Tensor) -> torch.Tensor:
        if attention_probs.ndim != 6:
            raise ValueError(
                "attention_probs must have shape [B,L,H,C,Q,K], "
                f"got {tuple(attention_probs.shape)}."
            )
        batch, layers, heads, channels, queries, keys = [int(value) for value in attention_probs.shape]
        if channels != int(self.config.attention_input_channels) or keys != int(self.config.attention_key_count):
            raise ValueError("attention probability shape does not match the configured channels/key count.")
        raw = attention_probs.permute(0, 1, 2, 4, 3, 5).reshape(batch, layers * heads * queries, -1)
        raw = _safe_tensor(raw).to(dtype=_parameter_dtype(self.attention_proj))
        tokens = self.attention_proj(raw)
        coords = _coordinate_grid(layers, heads, queries, device=tokens.device, dtype=tokens.dtype)
        coords = self.attention_coord_proj(coords.to(dtype=_parameter_dtype(self.attention_coord_proj)))
        tokens = tokens + coords.unsqueeze(0).to(dtype=tokens.dtype)
        return tokens + self._type(1, device=tokens.device, dtype=tokens.dtype)

    def forward(
        self,
        latent: torch.Tensor,
        attention_probs: torch.Tensor,
        *,
        context_features: torch.Tensor,
        prompt_embedding: torch.Tensor,
    ) -> SharedTensorFeatures:
        latent_source = self._latent_tokens(latent)
        attention_source = self._attention_tokens(attention_probs)
        batch_size = int(latent_source.shape[0])
        if int(attention_source.shape[0]) != batch_size:
            raise ValueError("latent and attention batch sizes differ.")
        if context_features.ndim != 2 or tuple(context_features.shape) != (
            batch_size,
            int(self.config.context_dim),
        ):
            raise ValueError(
                f"context_features must have shape [{batch_size},{self.config.context_dim}], "
                f"got {tuple(context_features.shape)}."
            )
        if prompt_embedding.ndim != 2 or tuple(prompt_embedding.shape) != (
            batch_size,
            int(self.config.text_embedding_dim),
        ):
            raise ValueError(
                f"prompt_embedding must have shape [{batch_size},{self.config.text_embedding_dim}], "
                f"got {tuple(prompt_embedding.shape)}."
            )

        latent_queries = self.latent_queries.to(
            device=latent_source.device,
            dtype=latent_source.dtype,
        ).expand(batch_size, -1, -1)
        attention_queries = self.attention_queries.to(
            device=attention_source.device,
            dtype=attention_source.dtype,
        ).expand(batch_size, -1, -1)
        for layer in self.latent_cross:
            latent_queries = layer(latent_queries, latent_source)
        for layer in self.attention_cross:
            attention_queries = layer(attention_queries, attention_source)
        latent_queries = latent_queries + self._type(0, device=latent_queries.device, dtype=latent_queries.dtype)
        attention_queries = attention_queries + self._type(
            1,
            device=attention_queries.device,
            dtype=attention_queries.dtype,
        )

        text_hidden = self.text_proj(
            _safe_tensor(prompt_embedding).to(
                device=latent_source.device,
                dtype=_parameter_dtype(self.text_proj),
            )
        )
        context_hidden = self.context_proj(
            _safe_tensor(context_features).to(
                device=latent_source.device,
                dtype=_parameter_dtype(self.context_proj),
            )
        )
        text_token = text_hidden.unsqueeze(1) + self._type(
            2,
            device=latent_source.device,
            dtype=text_hidden.dtype,
        )
        context_token = context_hidden.unsqueeze(1) + self._type(
            3,
            device=latent_source.device,
            dtype=context_hidden.dtype,
        )
        global_token = self.global_token.to(
            device=latent_source.device,
            dtype=latent_source.dtype,
        ).expand(batch_size, -1, -1)
        global_token = global_token + self._type(
            4,
            device=latent_source.device,
            dtype=latent_source.dtype,
        )
        encoded = self.fusion(
            torch.cat(
                [global_token, latent_queries, attention_queries, text_token, context_token],
                dim=1,
            )
        )
        latent_start = 1
        attention_start = latent_start + int(latent_queries.shape[1])
        text_index = attention_start + int(attention_queries.shape[1])
        context_index = text_index + 1
        return SharedTensorFeatures(
            global_hidden=self.output_norm(encoded[:, 0]),
            latent_hidden=self.output_norm(encoded[:, latent_start:attention_start].mean(dim=1)),
            attention_hidden=self.output_norm(encoded[:, attention_start:text_index].mean(dim=1)),
            text_hidden=self.output_norm(encoded[:, text_index]),
            context_hidden=self.output_norm(encoded[:, context_index]),
        )


class TaskAdapter(nn.Module):
    def __init__(self, *, input_dim: int, width: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(width)),
            nn.GELU(),
        )
        self.adapter = nn.Sequential(
            nn.LayerNorm(int(width)),
            nn.Linear(int(width), int(width)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(width), int(width)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(features)
        return hidden + self.adapter(hidden)


class SharedResolutionSparsePolicy(nn.Module):
    model_type = "shared_perceiver_resolution_sparse"

    def __init__(self, config: SharedResolutionSparsePolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = SharedPerceiverTensorBackbone(config.backbone)
        width = int(config.adapter_width)
        self.switch_adapter = TaskAdapter(
            input_dim=self.backbone.output_dim,
            width=width,
            dropout=float(config.dropout),
        )
        self.sparse_window_embedding = nn.Embedding(int(config.sparse_window_count), width)
        self.sparse_decision_proj = nn.Sequential(
            nn.LayerNorm(int(config.sparse_decision_feature_dim)),
            nn.Linear(int(config.sparse_decision_feature_dim), width),
            nn.GELU(),
        )
        self.sparse_adapter = TaskAdapter(
            input_dim=self.backbone.output_dim + width * 2,
            width=width,
            dropout=float(config.dropout),
        )
        self.switch_head = CandidateScoringHead(
            hidden_dim=width,
            candidate_feature_dim=int(config.switch_candidate_feature_dim),
            width=width,
            dropout=float(config.dropout),
        )
        self.sparse_head = CandidateScoringHead(
            hidden_dim=width,
            candidate_feature_dim=int(config.sparse_candidate_feature_dim),
            width=width,
            dropout=float(config.dropout),
        )

    def encode(
        self,
        latent: torch.Tensor,
        attention_probs: torch.Tensor,
        *,
        context_features: torch.Tensor,
        prompt_embedding: torch.Tensor,
    ) -> SharedTensorFeatures:
        return self.backbone(
            latent,
            attention_probs,
            context_features=context_features,
            prompt_embedding=prompt_embedding,
        )

    def forward_switch(
        self,
        latent: torch.Tensor,
        attention_probs: torch.Tensor,
        *,
        context_features: torch.Tensor,
        prompt_embedding: torch.Tensor,
        candidate_features: torch.Tensor,
        detach_backbone: bool = False,
    ) -> torch.Tensor:
        features = self.encode(
            latent,
            attention_probs,
            context_features=context_features,
            prompt_embedding=prompt_embedding,
        )
        return self.forward_switch_from_features(
            features,
            candidate_features=candidate_features,
            detach_backbone=detach_backbone,
        )

    def forward_switch_from_features(
        self,
        features: SharedTensorFeatures,
        *,
        candidate_features: torch.Tensor,
        detach_backbone: bool = False,
    ) -> torch.Tensor:
        shared_features = features.concatenate()
        if bool(detach_backbone):
            shared_features = shared_features.detach()
        hidden = self.switch_adapter(shared_features)
        return self.switch_head(hidden, candidate_features)

    def forward_sparse(
        self,
        latent: torch.Tensor,
        attention_probs: torch.Tensor,
        *,
        context_features: torch.Tensor,
        prompt_embedding: torch.Tensor,
        window_indices: torch.Tensor,
        decision_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        features = self.encode(
            latent,
            attention_probs,
            context_features=context_features,
            prompt_embedding=prompt_embedding,
        )
        return self.forward_sparse_from_features(
            features,
            window_indices=window_indices,
            decision_features=decision_features,
            candidate_features=candidate_features,
        )

    def forward_sparse_from_features(
        self,
        features: SharedTensorFeatures,
        *,
        window_indices: torch.Tensor,
        decision_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        device = features.global_hidden.device
        indices = window_indices.to(device=device, dtype=torch.long).reshape(-1)
        window_hidden = self.sparse_window_embedding(indices)
        decision_hidden = self.sparse_decision_proj(
            decision_features.to(device=device, dtype=_parameter_dtype(self.sparse_decision_proj))
        )
        hidden = self.sparse_adapter(
            torch.cat(
                [features.concatenate(), window_hidden, decision_hidden.to(dtype=window_hidden.dtype)],
                dim=-1,
            )
        )
        return self.sparse_head(hidden, candidate_features)

