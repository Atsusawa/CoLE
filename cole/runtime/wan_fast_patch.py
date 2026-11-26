# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# CoLE adapts Diffusers' Wan transformer forwards for its fused kernels.

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel, WanTransformerBlock
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

from cole.runtime.wan_fast_support import (
    ENABLE_FAST_KERNEL,
    layernorm_forward,
    modulate_gate_residual_forward,
    modulate_shift_forward,
)


class WanTransformerBlock_CoLE(WanTransformerBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        timestep: int = 0,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        if isinstance(self.norm1, FP32LayerNorm):
            norm_hidden_states = layernorm_forward(
                hidden_states, self.norm1.weight, self.norm1.bias, self.norm1.eps, self.norm1.elementwise_affine
            )
        else:
            raise ValueError(f"Unsupported norm type: {type(self.norm1)}")

        if ENABLE_FAST_KERNEL:
            norm_hidden_states = modulate_shift_forward(
                norm_hidden_states, scale_msa, shift_msa, output_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = (norm_hidden_states * (1 + scale_msa) + shift_msa).type_as(hidden_states)

        try:
            attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, timestep=timestep)
        except TypeError as exc:
            if "unexpected keyword argument 'timestep'" not in str(exc):
                raise
            attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)

        if ENABLE_FAST_KERNEL:
            hidden_states = modulate_gate_residual_forward(
                hidden_states, attn_output, gate_msa, output_dtype=hidden_states.dtype
            )
        else:
            hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        if isinstance(self.norm2, FP32LayerNorm):
            norm_hidden_states = layernorm_forward(
                hidden_states, self.norm2.weight, self.norm2.bias, self.norm2.eps, self.norm2.elementwise_affine
            ).type_as(hidden_states)
        elif isinstance(self.norm2, torch.nn.Identity):
            norm_hidden_states = hidden_states
        else:
            raise ValueError(f"Unsupported norm type: {type(self.norm2)}")

        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        if isinstance(self.norm3, FP32LayerNorm):
            norm_hidden_states = layernorm_forward(
                hidden_states, self.norm3.weight, self.norm3.bias, self.norm3.eps, self.norm3.elementwise_affine
            )
        else:
            raise ValueError(f"Unsupported norm type: {type(self.norm3)}")

        if ENABLE_FAST_KERNEL:
            norm_hidden_states = modulate_shift_forward(
                norm_hidden_states, c_scale_msa, c_shift_msa, output_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = (norm_hidden_states * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)

        ff_output = self.ffn(norm_hidden_states)
        if ENABLE_FAST_KERNEL:
            hidden_states = modulate_gate_residual_forward(
                hidden_states, ff_output, c_gate_msa, output_dtype=hidden_states.dtype
            )
        else:
            hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states


class WanTransformer3DModel_CoLE(WanTransformer3DModel):
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
            cole_timestep_int = attention_kwargs.pop("cole_timestep_int", None)
        else:
            lora_scale = 1.0
            cole_timestep_int = None

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        if hasattr(self.rope, "set_timestep"):
            normalized_timestep = 1.0
            if cole_timestep_int is not None:
                timestep_value = float(cole_timestep_int)
                normalized_timestep = max(0.0, min(1.0, timestep_value / 1000.0))
            elif isinstance(timestep, torch.Tensor) and timestep.numel() > 0:
                timestep_value = float(timestep.reshape(-1)[0].item())
                normalized_timestep = max(0.0, min(1.0, timestep_value / 1000.0))
            self.rope.set_timestep(normalized_timestep)
        rotary_emb = self.rope(hidden_states)
        if ENABLE_FAST_KERNEL:
            if isinstance(rotary_emb, (tuple, list)):
                rot_real, rot_imag = rotary_emb
                if torch.is_complex(rot_real):
                    rot_real, rot_imag = rot_real.real, rot_real.imag
                rotary_emb = (rot_real.squeeze().contiguous().to(torch.float32), rot_imag.squeeze().contiguous().to(torch.float32))
            else:
                rotary_emb = (
                    rotary_emb.real.squeeze().contiguous().to(torch.float32),
                    rotary_emb.imag.squeeze().contiguous().to(torch.float32),
                )

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        block_timestep = int(cole_timestep_int) if cole_timestep_int is not None else timestep

        def _compute_blocks(value: torch.Tensor) -> torch.Tensor:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                for block in self.blocks:
                    value = self._gradient_checkpointing_func(
                        block, value, encoder_hidden_states, timestep_proj, rotary_emb
                    )
                return value
            for block in self.blocks:
                value = block(
                    value,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    timestep=block_timestep,
                )
            return value

        hidden_states = _compute_blocks(hidden_states)

        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def replace_sparse_forward() -> None:
    WanTransformerBlock.forward = WanTransformerBlock_CoLE.forward
    WanTransformer3DModel.forward = WanTransformer3DModel_CoLE.forward
