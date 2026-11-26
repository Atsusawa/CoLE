from __future__ import annotations

import torch


def mean_pool_prompt_embedding(prompt_embeds: torch.Tensor) -> list[float]:
    """Convert prompt conditioning tensors into a fixed-width controller feature vector."""

    if not isinstance(prompt_embeds, torch.Tensor):
        raise TypeError(f"prompt_embeds must be a torch.Tensor, got {type(prompt_embeds)!r}")
    if prompt_embeds.ndim == 0:
        raise ValueError("prompt_embeds must have at least one dimension")

    tensor = prompt_embeds.detach().to(dtype=torch.float32)
    if tensor.ndim == 1:
        pooled = tensor
    else:
        hidden_dim = int(tensor.shape[-1])
        pooled = tensor.reshape(-1, hidden_dim).mean(dim=0)
    return [float(value) for value in pooled.cpu().tolist()]
