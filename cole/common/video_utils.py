from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
try:
    from diffusers.utils import export_to_video
except ImportError:  # pragma: no cover
    export_to_video = None
from PIL import Image

from cole.common.finite_guard import finite_guard_enabled

def _assert_finite_video_tensor(video: torch.Tensor, *, max_examples: int = 8) -> None:
    if not finite_guard_enabled():
        return
    tensor = video.detach().cpu().float()
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all().item()):
        return
    bad_indices = torch.nonzero(~finite_mask, as_tuple=False)
    examples: list[str] = []
    for raw_idx in bad_indices[: int(max_examples)]:
        idx = tuple(int(value) for value in raw_idx.tolist())
        value = float(tensor[idx].item())
        examples.append(f"{idx}={value!r}")
    bad_count = int(bad_indices.shape[0])
    total = int(tensor.numel())
    raise ValueError(
        f"Video tensor contains non-finite values before uint8 conversion "
        f"shape={tuple(tensor.shape)} bad={bad_count}/{total} examples={examples}"
    )


def video_tensor_to_uint8_frames(video: torch.Tensor) -> list[np.ndarray]:
    """Convert (F,C,H,W) or (1,F,C,H,W) tensor in [0,1] to list of HWC uint8 frames."""
    tensor = video.detach().cpu().float()
    _assert_finite_video_tensor(tensor)
    if tensor.dim() == 5:
        tensor = tensor[0]
    if tensor.dim() != 4:
        raise ValueError(f"Expected video tensor with 4 or 5 dims, got shape={tuple(tensor.shape)}")
    tensor = tensor.clamp(0.0, 1.0)
    frames = tensor.permute(0, 2, 3, 1).numpy()
    return [np.clip(frame * 255.0, 0, 255).astype(np.uint8) for frame in frames]


def video_tensor_to_pil_frames(video: torch.Tensor) -> list[Image.Image]:
    """Convert a video tensor to RGB PIL frames for diffusers.export_to_video.

    Passing uint8 numpy frames would be scaled a second time by diffusers.
    """
    return [Image.fromarray(frame) for frame in video_tensor_to_uint8_frames(video)]


def save_video_tensor(
    video: torch.Tensor,
    path: str | Path,
    *,
    fps: int = 16,
    quality: float | None = None,
    macro_block_size: int | None = None,
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if export_to_video is not None:
        export_kwargs: dict[str, float | int] = {}
        if quality is not None:
            export_kwargs["quality"] = float(quality)
        if macro_block_size is not None:
            export_kwargs["macro_block_size"] = int(macro_block_size)
        return export_to_video(
            video_tensor_to_pil_frames(video),
            str(target),
            fps=fps,
            **export_kwargs,
        )
    try:
        import imageio.v3 as iio
        frames = video_tensor_to_uint8_frames(video)
        iio.imwrite(str(target), frames, fps=fps)
        return str(target)
    except Exception as exc:
        raise RuntimeError(f"Failed to save video to {target}: {exc}") from exc
