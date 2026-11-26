import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import warnings
from PIL import Image

from cole.common import video_utils


def test_video_tensor_to_pil_frames_preserves_pixels():
    video = torch.tensor(
        [
            [
                [[0.0, 1.0], [0.5, 0.25]],
                [[0.25, 0.5], [1.0, 0.0]],
                [[1.0, 0.0], [0.25, 0.5]],
            ]
        ],
        dtype=torch.float32,
    )

    frames = video_utils.video_tensor_to_pil_frames(video)

    assert len(frames) == 1
    assert isinstance(frames[0], Image.Image)
    np.testing.assert_array_equal(
        np.array(frames[0]),
        np.array(
            [
                [[0, 63, 255], [255, 127, 0]],
                [[127, 255, 63], [63, 0, 127]],
            ],
            dtype=np.uint8,
        ),
    )


def test_save_video_tensor_uses_pil_frames(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_export(frames, output_path, fps):
        captured["frames"] = frames
        captured["output_path"] = output_path
        captured["fps"] = fps
        return output_path

    monkeypatch.setattr(video_utils, "export_to_video", _fake_export)

    video = torch.zeros((1, 1, 3, 2, 2), dtype=torch.float32)
    out_path = tmp_path / "sample.mp4"

    result = video_utils.save_video_tensor(video, out_path, fps=12)

    assert result == str(out_path)
    assert captured["output_path"] == str(out_path)
    assert captured["fps"] == 12
    assert all(isinstance(frame, Image.Image) for frame in captured["frames"])


def test_save_video_tensor_forwards_explicit_encoding_settings(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_export(frames, output_path, fps, quality, macro_block_size):
        captured["frames"] = frames
        captured["output_path"] = output_path
        captured["fps"] = fps
        captured["quality"] = quality
        captured["macro_block_size"] = macro_block_size
        return output_path

    monkeypatch.setattr(video_utils, "export_to_video", _fake_export)
    video = torch.zeros((1, 1, 3, 2, 2), dtype=torch.float32)
    out_path = tmp_path / "formal.mp4"

    result = video_utils.save_video_tensor(
        video,
        out_path,
        fps=16,
        quality=9.0,
        macro_block_size=1,
    )

    assert result == str(out_path)
    assert captured["quality"] == 9.0
    assert captured["macro_block_size"] == 1


def test_video_tensor_to_uint8_frames_rejects_nonfinite_values_when_guard_enabled(monkeypatch):
    monkeypatch.setenv("COLE_ENABLE_FINITE_GUARD", "1")
    video = torch.zeros((1, 1, 3, 2, 2), dtype=torch.float32)
    video[0, 0, 0, 0, 0] = float('nan')

    with pytest.raises(ValueError, match="non-finite values"):
        video_utils.video_tensor_to_uint8_frames(video)


def test_video_tensor_to_uint8_frames_allows_nonfinite_values_when_guard_disabled(monkeypatch):
    monkeypatch.delenv("COLE_ENABLE_FINITE_GUARD", raising=False)
    video = torch.zeros((1, 1, 3, 2, 2), dtype=torch.float32)
    video[0, 0, 0, 0, 0] = float('nan')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        frames = video_utils.video_tensor_to_uint8_frames(video)

    assert len(frames) == 1
