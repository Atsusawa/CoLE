from __future__ import annotations

import os
import random
from typing import Optional


def parse_device_arg(value: str) -> tuple[Optional[str], str]:
    raw = str(value).strip()
    if not raw:
        return None, "cuda:0"
    lowered = raw.lower()
    if lowered == "cpu":
        return None, "cpu"
    if lowered.startswith("cuda"):
        return None, lowered
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if parts and all(part.isdigit() for part in parts):
        return ",".join(parts), "cuda:0"
    raise ValueError(f"Unsupported device string: {value}")


def apply_visible_devices(visible_devices: Optional[str]) -> None:
    if visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices


def configure_torch_runtime(
    *,
    seed: int,
    device_str: str,
    use_tf32: bool = False,
    cudnn_benchmark: bool = False,
    deterministic_algorithms: bool = False,
    deterministic_warn_only: bool = False,
) -> None:
    import torch

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device_str.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device_str))
        torch.cuda.manual_seed_all(int(seed))
    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(int(seed))

    torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
    torch.backends.cudnn.allow_tf32 = bool(use_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if bool(use_tf32) else "highest")

    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.backends.cudnn.deterministic = bool(deterministic_algorithms)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(
            bool(deterministic_algorithms),
            warn_only=bool(deterministic_warn_only),
        )
