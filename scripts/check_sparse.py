#!/usr/bin/env python3
"""Run CoLE's sm_120 sparse attention backend on BF16 tensors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch

    from cole.core.types import MaskPlan
    from cole.runtime.backends.sparse_backend import SparseAttentionBackend, load_sparse_kernel

    try:
        device = torch.device("cuda:0" if args.device == "cuda" else args.device)
        load_sparse_kernel(device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    plan = MaskPlan(
        backend_name="sparse", profile_name="sparse_mid",
        sparse_mode="topk", sparse_value=0.50, estimated_density=0.50,
        selected_units=0, plan_signature=("sparse_check",), sparse_return_sparsity=True,
    )
    torch.manual_seed(0)
    with torch.inference_mode():
        q, k, v = torch.randn((3, 1, 2, 512, 128), device=device, dtype=torch.bfloat16).unbind()
        output, feedback = SparseAttentionBackend().run(q, k, v, plan)
        torch.cuda.synchronize(device)
        if output.shape != q.shape or not torch.isfinite(output).all().item():
            raise RuntimeError("Sparse attention returned unexpected or non-finite output.")
        if not 0.0 <= feedback.actual_density <= 1.0:
            raise RuntimeError("Sparse attention returned an invalid QK block density.")
    print(json.dumps({
        "device": torch.cuda.get_device_name(device),
        "arch": "sm_120",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "backend": feedback.backend_name,
        "topk": plan.sparse_value,
        "qk_block_density": feedback.actual_density,
    }, indent=2))


if __name__ == "__main__":
    main()
