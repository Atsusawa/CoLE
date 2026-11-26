from __future__ import annotations

import functools
import os
import time
from collections.abc import Callable

import torch

from cole.core.feedback import ExecFeedback
from cole.core.types import MaskPlan

from .base import AttentionBackend


@functools.cache
def load_sparse_kernel(device: torch.device) -> Callable[..., torch.Tensor | tuple[torch.Tensor, float]]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("CoLE sparse attention requires a visible NVIDIA CUDA GPU with compute capability 12.0 (sm_120).")
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) != (12, 0):
        raise ValueError(f"CoLE sparse attention requires sm_120. GPU {device} has compute capability {major}.{minor}.")

    try:
        import spas_sage_attn
        from spas_sage_attn.core import SAGE2PP_ENABLED
    except (ImportError, OSError) as exc:
        raise ValueError(
            f"Could not load the sparse CUDA extension. Follow the build instructions in the README. {exc}"
        ) from exc

    kernel = getattr(spas_sage_attn, "spas_sage2_attn_meansim_topk_cuda", None)
    if not callable(kernel) or not SAGE2PP_ENABLED:
        raise ValueError("CoLE requires the Sage2++ top-k kernel. Rebuild the sparse CUDA extension for sm_120.")

    if os.environ.get("COLE_TRACE_SPARSE_DISPATCH") == "1":
        print(
            f"[cole-sparse-dispatch] device={device} arch=sm_120 entrypoint={kernel.__name__} "
            f"sage2pp_enabled={SAGE2PP_ENABLED} module_file={spas_sage_attn.__file__}",
            flush=True,
        )
    return kernel


class SparseAttentionBackend(AttentionBackend):
    """Block-sparse top-k attention for sm_120."""

    def __init__(self) -> None:
        self._has_run = False

    def supports(self, mask_plan: MaskPlan, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
        return (
            mask_plan.backend_name == "sparse"
            and mask_plan.sparse_mode == "topk"
            and q.is_cuda and k.device == q.device and v.device == q.device
            and q.ndim == 4 and q.shape == k.shape == v.shape
            and q.shape[-2] >= 128 and q.shape[-1] in (64, 128)
        )

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask_plan: MaskPlan) -> tuple[torch.Tensor, ExecFeedback]:
        if mask_plan.sparse_mode != "topk" or mask_plan.sparse_value is None:
            raise ValueError("CoLE sparse attention requires a top-k profile.")
        prepare_start = time.perf_counter()
        kernel = load_sparse_kernel(q.device)
        prep_ms = (time.perf_counter() - prepare_start) * 1000.0

        run_start = time.perf_counter()
        result = kernel(
            q.contiguous(), k.contiguous(), v.contiguous(),
            topk=float(mask_plan.sparse_value),
            return_sparsity=mask_plan.sparse_return_sparsity,
        )
        run_ms = (time.perf_counter() - run_start) * 1000.0
        density = mask_plan.estimated_density
        if mask_plan.sparse_return_sparsity:
            output, sparsity = result
            density = 1.0 - float(sparsity)
        else:
            output = result

        feedback = ExecFeedback(
            backend_prepare_ms=prep_ms,
            backend_run_ms=run_ms,
            total_attn_ms=prep_ms + run_ms,
            actual_density=density,
            selected_units=mask_plan.selected_units,
            backend_name="sparse",
            executed_plan_signature=str(mask_plan.plan_signature),
            executed_plan_density=density,
            backend_is_sparse=True,
            backend_first_call=not self._has_run,
        )
        self._has_run = True
        return output, feedback
