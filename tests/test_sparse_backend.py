from __future__ import annotations

import sys
from dataclasses import replace
from types import ModuleType

import pytest
import torch

from cole.core.types import MaskPlan
from cole.runtime.attention_bridge import AttentionBridge
from cole.runtime.backends import sparse_backend


@pytest.fixture
def fake_cuda(monkeypatch):
    sparse_backend.load_sparse_kernel.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    yield
    sparse_backend.load_sparse_kernel.cache_clear()


def test_load_sparse_kernel_rejects_other_cuda_architectures(fake_cuda, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))

    with pytest.raises(ValueError, match=r"requires sm_120.*9\.0"):
        sparse_backend.load_sparse_kernel(torch.device("cuda:0"))


def test_load_sparse_kernel_requires_cuda():
    with pytest.raises(ValueError, match="visible NVIDIA CUDA GPU"):
        sparse_backend.load_sparse_kernel(torch.device("cpu"))


@pytest.mark.parametrize("sage2pp_enabled", [True, False])
def test_load_sparse_kernel_uses_sage2pp_topk(fake_cuda, monkeypatch, sage2pp_enabled):
    module = ModuleType("spas_sage_attn")
    core = ModuleType("spas_sage_attn.core")

    def topk(q, k, v, **kwargs):
        return q

    module.spas_sage2_attn_meansim_topk_cuda = topk
    core.SAGE2PP_ENABLED = sage2pp_enabled
    monkeypatch.setitem(sys.modules, "spas_sage_attn", module)
    monkeypatch.setitem(sys.modules, "spas_sage_attn.core", core)

    if sage2pp_enabled:
        assert sparse_backend.load_sparse_kernel(torch.device("cuda:0")) is topk
    else:
        with pytest.raises(ValueError, match=r"requires the Sage2\+\+ top-k kernel"):
            sparse_backend.load_sparse_kernel(torch.device("cuda:0"))


@pytest.fixture
def sparse_plan():
    return MaskPlan(
        backend_name="sparse", profile_name="sparse_mid",
        sparse_mode="topk", sparse_value=0.50, estimated_density=0.50,
        selected_units=1, plan_signature=("sparse_mid",),
    )


@pytest.mark.parametrize("measure_density", [False, True])
def test_sparse_backend_forwards_topk_and_reports_density(monkeypatch, sparse_plan, measure_density):
    def kernel(q, k, v, *, topk, return_sparsity):
        assert topk == 0.50
        assert return_sparsity is measure_density
        return (q, 0.25) if return_sparsity else q

    monkeypatch.setattr(sparse_backend, "load_sparse_kernel", lambda device: kernel)
    query = torch.randn(1, 2, 128, 128)
    backend = sparse_backend.SparseAttentionBackend()
    plan = replace(sparse_plan, sparse_return_sparsity=measure_density)

    output, feedback = backend.run(query, query, query, plan)

    assert torch.equal(output, query)
    assert feedback.backend_is_sparse
    assert feedback.backend_name == "sparse"
    assert feedback.actual_density == (0.75 if measure_density else 0.50)
    assert feedback.backend_first_call
    assert not backend.run(query, query, query, plan)[1].backend_first_call


def test_sparse_plan_does_not_select_dense_backend_on_cpu(sparse_plan):
    query = torch.zeros(1, 2, 128, 128)

    with pytest.raises(RuntimeError, match="backend_name=sparse"):
        AttentionBridge().run(query, query, query, sparse_plan)
