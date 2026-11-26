from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SparseProfileSpec:
    name: str
    backend_name: str
    sparse_mode: str
    sparse_value: float | None
    estimated_density: float

    @property
    def is_dense(self) -> bool:
        return self.backend_name == "dense"


SPARSE_PROFILE_SPECS: dict[str, SparseProfileSpec] = {
    "dense_full": SparseProfileSpec(
        name="dense_full",
        backend_name="dense",
        sparse_mode="full",
        sparse_value=None,
        estimated_density=1.0,
    ),
    "sparse_p90": SparseProfileSpec(
        name="sparse_p90",
        backend_name="sparse",
        sparse_mode="topk",
        sparse_value=0.90,
        estimated_density=0.90,
    ),
    "sparse_p80": SparseProfileSpec(
        name="sparse_p80",
        backend_name="sparse",
        sparse_mode="topk",
        sparse_value=0.80,
        estimated_density=0.80,
    ),
    "sparse_p70": SparseProfileSpec(
        name="sparse_p70",
        backend_name="sparse",
        sparse_mode="topk",
        sparse_value=0.70,
        estimated_density=0.70,
    ),
    "sparse_safe": SparseProfileSpec(
        name="sparse_safe",
        backend_name="sparse",
        sparse_mode="topk",
        sparse_value=0.60,
        estimated_density=0.60,
    ),
    "sparse_mid": SparseProfileSpec(
        name="sparse_mid",
        backend_name="sparse",
        sparse_mode="topk",
        sparse_value=0.50,
        estimated_density=0.50,
    ),
    "sparse_fast": SparseProfileSpec(
        name="sparse_fast",
        backend_name="sparse",
        sparse_mode="topk",
        sparse_value=0.40,
        estimated_density=0.40,
    ),
    "sparse_aggr": SparseProfileSpec(
        name="sparse_aggr",
        backend_name="sparse",
        sparse_mode="topk",
        sparse_value=0.30,
        estimated_density=0.30,
    ),
    "sparse_p20": SparseProfileSpec(
        name="sparse_p20",
        backend_name="sparse",
        sparse_mode="topk",
        sparse_value=0.20,
        estimated_density=0.20,
    ),
    "sparse_p10": SparseProfileSpec(
        name="sparse_p10",
        backend_name="sparse",
        sparse_mode="topk",
        sparse_value=0.10,
        estimated_density=0.10,
    ),
}

SPARSE_PROFILE_ORDER: tuple[str, ...] = tuple(SPARSE_PROFILE_SPECS.keys())
_SPARSE_TOPK_TO_PROFILE: dict[float, str] = {
    round(float(spec.sparse_value), 10): str(name)
    for name, spec in SPARSE_PROFILE_SPECS.items()
    if str(spec.sparse_mode) == "topk" and spec.sparse_value is not None
}


def get_sparse_profile_spec(name: str) -> SparseProfileSpec:
    if name not in SPARSE_PROFILE_SPECS:
        return SPARSE_PROFILE_SPECS["dense_full"]
    return SPARSE_PROFILE_SPECS[name]


def get_sparse_profile_index(name: str) -> int:
    try:
        return SPARSE_PROFILE_ORDER.index(name)
    except ValueError:
        return 0


def get_sparse_profile_topk(name: str) -> float | None:
    spec = get_sparse_profile_spec(str(name))
    if str(spec.sparse_mode) != "topk" or spec.sparse_value is None:
        return None
    return round(float(spec.sparse_value), 10)


def get_sparse_profile_name_for_topk(topk: float | None) -> str | None:
    if topk is None:
        return None
    try:
        normalized = round(float(topk), 10)
    except (TypeError, ValueError):
        return None
    return _SPARSE_TOPK_TO_PROFILE.get(normalized)
