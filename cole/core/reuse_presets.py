from __future__ import annotations

from typing import Any, Iterable


# Relative latent-delta threshold, maximum consecutive reuse, and full refresh steps.
PRESETS = {
    "safe": (0.06, 2, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 30, 33, 36, 39, 42, 45, 48, 49)),
    "balanced": (0.10, 3, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 31, 34, 37, 40, 43, 46, 48, 49)),
    "fast": (0.15, 4, (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 17, 18, 19, 20, 21, 22, 23, 28, 32, 36, 40, 44, 48, 49)),
    "turbo": (0.20, 5, (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 17, 18, 19, 20, 21, 22, 23, 29, 35, 41, 47, 49)),
    "ultra": (0.30, 7, (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 17, 18, 19, 20, 21, 22, 23, 31, 39, 47, 49)),
}


def build_reuse_config(
    action: str, *, total_steps: int, switch_step: int, force_full_steps: Iterable[int] = (),
) -> dict[str, Any]:
    forced = sorted({step for step in force_full_steps if 0 <= step < total_steps})
    if action == "off":
        return {"mode": "off", "switch_step": switch_step, "full_steps": forced, "cache_steps": []}
    threshold, cap, refresh_steps = PRESETS[action]
    full_steps = set(forced).union(step for step in refresh_steps if step < total_steps)
    return {
        "mode": "adaptive_output",
        "full_steps": sorted(full_steps),
        "cache_steps": [step for step in range(total_steps) if step not in full_steps],
        "max_consecutive_cached_steps": cap,
        "adaptive_delta_threshold": threshold,
        "switch_step": switch_step,
        "runtime_force_full_steps": forced,
    }
