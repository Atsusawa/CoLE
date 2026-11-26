from __future__ import annotations

import os


def finite_guard_enabled() -> bool:
    return os.environ.get("COLE_ENABLE_FINITE_GUARD", "0") == "1"
