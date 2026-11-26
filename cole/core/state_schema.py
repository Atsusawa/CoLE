from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StateSchema:
    """State group sizes."""

    groups: dict[str, int] = field(
        default_factory=lambda: {
            "phase": 5,
            "history": 5,
            "structure": 8,
            "motion": 4,
            "efficiency": 4,
            "attention": 4,
        }
    )

    @property
    def dim(self) -> int:
        return sum(self.groups.values())
