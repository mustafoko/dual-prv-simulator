"""Workspace and slot definitions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Slot:
    """Workspace slot definition."""
    name: str
    x: float
    y: float
    radius: float  # occupancy radius (meters)


def default_slots() -> Dict[str, Slot]:
    """Return default 2x2 grid workspace slots."""
    r = 0.10
    return {
        "A": Slot("A", x=-0.25, y=+0.25, radius=r),
        "B": Slot("B", x=+0.25, y=+0.25, radius=r),
        "C": Slot("C", x=-0.25, y=-0.25, radius=r),
        "D": Slot("D", x=+0.25, y=-0.25, radius=r),
    }


def dist2(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points."""
    return math.hypot(x1 - x2, y1 - y2)


def in_slot(px: float, py: float, slot: Slot) -> bool:
    """Check if point (px, py) is within slot."""
    return dist2(px, py, slot.x, slot.y) <= slot.radius
