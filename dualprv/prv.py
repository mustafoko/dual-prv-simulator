"""PRV (Predictive Runtime Verification) classes and fusion logic."""

from __future__ import annotations

import math
from typing import Dict, Optional

from .models import BranchOutput, FusionParams, Mode, SafetyParams, Verdict
from .workspace import Slot, dist2, in_slot


def protective_separation(safe: SafetyParams, v_r: float, v_h: float) -> float:
    """Calculate protective separation distance."""
    return (v_r + v_h) * safe.t_react + v_r * safe.t_stop + safe.margin


def dt_to_sep_violation_formal(safe: SafetyParams, d: float) -> float:
    """Calculate time to separation violation using formal method."""
    Sp = protective_separation(safe, safe.v_r_max, safe.v_h_max)
    if d <= Sp:
        return 0.0
    v_close = safe.v_r_max + safe.v_h_max
    return (d - Sp) / max(v_close, 1e-9)


def travel_time_to_slot(px: float, py: float, slot: Slot, v_assumed: float) -> float:
    """Calculate travel time to reach a slot."""
    d = dist2(px, py, slot.x, slot.y)
    if d <= slot.radius:
        return 0.0
    return max(0.0, (d - slot.radius) / max(v_assumed, 1e-9))


def dt_to_mutex_violation_formal(
    safe: SafetyParams, slots: Dict[str, Slot], hx: float, hy: float, rx: float, ry: float
) -> float:
    """Calculate time to mutual exclusion violation using formal method."""
    best = math.inf
    for s in slots.values():
        th = travel_time_to_slot(hx, hy, s, safe.v_h_max)
        tr = travel_time_to_slot(rx, ry, s, safe.v_r_max)
        best = min(best, max(th, tr))
    return best


class FormalPRV:
    """Formal PRV branch - conservative envelope prediction."""

    def __init__(self, safe: SafetyParams, slots: Dict[str, Slot]):
        self.safe = safe
        self.slots = slots

    def step(self, hx: float, hy: float, rx: float, ry: float) -> BranchOutput:
        """Execute one step of formal PRV monitoring."""
        Sp = protective_separation(self.safe, self.safe.v_r_max, self.safe.v_h_max)
        d = dist2(hx, hy, rx, ry)

        sep_violation_now = (d <= Sp)
        mutex_violation_now = any(in_slot(hx, hy, s) and in_slot(rx, ry, s) for s in self.slots.values())
        if sep_violation_now or mutex_violation_now:
            return BranchOutput(Verdict.FALSE, 0.0, True)

        dt_sep = dt_to_sep_violation_formal(self.safe, d)
        dt_mutex = dt_to_mutex_violation_formal(self.safe, self.slots, hx, hy, rx, ry)
        dt_vio = min(dt_sep, dt_mutex)

        v = Verdict.C_TRUE if math.isfinite(dt_vio) else Verdict.TRUE
        return BranchOutput(v, dt_vio, True)


class LearnedPRV:
    """
    Learned PRV branch - intent-based short-horizon advisory evidence.

    Uses intent_slot + (mimicked/trace) confidence and a simple risk heuristic:
    if intent matches robot goal, predict earlier risk.
    Coverage filtering: if confidence missing -> out of coverage.
    """

    def __init__(self, safe: SafetyParams, slots: Dict[str, Slot]):
        self.safe = safe
        self.slots = slots

    def step(self, hx: float, hy: float, rx: float, ry: float, intent_slot: str, p_k: Optional[float], goal_slot: str) -> BranchOutput:
        """Execute one step of learned PRV monitoring."""
        if p_k is None:
            return BranchOutput(Verdict.UNKNOWN, math.inf, False)

        Sp = protective_separation(self.safe, self.safe.v_r_max, self.safe.v_h_max)
        d = dist2(hx, hy, rx, ry)
        sep_violation_now = (d <= Sp)
        mutex_violation_now = any(in_slot(hx, hy, s) and in_slot(rx, ry, s) for s in self.slots.values())
        if sep_violation_now or mutex_violation_now:
            return BranchOutput(Verdict.FALSE, 0.0, True)

        # Heuristic learned prediction:
        # - If human intent equals robot goal, treat it as higher short-horizon risk => smaller dt.
        # - Otherwise more relaxed.
        dt_formal_sep = dt_to_sep_violation_formal(self.safe, d)
        if intent_slot in self.slots and intent_slot == goal_slot:
            dt_vio = 0.6 * dt_formal_sep
        else:
            dt_vio = 1.5 * dt_formal_sep

        # Feasibility filter: learned cannot be "earlier than physics" lower bound.
        if math.isfinite(dt_vio) and dt_vio + 1e-9 < dt_formal_sep:
            return BranchOutput(Verdict.UNKNOWN, math.inf, False)

        v = Verdict.C_TRUE if math.isfinite(dt_vio) else Verdict.TRUE
        return BranchOutput(v, dt_vio, True)


def fuse(formal: BranchOutput, learned: BranchOutput, p_k: Optional[float], fusion: FusionParams) -> Mode:
    """Fuse formal and learned branch outputs into mitigation mode."""
    if formal.verdict == Verdict.FALSE or formal.dt_violation <= fusion.tau_hard:
        return Mode.CRITICAL
    if learned.covered and (p_k is not None) and (p_k >= fusion.p_min) and (learned.dt_violation <= fusion.tau_soft):
        return Mode.ADVISORY
    return Mode.NOMINAL
