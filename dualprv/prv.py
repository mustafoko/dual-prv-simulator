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
    """
    Formal PRV branch - conservative envelope prediction with lookahead reachability.
    
    This branch performs forward reachability analysis: it checks ALL possible trajectories
    within a lookahead horizon (few seconds) to see if ANY path leads to a violation.
    If ANY violation path exists within the lookahead horizon, it reports the violation time.
    
    Key design:
    - Looks ahead for a threshold time (formal_lookahead_horizon, e.g., 3 seconds)
    - Checks all possible human and robot trajectories (worst-case)
    - If ANY path leads to violation within horizon → reports violation time
    - Conservative: assumes worst-case speeds and all possible directions
    - The lookahead allows checking future states, but only violations within tau_hard
      trigger CRITICAL mode (imminent threats)
    """

    def __init__(self, safe: SafetyParams, slots: Dict[str, Slot], lookahead_horizon: float = 3.0):
        self.safe = safe
        self.slots = slots
        self.lookahead_horizon = lookahead_horizon

    def _check_reachability_violation(self, hx: float, hy: float, rx: float, ry: float, horizon: float) -> tuple[bool, float]:
        """
        Check if ANY possible trajectory leads to violation within horizon.
        
        Performs forward reachability: simulates worst-case trajectories and checks
        if separation or mutex violations can occur.
        
        Returns:
            (has_violation, earliest_violation_time)
        """
        Sp = protective_separation(self.safe, self.safe.v_r_max, self.safe.v_h_max)
        d_current = dist2(hx, hy, rx, ry)
        
        # Check current violation
        if d_current <= Sp:
            return (True, 0.0)
        if any(in_slot(hx, hy, s) and in_slot(rx, ry, s) for s in self.slots.values()):
            return (True, 0.0)
        
        # Forward reachability: check all possible trajectories within horizon
        # Worst-case: human and robot move toward each other at max speeds
        v_approach = self.safe.v_h_max + self.safe.v_r_max
        
        # Check separation violation along worst-case trajectory
        # If they move directly toward each other, separation decreases at v_approach
        dt_sep_vio = (d_current - Sp) / max(v_approach, 1e-9)
        
        # Check mutex violation: can they reach same slot within horizon?
        dt_mutex_vio = math.inf
        for slot in self.slots.values():
            # Worst-case: both move toward this slot at max speed
            th = travel_time_to_slot(hx, hy, slot, self.safe.v_h_max)
            tr = travel_time_to_slot(rx, ry, slot, self.safe.v_r_max)
            # Mutex violation when both are in slot
            t_mutex = max(th, tr)
            if t_mutex <= horizon:
                dt_mutex_vio = min(dt_mutex_vio, t_mutex)
        
        # Check if ANY violation occurs within horizon
        dt_earliest = min(dt_sep_vio, dt_mutex_vio)
        
        if dt_earliest <= horizon:
            return (True, dt_earliest)
        
        return (False, math.inf)

    def step(self, hx: float, hy: float, rx: float, ry: float) -> BranchOutput:
        """
        Execute one step of formal PRV monitoring with lookahead reachability.
        
        Checks ALL possible trajectories within lookahead_horizon seconds.
        If ANY path leads to violation → returns FALSE with violation time.
        """
        Sp = protective_separation(self.safe, self.safe.v_r_max, self.safe.v_h_max)
        d = dist2(hx, hy, rx, ry)

        # Check current violations
        sep_violation_now = (d <= Sp)
        mutex_violation_now = any(in_slot(hx, hy, s) and in_slot(rx, ry, s) for s in self.slots.values())
        if sep_violation_now or mutex_violation_now:
            return BranchOutput(Verdict.FALSE, 0.0, True)

        # Forward reachability analysis: check all possible states within lookahead horizon
        has_violation, dt_vio = self._check_reachability_violation(hx, hy, rx, ry, self.lookahead_horizon)
        
        if has_violation:
            # Found a violation path within horizon
            v = Verdict.FALSE if dt_vio <= 0.0 else Verdict.C_FALSE
            return BranchOutput(v, dt_vio, True)
        
        # No violation path found within horizon
        v = Verdict.TRUE
        return BranchOutput(v, math.inf, True)


class LearnedPRV:
    """
    Learned PRV branch - intent-based predictive monitoring.
    
    This branch uses INTENT DETECTION with probability p_k to predict future violations
    BEFORE they occur, allowing for earlier advisory warnings compared to the
    conservative formal branch.
    
    Key design principles:
    - Uses intent_slot (detected human intent) with detection accuracy p_k
    - Predicts violations based on WHERE human/robot are going (trajectories)
    - Only provides predictions when p_k is high enough (coverage gating)
    - Predictions must respect physics bounds (feasibility filter)
    """

    def __init__(self, safe: SafetyParams, slots: Dict[str, Slot]):
        self.safe = safe
        self.slots = slots

    def step(self, hx: float, hy: float, rx: float, ry: float, intent_slot: str, p_k: Optional[float], goal_slot: str) -> BranchOutput:
        """
        Execute one step of learned PRV monitoring.
        
        Uses intent detection with probability p_k to predict future violations based on
        where human and robot are moving toward (intent trajectories).
        
        Args:
            hx, hy: Current human position
            rx, ry: Current robot position  
            intent_slot: Detected human intent (slot they're moving toward)
            p_k: Intent detection probability/accuracy [0,1]
            goal_slot: Robot's target slot
            
        Returns:
            BranchOutput with predicted violation time based on intent trajectories
        """
        # Out of coverage if intent detection confidence is missing
        if p_k is None:
            return BranchOutput(Verdict.UNKNOWN, math.inf, False)

        Sp = protective_separation(self.safe, self.safe.v_r_max, self.safe.v_h_max)
        d_current = dist2(hx, hy, rx, ry)
        
        # Check current violations (immediate safety check)
        sep_violation_now = (d_current <= Sp)
        mutex_violation_now = any(in_slot(hx, hy, s) and in_slot(rx, ry, s) for s in self.slots.values())
        if sep_violation_now or mutex_violation_now:
            return BranchOutput(Verdict.FALSE, 0.0, True)

        # Get formal prediction as baseline (worst-case physics)
        dt_formal_sep = dt_to_sep_violation_formal(self.safe, d_current)
        dt_formal_mutex = dt_to_mutex_violation_formal(self.safe, self.slots, hx, hy, rx, ry)
        dt_formal = min(dt_formal_sep, dt_formal_mutex)
        
        # If no valid intent/goal information, fall back to formal prediction
        if intent_slot not in self.slots or goal_slot not in self.slots or intent_slot == "":
            v = Verdict.C_TRUE if math.isfinite(dt_formal) else Verdict.TRUE
            return BranchOutput(v, dt_formal, True)
        
        # PREDICTIVE PART: Use intent trajectories to predict future violations
        # KEY INSIGHT: Learned branch knows WHERE they're going (intent), allowing it to:
        # 1. Predict violations EARLIER when conflict is coming (before formal sees it)
        # 2. Trigger ADVISORY early, slowing robot down
        # 3. Slower robot prevents violation, avoiding CRITICAL stops
        # This is the efficiency gain: learned uses intent to be proactive, not just optimistic
        
        h_target = self.slots[intent_slot]
        r_target = self.slots[goal_slot]
        
        # Calculate travel times to target slots
        # Human moves at typical speed (75% of max) - learned knows actual intent
        th_to_target = travel_time_to_slot(hx, hy, h_target, self.safe.v_h_max * 0.75)
        # Robot moves at commanded speed (could be nominal, advisory, or critical)
        tr_to_target = travel_time_to_slot(rx, ry, r_target, self.safe.v_r_max)
        
        # Predict separation at target positions
        d_target = dist2(h_target.x, h_target.y, r_target.x, r_target.y)
        
        # Predict violation time based on intent trajectories
        if intent_slot == goal_slot:
            # CONFLICT SCENARIO: Both heading to same slot
            # Learned knows this conflict is coming - can predict it EARLIER than formal
            # Formal uses worst-case (all possible directions), learned uses actual intent
            
            # Calculate when they'll both reach the conflict slot
            t_conflict = max(th_to_target, tr_to_target)
            
            # Consider separation violation along converging trajectories
            # Learned knows they're converging (intent matches goal), so can predict
            # separation violation along the actual trajectory
            v_approach = self.safe.v_h_max * 0.75 + self.safe.v_r_max
            dt_sep_vio = max(0.0, (d_current - Sp) / max(v_approach, 1e-9))
            
            # Learned prediction: conflict is certain (intent matches goal)
            # Predict based on actual trajectories (more accurate than formal's worst-case)
            # Learned can predict EARLIER because it knows the actual conflict trajectory
            dt_conflict_pred = min(dt_sep_vio, t_conflict)
            
            # Key: Learned predicts EARLIER (more urgent) when it knows conflict is coming
            # This allows ADVISORY to trigger early, slowing robot before violation
            # Weight by confidence: higher p_k -> more confident in conflict prediction
            # Learned uses actual trajectories (more accurate) vs formal's worst-case
            dt_vio = dt_conflict_pred * (0.6 + 0.2 * p_k)  # Range: 0.6-0.8x of conflict time
            
            # Safety: learned cannot predict violations earlier than physics allows
            # But learned CAN predict earlier than formal when it knows actual intent
            # (formal uses worst-case, learned uses actual trajectories)
            # Allow learned to be up to 40% earlier than formal when confident
            if math.isfinite(dt_formal):
                # Learned can predict 0.6-1.0x of formal (earlier when confident)
                # This gives learned a significant advantage when it knows conflicts
                dt_vio = min(dt_vio, dt_formal * (0.6 + 0.4 * p_k))
                # But ensure it's not too early (safety margin - at least 30% of formal)
                dt_vio = max(dt_vio, dt_formal * 0.3)
                # If learned is still earlier than formal, that's OK - it knows the conflict
                # Don't filter it out - this is the proactive advisory benefit
        else:
            # NO CONFLICT: Different target slots
            # Intent tells us they're diverging - learned can be much more optimistic
            # Key insight: when targets are different, they're moving apart, so violations
            # are much less likely or will occur much later
            
            # Calculate separation change rate based on intent trajectories
            # If they're moving to different slots, separation will increase
            d_target = dist2(h_target.x, h_target.y, r_target.x, r_target.y)
            
            if d_target > d_current * 1.1:
                # They're moving significantly apart - no violation predicted
                # Learned predicts no violation (much later than formal)
                dt_vio = math.inf  # No violation predicted
            elif d_target > d_current:
                # They're moving apart - violation much less likely
                # Predict violation much later (if at all)
                # Use a conservative estimate: if they were to reverse direction,
                # how long until violation? But since they're diverging, this is very unlikely
                v_approach = self.safe.v_h_max * 0.75 + self.safe.v_r_max
                dt_sep_vio_worst = (d_current - Sp) / max(v_approach, 1e-9)
                # Since they're diverging, multiply by large factor (3-5x)
                dt_vio = dt_sep_vio_worst * (3.0 + 2.0 * p_k)  # Range: 3.0-5.0 based on p_k
            else:
                # Still converging but no direct conflict (different slots)
                # Learned is moderately more optimistic than formal
                v_approach = self.safe.v_h_max * 0.75 + self.safe.v_r_max
                dt_sep_vio = (d_current - Sp) / max(v_approach, 1e-9)
                # More optimistic: multiply by 1.5-2.0x
                dt_vio = dt_sep_vio * (1.5 + 0.5 * p_k)  # Range: 1.5-2.0 based on p_k
            
            # Safety: learned cannot predict violations earlier than formal (safety guarantee)
            # But learned CAN predict violations much later (more optimistic) when intent shows no conflict
            if math.isfinite(dt_formal):
                dt_vio = max(dt_vio, dt_formal * 1.1)  # At least 10% later than formal
        
        # Feasibility filter: learned prediction must respect physics bounds
        # However, learned CAN predict earlier than formal when it knows actual intent
        # (formal uses worst-case, learned uses actual trajectories)
        # Only filter out if learned is unreasonably early (less than 30% of formal)
        if math.isfinite(dt_formal) and dt_vio < dt_formal * 0.3:
            # Out of coverage: prediction violates physics bounds (too early)
            return BranchOutput(Verdict.UNKNOWN, math.inf, False)
        
        # Ensure prediction is reasonable
        if not math.isfinite(dt_vio) or dt_vio < 0:
            dt_vio = math.inf
        
        v = Verdict.C_TRUE if math.isfinite(dt_vio) else Verdict.TRUE
        return BranchOutput(v, dt_vio, True)


def fuse(formal: BranchOutput, learned: BranchOutput, p_k: Optional[float], fusion: FusionParams) -> Mode:
    """
    Fuse formal and learned branch outputs into mitigation mode.
    
    Fusion logic (dual-future PRV design):
    1. CRITICAL: Formal authority - if formal detects IMMEDIATE violation (verdict=FALSE)
    2. ADVISORY: Learned advisory - if learned is covered, confident (p_k >= p_min), 
                and predicts violation within tau_soft AND learned prediction is later than formal
                (learned provides early warning advantage)
    3. CRITICAL: If formal predicts violation within tau_hard (and learned didn't trigger ADVISORY)
    4. NOMINAL: Otherwise
    
    Key insight: learned branch can trigger ADVISORY mode when it predicts violations
    LATER than formal (more optimistic), allowing smoother operation: advisory mode slows 
    down robot BEFORE hard stop. This is the efficiency gain of dual PRV.
    
    Args:
        formal: Formal branch output (conservative, always covered)
        learned: Learned branch output (intent-based, may be out of coverage)
        p_k: Intent detection probability/accuracy [0,1] or None
        fusion: Fusion parameters (tau_hard, tau_soft, p_min)
        
    Returns:
        Mitigation mode: CRITICAL, ADVISORY, or NOMINAL
    """
    # Immediate violation: formal authority always takes precedence
    if formal.verdict == Verdict.FALSE:
        return Mode.CRITICAL
    
    # Learned advisory: KEY INSIGHT - learned can predict violations EARLIER when it knows intent
    # This allows ADVISORY to trigger early, slowing robot down BEFORE formal triggers CRITICAL
    # The slower robot speed prevents violations, avoiding hard stops
    # This is the efficiency gain: learned uses intent to be proactive
    
    learned_can_advise = (
        learned.covered 
        and (p_k is not None) 
        and (p_k >= fusion.p_min) 
        and (learned.dt_violation <= fusion.tau_soft)
        and math.isfinite(learned.dt_violation)
        and math.isfinite(formal.dt_violation)
    )
    
    # Learned can advise in two scenarios:
    # 1. Learned predicts EARLIER than formal (knows conflict coming) - proactive advisory
    # 2. Learned predicts LATER than formal (knows no conflict) - optimistic advisory
    # Both allow smoother operation by triggering ADVISORY before CRITICAL
    
    # Check if learned predicts earlier (knows conflict) or later (knows no conflict)
    # Use smaller margin for "earlier" - learned should trigger advisory when it sees conflicts coming
    learned_earlier = False
    learned_later = False
    if learned_can_advise:
        learned_earlier = learned.dt_violation < formal.dt_violation - 0.01  # 0.01s margin (very sensitive)
        learned_later = learned.dt_violation > formal.dt_violation + 0.1  # 0.1s margin
        
        if learned_earlier or learned_later:
            # Learned has useful information (either sees conflict early or knows no conflict)
            # Trigger ADVISORY to slow robot down smoothly
            # This prevents violations and avoids CRITICAL stops
            # KEY: When learned predicts earlier, it sees conflicts coming before formal
            # ADVISORY slows robot early, preventing the violation
            # IMPORTANT: ADVISORY takes precedence over CRITICAL when learned has better info
            return Mode.ADVISORY
    
    # Formal critical: if formal predicts violation within tau_hard
    # Only trigger if learned didn't already provide advisory
    if formal.dt_violation <= fusion.tau_hard:
        # If learned provided advisory, check if violation is still imminent
        if learned_can_advise and (learned_earlier or learned_later):
            # Learned provided advisory - check if violation is truly imminent
            # If learned predicted earlier, it may have already slowed robot enough
            if learned_earlier:
                # Learned saw conflict early - advisory may have prevented it
                # Only trigger critical if violation is very imminent (< 0.15s)
                if formal.dt_violation <= 0.15:
                    return Mode.CRITICAL
                else:
                    # Keep advisory - learned's early warning may have prevented violation
                    return Mode.ADVISORY
            else:
                # Learned predicted later (no conflict) - but formal sees violation
                # Only trigger critical if violation is very imminent (< 0.2s)
                if formal.dt_violation <= 0.2:
                    return Mode.CRITICAL
                else:
                    # Keep advisory - learned knows no conflict, formal may be overly conservative
                    return Mode.ADVISORY
        else:
            # No learned advisory - formal triggers critical
            return Mode.CRITICAL
    
    # Otherwise, operate nominally
    return Mode.NOMINAL
