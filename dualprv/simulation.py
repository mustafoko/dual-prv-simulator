"""Simulation engine and robot/human dynamics."""

from __future__ import annotations

import csv
import math
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    BranchOutput,
    ControllerParams,
    FusionParams,
    HumanState,
    Mode,
    RobotState,
    RunMetrics,
    SafetyParams,
    SimParams,
    TraceRow,
)
from .prv import FormalPRV, LearnedPRV, fuse, protective_separation
from .workspace import Slot, dist2, in_slot


def step_robot(robot: RobotState, slots: Dict[str, Slot], dt: float, v_cmd: float, ctrl: ControllerParams) -> RobotState:
    """Step robot dynamics forward by dt."""
    if robot.goal_slot not in slots:
        raise ValueError(f"Unknown robot_goal_slot: {robot.goal_slot}")
    gx, gy = slots[robot.goal_slot].x, slots[robot.goal_slot].y

    # Speed update with decel bound
    if v_cmd < robot.v:
        robot.v = max(v_cmd, robot.v - ctrl.max_decel * dt)
    else:
        robot.v = v_cmd  # optimistic accel

    # Move toward goal center
    dx, dy = gx - robot.x, gy - robot.y
    d = math.hypot(dx, dy)
    if d <= 1e-9 or robot.v <= 1e-9:
        return robot
    step = min(d, robot.v * dt)
    robot.x += (dx / d) * step
    robot.y += (dy / d) * step
    return robot


def mimic_intent_conf(intent_conf: Optional[float], intent_slot: str, goal_slot: str) -> Optional[float]:
    """
    If intent_conf missing, mimic a plausible confidence:
    - higher when human intent conflicts with robot goal (more informative),
    - lower otherwise.
    """
    if intent_conf is not None:
        return max(0.0, min(1.0, intent_conf))
    if intent_slot == "" or intent_slot not in {"A", "B", "C", "D"}:
        return None
    # heuristic: conflicts -> high confidence
    return 0.90 if intent_slot == goal_slot else 0.75


def step_human_synth(h: HumanState, slots: Dict[str, Slot], dt: float, intent_slot: str, safe: SafetyParams) -> HumanState:
    """
    Synthesise human motion that tends to move toward the intent slot center
    at a typical speed fraction of v_h_max.
    """
    v_typ = 0.75 * safe.v_h_max

    if intent_slot not in slots:
        # No intent: drift slowly toward center (0,0)
        tx, ty = 0.0, 0.0
        v_typ = 0.25 * safe.v_h_max
    else:
        tx, ty = slots[intent_slot].x, slots[intent_slot].y

    dx, dy = tx - h.x, ty - h.y
    d = math.hypot(dx, dy)
    if d <= 1e-9:
        return h
    step = min(d, v_typ * dt)
    h.x += (dx / d) * step
    h.y += (dy / d) * step
    return h


def build_letter(
    slots: Dict[str, Slot],
    hx: float, hy: float,
    rx: float, ry: float,
    intent_slot: str,
    goal_slot: str,
    sep_safe: bool,
    r_stopped: bool
) -> Set[str]:
    """
    Construct the letter a_k ⊆ AP (atomic propositions).
    Returns a set of proposition strings.
    """
    a: Set[str] = set()

    for s in slots.keys():
        if in_slot(hx, hy, slots[s]):
            a.add(f"H_at_{s}")
        if in_slot(rx, ry, slots[s]):
            a.add(f"R_at_{s}")
        if intent_slot == s:
            a.add(f"H_intent_{s}")
        if goal_slot == s:
            a.add(f"R_goal_{s}")

    if sep_safe:
        a.add("sep_safe")
    if r_stopped:
        a.add("R_stopped")

    return a


def serialize_letter(a: Set[str]) -> str:
    """Serialize letter set to semicolon-separated string (stable order)."""
    return ";".join(sorted(a))


def simulate(
    trace: List[TraceRow],
    slots: Dict[str, Slot],
    safe: SafetyParams,
    ctrl: ControllerParams,
    fusion_params: FusionParams,
    sim_params: SimParams,
    mode: str,  # "none" | "conservative" | "dual"
    robot_start: Tuple[float, float] = (0.0, 0.0),
    human_start: Tuple[float, float] = (-1.2, 0.0),  # Farther apart to allow predictive warnings
    export_debug_csv: Optional[str] = None,
) -> RunMetrics:
    """Run simulation and return metrics."""
    if mode not in {"none", "conservative", "dual"}:
        raise ValueError("mode must be one of: none, conservative, dual")

    # Formal branch uses lookahead reachability analysis
    formal_lookahead = sim_params.formal_lookahead_horizon
    formal = FormalPRV(safe, slots, lookahead_horizon=formal_lookahead) if mode in {"conservative", "dual"} else None
    learned = LearnedPRV(safe, slots) if mode == "dual" else None

    robot = RobotState(x=robot_start[0], y=robot_start[1], v=0.0, goal_slot=trace[0].robot_goal_slot)
    human = HumanState(x=human_start[0], y=human_start[1])

    hard_stops = 0
    time_nominal = 0.0
    time_advisory = 0.0
    time_critical = 0.0

    sep_viol = 0
    mutex_viol = 0
    missed_viol = 0
    min_sep_margin = math.inf

    last_mode = Mode.NOMINAL
    critical_hold_left = 0.0

    debug_writer = None
    debug_file = None
    if export_debug_csv:
        debug_file = open(export_debug_csv, "w", newline="")
        debug_writer = csv.DictWriter(
            debug_file,
            fieldnames=[
                "t", "hx", "hy", "rx", "ry", "robot_v", "goal_slot", "intent_slot", "p_k",
                "Sp", "d", "sep_margin", "sep_safe", "mutex_violation",
                "mode", "a_k",
                "formal_verdict", "formal_dt_vio",
                "learned_covered", "learned_dt_vio"
            ],
        )
        debug_writer.writeheader()

    for k in range(1, len(trace)):
        prev = trace[k - 1]
        cur = trace[k]
        dt = cur.t - prev.t
        if dt <= 0:
            continue

        # Exogenous goal & intent from trace
        robot.goal_slot = cur.robot_goal_slot
        intent_slot = cur.intent_slot

        # Mimic confidence if missing
        p_k = mimic_intent_conf(cur.intent_conf, intent_slot, robot.goal_slot)

        # Human position: if provided, use it; else synthesise
        if cur.hx is not None and cur.hy is not None:
            human.x, human.y = cur.hx, cur.hy
        else:
            human = step_human_synth(human, slots, dt, intent_slot, safe)

        # Current separation & propositions
        Sp = protective_separation(safe, safe.v_r_max, safe.v_h_max)
        d = dist2(human.x, human.y, robot.x, robot.y)
        sep_margin = d - Sp
        min_sep_margin = min(min_sep_margin, sep_margin)

        sep_safe = (d >= Sp)
        r_stopped = (robot.v <= sim_params.robot_stopped_eps)

        mutex_violation_now = any(in_slot(human.x, human.y, s) and in_slot(robot.x, robot.y, s) for s in slots.values())

        # Decide mode
        if mode == "none":
            m = Mode.NOMINAL
            f_out = None
            l_out = None
        else:
            assert formal is not None
            f_out = formal.step(human.x, human.y, robot.x, robot.y)

            if mode == "conservative":
                from .models import Verdict
                # Conservative mode: formal branch checks all possible states within lookahead horizon
                # It triggers CRITICAL if:
                # 1. Immediate violation (verdict=FALSE), OR
                # 2. Violation predicted within tau_hard (imminent threat)
                # The lookahead horizon allows formal to check future states, but only
                # violations within tau_hard trigger CRITICAL (hard stop)
                if f_out.verdict == Verdict.FALSE:
                    m = Mode.CRITICAL  # Immediate violation
                elif f_out.verdict == Verdict.C_FALSE and f_out.dt_violation <= fusion_params.tau_hard:
                    m = Mode.CRITICAL  # Violation within tau_hard (imminent)
                else:
                    m = Mode.NOMINAL  # Violation exists but not imminent
                l_out = None
            else:
                assert learned is not None
                l_out = learned.step(human.x, human.y, robot.x, robot.y, intent_slot, p_k, robot.goal_slot)
                m = fuse(f_out, l_out, p_k, fusion_params)

        # Optional critical hold
        if sim_params.critical_hold_s > 0.0:
            if m == Mode.CRITICAL:
                critical_hold_left = sim_params.critical_hold_s
            else:
                if critical_hold_left > 0.0:
                    m = Mode.CRITICAL
                    critical_hold_left = max(0.0, critical_hold_left - dt)

        # Hard-stop events
        if m == Mode.CRITICAL and last_mode != Mode.CRITICAL:
            hard_stops += 1
        last_mode = m

        # Commanded speed from mitigation mode
        if m == Mode.NOMINAL:
            v_cmd = ctrl.v_nominal
            time_nominal += dt
        elif m == Mode.ADVISORY:
            v_cmd = ctrl.v_advisory
            time_advisory += dt
        else:
            v_cmd = 0.0
            time_critical += dt

        # Step robot under mitigation (closed-loop effect!)
        robot = step_robot(robot, slots, dt, v_cmd, ctrl)

        # Recompute violations after robot moves this tick
        d2 = dist2(human.x, human.y, robot.x, robot.y)
        sep_bad = (d2 <= Sp)
        mutex_bad = any(in_slot(human.x, human.y, s) and in_slot(robot.x, robot.y, s) for s in slots.values())

        if sep_bad:
            sep_viol += 1
        if mutex_bad:
            mutex_viol += 1
        if (sep_bad or mutex_bad) and m != Mode.CRITICAL:
            missed_viol += 1

        # Build a_k and optionally export
        a_k = build_letter(
            slots, human.x, human.y, robot.x, robot.y,
            intent_slot=intent_slot,
            goal_slot=robot.goal_slot,
            sep_safe=(not sep_bad),
            r_stopped=(robot.v <= sim_params.robot_stopped_eps),
        )

        if debug_writer is not None:
            debug_writer.writerow({
                "t": f"{cur.t:.3f}",
                "hx": f"{human.x:.4f}",
                "hy": f"{human.y:.4f}",
                "rx": f"{robot.x:.4f}",
                "ry": f"{robot.y:.4f}",
                "robot_v": f"{robot.v:.4f}",
                "goal_slot": robot.goal_slot,
                "intent_slot": intent_slot,
                "p_k": "" if p_k is None else f"{p_k:.3f}",
                "Sp": f"{Sp:.4f}",
                "d": f"{d2:.4f}",
                "sep_margin": f"{(d2-Sp):.4f}",
                "sep_safe": "1" if (not sep_bad) else "0",
                "mutex_violation": "1" if mutex_bad else "0",
                "mode": m.value,
                "a_k": serialize_letter(a_k),
                "formal_verdict": "" if f_out is None else f_out.verdict.value,
                "formal_dt_vio": "" if f_out is None else (f"{f_out.dt_violation:.4f}" if math.isfinite(f_out.dt_violation) else "inf"),
                "learned_covered": "" if l_out is None else ("1" if l_out.covered else "0"),
                "learned_dt_vio": "" if l_out is None else (f"{l_out.dt_violation:.4f}" if math.isfinite(l_out.dt_violation) else "inf"),
            })

    if debug_file is not None:
        debug_file.close()

    total_viol = sep_viol + mutex_viol
    return RunMetrics(
        mode=mode,
        tau_hard=fusion_params.tau_hard,
        tau_soft=fusion_params.tau_soft,
        p_min=fusion_params.p_min,
        hard_stops=hard_stops,
        time_nominal=time_nominal,
        time_advisory=time_advisory,
        time_critical=time_critical,
        min_sep_margin=min_sep_margin if math.isfinite(min_sep_margin) else float("nan"),
        sep_violations=sep_viol,
        mutex_violations=mutex_viol,
        total_violations=total_viol,
        missed_violations=missed_viol,
        steps=len(trace) - 1,
    )
