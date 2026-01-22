"""CSV I/O operations for trace files and metrics."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict
from typing import List, Optional, Tuple

from .models import RunMetrics, TraceRow


def _to_float_opt(s: str) -> Optional[float]:
    """Convert string to float or None if empty."""
    s = (s or "").strip()
    if s == "":
        return None
    return float(s)


def load_trace_csv(path: str) -> List[TraceRow]:
    """Load trace CSV file and return list of TraceRow objects."""
    rows: List[TraceRow] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Empty CSV or missing header row.")

        required = {"t", "intent_slot", "intent_conf", "robot_goal_slot"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Trace CSV missing required headers: {sorted(missing)}")

        has_h = ("hx" in reader.fieldnames) and ("hy" in reader.fieldnames)

        last_t = None
        for line_no, r in enumerate(reader, start=2):
            t = float(r["t"])
            if last_t is not None and t + 1e-12 < last_t:
                raise ValueError(f"Non-monotone time at line {line_no}: t={t} < last_t={last_t}")
            last_t = t

            row = TraceRow(
                t=t,
                intent_slot=(r.get("intent_slot") or "").strip(),
                intent_conf=_to_float_opt(r.get("intent_conf") or ""),
                robot_goal_slot=(r.get("robot_goal_slot") or "").strip(),
            )
            if has_h:
                row.hx = _to_float_opt(r.get("hx") or "")
                row.hy = _to_float_opt(r.get("hy") or "")
            rows.append(row)

    if len(rows) < 2:
        raise ValueError("Trace must contain at least 2 rows.")
    return rows


def write_metrics_csv(path: str, metrics: List[RunMetrics]) -> None:
    """Write metrics to CSV file."""
    if not metrics:
        return
    fieldnames = list(asdict(metrics[0]).keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in metrics:
            w.writerow(asdict(m))


def generate_minimal_trace(path: str, dt: float = 0.2, T: float = 35.0) -> None:
    """
    Generate a minimal trace CSV file (default scenario).
    
    Generates only t, intent_slot, intent_conf, robot_goal_slot.
    No positions. Those will be synthesised inside the simulator.
    """
    n = int(T / dt) + 1

    goals = ["A", "B", "C", "D"]
    g_idx = 0

    segments = [
        (0.0,  "B", 0.90),
        (8.0,  "A", 0.92),
        (16.0, "D", 0.75),
        (24.0, "C", 0.88),
        (30.0, "B", 0.95),
    ]

    def current_segment(t: float) -> Tuple[str, Optional[float]]:
        seg = segments[0]
        for s in segments:
            if t >= s[0]:
                seg = s
        return seg[1], seg[2]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "intent_slot", "intent_conf", "robot_goal_slot"])
        w.writeheader()
        for i in range(n):
            t = i * dt

            if i % int(6.0 / dt) == 0 and i > 0:
                g_idx = (g_idx + 1) % len(goals)
            robot_goal = goals[g_idx]

            intent_slot, conf = current_segment(t)

            # Drop confidence for a short interval to test "out-of-coverage"
            conf_out: Optional[float] = conf
            if 12.0 <= t <= 13.0:
                conf_out = None

            w.writerow({
                "t": f"{t:.3f}",
                "intent_slot": intent_slot,
                "intent_conf": "" if conf_out is None else f"{conf_out:.3f}",
                "robot_goal_slot": robot_goal,
            })

    print(f"Wrote minimal trace: {path}")


def generate_low_conflict_trace(path: str, dt: float = 0.2, T: float = 40.0) -> None:
    """
    Generate trace with low conflict between human and robot goals.
    Human and robot tend to avoid same slots.
    """
    n = int(T / dt) + 1
    goals = ["A", "B", "C", "D"]
    
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "intent_slot", "intent_conf", "robot_goal_slot"])
        w.writeheader()
        for i in range(n):
            t = i * dt
            
            # Robot cycles through goals every 8 seconds
            robot_goal_idx = (i // int(8.0 / dt)) % len(goals)
            robot_goal = goals[robot_goal_idx]
            
            # Human avoids robot's current goal
            human_slots = [g for g in goals if g != robot_goal]
            human_idx = (i // int(5.0 / dt)) % len(human_slots)
            intent_slot = human_slots[human_idx]
            conf = 0.85  # Good confidence
            
            w.writerow({
                "t": f"{t:.3f}",
                "intent_slot": intent_slot,
                "intent_conf": f"{conf:.3f}",
                "robot_goal_slot": robot_goal,
            })
    
    print(f"Wrote low-conflict trace: {path}")


def generate_high_conflict_trace(path: str, dt: float = 0.2, T: float = 40.0) -> None:
    """
    Generate trace with high conflict - human and robot often target same slots.
    This scenario should benefit more from dual PRV.
    """
    n = int(T / dt) + 1
    goals = ["A", "B", "C", "D"]
    
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "intent_slot", "intent_conf", "robot_goal_slot"])
        w.writeheader()
        for i in range(n):
            t = i * dt
            
            # Robot cycles through goals
            robot_goal_idx = (i // int(7.0 / dt)) % len(goals)
            robot_goal = goals[robot_goal_idx]
            
            # Human often matches robot goal (conflict scenario)
            if i % int(3.0 / dt) < int(1.5 / dt):
                intent_slot = robot_goal  # Conflict!
                conf = 0.92  # High confidence about conflict
            else:
                intent_slot = goals[(robot_goal_idx + 1) % len(goals)]
                conf = 0.80
            
            w.writerow({
                "t": f"{t:.3f}",
                "intent_slot": intent_slot,
                "intent_conf": f"{conf:.3f}",
                "robot_goal_slot": robot_goal,
            })
    
    print(f"Wrote high-conflict trace: {path}")


def generate_intermittent_coverage_trace(path: str, dt: float = 0.2, T: float = 40.0) -> None:
    """
    Generate trace with intermittent coverage gaps (missing confidence).
    Tests how dual PRV handles out-of-coverage scenarios.
    """
    n = int(T / dt) + 1
    goals = ["A", "B", "C", "D"]
    
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "intent_slot", "intent_conf", "robot_goal_slot"])
        w.writeheader()
        for i in range(n):
            t = i * dt
            
            robot_goal_idx = (i // int(6.0 / dt)) % len(goals)
            robot_goal = goals[robot_goal_idx]
            
            intent_slot = goals[(i // int(4.0 / dt)) % len(goals)]
            
            # Periodic coverage gaps
            has_coverage = (i // int(2.0 / dt)) % 3 != 2
            conf: Optional[float] = 0.88 if has_coverage else None
            
            w.writerow({
                "t": f"{t:.3f}",
                "intent_slot": intent_slot,
                "intent_conf": "" if conf is None else f"{conf:.3f}",
                "robot_goal_slot": robot_goal,
            })
    
    print(f"Wrote intermittent-coverage trace: {path}")


def generate_rapid_changes_trace(path: str, dt: float = 0.2, T: float = 40.0) -> None:
    """
    Generate trace with rapid goal changes.
    Tests responsiveness of PRV systems.
    """
    n = int(T / dt) + 1
    goals = ["A", "B", "C", "D"]
    
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "intent_slot", "intent_conf", "robot_goal_slot"])
        w.writeheader()
        for i in range(n):
            t = i * dt
            
            # Rapid robot goal changes
            robot_goal_idx = (i // int(2.5 / dt)) % len(goals)
            robot_goal = goals[robot_goal_idx]
            
            # Human also changes frequently
            intent_idx = (i // int(2.0 / dt)) % len(goals)
            intent_slot = goals[intent_idx]
            conf = 0.85
            
            w.writerow({
                "t": f"{t:.3f}",
                "intent_slot": intent_slot,
                "intent_conf": f"{conf:.3f}",
                "robot_goal_slot": robot_goal,
            })
    
    print(f"Wrote rapid-changes trace: {path}")


def generate_early_warning_trace(path: str, dt: float = 0.2, T: float = 50.0) -> None:
    """
    Generate trace designed to show early warning benefit of dual PRV.
    Human intent provides early warning of conflicts before they occur.
    """
    n = int(T / dt) + 1
    goals = ["A", "B", "C", "D"]
    
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "intent_slot", "intent_conf", "robot_goal_slot"])
        w.writeheader()
        for i in range(n):
            t = i * dt
            
            # Robot goal changes every 10 seconds
            robot_goal_idx = (i // int(10.0 / dt)) % len(goals)
            robot_goal = goals[robot_goal_idx]
            
            # Human intent predicts robot's NEXT goal 5 seconds early
            # This gives learned branch early warning advantage
            next_goal_idx = (robot_goal_idx + 1) % len(goals)
            next_goal = goals[next_goal_idx]
            
            # Human shows intent toward next goal early (high confidence)
            if t < 5.0 or (t % 10.0) < 5.0:
                intent_slot = next_goal  # Predicting future conflict
                conf = 0.95  # Very high confidence
            else:
                intent_slot = robot_goal  # Current conflict
                conf = 0.90
            
            w.writerow({
                "t": f"{t:.3f}",
                "intent_slot": intent_slot,
                "intent_conf": f"{conf:.3f}",
                "robot_goal_slot": robot_goal,
            })
    
    print(f"Wrote early-warning trace: {path}")


def generate_varying_confidence_trace(path: str, dt: float = 0.2, T: float = 45.0) -> None:
    """
    Generate trace with varying intent detection confidence.
    Tests how dual PRV adapts to confidence levels.
    """
    n = int(T / dt) + 1
    goals = ["A", "B", "C", "D"]
    
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "intent_slot", "intent_conf", "robot_goal_slot"])
        w.writeheader()
        for i in range(n):
            t = i * dt
            
            robot_goal_idx = (i // int(7.0 / dt)) % len(goals)
            robot_goal = goals[robot_goal_idx]
            
            intent_slot = goals[(i // int(5.0 / dt)) % len(goals)]
            
            # Varying confidence: high -> medium -> low -> high cycle
            conf_phase = (i // int(3.0 / dt)) % 4
            if conf_phase == 0:
                conf = 0.95  # Very high
            elif conf_phase == 1:
                conf = 0.85  # High
            elif conf_phase == 2:
                conf = 0.75  # Medium (below p_min=0.8, should reduce advisory)
            else:
                conf = None  # Out of coverage
            
            w.writerow({
                "t": f"{t:.3f}",
                "intent_slot": intent_slot,
                "intent_conf": "" if conf is None else f"{conf:.3f}",
                "robot_goal_slot": robot_goal,
            })
    
    print(f"Wrote varying-confidence trace: {path}")


def generate_cooperative_trace(path: str, dt: float = 0.2, T: float = 50.0) -> None:
    """
    Generate trace where human and robot coordinate (low conflict, high efficiency).
    Dual PRV should show significant efficiency gains here.
    """
    n = int(T / dt) + 1
    goals = ["A", "B", "C", "D"]
    
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "intent_slot", "intent_conf", "robot_goal_slot"])
        w.writeheader()
        for i in range(n):
            t = i * dt
            
            # Robot cycles through goals
            robot_goal_idx = (i // int(8.0 / dt)) % len(goals)
            robot_goal = goals[robot_goal_idx]
            
            # Human coordinates: moves to different slot, high confidence
            # This allows learned branch to predict no conflict
            human_slots = [g for g in goals if g != robot_goal]
            human_idx = (i // int(6.0 / dt)) % len(human_slots)
            intent_slot = human_slots[human_idx]
            conf = 0.93  # High confidence about coordination
            
            w.writerow({
                "t": f"{t:.3f}",
                "intent_slot": intent_slot,
                "intent_conf": f"{conf:.3f}",
                "robot_goal_slot": robot_goal,
            })
    
    print(f"Wrote cooperative trace: {path}")


def generate_challenging_trace(path: str, dt: float = 0.2, T: float = 60.0) -> None:
    """
    Generate challenging trace with complex conflict patterns.
    Tests robustness of PRV systems.
    """
    n = int(T / dt) + 1
    goals = ["A", "B", "C", "D"]
    
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "intent_slot", "intent_conf", "robot_goal_slot"])
        w.writeheader()
        for i in range(n):
            t = i * dt
            
            # Complex robot goal pattern
            if t < 15.0:
                robot_goal = "A"
            elif t < 25.0:
                robot_goal = "B"
            elif t < 35.0:
                robot_goal = "C"
            elif t < 45.0:
                robot_goal = "D"
            else:
                robot_goal_idx = (i // int(4.0 / dt)) % len(goals)
                robot_goal = goals[robot_goal_idx]
            
            # Human intent: sometimes conflicts, sometimes avoids
            phase = int(t / 10.0) % 4
            if phase == 0:
                intent_slot = robot_goal  # Conflict
                conf = 0.90
            elif phase == 1:
                intent_slot = goals[(goals.index(robot_goal) + 1) % len(goals)]  # Avoid
                conf = 0.88
            elif phase == 2:
                intent_slot = robot_goal  # Conflict again
                conf = 0.92
            else:
                intent_slot = ""  # No intent
                conf = None
            
            w.writerow({
                "t": f"{t:.3f}",
                "intent_slot": intent_slot,
                "intent_conf": "" if conf is None else f"{conf:.3f}",
                "robot_goal_slot": robot_goal,
            })
    
    print(f"Wrote challenging trace: {path}")


def generate_long_trace(path: str, dt: float = 0.2, T: float = 120.0) -> None:
    """
    Generate long trace for extended simulation.
    """
    n = int(T / dt) + 1
    goals = ["A", "B", "C", "D"]
    
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "intent_slot", "intent_conf", "robot_goal_slot"])
        w.writeheader()
        for i in range(n):
            t = i * dt
            
            # Robot goal changes periodically
            robot_goal_idx = (i // int(15.0 / dt)) % len(goals)
            robot_goal = goals[robot_goal_idx]
            
            # Human intent varies
            intent_idx = (i // int(12.0 / dt)) % len(goals)
            intent_slot = goals[intent_idx]
            
            # Confidence varies with time
            conf = 0.80 + 0.15 * math.sin(t / 10.0)
            conf = max(0.75, min(0.95, conf))
            
            w.writerow({
                "t": f"{t:.3f}",
                "intent_slot": intent_slot,
                "intent_conf": f"{conf:.3f}",
                "robot_goal_slot": robot_goal,
            })
    
    print(f"Wrote long trace: {path}")


def generate_proactive_advisory_trace(path: str, dt: float = 0.2, T: float = 50.0) -> None:
    """
    Generate trace specifically designed to show proactive advisory benefit.
    
    Scenario: Human intent shows conflicts coming early (high confidence).
    Learned branch sees conflicts before formal does, triggers ADVISORY early.
    ADVISORY slows robot, preventing violations and avoiding CRITICAL stops.
    Conservative mode would trigger CRITICAL too often.
    """
    n = int(T / dt) + 1
    goals = ["A", "B", "C", "D"]
    
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "intent_slot", "intent_conf", "robot_goal_slot"])
        w.writeheader()
        for i in range(n):
            t = i * dt
            
            # Robot goal changes every 8 seconds
            robot_goal_idx = (i // int(8.0 / dt)) % len(goals)
            robot_goal = goals[robot_goal_idx]
            
            # Human intent shows conflicts EARLY (before robot reaches goal)
            # This allows learned branch to predict conflicts early and trigger ADVISORY
            # Pattern: Human shows intent toward robot's goal 2-3 seconds before robot reaches it
            time_in_cycle = t % 8.0
            
            if 0.0 <= time_in_cycle < 2.0:
                # Early phase: Human shows intent toward robot's goal (conflict coming)
                # Learned sees this early and can trigger ADVISORY before formal sees violation
                intent_slot = robot_goal  # Conflict!
                conf = 0.94  # Very high confidence - learned knows conflict is coming
            elif 2.0 <= time_in_cycle < 5.0:
                # Middle phase: Conflict is happening, but learned already slowed robot
                intent_slot = robot_goal  # Still conflict
                conf = 0.92
            else:
                # Late phase: Human moves away, no conflict
                intent_slot = goals[(robot_goal_idx + 1) % len(goals)]
                conf = 0.88
            
            w.writerow({
                "t": f"{t:.3f}",
                "intent_slot": intent_slot,
                "intent_conf": f"{conf:.3f}",
                "robot_goal_slot": robot_goal,
            })
    
    print(f"Wrote proactive-advisory trace: {path}")
