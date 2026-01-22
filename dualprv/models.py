"""Data models for the Dual-Future PRV simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Verdict(str, Enum):
    """PRV verdict types."""
    TRUE = "true"
    FALSE = "false"
    C_TRUE = "c_true"
    C_FALSE = "c_false"
    UNKNOWN = "?"


class Mode(str, Enum):
    """Mitigation modes."""
    NOMINAL = "Nominal"
    ADVISORY = "Advisory"
    CRITICAL = "Critical"


@dataclass
class BranchOutput:
    """Output from a PRV branch (formal or learned)."""
    verdict: Verdict
    dt_violation: float   # earliest time-to-violation (s), inf if none
    covered: bool         # learned coverage flag; formal is always True


@dataclass
class SafetyParams:
    """Safety parameters."""
    v_h_max: float = 1.2      # m/s (hand max)
    v_r_max: float = 1.0      # m/s (robot max)
    t_react: float = 0.10     # s (detection+compute latency)
    t_stop: float = 0.40      # s (robot stopping time bound)
    margin: float = 0.05      # m (uncertainty + intrusion + margin)


@dataclass
class ControllerParams:
    """Controller parameters."""
    v_nominal: float = 0.8    # m/s
    v_advisory: float = 0.3   # m/s
    max_decel: float = 2.0    # m/s^2


@dataclass
class FusionParams:
    """Fusion parameters."""
    tau_hard: float = 0.5     # s
    tau_soft: float = 1.0     # s
    p_min: float = 0.8        # confidence threshold


@dataclass
class SimParams:
    """Simulation parameters."""
    critical_hold_s: float = 0.0
    vel_window: int = 3
    robot_stopped_eps: float = 0.02  # m/s threshold for R_stopped proposition
    formal_lookahead_horizon: float = 3.0  # seconds - formal branch checks all possible states within this horizon


@dataclass
class TraceRow:
    """Single row from trace CSV."""
    t: float
    intent_slot: str
    intent_conf: Optional[float]
    robot_goal_slot: str
    hx: Optional[float] = None
    hy: Optional[float] = None


@dataclass
class RobotState:
    """Robot state."""
    x: float
    y: float
    v: float
    goal_slot: str


@dataclass
class RobotState:
    """Robot state."""
    x: float
    y: float
    v: float
    goal_slot: str


@dataclass
class HumanState:
    """Human state."""
    x: float
    y: float


@dataclass
class RunMetrics:
    """Metrics from a simulation run."""
    mode: str
    tau_hard: float
    tau_soft: float
    p_min: float

    hard_stops: int
    time_nominal: float
    time_advisory: float
    time_critical: float

    min_sep_margin: float
    sep_violations: int
    mutex_violations: int
    total_violations: int
    missed_violations: int
    steps: int
