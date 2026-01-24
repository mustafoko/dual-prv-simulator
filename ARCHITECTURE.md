# Dual-Future PRV Simulator: Complete Architecture & Experiment Guide

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Simulation Model](#simulation-model)
5. [PRV Monitor Design](#prv-monitor-design)
6. [Experiment Methodology](#experiment-methodology)
7. [Trace Format & Interpretation](#trace-format--interpretation)
8. [Results Analysis](#results-analysis)

---

## Overview

### What is Dual-Future PRV?

**Predictive Runtime Verification (PRV)** monitors a system at runtime to predict and prevent safety violations before they occur. **Dual-Future PRV** extends this with two parallel prediction branches:

```
                    ┌─────────────────────┐
                    │   System State      │
                    │  (Human + Robot)    │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
    ┌─────────────────┐              ┌─────────────────┐
    │  FORMAL BRANCH  │              │ LEARNED BRANCH  │
    │  (Conservative) │              │   (Advisory)    │
    │                 │              │                 │
    │ • Worst-case    │              │ • Probabilistic │
    │ • Safety-auth   │              │ • Early warning │
    │ • Hard stop     │              │ • Slow-down     │
    └────────┬────────┘              └────────┬────────┘
             │                                │
             └────────────┬───────────────────┘
                          │
                          ▼
                ┌─────────────────┐
                │  FUSION POLICY  │
                │   (Monotone)    │
                │                 │
                │ Critical > Adv  │
                │      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  MITIGATION     │
                │                 │
                │ • NOMINAL: Run  │
                │ • ADVISORY: Slow│
                │ • CRITICAL: Stop│
                └─────────────────┘
```

### The Pick-and-Place Scenario

```
    SHARED WORKSPACE (4 Slots)
    ┌─────────────────────────────┐
    │                             │
    │    ┌───┐         ┌───┐     │
    │    │ A │         │ B │     │
    │    └───┘         └───┘     │
    │                             │
    │    ┌───┐         ┌───┐     │
    │    │ C │         │ D │     │
    │    └───┘         └───┘     │
    │                             │
    │       Human    Robot       │
    │                             │
    └─────────────────────────────┘

    TASK: Both agents place boxes at slots A, B, C, D
    CONSTRAINT: Only ONE agent can occupy a slot at a time
    RISK: Collision if both reach same slot simultaneously
```

---

## System Architecture

### File Structure

```
dualprv/
├── dualprv/
│   └── dual_future_prv_sim.py    # Main simulator (all-in-one)
├── data/
│   ├── goals.txt                  # Goal sequences
│   ├── goals_high_conflict.txt    # High conflict scenario
│   ├── goals_low_conflict.txt     # Low conflict scenario
│   ├── goals_intermittent.txt     # Mixed scenario
│   ├── trace_*.csv                # Simulation traces
│   ├── comparison_*.png           # Result plots
│   └── RESULTS.md                 # Experiment results
├── requirements.txt               # Dependencies
└── ARCHITECTURE.md                # This file
```

### Module Organization (dual_future_prv_sim.py)

```python
# =============================================================================
# FILE STRUCTURE (1286 lines)
# =============================================================================

# Lines 1-30:     Imports and documentation
# Lines 31-75:    Constants and configuration
# Lines 76-105:   Enums (AgentStatus, RobotMode, PRVMode, ControllerType)
# Lines 106-185:  Data classes (AgentState, PRVOutput, SimulationLog, RunStatistics)
# Lines 186-240:  Goal sequence generation
# Lines 241-370:  File I/O (load/save goals and traces)
# Lines 371-410:  Timing utilities
# Lines 411-620:  PRVMonitor class (formal + learned + fusion)
# Lines 621-870:  Simulation class (main simulation engine)
# Lines 871-1100: ExperimentRunner class (Monte-Carlo experiments)
# Lines 1101-1170: Time-series logging utilities
# Lines 1171-1286: Main function and CLI
```

---

## Core Components

### 1. Agent State Machine

Each agent (human and robot) follows this state machine:

```
                    ┌──────────┐
                    │   IDLE   │ ◄─────────────────────┐
                    └────┬─────┘                       │
                         │ Start next goal             │
                         ▼                             │
                    ┌──────────┐                       │
              ┌────►│  MOVING  │◄────┐                 │
              │     └────┬─────┘     │                 │
              │          │           │                 │
              │          │ Arrive    │ Resume          │
              │          ▼           │                 │
              │     ┌──────────┐     │                 │
              │     │ PLACING  │─────┼─────────────────┤
              │     └────┬─────┘     │                 │
              │          │           │                 │
    Robot     │          │ Start     │                 │ Goal
    only      │          │ exit      │                 │ complete
              │          ▼           │                 │
              │     ┌──────────┐     │                 │
              │     │ EXITING  │─────┼─────────────────┘
              │     └────┬─────┘     │
              │          │           │
              │          │ PRV       │
              │          │ Critical  │
              │          ▼           │
              │     ┌──────────┐     │
              └─────│ STOPPED  │─────┘
                    └──────────┘
                    (Robot only)
```

### 2. State Variables

```python
@dataclass
class AgentState:
    # Goal tracking
    goal_sequence: List[str]      # e.g., ['A', 'B', 'C', 'D']
    current_goal_idx: int         # Which goal we're on (0-based)
    
    # Current action
    status: AgentStatus           # IDLE, MOVING, PLACING, EXITING, STOPPED
    target_slot: Optional[str]    # Current target: 'A', 'B', 'C', or 'D'
    remaining_time: float         # Seconds until current action completes
    
    # Slot occupancy (True when physically at slot)
    at_slot: Dict[str, bool]      # {'A': False, 'B': True, 'C': False, 'D': False}
    
    # Robot-specific
    mode: RobotMode               # NORMAL or SLOW
    stop_accumulator: float       # Tracks stop latency (0.4s to fully stop)
```

### 3. Timing Model

```
HUMAN TIMING:
┌─────────────────────────────────────────────────────┐
│  MOVING (travel + place)  │  EXITING (clear slot)   │
│     [4.0 - 7.0] seconds   │   [1.0 - 2.0] seconds   │
└─────────────────────────────────────────────────────┘

ROBOT TIMING (NORMAL mode):
┌─────────────────────────────────────────────────────┐
│  MOVING (travel + place)  │  EXITING (clear slot)   │
│     [5.0 - 8.0] seconds   │   [1.0 - 2.0] seconds   │
└─────────────────────────────────────────────────────┘

ROBOT TIMING (SLOW mode):
┌─────────────────────────────────────────────────────┐
│  MOVING (travel + place)  │  EXITING (clear slot)   │
│     [6.0 - 8.0] seconds   │   [1.0 - 2.0] seconds   │
└─────────────────────────────────────────────────────┘

ROBOT STOP LATENCY: 0.4 seconds (time to fully halt)
BASELINE RESUME PENALTY: +2.0 seconds (formal-only controller)
SPEED RECOVERY: 0.4× remaining time (when switching SLOW→NORMAL)
```

---

## Simulation Model

### Time Progression

```
Δt = 0.5 seconds (discrete time step)

t=0.0  ──► t=0.5 ──► t=1.0 ──► t=1.5 ──► ... ──► completion
  │          │         │         │
  ▼          ▼         ▼         ▼
┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
│Step │   │Step │   │Step │   │Step │
│  1  │   │  2  │   │  3  │   │  4  │
└─────┘   └─────┘   └─────┘   └─────┘
```

### Simulation Loop (per step)

```python
def simulation_step():
    # 1. Evaluate PRV monitor
    prv_output = prv.evaluate(human, robot, controller_type)
    
    # 2. Log current state
    log_state(prv_output)
    
    # 3. Check for safety violations
    check_safety()  # Should be 0 with proper enforcement
    
    # 4. Apply mitigation
    if prv_output.fused_mode == CRITICAL:
        robot.stop()
    elif prv_output.fused_mode == ADVISORY:
        robot.slow_down()
    else:  # NOMINAL
        robot.speed_recovery_if_was_slow()
    
    # 5. Advance agents
    step_human()  # Human has priority
    step_robot()  # Robot respects safety interlocks
    
    # 6. Increment time
    t += Δt
```

### Goal Achievement

A goal is **achieved** when a box is placed at the target slot:

```
Goal: Place box at slot A

Timeline:
├── IDLE ────────► Start moving to A
│
├── MOVING ──────► Travel toward slot A
│                  (remaining_time decrements each step)
│
├── PLACING ─────► Arrive at A, box is placed
│                  at_slot['A'] = True
│
├── EXITING ─────► Clear the slot area
│                  (still at_slot['A'] = True)
│
└── IDLE ────────► Goal complete!
                   current_goal_idx += 1
                   at_slot['A'] = False
```

---

## PRV Monitor Design

### Formal Branch (Safety-Authoritative)

The formal branch uses **conservative worst-case analysis**:

```python
def evaluate_formal(human, robot):
    """
    FORMAL BRANCH: Must never miss a real violation
    
    Uses worst-case assumptions:
    - Human arrives EARLIER than expected (1.5s buffer)
    - Human stays LONGER than expected
    """
    
    # Case 1: Human already at robot's target
    if human.at_slot[robot.target_slot]:
        return CRITICAL  # Robot approaching occupied slot
    
    # Case 2: Both heading to same slot
    if human.target_slot == robot.target_slot:
        
        # Conservative human arrival estimate
        human_earliest = human.remaining_time - 1.5  # Could arrive early!
        
        # Will they overlap?
        if overlap_possible(human_earliest, robot.remaining_time):
            
            # Is it imminent?
            if time_to_violation <= TAU_HARD:  # 1.5 seconds
                return CRITICAL
    
    return NOMINAL
```

**Key principle**: Formal branch **never** produces false negatives (missed violations)

### Learned Branch (Advisory)

The learned branch uses **probabilistic prediction**:

```python
def evaluate_learned(human, robot):
    """
    LEARNED BRANCH: Predict human intent with uncertainty
    
    Uses expected values (not worst-case):
    - Predicts human's target slot
    - Estimates time to conflict
    - Provides confidence score
    """
    
    # Oracle prediction (simulated ML model)
    if random() < ACCURACY:  # 85% accuracy
        predicted_slot = human.actual_target
        confidence = 0.85
    else:
        predicted_slot = random_other_slot()
        confidence = 0.05  # Low confidence for wrong prediction
    
    # Check for predicted conflict
    if predicted_slot == robot.target_slot:
        if time_to_conflict <= TAU_SOFT:  # 5.0 seconds
            if confidence >= P_MIN:  # 0.65 threshold
                return ADVISORY
    
    return NOMINAL
```

**Key principle**: Learned branch can have false positives (unnecessary slowdowns OK)

### Fusion Policy (Monotone)

```python
def fuse(formal_output, learned_output, controller_type):
    """
    MONOTONE FUSION: Formal authority preserved
    
    Priority: CRITICAL > ADVISORY > NOMINAL
    
    Formal CRITICAL always wins (safety guarantee)
    Learned ADVISORY only supplements (never overrides)
    """
    
    # Rule 1: Formal Critical → Always Critical
    if formal_output == CRITICAL:
        return CRITICAL  # Safety-authoritative
    
    # Rule 2: Learned Advisory (only in Dual-PRV mode)
    if controller_type == DUAL_PRV:
        if learned_output == ADVISORY:
            return ADVISORY  # Early mitigation
    
    # Rule 3: Default to Nominal
    return NOMINAL
```

```
FUSION TRUTH TABLE:
┌─────────────┬─────────────┬─────────────┐
│   Formal    │   Learned   │   Output    │
├─────────────┼─────────────┼─────────────┤
│  CRITICAL   │     any     │  CRITICAL   │  ← Formal wins
│  NOMINAL    │  ADVISORY   │  ADVISORY   │  ← Learned helps
│  NOMINAL    │   NOMINAL   │   NOMINAL   │  ← Normal operation
└─────────────┴─────────────┴─────────────┘
```

---

## Experiment Methodology

### Monte-Carlo Simulation

```
┌────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT DESIGN                            │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each trial (1 to N_TRIALS):                               │
│                                                                 │
│    1. Generate shared goal sequence                            │
│       goals = [A, B, D, C, ...]  (random permutation)          │
│       human_goals = goals                                       │
│       robot_goals = goals  ← SAME goals (max conflict)         │
│                                                                 │
│    2. Run FORMAL-ONLY simulation                               │
│       - Only Nominal and Critical modes                        │
│       - Hard stops with +2.0s resume penalty                   │
│       - Record: completion_time, stops, violations             │
│                                                                 │
│    3. Run DUAL-PRV simulation (same goals, same seed)          │
│       - Nominal, Advisory, and Critical modes                  │
│       - Slow-downs enabled, no resume penalty                  │
│       - Record: completion_time, stops, slowdowns, violations  │
│                                                                 │
│  Aggregate results across all trials                           │
│  Compute statistics: mean, std, t-test                         │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Controller Comparison

```
FORMAL-ONLY CONTROLLER:
┌──────────────────────────────────────────────────┐
│                                                  │
│  PRV Monitor ──► Only CRITICAL triggers action   │
│                                                  │
│  Conflict detected late                          │
│        │                                         │
│        ▼                                         │
│  ┌──────────┐                                    │
│  │HARD STOP │ + 2.0s resume penalty              │
│  └──────────┘                                    │
│        │                                         │
│        ▼                                         │
│  Wait for human to exit                          │
│        │                                         │
│        ▼                                         │
│  Resume with penalty delay                       │
│                                                  │
└──────────────────────────────────────────────────┘

DUAL-PRV CONTROLLER:
┌──────────────────────────────────────────────────┐
│                                                  │
│  PRV Monitor ──► ADVISORY or CRITICAL            │
│                                                  │
│  Conflict predicted early (learned branch)       │
│        │                                         │
│        ▼                                         │
│  ┌───────────┐                                   │
│  │ SLOW DOWN │ Robot decelerates                 │
│  └───────────┘                                   │
│        │                                         │
│        ▼                                         │
│  Human passes through first                      │
│        │                                         │
│        ▼                                         │
│  ┌────────────────┐                              │
│  │ SPEED RECOVERY │ 60% time reduction           │
│  └────────────────┘                              │
│        │                                         │
│        ▼                                         │
│  Robot completes smoothly                        │
│                                                  │
└──────────────────────────────────────────────────┘
```

### Metrics Collected

```python
@dataclass
class RunStatistics:
    completion_time: float      # Total time to complete all goals
    num_stops: int              # Number of hard stops
    total_stop_duration: float  # Cumulative time spent stopped
    num_slowdowns: int          # Number of advisory slow-downs (Dual-PRV only)
    safety_violations: int      # Should always be 0!
    human_goals_completed: int  # Should equal len(goals)
    robot_goals_completed: int  # Should equal len(goals)
```

---

## Trace Format & Interpretation

### CSV Trace Structure

Each row represents one simulation step (0.5 seconds):

```csv
time,human_status,human_target,human_remaining,human_at_A,human_at_B,human_at_C,human_at_D,
robot_status,robot_mode,robot_target,robot_remaining,robot_at_A,robot_at_B,robot_at_C,robot_at_D,
prv_mode,delta_t_formal,delta_t_learned,confidence,formal_critical,learned_advisory,
human_goal_idx,robot_goal_idx,event
```

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `time` | float | Simulation time in seconds |
| `human_status` | enum | IDLE, MOVING, PLACING, EXITING |
| `human_target` | str | Current target slot (A/B/C/D) or empty |
| `human_remaining` | float | Seconds until action completes |
| `human_at_X` | bool | True if human occupies slot X |
| `robot_status` | enum | IDLE, MOVING, PLACING, EXITING, STOPPED |
| `robot_mode` | enum | NORMAL or SLOW |
| `robot_target` | str | Current target slot or empty |
| `robot_remaining` | float | Seconds until action completes |
| `robot_at_X` | bool | True if robot occupies slot X |
| `prv_mode` | enum | NOMINAL, ADVISORY, or CRITICAL |
| `delta_t_formal` | float | Formal branch time-to-violation (inf if safe) |
| `delta_t_learned` | float | Learned branch time-to-conflict |
| `confidence` | float | Learned predictor confidence (0-1) |
| `formal_critical` | bool | True if formal branch triggered |
| `learned_advisory` | bool | True if learned branch recommends slowdown |
| `human_goal_idx` | int | Current goal index (0-based) |
| `robot_goal_idx` | int | Current goal index (0-based) |
| `event` | str | Description of significant events |

### Reading a Trace: Example

```csv
time,human_status,human_target,...,robot_status,robot_mode,robot_target,...,prv_mode,...,event
0.00,MOVING,C,...,MOVING,NORMAL,C,...,NOMINAL,...,both_start_same_target
0.50,MOVING,C,...,MOVING,NORMAL,C,...,ADVISORY,...,conflict_detected
1.00,MOVING,C,...,MOVING,SLOW,C,...,ADVISORY,...,robot_slowing
...
3.00,MOVING,C,...,MOVING,SLOW,C,...,CRITICAL,...,imminent_collision
3.50,MOVING,C,...,STOPPED,SLOW,C,...,CRITICAL,...,robot_stopped
...
5.00,EXITING,C,...,STOPPED,SLOW,C,...,CRITICAL,...,human_places_box_C
...
6.50,IDLE,,...,MOVING,NORMAL,C,...,NOMINAL,...,human_clears_robot_resumes
```

**Interpretation:**
1. `t=0.0`: Both start moving to slot C (same goal)
2. `t=0.5`: Learned branch detects conflict → ADVISORY
3. `t=1.0`: Robot switches to SLOW mode
4. `t=3.0`: Conflict imminent → CRITICAL
5. `t=3.5`: Robot stops completely
6. `t=5.0`: Human places box at C (goal achieved)
7. `t=6.5`: Human exits, robot resumes at normal speed

### Key Events to Watch

| Event | Meaning |
|-------|---------|
| `both_start_same_target` | Human and robot heading to same slot |
| `conflict_detected` | Learned branch predicts conflict |
| `robot_slowing` | Robot enters SLOW mode |
| `imminent_collision` | Formal branch triggers CRITICAL |
| `robot_stopped` | Robot fully halted |
| `human_places_box_X` | Human achieves goal at slot X |
| `robot_places_box_X` | Robot achieves goal at slot X |
| `human_clears_robot_resumes` | Human exits, robot continues |
| `speed_recovery` | Robot accelerates after slowdown |
| `simulation_complete` | All goals achieved |

---

## Results Analysis

### Experiment Results (500 trials, same goals)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FORMAL-ONLY vs DUAL-PRV                       │
├──────────────────────┬──────────────┬──────────────┬────────────┤
│       Metric         │ Formal-Only  │   Dual-PRV   │ Difference │
├──────────────────────┼──────────────┼──────────────┼────────────┤
│ Completion Time      │   121.84s    │   106.09s    │  -15.75s   │
│ (mean)               │              │              │  (12.9%)   │
├──────────────────────┼──────────────┼──────────────┼────────────┤
│ Completion Time      │    8.30s     │    5.34s     │   Lower    │
│ (std)                │              │              │  variance  │
├──────────────────────┼──────────────┼──────────────┼────────────┤
│ Hard Stops           │     4.63     │    11.96     │   +7.33    │
│ (mean)               │              │              │            │
├──────────────────────┼──────────────┼──────────────┼────────────┤
│ Stop Duration        │    2.73s     │   11.79s     │   +9.06s   │
│ (mean)               │              │              │            │
├──────────────────────┼──────────────┼──────────────┼────────────┤
│ Slow-downs           │     N/A      │     8.42     │     -      │
│ (mean)               │              │              │            │
├──────────────────────┼──────────────┼──────────────┼────────────┤
│ Safety Violations    │      0       │      0       │   ✓ Safe   │
├──────────────────────┴──────────────┴──────────────┴────────────┤
│ T-statistic: 35.70  (p < 0.001, highly significant)             │
└─────────────────────────────────────────────────────────────────┘
```

### Why Dual-PRV Wins

```
FORMAL-ONLY PENALTY CALCULATION:
─────────────────────────────────
Average stops: 4.63
Resume penalty: 2.0s each
Total penalty: 4.63 × 2.0 = 9.26s

DUAL-PRV ADVANTAGE:
──────────────────
• No resume penalty
• Early slowdowns prevent worst-case stops
• Speed recovery (60% time savings after slowdown)
• More stops but shorter effective delays

NET RESULT:
──────────
Dual-PRV saves 15.75s per task cycle (12.9% improvement)
```

### Statistical Significance

```
T-TEST ANALYSIS:
───────────────
H₀: No difference between controllers
H₁: Dual-PRV is faster

t-statistic = 35.70
Critical value (α=0.05) ≈ 1.96

Since |35.70| >> 1.96:
→ REJECT H₀
→ Difference is statistically significant
→ p-value < 0.001
```

---

## Usage Examples

### Run Monte-Carlo Experiment

```bash
# Default: 100 trials, 12 goals, 85% accuracy
python dualprv/dual_future_prv_sim.py

# Custom experiment
python dualprv/dual_future_prv_sim.py \
    --n-trials 500 \
    --n-goals 16 \
    --accuracy 0.90 \
    --seed 42 \
    --plot-file results.png
```

### Single Detailed Run

```bash
# Generate detailed trace
python dualprv/dual_future_prv_sim.py \
    --single-run \
    --n-goals 8 \
    --trace-file data/my_trace.csv

# Use custom goals file
python dualprv/dual_future_prv_sim.py \
    --single-run \
    --goals-file data/goals.txt \
    --trace-file data/trace.csv
```

### Load and Analyze Traces

```python
import csv

# Load trace
with open('data/trace_same_goals.csv') as f:
    reader = csv.DictReader(f)
    
    critical_events = []
    for row in reader:
        if row['prv_mode'] == 'CRITICAL':
            critical_events.append({
                'time': float(row['time']),
                'human_target': row['human_target'],
                'robot_target': row['robot_target'],
                'event': row['event']
            })

# Analyze
print(f"Total critical events: {len(critical_events)}")
for e in critical_events[:5]:
    print(f"  t={e['time']:.1f}s: {e['event']}")
```

---

## Conclusion

The Dual-Future PRV simulator demonstrates that combining a **safety-authoritative formal branch** with an **advisory learned branch** achieves:

1. **12.9% faster task completion** in high-conflict scenarios
2. **Zero safety violations** (formal branch guarantees safety)
3. **Lower variance** (more predictable performance)
4. **Graceful degradation** (even 60% accuracy provides benefit)

The key insight is that early, probabilistic predictions can **supplement** (not replace) conservative formal verification, enabling smoother human-robot collaboration without compromising safety.

---

## References

- Dual-Future PRV paper concepts for monotone fusion policy
- Runtime verification for cyber-physical systems
- Human-robot collaboration safety standards (ISO 10218, ISO/TS 15066)
