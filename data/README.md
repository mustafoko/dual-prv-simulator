# Data Files for Dual-Future PRV Simulator

This folder contains goal sequences and simulation traces for the HRC pick-and-place simulation.

## Important: Same Goals Constraint

**Both human and robot always have the SAME goal sequence.** This means:
- They always target the same slot
- Every goal creates a potential conflict
- The PRV system must manage every transition

## Goal Files (*.txt)

| File | Goals | Description |
|------|-------|-------------|
| `goals.txt` | 12 | Default scenario |
| `goals_short.txt` | 4 | Quick testing |
| `goals_medium.txt` | 8 | Standard tests |
| `goals_long.txt` | 12 | Extended tests |

### Format
```
# Comment lines start with #
human: A, B, C, D, ...
robot: A, B, C, D, ...   <- SAME as human
```

### Usage
```bash
python dualprv/dual_future_prv_sim.py --single-run --goals-file data/goals_short.txt
```

## What Varies Between Runs?

Since both agents have the same goals, conflict is guaranteed at every step. What varies:

1. **Timing randomness** - Travel times are sampled from distributions:
   - Human: [4-7]s travel, [1-2]s exit
   - Robot: [5-8]s normal, [6-8]s slow

2. **Who arrives first** - Depends on random timing samples

3. **PRV predictions** - Learned branch has 85% accuracy (some mispredictions)

4. **Goal sequence order** - Different orderings affect total completion time

## Trace Files (*.csv)

Pre-computed simulation traces showing step-by-step state evolution.

| File | Description |
|------|-------------|
| `trace_example.csv` | Minimal template |
| `trace_same_goals.csv` | Standard run with same goals |
| `trace_output.csv` | Generated trace |

### Trace Columns

**Agent State:**
- `time` - Simulation time in seconds
- `human_status` / `robot_status` - IDLE, MOVING, PLACING, EXITING, STOPPED
- `human_target` / `robot_target` - Current target slot (A, B, C, D)
- `human_remaining` / `robot_remaining` - Time to complete current action
- `human_at_X` / `robot_at_X` - True if agent is at slot X

**PRV Monitor:**
- `prv_mode` - NOMINAL, ADVISORY, or CRITICAL
- `delta_t_formal` - Formal branch time-to-violation estimate
- `delta_t_learned` - Learned branch time-to-conflict estimate
- `confidence` - Learned predictor confidence (0-1)

**Progress:**
- `human_goal_idx` / `robot_goal_idx` - Current goal index (0-based)
- `event` - Description of significant events

### Goal Achievement

A goal is achieved when the agent **places a box** at the target location:

```
MOVING → arrives → PLACING → EXITING → IDLE (goal complete)
                      ↓
              at_slot[X] = True (box placed)
```

### Example Usage

```bash
# Run simulation and save trace
python dualprv/dual_future_prv_sim.py --single-run \
    --goals-file data/goals_medium.txt \
    --trace-file data/my_trace.csv
```

```python
# Load and analyze trace
import csv

with open('data/trace_same_goals.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['prv_mode'] == 'CRITICAL':
            print(f"t={row['time']}: {row['event']}")
```
