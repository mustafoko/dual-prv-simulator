# Data Files for Dual-Future PRV Simulator

This folder contains goal sequences and pre-computed trace files for the HRC pick-and-place simulation.

## Goal Files (*.txt)

Define the target slot sequences for human and robot. Each goal represents a box placement task.

| File | Description | Conflict Level |
|------|-------------|----------------|
| `goals.txt` | Default 12-goal scenario | Mixed |
| `goals_high_conflict.txt` | Same sequence for both agents | High |
| `goals_low_conflict.txt` | Complementary sequences | Low |
| `goals_intermittent.txt` | Partially overlapping sequences | Medium |

### Format
```
# Comment lines start with #
human: A, B, C, D, ...
robot: B, C, A, D, ...
```

### Usage
```bash
python dualprv/dual_future_prv_sim.py --single-run --goals-file data/goals_high_conflict.txt
```

## Trace Files (*.csv)

Pre-computed simulation traces showing step-by-step state evolution. Each row represents one simulation time step (0.5s).

| File | Description | Key Events |
|------|-------------|------------|
| `trace_example.csv` | Minimal template | - |
| `trace_output.csv` | Generated from simulation | Varies |
| `trace_high_conflict.csv` | Maximum contention scenario | Many stops, slowdowns |
| `trace_low_conflict.csv` | Minimal contention | Smooth operation |
| `trace_intermittent.csv` | Mixed contention | Some conflicts |

### Columns

**Agent State:**
- `time` - Simulation time in seconds
- `human_status` / `robot_status` - IDLE, MOVING, PLACING, EXITING, STOPPED
- `human_target` / `robot_target` - Current target slot (A, B, C, D)
- `human_remaining` / `robot_remaining` - Time to complete current action
- `human_at_X` / `robot_at_X` - True if agent is at slot X (placing/exiting)

**Robot Mode:**
- `robot_mode` - NORMAL or SLOW

**PRV Monitor:**
- `prv_mode` - NOMINAL, ADVISORY, or CRITICAL
- `delta_t_formal` - Formal branch time-to-violation estimate
- `delta_t_learned` - Learned branch time-to-conflict estimate
- `confidence` - Learned predictor confidence (0-1)
- `formal_critical` - True if formal branch triggered critical
- `learned_advisory` - True if learned branch recommends advisory

**Progress:**
- `human_goal_idx` / `robot_goal_idx` - Current goal index (0-based)
- `event` - Description of significant events

### Goal Achievement

A goal is achieved when the agent **places a box** at the target location:
1. Agent moves to target slot (MOVING state)
2. Agent arrives and places box (transitions through PLACING to EXITING)
3. Agent exits the slot (EXITING state with `at_slot=True`)
4. Goal complete, agent becomes IDLE and starts next goal

### Example Usage

Load trace for analysis:
```python
import csv

with open('data/trace_high_conflict.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['prv_mode'] == 'CRITICAL':
            print(f"t={row['time']}: Critical - {row['event']}")
```

Generate new trace:
```bash
python dualprv/dual_future_prv_sim.py --single-run \
    --goals-file data/goals_intermittent.txt \
    --trace-file data/my_trace.csv
```
