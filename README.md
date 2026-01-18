# Dual-Future PRV Simulator (UR10 Pick-and-Place) — README

This project is a **trace-driven simulation framework** for evaluating **Dual-Future Predictive Runtime Verification (PRV)** in a simplified UR10 human–robot collaborative pick-and-place setting.

It lets you run and compare three runtime-assurance configurations:

- **`none`**: no PRV enforcement (robot always runs nominally)
- **`conservative`**: single-branch **formal** (conservative envelope) PRV
- **`dual`**: **dual-future PRV** (formal authority + learned advisory escalation)

The simulator consumes a **minimal CSV trace** and, if needed, **mimics missing values** by synthesizing:
- human motion consistent with intent (hand position)
- confidence values for the learned branch (if missing)

It then runs the monitoring loop, produces mitigation decisions, and reports metrics such as:
- number of hard-stops
- time spent in advisory / critical
- minimum separation margin
- missed violations (should be **0** if formal authority is correct and thresholds are consistent)

---

## Contents

- `df_prv_sim.py` — main executable script
- (optional) `minimal.csv` — an example generated trace
- (optional) `debug_dual.csv` — per-tick debug output (includes synthesized positions and `a_k`)
- (optional) `sweep.csv` — parameter sweep results

---

## Requirements

- Python **3.9+** recommended (works with 3.8+ in most cases)
- No external dependencies (uses only the Python standard library)

Tested with:
- Linux / macOS / Windows (any environment with Python)

---

## Conceptual model

### What the simulator does
At each sampling tick \(t_k\):

1. **Read trace inputs**: `intent_slot`, `intent_conf` (optional), `robot_goal_slot`
2. **Synthesize missing state** (if needed): human position \((h_x,h_y)\)
3. **Simulate robot motion** \((r_x,r_y)\) and speed influenced by mitigation mode
4. Compute safety quantities:
   - separation distance \(d(t_k)\)
   - protective separation requirement \(S_p(t_k)\)
   - mutual-exclusion co-occupancy per slot
5. Run PRV monitors depending on mode:
   - Formal branch: conservative prediction of earliest time-to-violation
   - Learned branch: intent-based short-horizon advisory evidence (with coverage gating)
6. **Fuse** into mitigation mode:
   - `Critical` if formal indicates violation or \(\Delta t_f \le \tau_{\text{hard}}\)
   - else `Advisory` if learned is covered, confident, and \(\Delta t_\ell \le \tau_{\text{soft}}\)
   - else `Nominal`
7. Record metrics and (optionally) export per-tick debug output including the constructed letter \(a_k \subseteq AP\).

### Atomic propositions (`a_k`)
The simulator internally constructs a set of propositions per tick, such as:
- `H_at_A`, `R_at_A`
- `H_intent_A`, `R_goal_A`
- `sep_safe`, `R_stopped`

These are exported as a semicolon-separated string in the debug CSV.

---

## Input trace format

### Minimal required CSV headers
Your trace file must include:

```csv
t,intent_slot,intent_conf,robot_goal_slot
```

---

## Running Comparisons and Generating Plots

### Quick Start

Run simulations across multiple scenarios and generate comparison plots:

```bash
python3 run_comparisons.py
```

This will:
1. Generate 5 different trace scenarios in `data/`:
   - `minimal.csv` - Default scenario with periodic goal changes
   - `low_conflict.csv` - Human and robot avoid same slots
   - `high_conflict.csv` - Human and robot often target same slots
   - `intermittent_coverage.csv` - Periodic coverage gaps in learned branch
   - `rapid_changes.csv` - Rapid goal changes
2. Run simulations for each trace with all three modes (none, conservative, dual)
3. Generate comparison plots in `results/`:
   - `comparison.png` - Comprehensive 4-panel comparison
   - `efficiency_summary.png` - Efficiency gain visualization
   - `summary.csv` - Detailed metrics for all runs

### Installing Plotting Dependencies

For plotting features, install matplotlib:

```bash
pip install matplotlib
```

The simulation and CSV generation work without matplotlib, but plots require it.

### Plot Features

The generated plots show:
- **Hard Stops Comparison**: Number of hard stops for each mode across scenarios
- **Advisory Mode Usage**: Time spent in advisory mode (unique to dual PRV)
- **Critical Mode Usage**: Time spent in critical (hard stop) mode
- **Efficiency Gain**: Hard stops avoided by dual PRV vs conservative mode

### Customizing Comparisons

```bash
# Use custom parameters
python3 run_comparisons.py --tau-soft 1.5 --p-min 0.9

# Specify custom directories
python3 run_comparisons.py --data-dir my_traces --output-dir my_results
```

### Generating Custom Trace Files

You can generate individual trace files programmatically:

```python
from dualprv import io

# Generate a low-conflict scenario
io.generate_low_conflict_trace("data/my_low_conflict.csv")

# Generate a high-conflict scenario  
io.generate_high_conflict_trace("data/my_high_conflict.csv")

# Generate with custom time parameters
io.generate_intermittent_coverage_trace("data/custom.csv", dt=0.1, T=50.0)
```

---

## Author

Mustafa Adam

---

## License

This software is provided under the license GPL.