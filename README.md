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


## Author

Mustafa Adam

---

## License

This software is provided under the license GPL.
