#!/usr/bin/env python3
"""
Dual-Future PRV simulation framework - main entry point.

This is a trace-driven simulation framework for evaluating Dual-Future
Predictive Runtime Verification (PRV) in a simplified UR10 human–robot
collaborative pick-and-place setting.
"""

from __future__ import annotations

import argparse
import itertools
from typing import List

from dualprv import io, models, simulation, workspace


def print_summary(metrics: models.RunMetrics) -> None:
    """Print summary of simulation metrics."""
    print(f"\n=== Results [{metrics.mode}] tau_hard={metrics.tau_hard:.3f} tau_soft={metrics.tau_soft:.3f} p_min={metrics.p_min:.2f} ===")
    print(f"steps: {metrics.steps}")
    print(f"hard_stops: {metrics.hard_stops}")
    print(f"time_nominal:  {metrics.time_nominal:.3f}s")
    print(f"time_advisory: {metrics.time_advisory:.3f}s")
    print(f"time_critical: {metrics.time_critical:.3f}s")
    print(f"min_sep_margin (d-Sp): {metrics.min_sep_margin:.4f} m")
    print(f"sep_violations: {metrics.sep_violations}")
    print(f"mutex_violations: {metrics.mutex_violations}")
    print(f"total_violations: {metrics.total_violations}")
    print(f"missed_violations (violation while not Critical): {metrics.missed_violations}")


def parse_float_list(s: str) -> List[float]:
    """Parse comma-separated float string into list."""
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [float(p) for p in parts]


def main() -> None:
    """Main entry point for CLI."""
    ap = argparse.ArgumentParser(
        description="Dual-Future PRV Simulator - trace-driven simulation framework"
    )
    ap.add_argument("--trace", type=str, default="", help="Path to minimal trace CSV.")
    ap.add_argument("--mode", type=str, default="dual", choices=["none", "conservative", "dual"])

    ap.add_argument("--tau-hard", type=float, default=0.5)
    ap.add_argument("--tau-soft", type=float, default=1.0)
    ap.add_argument("--p-min", type=float, default=0.8)

    ap.add_argument("--sweep", action="store_true", help="Sweep tau_soft and p_min.")
    ap.add_argument("--tau-soft-grid", type=str, default="0.5,1.0,1.5")
    ap.add_argument("--p-min-grid", type=str, default="0.6,0.8,0.9")
    ap.add_argument("--out-csv", type=str, default="", help="Write sweep results to CSV.")

    ap.add_argument("--export-debug", type=str, default="", help="Write per-tick debug CSV (includes a_k).")

    ap.add_argument("--generate-minimal-trace", type=str, default="", help="Generate a minimal trace CSV and exit.")
    ap.add_argument("--dt", type=float, default=0.2, help="Generator dt (if generating minimal trace).")
    ap.add_argument("--T", type=float, default=35.0, help="Generator horizon seconds (if generating minimal trace).")

    # Tuning knobs
    ap.add_argument("--v-nominal", type=float, default=0.8)
    ap.add_argument("--v-advisory", type=float, default=0.3)
    ap.add_argument("--v-h-max", type=float, default=1.2)
    ap.add_argument("--v-r-max", type=float, default=1.0)
    ap.add_argument("--t-react", type=float, default=0.10)
    ap.add_argument("--t-stop", type=float, default=0.40)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--critical-hold", type=float, default=0.0)
    ap.add_argument("--formal-lookahead", type=float, default=3.0, help="Formal branch lookahead horizon (seconds)")

    args = ap.parse_args()

    slots = workspace.default_slots()

    if args.generate_minimal_trace:
        io.generate_minimal_trace(args.generate_minimal_trace, dt=args.dt, T=args.T)
        return

    if not args.trace:
        raise SystemExit("Error: --trace is required (or use --generate-minimal-trace).")

    trace = io.load_trace_csv(args.trace)

    safe = models.SafetyParams(
        v_h_max=args.v_h_max,
        v_r_max=args.v_r_max,
        t_react=args.t_react,
        t_stop=args.t_stop,
        margin=args.margin,
    )
    ctrl = models.ControllerParams(v_nominal=args.v_nominal, v_advisory=args.v_advisory)
    sim_params = models.SimParams(
        critical_hold_s=args.critical_hold,
        formal_lookahead_horizon=args.formal_lookahead
    )

    if not args.sweep:
        fusion_params = models.FusionParams(tau_hard=args.tau_hard, tau_soft=args.tau_soft, p_min=args.p_min)
        metrics = simulation.simulate(
            trace=trace,
            slots=slots,
            safe=safe,
            ctrl=ctrl,
            fusion_params=fusion_params,
            sim_params=sim_params,
            mode=args.mode,
            export_debug_csv=(args.export_debug or None),
        )
        print_summary(metrics)
        if args.export_debug:
            print(f"\nWrote debug CSV (includes a_k): {args.export_debug}")
        return

    tau_soft_grid = parse_float_list(args.tau_soft_grid)
    p_min_grid = parse_float_list(args.p_min_grid)

    all_metrics: List[models.RunMetrics] = []
    for tau_soft, p_min in itertools.product(tau_soft_grid, p_min_grid):
        fusion_params = models.FusionParams(tau_hard=args.tau_hard, tau_soft=tau_soft, p_min=p_min)

        m_none = simulation.simulate(trace, slots, safe, ctrl, fusion_params, sim_params, mode="none")
        m_cons = simulation.simulate(trace, slots, safe, ctrl, fusion_params, sim_params, mode="conservative")
        m_dual = simulation.simulate(trace, slots, safe, ctrl, fusion_params, sim_params, mode="dual")

        all_metrics.extend([m_none, m_cons, m_dual])

        avoided = m_cons.hard_stops - m_dual.hard_stops
        print(
            f"tau_soft={tau_soft:.2f} p_min={p_min:.2f} | "
            f"hard_stops(cons)={m_cons.hard_stops} hard_stops(dual)={m_dual.hard_stops} avoided={avoided} | "
            f"time_adv(dual)={m_dual.time_advisory:.2f}s | missed_vio(dual)={m_dual.missed_violations}"
        )

    if args.out_csv:
        io.write_metrics_csv(args.out_csv, all_metrics)
        print(f"\nWrote sweep CSV: {args.out_csv}")
    else:
        print("\n(sweep complete) Use --out-csv to write results.")


if __name__ == "__main__":
    main()
