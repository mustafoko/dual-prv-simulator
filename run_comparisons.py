#!/usr/bin/env python3
"""
Run simulations across multiple trace scenarios and generate comparison plots.

This script:
1. Generates multiple trace files with different scenarios
2. Runs simulations for each trace with all three modes (none, conservative, dual)
3. Generates comparison plots showing efficiency gains
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

from dualprv import io, models, plotting, simulation, workspace


def main() -> None:
    """Run comparison simulations and generate plots."""
    parser = argparse.ArgumentParser(
        description="Run simulations across multiple scenarios and generate plots"
    )
    parser.add_argument("--data-dir", type=str, default="data", 
                       help="Directory for trace files (default: data)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory for output plots (default: results)")
    parser.add_argument("--tau-hard", type=float, default=0.5)
    parser.add_argument("--tau-soft", type=float, default=1.5)  # Increased to allow more advisory time
    parser.add_argument("--p-min", type=float, default=0.8)
    parser.add_argument("--formal-lookahead", type=float, default=3.0,
                       help="Formal branch lookahead horizon (seconds) for reachability analysis")
    
    # Tuning parameters
    parser.add_argument("--v-nominal", type=float, default=0.8)
    parser.add_argument("--v-advisory", type=float, default=0.3)
    parser.add_argument("--v-h-max", type=float, default=1.2)
    parser.add_argument("--v-r-max", type=float, default=1.0)
    parser.add_argument("--t-react", type=float, default=0.10)
    parser.add_argument("--t-stop", type=float, default=0.40)
    parser.add_argument("--margin", type=float, default=0.05)

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Dual PRV Simulation Comparison")
    print("=" * 70)

    # Define trace scenarios - expanded set for comprehensive testing
    trace_scenarios = {
        "default": ("minimal.csv", io.generate_minimal_trace),
        "low_conflict": ("low_conflict.csv", io.generate_low_conflict_trace),
        "high_conflict": ("high_conflict.csv", io.generate_high_conflict_trace),
        "intermittent": ("intermittent_coverage.csv", io.generate_intermittent_coverage_trace),
        "rapid": ("rapid_changes.csv", io.generate_rapid_changes_trace),
        "early_warning": ("early_warning.csv", io.generate_early_warning_trace),
        "varying_conf": ("varying_confidence.csv", io.generate_varying_confidence_trace),
        "cooperative": ("cooperative.csv", io.generate_cooperative_trace),
        "challenging": ("challenging.csv", io.generate_challenging_trace),
        "long": ("long_trace.csv", io.generate_long_trace),
        "proactive_advisory": ("proactive_advisory.csv", io.generate_proactive_advisory_trace),
    }

    print("\n1. Generating trace files...")
    trace_paths = {}
    for scenario_name, (filename, generator) in trace_scenarios.items():
        path = os.path.join(args.data_dir, filename)
        generator(path)
        trace_paths[scenario_name] = path

    print("\n2. Running simulations...")
    
    # Initialize parameters
    slots = workspace.default_slots()
    safe = models.SafetyParams(
        v_h_max=args.v_h_max,
        v_r_max=args.v_r_max,
        t_react=args.t_react,
        t_stop=args.t_stop,
        margin=args.margin,
    )
    ctrl = models.ControllerParams(v_nominal=args.v_nominal, v_advisory=args.v_advisory)
    sim_params = models.SimParams(formal_lookahead_horizon=args.formal_lookahead)
    fusion_params = models.FusionParams(
        tau_hard=args.tau_hard,
        tau_soft=args.tau_soft,
        p_min=args.p_min
    )

    # Run simulations for each trace and mode
    results: Dict[str, Dict[str, models.RunMetrics]] = {}
    
    modes = ["none", "conservative", "dual"]
    
    for scenario_name, trace_path in trace_paths.items():
        print(f"\n  Scenario: {scenario_name}")
        trace = io.load_trace_csv(trace_path)
        results[scenario_name] = {}
        
        for mode in modes:
            metrics = simulation.simulate(
                trace=trace,
                slots=slots,
                safe=safe,
                ctrl=ctrl,
                fusion_params=fusion_params,
                sim_params=sim_params,
                mode=mode,
            )
            results[scenario_name][mode] = metrics
            
            # Print summary
            print(f"    {mode:12s}: hard_stops={metrics.hard_stops:3d}, "
                  f"advisory={metrics.time_advisory:6.2f}s, "
                  f"critical={metrics.time_critical:6.2f}s")
            
            if mode == "dual" and "conservative" in results[scenario_name]:
                cons_metrics = results[scenario_name]["conservative"]
                avoided = cons_metrics.hard_stops - metrics.hard_stops
                if cons_metrics.hard_stops > 0:
                    pct = (avoided / cons_metrics.hard_stops) * 100
                    print(f"      → Efficiency: {avoided} hard stops avoided ({pct:.1f}% reduction)")

    print("\n3. Generating plots...")
    
    # Generate comparison plots
    comparison_plot = os.path.join(args.output_dir, "comparison.png")
    plotting.plot_comparison(results, output_path=comparison_plot)
    
    efficiency_plot = os.path.join(args.output_dir, "efficiency_summary.png")
    plotting.plot_efficiency_summary(results, output_path=efficiency_plot)

    # Write summary CSV
    summary_csv = os.path.join(args.output_dir, "summary.csv")
    all_metrics = []
    for scenario_name, scenario_results in results.items():
        for mode, metrics in scenario_results.items():
            all_metrics.append(metrics)
    io.write_metrics_csv(summary_csv, all_metrics)
    print(f"  Saved summary CSV: {summary_csv}")

    print("\n" + "=" * 70)
    print("Comparison complete!")
    print(f"Results saved to: {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
