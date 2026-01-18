"""Plotting utilities for simulation results and comparisons."""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .models import RunMetrics


def plot_comparison(
    results: Dict[str, Dict[str, RunMetrics]],
    output_path: Optional[str] = None,
    figsize: tuple = (14, 10)
) -> None:
    """
    Plot comparison of simulation results across different scenarios and modes.
    
    Args:
        results: Dict mapping trace_name -> Dict mapping mode -> RunMetrics
        output_path: Path to save figure (if None, displays interactively)
        figsize: Figure size (width, height)
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available. Install with: pip install matplotlib")
        return

    trace_names = sorted(results.keys())
    modes = ["none", "conservative", "dual"]
    mode_labels = ["None", "Conservative", "Dual PRV"]
    colors = {"none": "#d62728", "conservative": "#ff7f0e", "dual": "#2ca02c"}

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Dual PRV Performance Comparison", fontsize=16, fontweight="bold")

    # Extract metrics
    hard_stops = {mode: [] for mode in modes}
    time_advisory = {mode: [] for mode in modes}
    time_critical = {mode: [] for mode in modes}
    efficiency_gain = []  # Dual vs Conservative hard stops reduction

    for trace_name in trace_names:
        trace_results = results[trace_name]
        for mode in modes:
            if mode in trace_results:
                metrics = trace_results[mode]
                hard_stops[mode].append(metrics.hard_stops)
                time_advisory[mode].append(metrics.time_advisory)
                time_critical[mode].append(metrics.time_critical)
            else:
                hard_stops[mode].append(0)
                time_advisory[mode].append(0.0)
                time_critical[mode].append(0.0)
        
        # Calculate efficiency gain (hard stops avoided)
        if "conservative" in trace_results and "dual" in trace_results:
            avoided = trace_results["conservative"].hard_stops - trace_results["dual"].hard_stops
            efficiency_gain.append(avoided)
        else:
            efficiency_gain.append(0)

    x = range(len(trace_names))
    x_pos = [i - 0.2 for i in x]
    width = 0.25

    # Plot 1: Hard Stops Comparison
    ax1 = axes[0, 0]
    for i, mode in enumerate(modes):
        ax1.bar([p + i * width for p in x_pos], hard_stops[mode], width,
                label=mode_labels[i], color=colors[mode], alpha=0.8)
    ax1.set_xlabel("Scenario", fontweight="bold")
    ax1.set_ylabel("Hard Stops", fontweight="bold")
    ax1.set_title("Hard Stops by Mode", fontweight="bold")
    ax1.set_xticks([p + width for p in x_pos])
    ax1.set_xticklabels(trace_names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Time in Advisory Mode
    ax2 = axes[0, 1]
    for i, mode in enumerate(modes):
        ax2.bar([p + i * width for p in x_pos], time_advisory[mode], width,
                label=mode_labels[i], color=colors[mode], alpha=0.8)
    ax2.set_xlabel("Scenario", fontweight="bold")
    ax2.set_ylabel("Time in Advisory (s)", fontweight="bold")
    ax2.set_title("Advisory Mode Usage", fontweight="bold")
    ax2.set_xticks([p + width for p in x_pos])
    ax2.set_xticklabels(trace_names, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Plot 3: Time in Critical Mode
    ax3 = axes[1, 0]
    for i, mode in enumerate(modes):
        ax3.bar([p + i * width for p in x_pos], time_critical[mode], width,
                label=mode_labels[i], color=colors[mode], alpha=0.8)
    ax3.set_xlabel("Scenario", fontweight="bold")
    ax3.set_ylabel("Time in Critical (s)", fontweight="bold")
    ax3.set_title("Critical Mode Usage", fontweight="bold")
    ax3.set_xticks([p + width for p in x_pos])
    ax3.set_xticklabels(trace_names, rotation=45, ha="right")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # Plot 4: Efficiency Gain (Hard Stops Avoided)
    ax4 = axes[1, 1]
    bars = ax4.bar(x, efficiency_gain, color="#2ca02c", alpha=0.8)
    ax4.set_xlabel("Scenario", fontweight="bold")
    ax4.set_ylabel("Hard Stops Avoided", fontweight="bold")
    ax4.set_title("Dual PRV Efficiency Gain\n(Hard Stops Avoided vs Conservative)", fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(trace_names, rotation=45, ha="right")
    ax4.grid(axis="y", alpha=0.3)
    ax4.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight="bold")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot to: {output_path}")
    else:
        plt.show()


def plot_efficiency_summary(results: Dict[str, Dict[str, RunMetrics]], output_path: Optional[str] = None) -> None:
    """Create a summary plot showing efficiency improvements."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available. Install with: pip install matplotlib")
        return

    trace_names = sorted(results.keys())
    
    hard_stops_cons = []
    hard_stops_dual = []
    avoided = []
    advisory_time = []
    
    for trace_name in trace_names:
        trace_results = results[trace_name]
        if "conservative" in trace_results and "dual" in trace_results:
            cons = trace_results["conservative"]
            dual = trace_results["dual"]
            hard_stops_cons.append(cons.hard_stops)
            hard_stops_dual.append(dual.hard_stops)
            avoided.append(cons.hard_stops - dual.hard_stops)
            advisory_time.append(dual.time_advisory)
        else:
            hard_stops_cons.append(0)
            hard_stops_dual.append(0)
            avoided.append(0)
            advisory_time.append(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Dual PRV Efficiency Analysis", fontsize=14, fontweight="bold")

    x = range(len(trace_names))
    
    # Left plot: Hard stops comparison
    ax1 = axes[0]
    width = 0.35
    ax1.bar([i - width/2 for i in x], hard_stops_cons, width, label="Conservative", 
            color="#ff7f0e", alpha=0.8)
    ax1.bar([i + width/2 for i in x], hard_stops_dual, width, label="Dual PRV", 
            color="#2ca02c", alpha=0.8)
    ax1.set_xlabel("Scenario", fontweight="bold")
    ax1.set_ylabel("Hard Stops", fontweight="bold")
    ax1.set_title("Hard Stops: Conservative vs Dual PRV", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(trace_names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Right plot: Efficiency gains
    ax2 = axes[1]
    bars = ax2.bar(x, avoided, color="#2ca02c", alpha=0.8)
    ax2.set_xlabel("Scenario", fontweight="bold")
    ax2.set_ylabel("Hard Stops Avoided", fontweight="bold")
    ax2.set_title("Efficiency Gain (Dual PRV)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(trace_names, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0 and hard_stops_cons[i] > 0:
            pct = (height / hard_stops_cons[i]) * 100
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({pct:.0f}%)',
                    ha='center', va='bottom', fontweight="bold")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved efficiency summary to: {output_path}")
    else:
        plt.show()
