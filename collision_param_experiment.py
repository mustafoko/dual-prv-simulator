#!/usr/bin/env python3
"""
Experiment: Vary Human Plan Change Probability (Collision Parameter)

Test collision parameter from 0% to 100% in 10% increments.
Run 1000 trials for each value and plot average hard stops.
"""

import random
import statistics
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Import from the main simulation
import sys
sys.path.insert(0, '/home/mustafa/.cursor/worktrees/dualprv/oxo')
from dual_future_prv_sim import (
    SLOTS, N_TRIALS, Simulation, Policy, TrialResult
)


def generate_trials_with_param(n_trials: int, human_change_prob: float) -> List[Tuple[List[str], List[str]]]:
    """
    Generate trial plans with configurable human change probability.
    
    Args:
        n_trials: Number of trials to generate
        human_change_prob: Probability that human changes plan (0.0 to 1.0)
    
    Returns:
        List of (human_plan, robot_plan) tuples
    """
    trials = []
    
    # Trial 1: Opposite plans (minimal conflict)
    human_plan = ['A', 'B', 'C', 'D']
    robot_plan = ['D', 'C', 'B', 'A']
    trials.append((human_plan.copy(), robot_plan.copy()))
    
    prev_human_plan = human_plan.copy()
    
    for _ in range(1, n_trials):
        # Human: probability-based repeat or random
        if random.random() < human_change_prob:
            # Human changes to random permutation
            human_plan = SLOTS.copy()
            random.shuffle(human_plan)
            human_changed = True
        else:
            # Human repeats
            human_plan = prev_human_plan.copy()
            human_changed = False
        
        # Robot planning based on assumption about human
        if human_changed:
            # Human changed - robot's assumption was WRONG
            # Robot gets a new random plan (doesn't know what human is doing)
            robot_plan = SLOTS.copy()
            random.shuffle(robot_plan)
        else:
            # Human repeated - robot's assumption was correct
            # Robot plans to avoid expected human path (go opposite)
            robot_plan = prev_human_plan[::-1]
        
        trials.append((human_plan.copy(), robot_plan.copy()))
        prev_human_plan = human_plan.copy()
    
    return trials


def run_experiments_with_param(trials: List[Tuple[List[str], List[str]]], seed: int = 42):
    """
    Run experiments for both policies on the same trials.
    
    Returns results for Formal-only and Dual-PRV.
    """
    formal_results = []
    dual_results = []
    
    for i, (human_plan, robot_plan) in enumerate(trials):
        # Run Formal-only
        random.seed(seed + i * 1000)
        sim_formal = Simulation(human_plan, robot_plan, Policy.FORMAL_ONLY)
        formal_results.append(sim_formal.run())
        
        # Run Dual-PRV with same seed for fair comparison
        random.seed(seed + i * 1000)
        sim_dual = Simulation(human_plan, robot_plan, Policy.DUAL_PRV)
        dual_results.append(sim_dual.run())
    
    return formal_results, dual_results


def main():
    """Run collision parameter experiment."""
    # Test parameters: 0%, 10%, 20%, ..., 100%
    collision_params = [i/100.0 for i in range(0, 101, 10)]
    n_trials_per_param = 1000
    
    print("=" * 80)
    print("COLLISION PARAMETER EXPERIMENT")
    print("=" * 80)
    print(f"Testing human plan change probability from 0% to 100% in 10% steps")
    print(f"Running {n_trials_per_param} trials for each parameter value")
    print("=" * 80)
    print()
    
    # Store results
    results = {
        'params': [],
        'formal_hard_stops_mean': [],
        'formal_hard_stops_std': [],
        'formal_time_mean': [],
        'formal_time_std': [],
        'dual_hard_stops_mean': [],
        'dual_hard_stops_std': [],
        'dual_slowdowns_mean': [],
        'dual_slowdowns_std': [],
        'dual_time_mean': [],
        'dual_time_std': [],
    }
    
    # Run experiments
    for param in collision_params:
        param_pct = param * 100
        print(f"Testing collision parameter = {param_pct:.0f}%...", end=' ', flush=True)
        
        # Generate trials with this parameter
        random.seed(42)  # Same seed for reproducibility
        trials = generate_trials_with_param(n_trials_per_param, param)
        
        # Run experiments
        formal_results, dual_results = run_experiments_with_param(trials, seed=42)
        
        # Extract metrics
        formal_stops = [r.hard_stops for r in formal_results]
        formal_times = [r.completion_time for r in formal_results]
        dual_stops = [r.hard_stops for r in dual_results]
        dual_slows = [r.slowdowns for r in dual_results]
        dual_times = [r.completion_time for r in dual_results]
        
        # Store results
        results['params'].append(param_pct)
        results['formal_hard_stops_mean'].append(statistics.mean(formal_stops))
        results['formal_hard_stops_std'].append(statistics.stdev(formal_stops) if len(formal_stops) > 1 else 0)
        results['formal_time_mean'].append(statistics.mean(formal_times))
        results['formal_time_std'].append(statistics.stdev(formal_times) if len(formal_times) > 1 else 0)
        results['dual_hard_stops_mean'].append(statistics.mean(dual_stops))
        results['dual_hard_stops_std'].append(statistics.stdev(dual_stops) if len(dual_stops) > 1 else 0)
        results['dual_slowdowns_mean'].append(statistics.mean(dual_slows))
        results['dual_slowdowns_std'].append(statistics.stdev(dual_slows) if len(dual_slows) > 1 else 0)
        results['dual_time_mean'].append(statistics.mean(dual_times))
        results['dual_time_std'].append(statistics.stdev(dual_times) if len(dual_times) > 1 else 0)
        
        print(f"Done (Formal stops: {results['formal_hard_stops_mean'][-1]:.4f}, "
              f"Dual stops: {results['dual_hard_stops_mean'][-1]:.4f})")
    
    print()
    print("=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    print()
    print(f"{'Collision':>10} | {'Formal Hard Stops':>20} | {'Dual Hard Stops':>20} | "
          f"{'Dual Slowdowns':>20} | {'Improvement':>12}")
    print(f"{'Param (%)':>10} | {'Mean':>10} {'Std':>9} | {'Mean':>10} {'Std':>9} | "
          f"{'Mean':>10} {'Std':>9} | {'(% fewer)':>12}")
    print("-" * 110)
    
    for i, param_pct in enumerate(results['params']):
        formal_mean = results['formal_hard_stops_mean'][i]
        formal_std = results['formal_hard_stops_std'][i]
        dual_mean = results['dual_hard_stops_mean'][i]
        dual_std = results['dual_hard_stops_std'][i]
        slow_mean = results['dual_slowdowns_mean'][i]
        slow_std = results['dual_slowdowns_std'][i]
        
        improvement = ((formal_mean - dual_mean) / formal_mean * 100) if formal_mean > 0 else 0
        
        print(f"{param_pct:>10.0f} | {formal_mean:>10.4f} {formal_std:>9.4f} | "
              f"{dual_mean:>10.4f} {dual_std:>9.4f} | {slow_mean:>10.4f} {slow_std:>9.4f} | "
              f"{improvement:>11.1f}%")
    
    print()
    print("=" * 80)
    print("CREATING PLOTS (3 SEPARATE IMAGES)")
    print("=" * 80)
    
    # =============================================================================
    # PLOT 1: Hard Stops vs Collision Parameter with % Improvement
    # =============================================================================
    print("\n1. Creating Hard Stops plot...")
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Calculate improvement percentages
    improvement_pct = []
    for i in range(len(results['params'])):
        formal_mean = results['formal_hard_stops_mean'][i]
        dual_mean = results['dual_hard_stops_mean'][i]
        if formal_mean > 0:
            improvement_pct.append((formal_mean - dual_mean) / formal_mean * 100)
        else:
            improvement_pct.append(0)
    
    # Primary axis: Hard stops
    ax1.plot(results['params'], results['formal_hard_stops_mean'], 
             'o-', linewidth=2.5, markersize=8, label='Formal-Only PRV', color='#e74c3c')
    ax1.plot(results['params'], results['dual_hard_stops_mean'], 
             's-', linewidth=2.5, markersize=8, label='Dual-PRV', color='#3498db')
    ax1.fill_between(results['params'], 
                     np.array(results['formal_hard_stops_mean']) - np.array(results['formal_hard_stops_std']),
                     np.array(results['formal_hard_stops_mean']) + np.array(results['formal_hard_stops_std']),
                     alpha=0.2, color='#e74c3c')
    ax1.fill_between(results['params'], 
                     np.array(results['dual_hard_stops_mean']) - np.array(results['dual_hard_stops_std']),
                     np.array(results['dual_hard_stops_mean']) + np.array(results['dual_hard_stops_std']),
                     alpha=0.2, color='#3498db')
    ax1.set_xlabel('Collision Parameter (% Human Plan Change)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Average Hard Stops per Trial', fontsize=13, fontweight='bold', color='black')
    ax1.set_title('Hard Stops vs Collision Parameter\n(with % Improvement)', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(-5, 105)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    
    # Secondary axis: % Improvement
    ax1_twin = ax1.twinx()
    ax1_twin.plot(results['params'], improvement_pct, 
                  'D-', linewidth=2.5, markersize=8, label='% Improvement', 
                  color='#27ae60', alpha=0.9, markeredgecolor='darkgreen', markeredgewidth=1.5)
    ax1_twin.set_ylabel('Dual-PRV Improvement (%)', fontsize=13, fontweight='bold', color='#27ae60')
    ax1_twin.tick_params(axis='y', labelcolor='#27ae60', labelsize=11)
    ax1_twin.set_ylim(-5, 105)
    ax1_twin.axhline(y=90, color='#27ae60', linestyle='--', linewidth=1.5, alpha=0.5, label='90% target')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    output_file_1 = '/home/mustafa/.cursor/worktrees/dualprv/oxo/plot1_hard_stops.png'
    fig1.savefig(output_file_1, dpi=300, bbox_inches='tight')
    output_file_1_pdf = '/home/mustafa/.cursor/worktrees/dualprv/oxo/plot1_hard_stops.pdf'
    fig1.savefig(output_file_1_pdf, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_file_1}")
    print(f"   Saved: {output_file_1_pdf}")
    plt.close(fig1)
    
    # =============================================================================
    # PLOT 2: Completion Time vs Collision Parameter
    # =============================================================================
    print("\n2. Creating Completion Time plot...")
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(results['params'], results['formal_time_mean'], 
             'o-', linewidth=2.5, markersize=8, label='Formal-Only PRV', color='#e74c3c')
    ax2.plot(results['params'], results['dual_time_mean'], 
             's-', linewidth=2.5, markersize=8, label='Dual-PRV', color='#3498db')
    ax2.fill_between(results['params'], 
                     np.array(results['formal_time_mean']) - np.array(results['formal_time_std']),
                     np.array(results['formal_time_mean']) + np.array(results['formal_time_std']),
                     alpha=0.2, color='#e74c3c')
    ax2.fill_between(results['params'], 
                     np.array(results['dual_time_mean']) - np.array(results['dual_time_std']),
                     np.array(results['dual_time_mean']) + np.array(results['dual_time_std']),
                     alpha=0.2, color='#3498db')
    ax2.set_xlabel('Collision Parameter (% Human Plan Change)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Average Completion Time (seconds)', fontsize=13, fontweight='bold')
    ax2.set_title('Completion Time vs Collision Parameter', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(-5, 105)
    ax2.tick_params(axis='both', labelsize=11)
    
    plt.tight_layout()
    output_file_2 = '/home/mustafa/.cursor/worktrees/dualprv/oxo/plot2_completion_time.png'
    fig2.savefig(output_file_2, dpi=300, bbox_inches='tight')
    output_file_2_pdf = '/home/mustafa/.cursor/worktrees/dualprv/oxo/plot2_completion_time.pdf'
    fig2.savefig(output_file_2_pdf, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_file_2}")
    print(f"   Saved: {output_file_2_pdf}")
    plt.close(fig2)
    
    # =============================================================================
    # PLOT 3: Dual-PRV Slowdowns vs Collision Parameter
    # =============================================================================
    print("\n3. Creating Slowdowns plot...")
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(results['params'], results['dual_slowdowns_mean'], 
             'D-', linewidth=2.5, markersize=8, label='Dual-PRV Slowdowns', color='#f39c12',
             markeredgecolor='#d68910', markeredgewidth=1.5)
    ax3.fill_between(results['params'], 
                     np.array(results['dual_slowdowns_mean']) - np.array(results['dual_slowdowns_std']),
                     np.array(results['dual_slowdowns_mean']) + np.array(results['dual_slowdowns_std']),
                     alpha=0.25, color='#f39c12')
    ax3.set_xlabel('Collision Parameter (% Human Plan Change)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Average Slowdowns per Trial', fontsize=13, fontweight='bold')
    ax3.set_title('Dual-PRV Slowdowns (Learned Interventions)', fontsize=14, fontweight='bold', pad=20)
    ax3.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(-5, 105)
    ax3.tick_params(axis='both', labelsize=11)
    
    plt.tight_layout()
    output_file_3 = '/home/mustafa/.cursor/worktrees/dualprv/oxo/plot3_slowdowns.png'
    fig3.savefig(output_file_3, dpi=300, bbox_inches='tight')
    output_file_3_pdf = '/home/mustafa/.cursor/worktrees/dualprv/oxo/plot3_slowdowns.pdf'
    fig3.savefig(output_file_3_pdf, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_file_3}")
    print(f"   Saved: {output_file_3_pdf}")
    
    print()
    print("=" * 80)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. plot1_hard_stops.png / .pdf")
    print("  2. plot2_completion_time.png / .pdf")
    print("  3. plot3_slowdowns.png / .pdf")
    print()
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()
