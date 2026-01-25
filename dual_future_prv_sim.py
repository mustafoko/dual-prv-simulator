#!/usr/bin/env python3
"""
Trace-driven HRC Pick-&-Place Simulator + Evaluation

Simulates a human and robot collaboratively placing boxes into 4 slots {A,B,C,D}.
Compares two monitoring/control approaches:
  - Formal-only PRV (predicts imminent crash → hard stop)
  - Dual-PRV (formal PRV + learned intent predictor → avoid conflicts via slow-down/replanning)

Run 1000 trials and report average hard stops and completion time.
"""

import random
import statistics
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum, auto


# =============================================================================
# CONSTANTS
# =============================================================================

SLOTS = ['A', 'B', 'C', 'D']

# Timing model (all in seconds)
HUMAN_MOVE_PLACE = 10.0    # Human move+place time
HUMAN_DWELL = 3.0          # Human dwell time on slot
HUMAN_TOTAL = 13.0         # Total time for human to visit a slot

ROBOT_MOVE_PLACE_NORMAL = 11.0   # Robot normal move+place (faster than human)
ROBOT_MOVE_PLACE_SLOW = 14.0     # Robot slow-down move+place
ROBOT_DWELL = 3.0                # Robot dwell time
ROBOT_TOTAL_NORMAL = 14.0        # Total time for robot to visit (normal)
ROBOT_TOTAL_SLOW = 17.0          # Total time for robot to visit (slow-down)

HARD_STOP_PENALTY = 100.0   # Seconds added when hard stop occurs
CRASH_PREDICTION_WINDOW = 2.0  # Formal PRV predicts crash 2s ahead

LEARNED_ACCURACY = 0.9     # Learned predictor accuracy

N_TRIALS = 10000
HUMAN_CHANGE_PROB = 0.3    # Probability human changes plan


# =============================================================================
# ENUMS
# =============================================================================

class Policy(Enum):
    """Control policy type."""
    FORMAL_ONLY = auto()
    DUAL_PRV = auto()


class AgentPhase(Enum):
    """Current phase of agent action."""
    IDLE = auto()
    MOVING = auto()
    DWELLING = auto()
    DONE = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrialResult:
    """Result of a single trial."""
    completion_time: float
    hard_stops: int
    slowdowns: int


@dataclass
class AgentState:
    """State of an agent during simulation."""
    plan: List[str]
    plan_idx: int = 0
    phase: AgentPhase = AgentPhase.IDLE
    target_slot: Optional[str] = None
    time_remaining: float = 0.0
    
    def current_target(self) -> Optional[str]:
        """Get current target slot from plan."""
        if self.plan_idx < len(self.plan):
            return self.plan[self.plan_idx]
        return None
    
    def advance_plan(self):
        """Move to next slot in plan."""
        self.plan_idx += 1
    
    def is_done(self) -> bool:
        """Check if agent has completed its plan."""
        return self.plan_idx >= len(self.plan)


# =============================================================================
# DUAL-PRV ARCHITECTURE
# =============================================================================

@dataclass
class PRVDecision:
    """Decision made by the Dual-PRV system."""
    action: str  # 'proceed_normal', 'slowdown', 'hard_stop'
    target_slot: Optional[str]
    robot_speed: float  # ROBOT_MOVE_PLACE_NORMAL or ROBOT_MOVE_PLACE_SLOW
    reason: str  # Explanation of decision


def dual_prv_decision(
    robot_plan: List[str],
    robot_plan_idx: int,
    human_target: Optional[str],
    human_phase: AgentPhase,
    human_time_remaining: float,
    robot_target: Optional[str],
    robot_phase: AgentPhase,
    robot_time_remaining: float,
    visited: dict,
    learned_accuracy: float = LEARNED_ACCURACY
) -> PRVDecision:
    """
    Dual-PRV Architecture: Combined Learned + Formal Predictive Runtime Verification
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    DUAL-PRV SYSTEM                          │
    │                                                             │
    │  ┌──────────────────────┐    ┌──────────────────────────┐  │
    │  │  LEARNED BRANCH      │    │   FORMAL BRANCH          │  │
    │  │  (Advisory)          │    │   (Safety-Critical)      │  │
    │  │                      │    │                          │  │
    │  │ • Predict human      │    │ • Detect imminent crash  │  │
    │  │   intent (90% acc)   │    │   (2s prediction window) │  │
    │  │ • Proactive conflict │    │ • Verify occupancy       │  │
    │  │   avoidance          │    │   overlap                │  │
    │  │ • Trigger slowdown   │    │ • Trigger hard stop      │  │
    │  │   (3s penalty)       │    │   (100s penalty)         │  │
    │  └──────────┬───────────┘    └──────────┬───────────────┘  │
    │             │                           │                   │
    │             └────────┬──────────────────┘                   │
    │                      ▼                                      │
    │              ┌───────────────┐                              │
    │              │ FUSION POLICY │                              │
    │              │ Formal > Learn│                              │
    │              └───────────────┘                              │
    └─────────────────────────────────────────────────────────────┘
    
    Args:
        robot_plan: Robot's planned sequence of slots
        robot_plan_idx: Current index in robot's plan
        human_target: Current target slot of human (None if idle/done)
        human_phase: Current phase of human (IDLE, MOVING, DWELLING, DONE)
        human_time_remaining: Time remaining in human's current action
        robot_target: Current target slot of robot (None if planning)
        robot_phase: Current phase of robot
        robot_time_remaining: Time remaining in robot's current action
        visited: Dictionary of which slots have been visited
        learned_accuracy: Accuracy of learned predictor (default 0.9)
    
    Returns:
        PRVDecision with action, target_slot, speed, and reason
    
    Decision Logic:
        1. LEARNED BRANCH (Proactive):
           - Predict human's next/current target
           - If conflict predicted: find alternative slot → slowdown
        
        2. FORMAL BRANCH (Reactive Safety):
           - Check for imminent collision (occupancy overlap within 2s)
           - If collision predicted: hard stop (overrides learned)
        
        3. FUSION:
           - Formal hard stop overrides learned slowdown
           - If no conflict: proceed normally
    """
    
    # =============================================================================
    # LEARNED BRANCH: Proactive Conflict Prediction & Avoidance
    # =============================================================================
    
    # Step 1: Predict human target with learned_accuracy
    predicted_human_target = human_target
    if predicted_human_target is None:
        # Human is idle/planning - predict from their plan (if we had it)
        # In this standalone function, we only have current target
        predicted_human_target = None
    
    # Simulate learned prediction accuracy
    if predicted_human_target is not None:
        if random.random() >= learned_accuracy:
            # Wrong prediction (10% of the time)
            other_slots = [s for s in SLOTS if s != predicted_human_target]
            if other_slots:
                predicted_human_target = random.choice(other_slots)
    
    # Step 2: Find robot's next unvisited target
    robot_next_target = None
    for i in range(robot_plan_idx, len(robot_plan)):
        slot = robot_plan[i]
        if not visited.get(slot, False):
            robot_next_target = slot
            break
    
    if robot_next_target is None:
        return PRVDecision('proceed_normal', None, 0, 'No unvisited slots')
    
    # Step 3: Check if learned branch predicts conflict
    learned_conflict = (robot_next_target == predicted_human_target)
    
    # Step 4: If conflict predicted, find alternative slot and trigger slowdown
    alternative_target = None
    if learned_conflict:
        # Try to find alternative unvisited slot
        for slot in SLOTS:
            if not visited.get(slot, False) and slot != robot_next_target:
                alternative_target = slot
                break
        
        if alternative_target:
            # Learned branch: avoid conflict via slowdown
            learned_decision = PRVDecision(
                action='slowdown',
                target_slot=alternative_target,
                robot_speed=ROBOT_MOVE_PLACE_SLOW,
                reason=f'Learned: Conflict predicted at {robot_next_target}, rerouting to {alternative_target}'
            )
        else:
            # No alternative - proceed to original target
            learned_decision = PRVDecision(
                action='proceed_normal',
                target_slot=robot_next_target,
                robot_speed=ROBOT_MOVE_PLACE_NORMAL,
                reason='Learned: Conflict predicted but no alternative available'
            )
    else:
        # No conflict predicted by learned branch
        learned_decision = PRVDecision(
            action='proceed_normal',
            target_slot=robot_next_target,
            robot_speed=ROBOT_MOVE_PLACE_NORMAL,
            reason='Learned: No conflict predicted'
        )
    
    # =============================================================================
    # FORMAL BRANCH: Reactive Collision Detection (Safety-Critical)
    # =============================================================================
    
    # Check if robot is moving and heading to same slot as human
    if robot_phase == AgentPhase.MOVING and robot_target is not None:
        if human_phase in [AgentPhase.MOVING, AgentPhase.DWELLING]:
            if robot_target == human_target:
                # Calculate occupancy overlap
                robot_arrival = robot_time_remaining
                
                if human_phase == AgentPhase.MOVING:
                    human_arrival = human_time_remaining
                    human_leave = human_arrival + HUMAN_DWELL
                else:  # DWELLING
                    human_arrival = 0
                    human_leave = human_time_remaining
                
                robot_leave = robot_arrival + ROBOT_DWELL
                
                # Check for overlap
                crash_predicted = not (robot_leave < human_arrival or human_leave < robot_arrival)
                time_to_crash = max(0, min(robot_arrival, human_arrival))
                
                # Formal PRV triggers within prediction window
                if crash_predicted and time_to_crash <= CRASH_PREDICTION_WINDOW:
                    # FORMAL BRANCH OVERRIDES LEARNED
                    return PRVDecision(
                        action='hard_stop',
                        target_slot=robot_target,
                        robot_speed=0,
                        reason=f'Formal: Imminent collision at {robot_target} in {time_to_crash:.1f}s → HARD STOP'
                    )
    
    # =============================================================================
    # FUSION: Return Learned Decision (Formal didn't override)
    # =============================================================================
    
    return learned_decision


# =============================================================================
# SIMULATION
# =============================================================================

class Simulation:
    """
    Discrete-time simulation of HRC pick-and-place task.
    """
    
    def __init__(self, human_plan: List[str], robot_plan: List[str], policy: Policy):
        self.human_plan = human_plan.copy()
        self.robot_plan = robot_plan.copy()
        self.policy = policy
        
        # Visited slots
        self.visited = {slot: False for slot in SLOTS}
        
        # Agent states
        self.human = AgentState(plan=human_plan.copy())
        self.robot = AgentState(plan=robot_plan.copy())
        
        # Time tracking
        self.time = 0.0
        
        # Statistics
        self.hard_stops = 0
        self.slowdowns = 0
        
        # Robot mode
        self.robot_slow_mode = False
    
    def run(self) -> TrialResult:
        """Run simulation until all slots are visited."""
        # Start both agents
        self._start_human_action()
        self._start_robot_action()
        
        while not self._all_visited():
            # Find next event time
            dt = self._next_event_time()
            if dt <= 0 or dt == float('inf'):
                break
            
            # Check for crash before advancing time
            self._check_and_handle_crash(dt)
            
            # Advance time
            self.time += dt
            self.human.time_remaining -= dt
            self.robot.time_remaining -= dt
            
            # Process completions
            self._process_human()
            self._process_robot()
        
        return TrialResult(
            completion_time=self.time,
            hard_stops=self.hard_stops,
            slowdowns=self.slowdowns
        )
    
    def _all_visited(self) -> bool:
        """Check if all slots are visited."""
        return all(self.visited.values())
    
    def _next_event_time(self) -> float:
        """Get time to next event."""
        times = []
        if self.human.time_remaining > 0:
            times.append(self.human.time_remaining)
        if self.robot.time_remaining > 0:
            times.append(self.robot.time_remaining)
        return min(times) if times else float('inf')
    
    def _start_human_action(self):
        """Start human moving to next unvisited slot in plan."""
        while not self.human.is_done():
            target = self.human.current_target()
            if target and not self.visited[target]:
                self.human.target_slot = target
                self.human.phase = AgentPhase.MOVING
                self.human.time_remaining = HUMAN_MOVE_PLACE
                return
            self.human.advance_plan()
        
        # No more slots to visit
        self.human.phase = AgentPhase.DONE
        self.human.target_slot = None
        self.human.time_remaining = 0
    
    def _start_robot_action(self):
        """Start robot moving to next slot based on policy."""
        if self.policy == Policy.DUAL_PRV:
            self._start_robot_dual_prv()
        else:
            self._start_robot_formal_only()
    
    def _start_robot_formal_only(self):
        """Start robot action under formal-only policy."""
        while not self.robot.is_done():
            target = self.robot.current_target()
            if target and not self.visited[target]:
                self.robot.target_slot = target
                self.robot.phase = AgentPhase.MOVING
                self.robot.time_remaining = ROBOT_MOVE_PLACE_NORMAL
                self.robot_slow_mode = False
                return
            self.robot.advance_plan()
        
        self.robot.phase = AgentPhase.DONE
        self.robot.target_slot = None
        self.robot.time_remaining = 0
    
    def _start_robot_dual_prv(self):
        """Start robot action under dual-PRV policy with learned prediction."""
        # Get predicted human target (with 90% accuracy)
        predicted_human_target = self._predict_human_target()
        
        # Find next unvisited slot from robot's plan
        original_target = None
        while not self.robot.is_done():
            target = self.robot.current_target()
            if target and not self.visited[target]:
                original_target = target
                break
            self.robot.advance_plan()
        
        if original_target is None:
            self.robot.phase = AgentPhase.DONE
            self.robot.target_slot = None
            self.robot.time_remaining = 0
            return
        
        # Check if learned predictor says conflict
        if original_target == predicted_human_target:
            # Try to find alternative unvisited slot
            alternative = self._find_alternative_slot(original_target)
            if alternative:
                self.robot.target_slot = alternative
                self.robot.phase = AgentPhase.MOVING
                self.robot.time_remaining = ROBOT_MOVE_PLACE_SLOW  # Slow-down mode
                self.robot_slow_mode = True
                self.slowdowns += 1
                return
        
        # No conflict predicted or no alternative - proceed normally
        self.robot.target_slot = original_target
        self.robot.phase = AgentPhase.MOVING
        self.robot.time_remaining = ROBOT_MOVE_PLACE_NORMAL
        self.robot_slow_mode = False
    
    def _predict_human_target(self) -> Optional[str]:
        """Learned predictor: predict human's current/next target with 90% accuracy."""
        actual_target = self.human.target_slot
        if actual_target is None:
            # Predict from human's plan
            for i in range(self.human.plan_idx, len(self.human.plan)):
                slot = self.human.plan[i]
                if not self.visited[slot]:
                    actual_target = slot
                    break
        
        if actual_target is None:
            return None
        
        # 90% accuracy
        if random.random() < LEARNED_ACCURACY:
            return actual_target
        else:
            # Wrong prediction - return random other slot
            others = [s for s in SLOTS if s != actual_target]
            return random.choice(others) if others else actual_target
    
    def _find_alternative_slot(self, avoid_slot: str) -> Optional[str]:
        """Find an alternative unvisited slot for robot."""
        # First try remaining slots in robot's plan
        for i in range(self.robot.plan_idx + 1, len(self.robot.plan)):
            slot = self.robot.plan[i]
            if not self.visited[slot] and slot != avoid_slot:
                return slot
        
        # Then try any unvisited slot
        for slot in SLOTS:
            if not self.visited[slot] and slot != avoid_slot:
                return slot
        
        return None
    
    def _check_and_handle_crash(self, dt: float):
        """Check for predicted crash and handle it."""
        if self.robot.phase != AgentPhase.MOVING:
            return
        if self.human.phase not in [AgentPhase.MOVING, AgentPhase.DWELLING]:
            return
        
        # Check if targeting same slot
        if self.robot.target_slot != self.human.target_slot:
            return
        
        # Compute arrival times (time from now until arrival)
        robot_arrival = self.robot.time_remaining
        
        if self.human.phase == AgentPhase.MOVING:
            human_arrival = self.human.time_remaining
            human_leave = human_arrival + HUMAN_DWELL
        else:  # DWELLING - human is already at slot
            human_arrival = 0
            human_leave = self.human.time_remaining  # time until human leaves
        
        # Robot occupancy period: [robot_arrival, robot_arrival + ROBOT_DWELL]
        robot_leave = robot_arrival + ROBOT_DWELL
        
        # Check for occupancy overlap - crash if both would be at slot at same time
        # Overlap exists if: NOT (robot_leave < human_arrival OR human_leave < robot_arrival)
        crash_predicted = not (robot_leave < human_arrival or human_leave < robot_arrival)
        
        # Formal PRV triggers 2 seconds before the crash would happen
        time_to_crash = max(0, min(robot_arrival, human_arrival))
        
        if crash_predicted and time_to_crash <= CRASH_PREDICTION_WINDOW:
            self._trigger_hard_stop()
    
    def _trigger_hard_stop(self):
        """Trigger a hard stop for the robot."""
        self.hard_stops += 1
        self.robot.time_remaining += HARD_STOP_PENALTY
    
    def _process_human(self):
        """Process human state after time advance."""
        if self.human.time_remaining <= 0:
            if self.human.phase == AgentPhase.MOVING:
                # Arrived at slot - start dwelling
                self.human.phase = AgentPhase.DWELLING
                self.human.time_remaining = HUMAN_DWELL
                # Mark slot as visited
                if self.human.target_slot:
                    self.visited[self.human.target_slot] = True
            elif self.human.phase == AgentPhase.DWELLING:
                # Done dwelling - move to next slot
                self.human.advance_plan()
                self._start_human_action()
    
    def _process_robot(self):
        """Process robot state after time advance."""
        if self.robot.time_remaining <= 0:
            if self.robot.phase == AgentPhase.MOVING:
                # Arrived at slot - start dwelling
                self.robot.phase = AgentPhase.DWELLING
                self.robot.time_remaining = ROBOT_DWELL
                # Mark slot as visited
                if self.robot.target_slot:
                    self.visited[self.robot.target_slot] = True
            elif self.robot.phase == AgentPhase.DWELLING:
                # Done dwelling - move to next slot
                self.robot.advance_plan()
                self._start_robot_action()


# =============================================================================
# TRIAL GENERATION
# =============================================================================

def generate_trials(n_trials: int = N_TRIALS) -> List[Tuple[List[str], List[str]]]:
    """
    Generate trial plans according to specification.
    
    Trial 1: Human A,B,C,D; Robot D,C,B,A
    Trials 2+: Human 70% repeat, 30% random; Robot assumes repeat, if changed gets random
    
    The key insight: when human changes plan (30% of trials), the robot's assumption
    is WRONG, which creates conflict potential. Robot gets a new random plan that
    may overlap with human's new random plan.
    
    Returns list of (human_plan, robot_plan) tuples.
    """
    trials = []
    
    # Trial 1: Opposite plans (minimal conflict)
    human_plan = ['A', 'B', 'C', 'D']
    robot_plan = ['D', 'C', 'B', 'A']
    trials.append((human_plan.copy(), robot_plan.copy()))
    
    prev_human_plan = human_plan.copy()
    
    for _ in range(1, n_trials):
        # Human: 70% repeat, 30% random
        if random.random() < HUMAN_CHANGE_PROB:
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


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiments(trials: List[Tuple[List[str], List[str]]], seed: int = 42):
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


def print_trial_details(trials: List[Tuple[List[str], List[str]]],
                        formal_results: List[TrialResult],
                        dual_results: List[TrialResult],
                        n_show: int = 50):
    """Print details of first n trials."""
    print("\n" + "=" * 100)
    print(f"FIRST {n_show} TRIAL DETAILS")
    print("=" * 100)
    print(f"{'Trial':>5} | {'Human Plan':<15} | {'Robot Plan':<15} | "
          f"{'Formal Stops':>12} | {'Formal Time':>11} | "
          f"{'Dual Stops':>10} | {'Dual Slows':>10} | {'Dual Time':>10}")
    print("-" * 100)
    
    for i in range(min(n_show, len(trials))):
        human_plan, robot_plan = trials[i]
        fr = formal_results[i]
        dr = dual_results[i]
        
        h_str = ','.join(human_plan)
        r_str = ','.join(robot_plan)
        
        print(f"{i+1:>5} | {h_str:<15} | {r_str:<15} | "
              f"{fr.hard_stops:>12} | {fr.completion_time:>10.1f}s | "
              f"{dr.hard_stops:>10} | {dr.slowdowns:>10} | {dr.completion_time:>9.1f}s")


def print_summary(formal_results: List[TrialResult], dual_results: List[TrialResult]):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    
    # Formal-only stats
    formal_stops = [r.hard_stops for r in formal_results]
    formal_times = [r.completion_time for r in formal_results]
    
    print("\nFORMAL-ONLY PRV:")
    print("-" * 40)
    print(f"  Hard Stops per Trial:")
    print(f"    Mean: {statistics.mean(formal_stops):.3f}")
    print(f"    Std:  {statistics.stdev(formal_stops):.3f}")
    print(f"    Min:  {min(formal_stops)}")
    print(f"    Max:  {max(formal_stops)}")
    print(f"  Completion Time:")
    print(f"    Mean: {statistics.mean(formal_times):.2f}s")
    print(f"    Std:  {statistics.stdev(formal_times):.2f}s")
    print(f"    Min:  {min(formal_times):.1f}s")
    print(f"    Max:  {max(formal_times):.1f}s")
    
    # Dual-PRV stats
    dual_stops = [r.hard_stops for r in dual_results]
    dual_times = [r.completion_time for r in dual_results]
    dual_slows = [r.slowdowns for r in dual_results]
    
    print("\nDUAL-PRV (Formal + Learned):")
    print("-" * 40)
    print(f"  Hard Stops per Trial:")
    print(f"    Mean: {statistics.mean(dual_stops):.3f}")
    print(f"    Std:  {statistics.stdev(dual_stops):.3f}")
    print(f"    Min:  {min(dual_stops)}")
    print(f"    Max:  {max(dual_stops)}")
    print(f"  Slow-downs per Trial:")
    print(f"    Mean: {statistics.mean(dual_slows):.3f}")
    print(f"    Std:  {statistics.stdev(dual_slows):.3f}")
    print(f"  Completion Time:")
    print(f"    Mean: {statistics.mean(dual_times):.2f}s")
    print(f"    Std:  {statistics.stdev(dual_times):.2f}s")
    print(f"    Min:  {min(dual_times):.1f}s")
    print(f"    Max:  {max(dual_times):.1f}s")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    stop_reduction = statistics.mean(formal_stops) - statistics.mean(dual_stops)
    stop_reduction_pct = (stop_reduction / statistics.mean(formal_stops) * 100) if statistics.mean(formal_stops) > 0 else 0
    
    time_improvement = statistics.mean(formal_times) - statistics.mean(dual_times)
    time_improvement_pct = (time_improvement / statistics.mean(formal_times) * 100)
    
    print(f"\nHard Stop Reduction (Dual-PRV vs Formal-only):")
    print(f"  Absolute: {stop_reduction:.3f} fewer stops per trial")
    print(f"  Relative: {stop_reduction_pct:.1f}% reduction")
    
    print(f"\nCompletion Time Improvement:")
    print(f"  Absolute: {time_improvement:.2f}s faster per trial")
    print(f"  Relative: {time_improvement_pct:.1f}% improvement")
    
    print(f"\nTarget Check: Dual-PRV mean hard stops < 0.5?")
    if statistics.mean(dual_stops) < 0.5:
        print(f"  YES - {statistics.mean(dual_stops):.3f} < 0.5")
    else:
        print(f"  NO - {statistics.mean(dual_stops):.3f} >= 0.5")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Dual-Future PRV Simulator for HRC Pick-and-Place Task'
    )
    parser.add_argument('-n', '--n-trials', type=int, default=N_TRIALS,
                        help=f'Number of trials (default: {N_TRIALS})')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--show-trials', type=int, default=50,
                        help='Number of trial details to show (default: 50)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DUAL-FUTURE PRV SIMULATOR")
    print("Human-Robot Collaborative Pick-and-Place Task")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Slots: {SLOTS}")
    print(f"  Human: {HUMAN_MOVE_PLACE}s move+place, {HUMAN_DWELL}s dwell = {HUMAN_TOTAL}s total")
    print(f"  Robot (normal): {ROBOT_MOVE_PLACE_NORMAL}s move+place, {ROBOT_DWELL}s dwell = {ROBOT_TOTAL_NORMAL}s total")
    print(f"  Robot (slow): {ROBOT_MOVE_PLACE_SLOW}s move+place, {ROBOT_DWELL}s dwell = {ROBOT_TOTAL_SLOW}s total")
    print(f"  Hard stop penalty: {HARD_STOP_PENALTY}s")
    print(f"  Crash prediction window: {CRASH_PREDICTION_WINDOW}s")
    print(f"  Learned predictor accuracy: {LEARNED_ACCURACY*100:.0f}%")
    print(f"  Human plan change probability: {HUMAN_CHANGE_PROB*100:.0f}%")
    
    print(f"\nRunning {args.n_trials} trials...")
    print(f"  Random seed: {args.seed}")
    
    # Generate trials
    random.seed(args.seed)
    trials = generate_trials(args.n_trials)
    
    # Run experiments
    formal_results, dual_results = run_experiments(trials, seed=args.seed)
    
    # Print trial details
    print_trial_details(trials, formal_results, dual_results, n_show=args.show_trials)
    
    # Print summary
    print_summary(formal_results, dual_results)


if __name__ == '__main__':
    main()
