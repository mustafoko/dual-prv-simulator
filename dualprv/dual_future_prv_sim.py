#!/usr/bin/env python3
"""
Dual-Future Predictive Runtime Verification (PRV) Simulator
for Human-Robot Collaborative Pick-and-Place Task

This simulator implements:
- A discrete-time simulation of HRC with 4 shared slots (A, B, C, D)
- Dual-future PRV with formal (conservative) and learned (advisory) branches
- Fusion policy where learned advice can escalate to Advisory but never override Critical
- Experiments comparing formal-only baseline vs dual-PRV controller

Reference: Dual-Future PRV paper concepts for monotone fusion policy
"""

import random
import statistics
import csv
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import argparse

# Try to import matplotlib for optional plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Shared workspace slots
SLOTS = ['A', 'B', 'C', 'D']

# Time model
DELTA_T = 0.5  # Time step in seconds (0.5s for smoother overlap modeling)

# Human timing (in seconds)
HUMAN_TRAVEL_PLACE_MIN = 9.0
HUMAN_TRAVEL_PLACE_MAX = 11.78  # avg 10.39s
HUMAN_EXIT_MIN = 4.0
HUMAN_EXIT_MAX = 4.0

# Robot timing (in seconds)
ROBOT_TRAVEL_PLACE_NORMAL_MIN = 13.0
ROBOT_TRAVEL_PLACE_NORMAL_MAX = 17.86  # avg 15.43s
ROBOT_TRAVEL_PLACE_SLOW_MIN = 14.0  # avg 17s (1.57s penalty)
ROBOT_TRAVEL_PLACE_SLOW_MAX = 20.0
ROBOT_EXIT_MIN = 4.0
ROBOT_EXIT_MAX = 4.0

# Robot stop/resume parameters
ROBOT_STOP_LATENCY = 0.4  # seconds to fully stop
BASELINE_RESUME_PENALTY = 100.0  # extra seconds penalty after hard stop

# Speed recovery: remaining time multiplied by this factor when switching slow->normal
SPEED_RECOVERY_FACTOR = 0.4  # 60% reduction = multiply by 0.4

# PRV parameters
TAU_HARD = 1.0  # Hard threshold in seconds (trigger when close but with margin)
TAU_SOFT = 8.0  # Soft threshold for advisory - trigger early to prevent escalation
P_MIN = 0.60  # Minimum confidence for learned advisory
LEARNED_ACCURACY = 0.85  # Accuracy of learned predictor (oracle mode)

# Formal branch parameters
FORMAL_HORIZON_STEPS = 2  # Conservative prediction horizon in steps

# Learned branch parameters  
LEARNED_HORIZON_STEPS_MIN = 4  # Wider horizon for advisory
LEARNED_HORIZON_STEPS_MAX = 6


# =============================================================================
# ENUMS
# =============================================================================

class AgentStatus(Enum):
    """Status of an agent in the simulation."""
    IDLE = auto()
    MOVING = auto()
    PLACING = auto()
    EXITING = auto()
    STOPPED = auto()  # Robot only


class RobotMode(Enum):
    """Robot speed mode."""
    NORMAL = auto()
    SLOW = auto()


class PRVMode(Enum):
    """Fused PRV mode output."""
    NOMINAL = auto()
    ADVISORY = auto()
    CRITICAL = auto()


class ControllerType(Enum):
    """Type of controller being used."""
    FORMAL_ONLY = auto()
    DUAL_PRV = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AgentState:
    """State of a single agent (human or robot)."""
    goal_sequence: List[str]
    current_goal_idx: int = 0
    status: AgentStatus = AgentStatus.IDLE
    target_slot: Optional[str] = None
    remaining_time: float = 0.0
    at_slot: Dict[str, bool] = field(default_factory=lambda: {s: False for s in SLOTS})
    
    # Robot-specific fields
    mode: RobotMode = RobotMode.NORMAL
    stop_accumulator: float = 0.0  # For tracking stop latency
    
    @property
    def current_goal(self) -> Optional[str]:
        if self.current_goal_idx < len(self.goal_sequence):
            return self.goal_sequence[self.current_goal_idx]
        return None
    
    @property
    def completed(self) -> bool:
        return self.current_goal_idx >= len(self.goal_sequence)
    
    def clear_at_slots(self):
        for s in SLOTS:
            self.at_slot[s] = False


@dataclass
class PRVOutput:
    """Output from the PRV monitor."""
    # Formal branch
    delta_t_formal: float  # Estimated time to violation (inf if no risk)
    formal_critical: bool
    
    # Learned branch
    delta_t_learned: float  # Predicted time to conflict
    learned_advisory: bool
    confidence: float  # p_k
    predicted_human_slot: Optional[str]
    
    # Fused output
    fused_mode: PRVMode


@dataclass
class SimulationLog:
    """Log entry for a single simulation step."""
    t: float
    human_goal: Optional[str]
    human_status: AgentStatus
    human_target: Optional[str]
    human_remaining: float
    human_at_slots: Dict[str, bool]
    robot_goal: Optional[str]
    robot_status: AgentStatus
    robot_mode: RobotMode
    robot_target: Optional[str]
    robot_remaining: float
    robot_at_slots: Dict[str, bool]
    prv_output: PRVOutput
    

@dataclass
class RunStatistics:
    """Statistics from a single simulation run."""
    completion_time: float
    num_stops: int
    total_stop_duration: float
    num_slowdowns: int
    safety_violations: int
    human_goals_completed: int
    robot_goals_completed: int


# =============================================================================
# GOAL SEQUENCE GENERATION
# =============================================================================

def generate_goal_sequence(n_cycles: int = 3, allow_repeats: bool = False) -> List[str]:
    """
    Generate a goal sequence for an agent.
    
    Args:
        n_cycles: Number of times to cycle through all slots
        allow_repeats: If True, allows consecutive same slots
    
    Returns:
        List of slot names as goals
    """
    sequence = []
    for _ in range(n_cycles):
        cycle = SLOTS.copy()
        random.shuffle(cycle)
        sequence.extend(cycle)
    
    if not allow_repeats:
        # Remove consecutive duplicates
        filtered = [sequence[0]]
        for s in sequence[1:]:
            if s != filtered[-1]:
                filtered.append(s)
        return filtered
    
    return sequence


def generate_random_sequence(n_goals: int, allow_repeats: bool = False) -> List[str]:
    """
    Generate a random sequence of goals.
    
    Args:
        n_goals: Number of goals
        allow_repeats: If True, allows consecutive same slots
    
    Returns:
        List of slot names as goals
    """
    sequence = []
    for _ in range(n_goals):
        if allow_repeats or not sequence:
            sequence.append(random.choice(SLOTS))
        else:
            available = [s for s in SLOTS if s != sequence[-1]]
            sequence.append(random.choice(available))
    return sequence


# =============================================================================
# FILE I/O - GOALS AND TRACES
# =============================================================================

def load_goals_from_file(filepath: str) -> Tuple[List[str], List[str]]:
    """
    Load goal sequences from a text file.
    
    File format:
        # Comments start with #
        human: A, B, C, D, ...
        robot: B, C, A, D, ...
    
    Args:
        filepath: Path to the goals file
    
    Returns:
        Tuple of (human_goals, robot_goals)
    """
    human_goals = []
    robot_goals = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            if line.lower().startswith('human:'):
                goals_str = line.split(':', 1)[1]
                human_goals = [g.strip().upper() for g in goals_str.split(',')]
            elif line.lower().startswith('robot:'):
                goals_str = line.split(':', 1)[1]
                robot_goals = [g.strip().upper() for g in goals_str.split(',')]
    
    # Validate goals
    for goals, name in [(human_goals, 'human'), (robot_goals, 'robot')]:
        for g in goals:
            if g not in SLOTS:
                raise ValueError(f"Invalid slot '{g}' in {name} goals. Valid slots: {SLOTS}")
    
    if not human_goals:
        raise ValueError("No human goals found in file")
    if not robot_goals:
        raise ValueError("No robot goals found in file")
    
    return human_goals, robot_goals


def save_goals_to_file(filepath: str, human_goals: List[str], robot_goals: List[str]):
    """
    Save goal sequences to a text file.
    
    Args:
        filepath: Path to save the goals file
        human_goals: Human goal sequence
        robot_goals: Robot goal sequence
    """
    with open(filepath, 'w') as f:
        f.write("# Goal Sequences for Dual-Future PRV Simulation\n")
        f.write("# Format: One line per agent, comma-separated slot names\n")
        f.write(f"# Slots: {', '.join(SLOTS)}\n")
        f.write("# Lines starting with # are comments\n\n")
        f.write(f"human: {', '.join(human_goals)}\n\n")
        f.write(f"robot: {', '.join(robot_goals)}\n")


def save_trace_to_csv(filepath: str, logs: List['SimulationLog']):
    """
    Save simulation trace to a CSV file.
    
    Args:
        filepath: Path to save the CSV file
        logs: List of SimulationLog entries
    """
    if not logs:
        return
    
    fieldnames = [
        'time',
        'human_status', 'human_target', 'human_remaining',
        'human_at_A', 'human_at_B', 'human_at_C', 'human_at_D',
        'robot_status', 'robot_mode', 'robot_target', 'robot_remaining',
        'robot_at_A', 'robot_at_B', 'robot_at_C', 'robot_at_D',
        'prv_mode', 'delta_t_formal', 'delta_t_learned', 'confidence',
        'formal_critical', 'learned_advisory'
    ]
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for log in logs:
            row = {
                'time': f"{log.t:.2f}",
                'human_status': log.human_status.name,
                'human_target': log.human_target or '',
                'human_remaining': f"{log.human_remaining:.2f}",
                'human_at_A': log.human_at_slots.get('A', False),
                'human_at_B': log.human_at_slots.get('B', False),
                'human_at_C': log.human_at_slots.get('C', False),
                'human_at_D': log.human_at_slots.get('D', False),
                'robot_status': log.robot_status.name,
                'robot_mode': log.robot_mode.name,
                'robot_target': log.robot_target or '',
                'robot_remaining': f"{log.robot_remaining:.2f}",
                'robot_at_A': log.robot_at_slots.get('A', False),
                'robot_at_B': log.robot_at_slots.get('B', False),
                'robot_at_C': log.robot_at_slots.get('C', False),
                'robot_at_D': log.robot_at_slots.get('D', False),
                'prv_mode': log.prv_output.fused_mode.name,
                'delta_t_formal': f"{log.prv_output.delta_t_formal:.2f}" if log.prv_output.delta_t_formal < 1000 else 'inf',
                'delta_t_learned': f"{log.prv_output.delta_t_learned:.2f}" if log.prv_output.delta_t_learned < 1000 else 'inf',
                'confidence': f"{log.prv_output.confidence:.3f}",
                'formal_critical': log.prv_output.formal_critical,
                'learned_advisory': log.prv_output.learned_advisory
            }
            writer.writerow(row)


def load_trace_from_csv(filepath: str) -> List[Dict]:
    """
    Load simulation trace from a CSV file.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        List of dictionaries with trace data
    """
    traces = []
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string values to appropriate types
            trace = {
                'time': float(row['time']),
                'human_status': row['human_status'],
                'human_target': row['human_target'] or None,
                'human_remaining': float(row['human_remaining']),
                'human_at_slots': {
                    'A': row['human_at_A'] == 'True',
                    'B': row['human_at_B'] == 'True',
                    'C': row['human_at_C'] == 'True',
                    'D': row['human_at_D'] == 'True',
                },
                'robot_status': row['robot_status'],
                'robot_mode': row['robot_mode'],
                'robot_target': row['robot_target'] or None,
                'robot_remaining': float(row['robot_remaining']),
                'robot_at_slots': {
                    'A': row['robot_at_A'] == 'True',
                    'B': row['robot_at_B'] == 'True',
                    'C': row['robot_at_C'] == 'True',
                    'D': row['robot_at_D'] == 'True',
                },
                'prv_mode': row['prv_mode'],
                'delta_t_formal': float(row['delta_t_formal']) if row['delta_t_formal'] != 'inf' else float('inf'),
                'delta_t_learned': float(row['delta_t_learned']) if row['delta_t_learned'] != 'inf' else float('inf'),
                'confidence': float(row['confidence']),
                'formal_critical': row['formal_critical'] == 'True',
                'learned_advisory': row['learned_advisory'] == 'True'
            }
            traces.append(trace)
    return traces


# =============================================================================
# TIMING UTILITIES
# =============================================================================

def sample_human_travel_place() -> float:
    """Sample human travel+place duration."""
    return random.uniform(HUMAN_TRAVEL_PLACE_MIN, HUMAN_TRAVEL_PLACE_MAX)


def sample_human_exit() -> float:
    """Sample human exit duration."""
    return random.uniform(HUMAN_EXIT_MIN, HUMAN_EXIT_MAX)


def sample_robot_travel_place(mode: RobotMode) -> float:
    """Sample robot travel+place duration based on mode."""
    if mode == RobotMode.NORMAL:
        return random.uniform(ROBOT_TRAVEL_PLACE_NORMAL_MIN, ROBOT_TRAVEL_PLACE_NORMAL_MAX)
    else:  # SLOW
        return random.uniform(ROBOT_TRAVEL_PLACE_SLOW_MIN, ROBOT_TRAVEL_PLACE_SLOW_MAX)


def sample_robot_exit() -> float:
    """Sample robot exit duration."""
    return random.uniform(ROBOT_EXIT_MIN, ROBOT_EXIT_MAX)


# =============================================================================
# PRV MONITOR
# =============================================================================

class PRVMonitor:
    """
    Dual-Future PRV Monitor with formal and learned branches.
    
    Formal branch: Conservative reachability analysis (safety-authoritative)
    Learned branch: Probabilistic intent prediction (advisory)
    Fusion: Monotone policy - formal Critical always wins
    """
    
    def __init__(self, learned_accuracy: float = LEARNED_ACCURACY):
        self.learned_accuracy = learned_accuracy
    
    def evaluate(self, human: AgentState, robot: AgentState, 
                 controller_type: ControllerType) -> PRVOutput:
        """
        Evaluate PRV and return fused mode.
        
        Args:
            human: Current human state
            robot: Current robot state
            controller_type: Type of controller (affects whether learned is used)
        
        Returns:
            PRVOutput with all branch outputs and fused mode
        """
        # Formal branch evaluation
        delta_t_formal, formal_critical = self._evaluate_formal(human, robot)
        
        # Learned branch evaluation
        delta_t_learned, learned_advisory, confidence, predicted_slot = \
            self._evaluate_learned(human, robot)
        
        # Fusion policy
        fused_mode = self._fuse(formal_critical, learned_advisory, 
                                confidence, controller_type)
        
        return PRVOutput(
            delta_t_formal=delta_t_formal,
            formal_critical=formal_critical,
            delta_t_learned=delta_t_learned,
            learned_advisory=learned_advisory,
            confidence=confidence,
            predicted_human_slot=predicted_slot,
            fused_mode=fused_mode
        )
    
    def _evaluate_formal(self, human: AgentState, robot: AgentState) -> Tuple[float, bool]:
        """
        Formal branch: Conservative reachability analysis.
        
        Returns:
            Tuple of (delta_t_formal, critical_flag)
        """
        formal_critical = False
        delta_t_formal = float('inf')
        
        # Check for current violation (both at same slot) - immediate critical
        for slot in SLOTS:
            if human.at_slot.get(slot, False) and robot.at_slot.get(slot, False):
                return 0.0, True
        
        # No risk if either has no target
        if human.target_slot is None or robot.target_slot is None:
            return float('inf'), False
        
        # Check if robot is heading to a slot where human is present
        if robot.status == AgentStatus.MOVING:
            robot_target = robot.target_slot
            
            # Case 1: Human is currently at robot's target slot (PLACING or EXITING)
            if human.at_slot.get(robot_target, False):
                # Robot approaching occupied slot - always critical
                delta_t_formal = robot.remaining_time
                formal_critical = True
                return delta_t_formal, formal_critical
            
            # Case 2: Human is heading to same slot
            if human.target_slot == robot_target:
                # Compute time estimates
                
                # Human's expected arrival time
                if human.status == AgentStatus.PLACING:
                    human_earliest = 0.0  # Already there
                elif human.status == AgentStatus.EXITING:
                    human_earliest = 0.0  # Still at slot
                elif human.status == AgentStatus.MOVING:
                    human_earliest = human.remaining_time  # Use actual remaining time
                elif human.status == AgentStatus.IDLE and human.current_goal == robot_target:
                    human_earliest = HUMAN_TRAVEL_PLACE_MIN
                else:
                    human_earliest = float('inf')
                
                # Robot arrival time + time to complete placement
                robot_arrival = robot.remaining_time
                robot_exit_time = robot_arrival + ROBOT_EXIT_MIN  # Time for robot to finish and leave
                
                # KEY FIX: Only conflict if robot and human would be at slot simultaneously
                # If robot finishes BEFORE human arrives, no conflict!
                if robot_exit_time < human_earliest:
                    # Robot will finish and leave before human arrives - NO CONFLICT
                    delta_t_formal = float('inf')
                    formal_critical = False
                else:
                    # Potential overlap - check timing
                    delta_t_formal = min(human_earliest, robot_arrival)
                    
                    # Critical if human arrives before robot can finish (consistent evaluation)
                    if human_earliest <= robot_exit_time and delta_t_formal <= TAU_HARD:
                        formal_critical = True
        
        return delta_t_formal, formal_critical
    
    def _evaluate_learned(self, human: AgentState, robot: AgentState) \
            -> Tuple[float, bool, float, Optional[str]]:
        """
        Learned branch: Probabilistic intent prediction.
        
        Returns:
            Tuple of (delta_t_learned, advisory_flag, confidence, predicted_slot)
        """
        # Predict human's target slot with accuracy
        true_target = human.target_slot or human.current_goal
        
        if true_target is None:
            return float('inf'), False, 0.0, None
        
        # Oracle prediction with specified accuracy
        if random.random() < self.learned_accuracy:
            predicted_slot = true_target
            confidence = self.learned_accuracy
        else:
            # Wrong prediction
            other_slots = [s for s in SLOTS if s != true_target]
            predicted_slot = random.choice(other_slots)
            confidence = (1.0 - self.learned_accuracy) / 3.0  # Spread among 3 other slots
        
        # Check for conflict within advisory horizon
        if predicted_slot != robot.target_slot:
            return float('inf'), False, confidence, predicted_slot
        
        # Same predicted target - estimate time to conflict
        # Learned uses expected times, not worst-case
        
        human_expected_arrival = float('inf')
        if human.status == AgentStatus.MOVING:
            human_expected_arrival = human.remaining_time
        elif human.status == AgentStatus.PLACING:
            human_expected_arrival = 0.0
        elif human.status == AgentStatus.EXITING:
            human_expected_arrival = 0.0  # Still at slot
        elif human.status == AgentStatus.IDLE and human.current_goal == predicted_slot:
            # Human will start moving soon
            human_expected_arrival = (HUMAN_TRAVEL_PLACE_MIN + HUMAN_TRAVEL_PLACE_MAX) / 2
        
        robot_expected_arrival = float('inf')
        if robot.status == AgentStatus.MOVING:
            robot_expected_arrival = robot.remaining_time
        elif robot.status == AgentStatus.PLACING:
            robot_expected_arrival = 0.0
        
        # Time to learned conflict
        delta_t_learned = min(human_expected_arrival, robot_expected_arrival)
        
        # Robot's time to finish and exit the slot
        robot_exit_time = robot_expected_arrival + ROBOT_EXIT_MIN
        
        # KEY: Only trigger advisory if there's actual overlap potential
        # If robot finishes and exits before human arrives, no conflict!
        potential_conflict = (robot_exit_time >= human_expected_arrival)
        
        # Advisory if within soft horizon, confidence meets threshold, AND potential conflict
        advisory_horizon = TAU_SOFT
        learned_advisory = (
            potential_conflict and  # Only if robot can't finish first
            delta_t_learned <= advisory_horizon and 
            confidence >= P_MIN and
            robot.status == AgentStatus.MOVING
            # Note: Allow advisory even if already in SLOW mode - new conflict may need more delay
        )
        
        return delta_t_learned, learned_advisory, confidence, predicted_slot
    
    def _fuse(self, formal_critical: bool, learned_advisory: bool, 
              confidence: float, controller_type: ControllerType) -> PRVMode:
        """
        Monotone fusion policy.
        
        Formal Critical always wins (safety-authoritative).
        Learned Advisory only activates in DUAL_PRV mode with sufficient confidence.
        """
        # Formal Critical overrides everything
        if formal_critical:
            return PRVMode.CRITICAL
        
        # Learned Advisory only in dual-PRV mode
        if controller_type == ControllerType.DUAL_PRV:
            if learned_advisory and confidence >= P_MIN:
                return PRVMode.ADVISORY
        
        return PRVMode.NOMINAL


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

class Simulation:
    """
    Discrete-time simulation of HRC pick-and-place task with PRV.
    
    COOPERATIVE TASK: Both agents work together to visit all 4 slots.
    A slot only needs to be visited once by either agent.
    Task completes when all slots (A, B, C, D) have been visited.
    """
    
    def __init__(self, human_goals: List[str], robot_goals: List[str],
                 controller_type: ControllerType,
                 learned_accuracy: float = LEARNED_ACCURACY):
        """
        Initialize simulation.
        
        Args:
            human_goals: Initial goal preferences for human (may be reassigned)
            robot_goals: Initial goal preferences for robot (may be reassigned)
            controller_type: FORMAL_ONLY or DUAL_PRV
            learned_accuracy: Accuracy of learned predictor
        """
        self.human = AgentState(goal_sequence=human_goals.copy())
        self.robot = AgentState(goal_sequence=robot_goals.copy())
        self.controller_type = controller_type
        self.prv = PRVMonitor(learned_accuracy=learned_accuracy)
        
        self.t = 0.0
        self.logs: List[SimulationLog] = []
        
        # COOPERATIVE: Track which slots have been visited by ANYONE
        self.visited_slots: set = set()
        
        # Statistics
        self.num_stops = 0
        self.total_stop_duration = 0.0
        self.num_slowdowns = 0
        self.safety_violations = 0
        
        # State tracking for events
        self.robot_was_slow = False
        self.robot_stop_start_time: Optional[float] = None
    
    def _task_completed(self) -> bool:
        """Check if all slots have been visited."""
        return self.visited_slots == set(SLOTS)
    
    def _get_unvisited_slots(self) -> List[str]:
        """Get list of slots that haven't been visited yet."""
        return [s for s in SLOTS if s not in self.visited_slots]
    
    def run(self, max_time: float = 500.0) -> RunStatistics:
        """
        Run the simulation until all slots visited or max time.
        
        Args:
            max_time: Maximum simulation time
        
        Returns:
            RunStatistics for this run
        """
        # Start both agents with their first unvisited slot
        self._assign_next_goal(self.human, is_human=True)
        self._assign_next_goal(self.robot, is_human=False)
        
        while self.t < max_time:
            # Check for task completion (all slots visited)
            if self._task_completed():
                break
            
            # Evaluate PRV
            prv_output = self.prv.evaluate(self.human, self.robot, self.controller_type)
            
            # Log current state
            self._log_state(prv_output)
            
            # Check for safety violations
            self._check_safety()
            
            # Apply mitigation based on PRV output
            self._apply_mitigation(prv_output)
            
            # Advance simulation
            self._step()
            
            self.t += DELTA_T
        
        # Calculate effective completion time:
        # When robot stops, we lose parallel work capacity.
        # Each hard stop has a penalty (BASELINE_RESUME_PENALTY).
        # This penalty time represents work that robot couldn't do.
        # Human must compensate, but can't fully replace parallel work.
        # Lost efficiency = num_stops * penalty * collaboration_factor
        collaboration_loss_factor = 0.4  # Robot contributes ~40% of parallel work
        penalty_overhead = self.num_stops * BASELINE_RESUME_PENALTY * collaboration_loss_factor
        effective_completion = self.t + penalty_overhead
        
        return RunStatistics(
            completion_time=effective_completion,
            num_stops=self.num_stops,
            total_stop_duration=self.total_stop_duration,
            num_slowdowns=self.num_slowdowns,
            safety_violations=self.safety_violations,
            human_goals_completed=len([s for s in self.visited_slots]),
            robot_goals_completed=len([s for s in self.visited_slots])
        )
    
    def _assign_next_goal(self, agent: AgentState, is_human: bool):
        """Assign agent to visit an unvisited slot, preferring their goal sequence order."""
        unvisited = self._get_unvisited_slots()
        
        if not unvisited:
            # All slots visited - task complete
            agent.status = AgentStatus.IDLE
            agent.target_slot = None
            return
        
        # Pick from unvisited slots, preferring agent's goal sequence order
        target = None
        for goal in agent.goal_sequence[agent.current_goal_idx:]:
            if goal in unvisited:
                target = goal
                break
        
        # If preferred goals all done, pick any remaining unvisited
        if target is None and unvisited:
            target = unvisited[0]
        
        if target is None:
            agent.status = AgentStatus.IDLE
            agent.target_slot = None
            return
        
        agent.target_slot = target
        agent.status = AgentStatus.MOVING
        
        if is_human:
            agent.remaining_time = sample_human_travel_place()
        else:
            agent.remaining_time = sample_robot_travel_place(agent.mode)
    
    def _start_next_goal(self, agent: AgentState, is_human: bool):
        """Start agent moving toward their next goal (wrapper for compatibility)."""
        self._assign_next_goal(agent, is_human)
    
    def _step(self):
        """Advance simulation by one time step."""
        # Update human first (human has priority)
        self._step_agent(self.human, is_human=True)
        
        # Before robot step: check if robot would enter occupied slot
        # This is a safety interlock at the physical layer
        if (self.robot.status == AgentStatus.MOVING and 
            self.robot.remaining_time <= DELTA_T and
            self.robot.target_slot):
            target = self.robot.target_slot
            if self.human.at_slot.get(target, False):
                # Robot must wait - cannot enter occupied slot
                # Force a stop if not already stopped
                if self.robot.status != AgentStatus.STOPPED:
                    self.robot.status = AgentStatus.STOPPED
                    if self.robot_stop_start_time is None:
                        self.num_stops += 1
                        self.robot_stop_start_time = self.t
                return  # Don't step robot this cycle
        
        # Update robot
        self._step_agent(self.robot, is_human=False)
    
    def _step_agent(self, agent: AgentState, is_human: bool):
        """Step a single agent through time."""
        if agent.status == AgentStatus.IDLE:
            # Try to start next goal
            if not agent.completed:
                self._start_next_goal(agent, is_human)
            return
        
        if agent.status == AgentStatus.STOPPED:
            # Robot only - check if can resume
            if not is_human:
                contested_slot = agent.target_slot
                
                # Can resume when it's SAFE:
                # 1. Human is not physically at the slot, AND
                # 2. Human is not heading to the slot (or has different target)
                human_at_slot = self.human.at_slot.get(contested_slot, False)
                human_heading_to_slot = (self.human.target_slot == contested_slot and 
                                         self.human.status == AgentStatus.MOVING)
                
                can_resume = contested_slot and not human_at_slot and not human_heading_to_slot
                
                if can_resume:
                    # Record stop duration BEFORE adding penalty
                    if self.robot_stop_start_time is not None:
                        self.total_stop_duration += (self.t - self.robot_stop_start_time)
                        self.robot_stop_start_time = None
                    
                    # Check if contested slot was already visited by human
                    if contested_slot in self.visited_slots:
                        # Slot already done - reassign to unvisited slot
                        agent.status = AgentStatus.IDLE
                        self._assign_next_goal(agent, is_human=False)
                        # Add penalty for the stop
                        agent.remaining_time += BASELINE_RESUME_PENALTY
                    else:
                        # Continue to same slot with penalty
                        agent.remaining_time += BASELINE_RESUME_PENALTY
                        agent.status = AgentStatus.MOVING
            return
        
        # Decrement remaining time
        agent.remaining_time -= DELTA_T
        
        if agent.remaining_time <= 0:
            # Transition to next phase
            if agent.status == AgentStatus.MOVING:
                # About to arrive at slot
                target = agent.target_slot
                
                # For robot: check if human is at the target slot - must wait
                if not is_human and target:
                    if self.human.at_slot.get(target, False):
                        # Cannot enter - human is there. Treat as stopped at boundary.
                        agent.remaining_time = 0.1  # Wait at boundary
                        return
                
                # Safe to enter - start placing
                agent.status = AgentStatus.PLACING
                agent.clear_at_slots()
                if target:
                    agent.at_slot[target] = True
                # Place is instantaneous, then exit
                if is_human:
                    agent.remaining_time = sample_human_exit()
                else:
                    agent.remaining_time = sample_robot_exit()
                agent.status = AgentStatus.EXITING
                
            elif agent.status == AgentStatus.PLACING:
                # This shouldn't happen with current logic
                agent.status = AgentStatus.EXITING
                if is_human:
                    agent.remaining_time = sample_human_exit()
                else:
                    agent.remaining_time = sample_robot_exit()
                    
            elif agent.status == AgentStatus.EXITING:
                # Done with this goal - mark slot as visited
                completed_slot = agent.target_slot
                if completed_slot:
                    self.visited_slots.add(completed_slot)
                
                agent.clear_at_slots()
                agent.current_goal_idx += 1
                agent.status = AgentStatus.IDLE
                agent.target_slot = None
                
                # Reset robot mode to normal after completing action
                if not is_human:
                    agent.mode = RobotMode.NORMAL
                
                # Assign next unvisited slot (cooperative task)
                if not self._task_completed():
                    self._assign_next_goal(agent, is_human)
    
    def _apply_mitigation(self, prv_output: PRVOutput):
        """Apply mitigation based on PRV mode."""
        if prv_output.fused_mode == PRVMode.CRITICAL:
            # Hard stop - only process if robot is MOVING (not already STOPPED)
            if self.robot.status == AgentStatus.MOVING:
                # Account for stop latency - only increment if not already triggered
                self.robot.stop_accumulator += DELTA_T
                if self.robot.stop_accumulator >= ROBOT_STOP_LATENCY:
                    self.robot.status = AgentStatus.STOPPED
                    self.robot.stop_accumulator = 0.0
                    # Only count as hard stop if slow-down wasn't already handling this conflict
                    if not self.robot_was_slow:
                        self.num_stops += 1
                    # Reset slow mode tracking ONLY when stop actually happens
                    self.robot_was_slow = False
                    self.robot_stop_start_time = self.t
                    
        elif prv_output.fused_mode == PRVMode.ADVISORY:
            # Slow down - give human time to pass
            if self.robot.status == AgentStatus.MOVING:
                was_normal = (self.robot.mode == RobotMode.NORMAL)
                self.robot.mode = RobotMode.SLOW
                
                # Add enough delay for human to complete and exit the slot
                human_slot_time = HUMAN_EXIT_MAX + 2.0  # Time human needs to exit + buffer
                
                # Calculate delay needed for human to finish at contested slot
                if self.human.status == AgentStatus.MOVING:
                    # Human arriving + time at slot
                    human_done_time = self.human.remaining_time + human_slot_time
                elif self.human.status in [AgentStatus.PLACING, AgentStatus.EXITING]:
                    # Human already at slot, just wait for exit
                    human_done_time = human_slot_time
                else:
                    human_done_time = human_slot_time
                
                # Robot should arrive AFTER human is done
                if self.robot.remaining_time < human_done_time:
                    self.robot.remaining_time = human_done_time + 1.0  # Arrive after human leaves
                
                # Only count as new slow-down if transitioning from NORMAL
                if was_normal:
                    self.num_slowdowns += 1
                self.robot_was_slow = True
            # Reset stop accumulator
            self.robot.stop_accumulator = 0.0
                
        else:  # NOMINAL
            # Reset stop accumulator
            self.robot.stop_accumulator = 0.0
            
            # Return to normal if was slow
            if self.robot.mode == RobotMode.SLOW and self.robot.status == AgentStatus.MOVING:
                # Check if human has cleared the contested slot or is no longer heading there
                contested_slot = self.robot.target_slot
                human_cleared = True
                if contested_slot:
                    # Human cleared if not at slot and not heading to same slot
                    human_at_slot = self.human.at_slot.get(contested_slot, False)
                    human_heading_there = (self.human.target_slot == contested_slot and 
                                          self.human.status == AgentStatus.MOVING)
                    human_cleared = not human_at_slot and not human_heading_there
                
                if human_cleared:
                    # Speed recovery: reduce remaining time by 60%
                    self.robot.remaining_time *= SPEED_RECOVERY_FACTOR
                    self.robot.mode = RobotMode.NORMAL
                    self.robot_was_slow = False
    
    def _check_safety(self):
        """Check for safety violations (mutual exclusion)."""
        for slot in SLOTS:
            if self.human.at_slot.get(slot, False) and self.robot.at_slot.get(slot, False):
                # This should not happen with proper enforcement
                self.safety_violations += 1
                # Force robot to stop immediately as emergency measure
                if self.robot.status != AgentStatus.STOPPED:
                    self.robot.status = AgentStatus.STOPPED
                    self.robot.clear_at_slots()
                    if self.robot_stop_start_time is None:
                        self.robot_stop_start_time = self.t
    
    def _log_state(self, prv_output: PRVOutput):
        """Log current simulation state."""
        log = SimulationLog(
            t=self.t,
            human_goal=self.human.current_goal,
            human_status=self.human.status,
            human_target=self.human.target_slot,
            human_remaining=self.human.remaining_time,
            human_at_slots=self.human.at_slot.copy(),
            robot_goal=self.robot.current_goal,
            robot_status=self.robot.status,
            robot_mode=self.robot.mode,
            robot_target=self.robot.target_slot,
            robot_remaining=self.robot.remaining_time,
            robot_at_slots=self.robot.at_slot.copy(),
            prv_output=prv_output
        )
        self.logs.append(log)


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """
    Run Monte-Carlo experiments comparing controllers.
    """
    
    def __init__(self, n_trials: int = 100, n_goals_per_agent: int = 12,
                 learned_accuracy: float = LEARNED_ACCURACY, seed: Optional[int] = None):
        """
        Initialize experiment runner.
        
        Args:
            n_trials: Number of Monte-Carlo trials per controller
            n_goals_per_agent: Number of goals per agent per trial
            learned_accuracy: Accuracy of learned predictor
            seed: Random seed for reproducibility
        """
        self.n_trials = n_trials
        self.n_goals_per_agent = n_goals_per_agent
        self.learned_accuracy = learned_accuracy
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
    
    def run_experiments(self) -> Dict[str, List[RunStatistics]]:
        """
        Run experiments for both controllers.
        
        Returns:
            Dict mapping controller name to list of RunStatistics
        """
        results = {
            'formal_only': [],
            'dual_prv': []
        }
        
        # Track previous human order for robot to avoid
        prev_human_order = ['A', 'B', 'C', 'D']  # Initial human order
        
        # Track previous human order for robot to avoid
        prev_human_order = ['A', 'B', 'C', 'D']  # Initial human order
        
        for trial in range(self.n_trials):
            # HUMAN BEHAVIOR:
            # - First trial: A,B,C,D
            # - Next trials: 80% keep same order, 20% change randomly
            if trial == 0:
                human_goals = ['A', 'B', 'C', 'D']
            else:
                if random.random() < 0.8:
                    # 80%: Keep same order as previous trial
                    human_goals = prev_human_order.copy()
                else:
                    # 20%: Change to random order
                    human_goals = SLOTS.copy()
                    random.shuffle(human_goals)
            
            # ROBOT BEHAVIOR:
            # Use OPPOSITE of PREVIOUS human order (creates some conflicts when human changes)
            robot_goals = prev_human_order[::-1]
            
            # Save human order for next trial
            prev_human_order = human_goals.copy()
            
            # Run formal-only baseline
            if self.seed is not None:
                random.seed(self.seed + trial * 1000)
            sim_formal = Simulation(
                human_goals=human_goals,
                robot_goals=robot_goals,
                controller_type=ControllerType.FORMAL_ONLY,
                learned_accuracy=self.learned_accuracy
            )
            results['formal_only'].append(sim_formal.run())
            
            # Run dual-PRV
            if self.seed is not None:
                random.seed(self.seed + trial * 1000)
            sim_dual = Simulation(
                human_goals=human_goals,
                robot_goals=robot_goals,
                controller_type=ControllerType.DUAL_PRV,
                learned_accuracy=self.learned_accuracy
            )
            results['dual_prv'].append(sim_dual.run())
            
            if (trial + 1) % 20 == 0:
                print(f"  Completed {trial + 1}/{self.n_trials} trials...")
        
        return results
    
    def print_summary(self, results: Dict[str, List[RunStatistics]]):
        """Print summary statistics."""
        print("\n" + "=" * 70)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 70)
        
        for controller_name, stats_list in results.items():
            print(f"\n{controller_name.upper().replace('_', ' ')}:")
            print("-" * 40)
            
            completion_times = [s.completion_time for s in stats_list]
            num_stops = [s.num_stops for s in stats_list]
            stop_durations = [s.total_stop_duration for s in stats_list]
            safety_violations = [s.safety_violations for s in stats_list]
            
            print(f"  Completion Time:")
            print(f"    Mean: {statistics.mean(completion_times):.2f}s")
            print(f"    Std:  {statistics.stdev(completion_times):.2f}s")
            print(f"    Min:  {min(completion_times):.2f}s")
            print(f"    Max:  {max(completion_times):.2f}s")
            
            print(f"  Hard Stops:")
            print(f"    Mean: {statistics.mean(num_stops):.2f}")
            print(f"    Std:  {statistics.stdev(num_stops):.2f}")
            
            print(f"  Total Stop Duration:")
            print(f"    Mean: {statistics.mean(stop_durations):.2f}s")
            print(f"    Std:  {statistics.stdev(stop_durations):.2f}s")
            
            if controller_name == 'dual_prv':
                num_slowdowns = [s.num_slowdowns for s in stats_list]
                print(f"  Slow-down Events:")
                print(f"    Mean: {statistics.mean(num_slowdowns):.2f}")
                print(f"    Std:  {statistics.stdev(num_slowdowns):.2f}")
            
            print(f"  Safety Violations: {sum(safety_violations)}")
        
        # Comparison
        formal_times = [s.completion_time for s in results['formal_only']]
        dual_times = [s.completion_time for s in results['dual_prv']]
        
        improvement = statistics.mean(formal_times) - statistics.mean(dual_times)
        improvement_pct = (improvement / statistics.mean(formal_times)) * 100
        
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        print(f"Completion Time Improvement (Dual-PRV vs Formal-only):")
        print(f"  Absolute: {improvement:.2f}s faster")
        print(f"  Relative: {improvement_pct:.1f}% improvement")
        
        # Statistical significance (simple t-test approximation)
        n = len(formal_times)
        mean_diff = improvement
        var_formal = statistics.variance(formal_times)
        var_dual = statistics.variance(dual_times)
        se = ((var_formal + var_dual) / n) ** 0.5
        t_stat = mean_diff / se if se > 0 else float('inf')
        
        print(f"  T-statistic: {t_stat:.2f}")
        print(f"  (|t| > 2 suggests statistically significant difference)")
    
    def plot_results(self, results: Dict[str, List[RunStatistics]], 
                     output_file: Optional[str] = None):
        """Plot completion time distributions."""
        if not MATPLOTLIB_AVAILABLE:
            print("\nMatplotlib not available - skipping plots")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Completion times histogram
        ax1 = axes[0, 0]
        formal_times = [s.completion_time for s in results['formal_only']]
        dual_times = [s.completion_time for s in results['dual_prv']]
        
        ax1.hist(formal_times, bins=20, alpha=0.7, label='Formal-only', color='red')
        ax1.hist(dual_times, bins=20, alpha=0.7, label='Dual-PRV', color='blue')
        ax1.set_xlabel('Completion Time (s)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Completion Time Distribution')
        ax1.legend()
        ax1.axvline(statistics.mean(formal_times), color='red', linestyle='--', 
                   label=f'Formal mean: {statistics.mean(formal_times):.1f}s')
        ax1.axvline(statistics.mean(dual_times), color='blue', linestyle='--',
                   label=f'Dual mean: {statistics.mean(dual_times):.1f}s')
        
        # Number of stops
        ax2 = axes[0, 1]
        formal_stops = [s.num_stops for s in results['formal_only']]
        dual_stops = [s.num_stops for s in results['dual_prv']]
        
        x = ['Formal-only', 'Dual-PRV']
        means = [statistics.mean(formal_stops), statistics.mean(dual_stops)]
        stds = [statistics.stdev(formal_stops), statistics.stdev(dual_stops)]
        
        ax2.bar(x, means, yerr=stds, capsize=5, color=['red', 'blue'], alpha=0.7)
        ax2.set_ylabel('Number of Hard Stops')
        ax2.set_title('Hard Stops per Run (mean ± std)')
        
        # Stop duration
        ax3 = axes[1, 0]
        formal_dur = [s.total_stop_duration for s in results['formal_only']]
        dual_dur = [s.total_stop_duration for s in results['dual_prv']]
        
        means = [statistics.mean(formal_dur), statistics.mean(dual_dur)]
        stds = [statistics.stdev(formal_dur), statistics.stdev(dual_dur)]
        
        ax3.bar(x, means, yerr=stds, capsize=5, color=['red', 'blue'], alpha=0.7)
        ax3.set_ylabel('Total Stop Duration (s)')
        ax3.set_title('Stop Duration per Run (mean ± std)')
        
        # Slow-downs (dual only)
        ax4 = axes[1, 1]
        dual_slow = [s.num_slowdowns for s in results['dual_prv']]
        
        ax4.hist(dual_slow, bins=15, alpha=0.7, color='green')
        ax4.set_xlabel('Number of Slow-downs')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Dual-PRV Slow-down Events (mean: {statistics.mean(dual_slow):.1f})')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {output_file}")
        else:
            plt.show()


# =============================================================================
# TIME-SERIES LOGGING
# =============================================================================

def print_time_series_log(logs: List[SimulationLog], max_entries: int = 50):
    """Print time series log for debugging/analysis."""
    print("\n" + "=" * 100)
    print("TIME-SERIES LOG (first {} entries)".format(min(len(logs), max_entries)))
    print("=" * 100)
    
    header = (f"{'t':>6} | {'H_status':<10} | {'H_target':<8} | {'H_rem':>6} | "
              f"{'R_status':<10} | {'R_mode':<6} | {'R_target':<8} | {'R_rem':>6} | "
              f"{'PRV_mode':<10} | {'dt_f':>6} | {'dt_l':>6}")
    print(header)
    print("-" * 100)
    
    for log in logs[:max_entries]:
        h_target = log.human_target or '-'
        r_target = log.robot_target or '-'
        prv = log.prv_output
        dt_f = f"{prv.delta_t_formal:.1f}" if prv.delta_t_formal < 1000 else "inf"
        dt_l = f"{prv.delta_t_learned:.1f}" if prv.delta_t_learned < 1000 else "inf"
        
        row = (f"{log.t:>6.1f} | {log.human_status.name:<10} | {h_target:<8} | "
               f"{log.human_remaining:>6.1f} | {log.robot_status.name:<10} | "
               f"{log.robot_mode.name:<6} | {r_target:<8} | {log.robot_remaining:>6.1f} | "
               f"{prv.fused_mode.name:<10} | {dt_f:>6} | {dt_l:>6}")
        print(row)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Dual-Future PRV Simulator for HRC Pick-and-Place Task'
    )
    parser.add_argument('-n', '--n-trials', type=int, default=100,
                        help='Number of Monte-Carlo trials (default: 100)')
    parser.add_argument('-g', '--n-goals', type=int, default=12,
                        help='Number of goals per agent (default: 12)')
    parser.add_argument('-a', '--accuracy', type=float, default=0.85,
                        help='Learned predictor accuracy (default: 0.85)')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting')
    parser.add_argument('--plot-file', type=str, default=None,
                        help='Save plot to file instead of displaying')
    parser.add_argument('--single-run', action='store_true',
                        help='Run single simulation with detailed logging')
    parser.add_argument('--goals-file', type=str, default=None,
                        help='Load goals from text file (e.g., data/goals.txt)')
    parser.add_argument('--trace-file', type=str, default=None,
                        help='Save simulation trace to CSV file')
    parser.add_argument('--save-goals', type=str, default=None,
                        help='Save generated goals to text file')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DUAL-FUTURE PRV SIMULATOR")
    print("Human-Robot Collaborative Pick-and-Place Task")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Time step (Δt): {DELTA_T}s")
    print(f"  Slots: {SLOTS}")
    print(f"  Learned accuracy: {args.accuracy}")
    print(f"  PRV thresholds: τ_hard={TAU_HARD}s, τ_soft={TAU_SOFT}s, p_min={P_MIN}")
    
    if args.single_run:
        # Single detailed run for debugging
        print("\n" + "=" * 70)
        print("SINGLE RUN (Detailed)")
        print("=" * 70)
        
        random.seed(args.seed)
        
        # Load goals from file or generate
        if args.goals_file:
            print(f"\nLoading goals from: {args.goals_file}")
            human_goals, robot_goals = load_goals_from_file(args.goals_file)
        else:
            # COOPERATIVE TASK: Both work to visit all 4 slots
            # Goals are preferred order - task completes when all slots visited
            all_slots = SLOTS.copy()
            random.shuffle(all_slots)
            human_goals = all_slots.copy()
            
            all_slots_robot = SLOTS.copy()
            random.shuffle(all_slots_robot)
            robot_goals = all_slots_robot
        
        print(f"\nHuman goals: {human_goals}")
        print(f"Robot goals: {robot_goals}")
        
        # Save goals if requested
        if args.save_goals:
            save_goals_to_file(args.save_goals, human_goals, robot_goals)
            print(f"Goals saved to: {args.save_goals}")
        
        # Run dual-PRV
        sim = Simulation(
            human_goals=human_goals,
            robot_goals=robot_goals,
            controller_type=ControllerType.DUAL_PRV,
            learned_accuracy=args.accuracy
        )
        stats = sim.run()
        
        print(f"\nResults:")
        print(f"  Completion time: {stats.completion_time:.1f}s")
        print(f"  Hard stops: {stats.num_stops}")
        print(f"  Slow-downs: {stats.num_slowdowns}")
        print(f"  Safety violations: {stats.safety_violations}")
        
        # Save trace to CSV if requested
        if args.trace_file:
            save_trace_to_csv(args.trace_file, sim.logs)
            print(f"\nTrace saved to: {args.trace_file}")
        
        print_time_series_log(sim.logs)
        
    else:
        # Monte-Carlo experiments
        print(f"\nRunning Monte-Carlo experiments...")
        print(f"  Trials: {args.n_trials}")
        print(f"  Goals per agent: {args.n_goals}")
        print(f"  Random seed: {args.seed}")
        
        # Note: goals-file not used in MC mode (each trial generates random goals)
        if args.goals_file:
            print(f"  Note: --goals-file ignored in Monte-Carlo mode")
        
        runner = ExperimentRunner(
            n_trials=args.n_trials,
            n_goals_per_agent=args.n_goals,
            learned_accuracy=args.accuracy,
            seed=args.seed
        )
        
        results = runner.run_experiments()
        runner.print_summary(results)
        
        if not args.no_plot:
            runner.plot_results(results, output_file=args.plot_file)


if __name__ == '__main__':
    main()
