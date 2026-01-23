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
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional
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
HUMAN_TRAVEL_PLACE_MIN = 4.0
HUMAN_TRAVEL_PLACE_MAX = 7.0
HUMAN_EXIT_MIN = 1.0
HUMAN_EXIT_MAX = 2.0

# Robot timing (in seconds)
ROBOT_TRAVEL_PLACE_NORMAL_MIN = 5.0
ROBOT_TRAVEL_PLACE_NORMAL_MAX = 8.0
ROBOT_TRAVEL_PLACE_SLOW_MIN = 6.0
ROBOT_TRAVEL_PLACE_SLOW_MAX = 8.0
ROBOT_EXIT_MIN = 1.0
ROBOT_EXIT_MAX = 2.0

# Robot stop/resume parameters
ROBOT_STOP_LATENCY = 0.4  # seconds to fully stop
BASELINE_RESUME_PENALTY = 2.0  # extra seconds penalty in baseline after stop

# Speed recovery: remaining time multiplied by this factor when switching slow->normal
SPEED_RECOVERY_FACTOR = 0.4  # 60% reduction = multiply by 0.4

# PRV parameters
TAU_HARD = 1.5  # Hard threshold in seconds (tight for safety)
TAU_SOFT = 5.0  # Soft threshold for advisory in seconds (wider for early warning)
P_MIN = 0.65  # Minimum confidence for learned advisory
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
                # Robot approaching occupied slot - immediate critical
                delta_t_formal = robot.remaining_time
                formal_critical = True
                return delta_t_formal, formal_critical
            
            # Case 2: Human is heading to same slot
            if human.target_slot == robot_target:
                # Compute conservative time estimates
                
                # Human's earliest arrival time (conservative/worst-case)
                if human.status == AgentStatus.PLACING:
                    human_earliest = 0.0  # Already there
                elif human.status == AgentStatus.EXITING:
                    human_earliest = 0.0  # Still at slot
                elif human.status == AgentStatus.MOVING:
                    # Worst case: human arrives earlier than expected
                    # Use remaining time minus 1.5s uncertainty buffer
                    human_earliest = max(0.0, human.remaining_time - 1.5)
                elif human.status == AgentStatus.IDLE and human.current_goal == robot_target:
                    # Human will start moving to same target soon
                    human_earliest = HUMAN_TRAVEL_PLACE_MIN - 1.0
                else:
                    human_earliest = float('inf')
                
                # Human's latest departure time (when they'll leave the slot)
                human_latest_departure = float('inf')
                if human.status in [AgentStatus.PLACING, AgentStatus.EXITING]:
                    human_latest_departure = human.remaining_time + HUMAN_EXIT_MAX
                elif human.status == AgentStatus.MOVING:
                    human_latest_departure = human.remaining_time + HUMAN_EXIT_MAX
                
                # Robot arrival time
                robot_arrival = robot.remaining_time
                
                # Check if there's potential overlap
                # Violation if robot arrives while human is at slot
                # Conservative: robot arrives between human_earliest and human_latest_departure
                if robot_arrival >= human_earliest and robot_arrival <= human_latest_departure:
                    delta_t_formal = min(human_earliest, robot_arrival)
                elif human_earliest >= robot_arrival:
                    # Human may arrive while robot is placing
                    robot_departure = robot_arrival + ROBOT_EXIT_MAX
                    if human_earliest <= robot_departure:
                        delta_t_formal = robot_arrival
                
                # Critical if time to violation is below hard threshold
                if delta_t_formal <= TAU_HARD:
                    formal_critical = True
                
                # More aggressive critical trigger for imminent conflicts
                # If human is moving and will arrive within 2 steps, and robot is close too
                if human.status == AgentStatus.MOVING:
                    if human.remaining_time <= FORMAL_HORIZON_STEPS * DELTA_T + 1.0:
                        if robot.remaining_time <= FORMAL_HORIZON_STEPS * DELTA_T + 2.0:
                            formal_critical = True
                            delta_t_formal = min(delta_t_formal, human.remaining_time)
        
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
        
        # Advisory if within soft horizon and confidence meets threshold
        advisory_horizon = TAU_SOFT  # Use soft threshold directly
        learned_advisory = (
            delta_t_learned <= advisory_horizon and 
            confidence >= P_MIN and
            robot.status == AgentStatus.MOVING and
            robot.mode != RobotMode.SLOW  # Don't re-trigger if already slow
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
    """
    
    def __init__(self, human_goals: List[str], robot_goals: List[str],
                 controller_type: ControllerType,
                 learned_accuracy: float = LEARNED_ACCURACY):
        """
        Initialize simulation.
        
        Args:
            human_goals: Goal sequence for human
            robot_goals: Goal sequence for robot
            controller_type: FORMAL_ONLY or DUAL_PRV
            learned_accuracy: Accuracy of learned predictor
        """
        self.human = AgentState(goal_sequence=human_goals.copy())
        self.robot = AgentState(goal_sequence=robot_goals.copy())
        self.controller_type = controller_type
        self.prv = PRVMonitor(learned_accuracy=learned_accuracy)
        
        self.t = 0.0
        self.logs: List[SimulationLog] = []
        
        # Statistics
        self.num_stops = 0
        self.total_stop_duration = 0.0
        self.num_slowdowns = 0
        self.safety_violations = 0
        
        # State tracking for events
        self.robot_was_slow = False
        self.robot_stop_start_time: Optional[float] = None
    
    def run(self, max_time: float = 500.0) -> RunStatistics:
        """
        Run the simulation until completion or max time.
        
        Args:
            max_time: Maximum simulation time
        
        Returns:
            RunStatistics for this run
        """
        # Start both agents
        self._start_next_goal(self.human, is_human=True)
        self._start_next_goal(self.robot, is_human=False)
        
        while self.t < max_time:
            # Check for completion
            if self.human.completed and self.robot.completed:
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
        
        return RunStatistics(
            completion_time=self.t,
            num_stops=self.num_stops,
            total_stop_duration=self.total_stop_duration,
            num_slowdowns=self.num_slowdowns,
            safety_violations=self.safety_violations,
            human_goals_completed=self.human.current_goal_idx,
            robot_goals_completed=self.robot.current_goal_idx
        )
    
    def _start_next_goal(self, agent: AgentState, is_human: bool):
        """Start agent moving toward their next goal."""
        if agent.completed:
            agent.status = AgentStatus.IDLE
            agent.target_slot = None
            return
        
        agent.target_slot = agent.current_goal
        agent.status = AgentStatus.MOVING
        
        if is_human:
            agent.remaining_time = sample_human_travel_place()
        else:
            agent.remaining_time = sample_robot_travel_place(agent.mode)
    
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
                # Resume when human is out of contested slot
                if contested_slot and not self.human.at_slot.get(contested_slot, False):
                    # Add baseline resume penalty if using formal-only controller
                    if self.controller_type == ControllerType.FORMAL_ONLY:
                        agent.remaining_time += BASELINE_RESUME_PENALTY
                    agent.status = AgentStatus.MOVING
                    # Record stop duration
                    if self.robot_stop_start_time is not None:
                        self.total_stop_duration += (self.t - self.robot_stop_start_time)
                        self.robot_stop_start_time = None
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
                # Done with this goal
                agent.clear_at_slots()
                agent.current_goal_idx += 1
                agent.status = AgentStatus.IDLE
                agent.target_slot = None
                
                # Reset robot mode to normal after completing action
                if not is_human:
                    agent.mode = RobotMode.NORMAL
    
    def _apply_mitigation(self, prv_output: PRVOutput):
        """Apply mitigation based on PRV mode."""
        if prv_output.fused_mode == PRVMode.CRITICAL:
            # Hard stop
            if self.robot.status == AgentStatus.MOVING:
                # Account for stop latency
                self.robot.stop_accumulator += DELTA_T
                if self.robot.stop_accumulator >= ROBOT_STOP_LATENCY:
                    self.robot.status = AgentStatus.STOPPED
                    self.robot.stop_accumulator = 0.0
                    self.num_stops += 1
                    self.robot_stop_start_time = self.t
            # Reset slow mode tracking
            self.robot_was_slow = False
                    
        elif prv_output.fused_mode == PRVMode.ADVISORY:
            # Slow down
            if self.robot.status == AgentStatus.MOVING and self.robot.mode == RobotMode.NORMAL:
                self.robot.mode = RobotMode.SLOW
                # Adjust remaining time for slower speed
                # (Remaining time increases proportionally)
                slow_factor = (ROBOT_TRAVEL_PLACE_SLOW_MIN + ROBOT_TRAVEL_PLACE_SLOW_MAX) / \
                             (ROBOT_TRAVEL_PLACE_NORMAL_MIN + ROBOT_TRAVEL_PLACE_NORMAL_MAX)
                self.robot.remaining_time *= slow_factor
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
        
        for trial in range(self.n_trials):
            # Generate same goal sequences for fair comparison
            human_goals = generate_goal_sequence(n_cycles=self.n_goals_per_agent // 4 + 1)
            human_goals = human_goals[:self.n_goals_per_agent]
            
            robot_goals = generate_goal_sequence(n_cycles=self.n_goals_per_agent // 4 + 1)
            robot_goals = robot_goals[:self.n_goals_per_agent]
            
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
        human_goals = generate_goal_sequence(n_cycles=3)[:args.n_goals]
        robot_goals = generate_goal_sequence(n_cycles=3)[:args.n_goals]
        
        print(f"\nHuman goals: {human_goals}")
        print(f"Robot goals: {robot_goals}")
        
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
        
        print_time_series_log(sim.logs)
        
    else:
        # Monte-Carlo experiments
        print(f"\nRunning Monte-Carlo experiments...")
        print(f"  Trials: {args.n_trials}")
        print(f"  Goals per agent: {args.n_goals}")
        print(f"  Random seed: {args.seed}")
        
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
