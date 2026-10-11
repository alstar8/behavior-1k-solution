"""
Evaluation tricks for improving success rates on BEHAVIOR-1K tasks.

This module contains:
1. Correction rules: Task-specific problem resolution (stage reset, gripper recovery)
2. Gripper variation check: Disable compression when grippers have large movement

Enable via --apply-eval-tricks flag (default: False)
"""

import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================

# Gripper thresholds (normalized [-1, 1] range)
# Grippers are normalized from raw [0, 0.1] to [-1, 1]:
#   -1.0 = fully closed (raw 0.0)
#   -0.9 = mostly closed (raw 0.005)
#    0.0 = half open (raw 0.05)
#    0.9 = mostly open (raw 0.095)
#    1.0 = fully open (raw 0.1)
OPEN_THRESHOLD = 0.90
CLOSED_THRESHOLD = -0.98 # very strict threshold to avoid false positives

# Gripper variation threshold for compression check
GRIPPER_VARIATION_THRESHOLD = 0.2

# State indices
LEFT_GRIPPER_IDX = 14
RIGHT_GRIPPER_IDX = 22
LEFT_ARM_START_IDX = 7   # Left arm: state[7:14]
RIGHT_ARM_START_IDX = 15  # Right arm: state[15:22]
LEFT_ARM_END_IDX = 14
RIGHT_ARM_END_IDX = 22

# Base positions for arms (7 joints each)
BASE_POSITION_LEFT_ARM = np.array([-0.2, 0.0, 0.1, -1.4, 0.1, 0.7, 0.0])
BASE_POSITION_RIGHT_ARM = np.array([-0.2, 0.0, 0.1, -1.4, 0.1, 0.7, 0.0])


# ============================================================================
# GRIPPER CORRECTION POLICY
# ============================================================================

# Tasks where grippers should ALWAYS be open (any closure is an error)
ALWAYS_OPEN_LEFT_GRIPPER_TASKS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 47, 48}
ALWAYS_OPEN_RIGHT_GRIPPER_TASKS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 47, 48, 49}

# Tasks where closure is not allowed BEFORE specified stage
# Format: {task_id: {'left': min_stage, 'right': min_stage}}
MIN_STAGE_FOR_CLOSURE = {
    0: {'left': 2, 'right': 2},      # Radio: allow closing from stage 2+
    30: {'right': 6},                 # setting_the_fire
    31: {'right': 8},                 # clean_boxing_gloves
    32: {'right': 5},                 # wash_a_baseball_cap
    33: {'right': 11},                # wash_dog_toys
    40: {'right': 4},                 # make_microwave_popcorn
    41: {'left': 14, 'right': 14},   # cook_cabbage
    45: {'right': 10},                # cook_hot_dogs
    46: {'left': 8, 'right': 8},     # cook_bacon
    49: {'left': 14},                 # make_pizza
}

# Tasks where right gripper is ALWAYS allowed to close (exceptions)
RIGHT_GRIPPER_ALWAYS_ALLOWED = {38, 39}  # spraying_for_bugs, spraying_fruit_trees


# ============================================================================
# CORRECTION RULES - ADD NEW RULES HERE
# ============================================================================

def general_gripper_correction(
    task_id: int,
    stage: int,
    state: np.ndarray,
    actions: np.ndarray
) -> Optional[Tuple[np.ndarray, int]]:
    """
    General gripper correction rule for all tasks.
    
    Applies task-specific policies:
    1. Always-open tasks: Open gripper if closed (any stage)
    2. Stage-restricted tasks: Open gripper if closed before allowed stage
    3. Exceptions: Skip tasks where gripper is intentionally closed (e.g., spray bottles)
    
    Args:
        task_id: Current task ID
        stage: Current stage
        state: np.ndarray,
        actions: Predicted actions (horizon, 23)
    
    Returns:
        (corrected_actions, corrected_stage) if rule applies, None otherwise
    """
    # Get gripper states
    left_gripper = state[LEFT_GRIPPER_IDX]
    right_gripper = state[RIGHT_GRIPPER_IDX]
    
    left_closed = left_gripper < CLOSED_THRESHOLD
    right_closed = right_gripper < CLOSED_THRESHOLD
    
    # Determine if correction is needed
    left_needs_opening = False
    right_needs_opening = False
    
    # Check left gripper
    if left_closed:
        # Always open if task requires it
        if task_id in ALWAYS_OPEN_LEFT_GRIPPER_TASKS:
            left_needs_opening = True
        # Check stage restriction
        elif task_id in MIN_STAGE_FOR_CLOSURE:
            min_stage_left = MIN_STAGE_FOR_CLOSURE[task_id].get('left')
            if min_stage_left is not None and stage < min_stage_left:
                left_needs_opening = True
    
    # Check right gripper
    if right_closed:
        # Exception: Always allow for spray bottle tasks
        if task_id in RIGHT_GRIPPER_ALWAYS_ALLOWED:
            pass  # Never correct these tasks
        # Always open if task requires it
        elif task_id in ALWAYS_OPEN_RIGHT_GRIPPER_TASKS:
            right_needs_opening = True
        # Check stage restriction
        elif task_id in MIN_STAGE_FOR_CLOSURE:
            min_stage_right = MIN_STAGE_FOR_CLOSURE[task_id].get('right')
            if min_stage_right is not None and stage < min_stage_right:
                right_needs_opening = True
    
    # Apply correction if needed
    if left_needs_opening or right_needs_opening:
        logger.info(f"🔧 GENERAL GRIPPER CORRECTION: Task {task_id}, Stage {stage}")
        
        # Copy current state to all actions
        corrected_actions = np.tile(state, (actions.shape[0], 1))
        
        if left_needs_opening:
            logger.info(f"   Left gripper closed ({left_gripper:.4f}) → Opening to 1.0")
            corrected_actions[:, LEFT_GRIPPER_IDX] = 1.0
        
        if right_needs_opening:
            logger.info(f"   Right gripper closed ({right_gripper:.4f}) → Opening to 1.0")
            corrected_actions[:, RIGHT_GRIPPER_IDX] = 1.0
        
        # Keep stage unchanged
        return corrected_actions, stage
    
    return None  # No correction needed


# ============================================================================
# RULE REGISTRY - ORGANIZE BY TASK_ID
# ============================================================================

# Task-specific rules (these run BEFORE the general rule)
# Dictionary mapping task_id -> list of rule functions
TASK_SPECIFIC_RULES = {
    # Radio (task 0) used to reset stage 4 to stage 2 on every chunk, so the
    # final stage could never be held. Do not reintroduce unconditional stage resets.
}

# General rules that apply to ALL tasks (run after task-specific rules)
GENERAL_RULES = [
    general_gripper_correction,  # Applies to all tasks with gripper policies
]


# ============================================================================
# RULE APPLICATION
# ============================================================================

def apply_correction_rules(
    task_id: int,
    stage: int,
    state: np.ndarray,
    actions: np.ndarray
) -> Tuple[np.ndarray, int]:
    """
    Apply correction rules for the given task_id.
    
    Rules are applied in order until one matches:
    1. Task-specific rules (none registered at the moment)
    2. General rules (e.g., gripper correction policies)
    
    Only the first matching rule is applied per call.
    
    Args:
        task_id: Current task ID
        stage: Current stage
        state: Current state (23-dim)
        actions: Predicted actions (horizon, 23)
    
    Returns:
        (actions, stage) - corrected or original if no rule applies
    """
    # Try task-specific rules first
    task_specific_rules = TASK_SPECIFIC_RULES.get(task_id, [])
    for rule in task_specific_rules:
        result = rule(task_id, stage, state, actions)
        if result is not None:
            corrected_actions, corrected_stage = result
            return corrected_actions, corrected_stage
    
    # Try general rules (apply to all tasks)
    for rule in GENERAL_RULES:
        result = rule(task_id, stage, state, actions)
        if result is not None:
            corrected_actions, corrected_stage = result
            return corrected_actions, corrected_stage
    
    # No rules matched, return original
    return actions, stage


# ============================================================================
# GRIPPER VARIATION CHECK
# ============================================================================

def check_gripper_variation(
    actions: np.ndarray,
    num_actions_to_check: int
) -> Tuple[bool, float, float]:
    """
    Check if gripper actions have high variation.
    
    High variation in gripper positions suggests complex manipulation that
    should not be compressed via interpolation.
    
    Args:
        actions: Predicted actions (horizon, 23)
        num_actions_to_check: Number of actions to check (e.g., actions_to_execute)
    
    Returns:
        (has_high_variation, left_variation, right_variation)
    """
    # Check only the actions we plan to execute
    actions_to_check = actions[:num_actions_to_check]
    
    # Extract gripper actions
    left_gripper_actions = actions_to_check[:, LEFT_GRIPPER_IDX]
    right_gripper_actions = actions_to_check[:, RIGHT_GRIPPER_IDX]
    
    left_variation = float(np.max(left_gripper_actions) - np.min(left_gripper_actions))
    right_variation = float(np.max(right_gripper_actions) - np.min(right_gripper_actions))
    
    has_high_variation = (left_variation > GRIPPER_VARIATION_THRESHOLD or 
                         right_variation > GRIPPER_VARIATION_THRESHOLD)
    
    return has_high_variation, left_variation, right_variation

