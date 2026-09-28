"""Map a 2025 BEHAVIOR submission action onto the 2026 eval robot.

2025 checkpoints (IliaLarchenko/behavior_submission) were trained on the
2025 challenge demos. Those actions and the 2026 R1Pro controller use the
same 23-d layout:

    base(3), trunk(4), left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)

Checked against turning_on_radio demo frame 0 in both
``2025-challenge-demos`` and ``2026-challenge-demos``: the 23 values match,
including trunk joints at indices 3:7 and grippers at 14 and 22.

This adapter is the only place that rewrite is applied, and only when the
policy server is started with ``--convert-2025-actions``. 2026 checkpoints
leave the flag off.
"""

from __future__ import annotations

import numpy as np

# 2026 controller order: base, trunk, arm_left, gripper_left, arm_right, gripper_right.
# 2025 dataset action uses that same order, so the permutation is identity.
_ACTION_DIM = 23
_FROM_2025_TO_2026 = tuple(range(_ACTION_DIM))


def convert_2025_action_to_2026(actions: np.ndarray) -> np.ndarray:
    """Return ``actions`` in the 2026 robot action order.

    ``actions`` is ``(..., 23)`` or longer; dimensions past 23 are dropped.
    """
    array = np.asarray(actions)
    if array.shape[-1] < _ACTION_DIM:
        raise ValueError(f"Expected at least {_ACTION_DIM} action dims, got {array.shape}")
    index = np.asarray(_FROM_2025_TO_2026, dtype=np.int64)
    return np.ascontiguousarray(array[..., index])
