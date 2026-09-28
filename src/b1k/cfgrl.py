"""CFGRL labels for a flow-matching policy.

Forward dataset actions are optimality 2. A rewind action chunk is optimality 1.
About 10% of examples drop the label to 0 so the same network is the unconditional policy.
"""

from __future__ import annotations

import numpy as np

from b1k.shared.b1k_proprio import extract_state_from_proprio
from b1k.shared.b1k_proprio import legacy_proprio_to_compact

# DeltaActions mask (-3, 3, -1, 7, -1, 7, -1). Only these joints are absolute
# targets relative to the current proprio. Base velocity and the remaining
# absolute channels (last trunk joint, both grippers) stay as reversed commands.
DELTA_JOINT_DIMS = (3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21)
BASE_VELOCITY_DIMS = 3
OPT_DROPOUT = 0
OPT_REWIND = 1
OPT_FORWARD = 2


def rewind_absolute_actions(actions: np.ndarray, start_state: np.ndarray) -> np.ndarray:
    """Absolute action chunk that undoes `actions`, starting at the chunk's end state.

    Delta joints walk back through the commanded targets and finish at `start_state`.
    Base velocity is reversed and negated. Other absolute channels are reversed
    unchanged, because substituting proprio there leaves the normalized range.
    """
    actions = np.asarray(actions, dtype=np.float32)
    start_state = np.asarray(start_state, dtype=np.float32).reshape(-1)
    horizon, dims = actions.shape
    rev = np.array(actions[::-1], dtype=np.float32, copy=True)
    rev[:, :BASE_VELOCITY_DIMS] = -actions[::-1, :BASE_VELOCITY_DIMS]
    delta_dims = [index for index in DELTA_JOINT_DIMS if index < dims and index < start_state.shape[0]]
    if horizon == 1:
        rev[0, delta_dims] = start_state[delta_dims]
        return rev
    rev[:-1, delta_dims] = actions[-2::-1][:, delta_dims]
    rev[-1, delta_dims] = start_state[delta_dims]
    return rev


def state_from_proprio(proprio: np.ndarray) -> np.ndarray:
    return extract_state_from_proprio(legacy_proprio_to_compact(np.asarray(proprio)))
