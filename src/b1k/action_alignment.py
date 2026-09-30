"""Action-label alignment for 2026 demos and Comet RFT.

The observation stays on one row: images and the full proprioception vector
move together. Only the action label is taken from another row. On 2026 the
base is taken from a different row than the arms.

RFT is one shift for the whole command. The arm joints and the base velocity
in that row are the command just applied, so the command to send now is the
next row. 2026 is split. Arm and trunk joints match the command from two rows
earlier, so their label is the previous row. Base velocity matches the command
from one row earlier, so its label stays on this row. Grippers barely move in
these episodes, so they follow the arms.

The stored base velocity is about 0.75 times the x and y command because that
is the controller limit. This module copies that recorded command and does not
rescale it.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

# lead = where the label starts, relative to the observation row.
# The label window is action[t + lead : t + lead + horizon, slice].
ACTION_SLICES: dict[str, tuple[int, int]] = {
    "base": (0, 3),
    "trunk": (3, 7),
    "left_arm": (7, 14),
    "left_gripper": (14, 15),
    "right_arm": (15, 22),
    "right_gripper": (22, 23),
}
ACTION_DIM = 23

RFT_LEAD: dict[str, int] = {name: 1 for name in ACTION_SLICES}
D2026_LEAD: dict[str, int] = {
    "base": 0,
    "trunk": -1,
    "left_arm": -1,
    "left_gripper": -1,
    "right_arm": -1,
    "right_gripper": -1,
}


def leads_for_source(*, rft: bool) -> dict[str, int]:
    return dict(RFT_LEAD if rft else D2026_LEAD)


def valid_observation_range(length: int, leads: Mapping[str, int], horizon: int) -> tuple[int, int]:
    """Half-open range of observation rows whose label windows sit inside the episode."""
    if horizon < 1 or length < 1:
        return 0, 0
    start = 0
    end = length
    for lead in leads.values():
        lead = int(lead)
        start = max(start, -lead)
        end = min(end, length - horizon - lead + 1)
    if start >= end:
        return 0, 0
    return start, end


def assemble_action(
    actions: np.ndarray,
    t: int,
    horizon: int,
    leads: Mapping[str, int],
) -> np.ndarray | None:
    """Build one label chunk. ``t`` is the observation row inside ``actions``.

    ``actions`` may be a cropped block. ``t`` is then relative to that block and
    may be negative when every slice window still lies inside the block.
    Returns None when any slice would leave the array. Does not rescale base.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected action rows [T, D], got shape {actions.shape}")
    length, dim = actions.shape
    if dim < ACTION_DIM:
        raise ValueError(f"Expected at least {ACTION_DIM} action dims, got {dim}")
    if horizon < 1:
        return None
    for lead in leads.values():
        start = int(t) + int(lead)
        end = start + horizon
        if start < 0 or end > length:
            return None
    out = np.zeros((horizon, dim), dtype=np.float32)
    if dim > ACTION_DIM and 0 <= int(t) < length:
        out[:, ACTION_DIM:] = actions[int(t), ACTION_DIM:]
    for name, (lo, hi) in ACTION_SLICES.items():
        start = int(t) + int(leads[name])
        out[:, lo:hi] = actions[start : start + horizon, lo:hi]
    return out


def aligned_action_chunks(
    actions: np.ndarray,
    leads: Mapping[str, int],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Every in-range label chunk. Returns ``[N, horizon, D]`` and observation indices."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected action rows [T, D], got shape {actions.shape}")
    length, dim = int(actions.shape[0]), int(actions.shape[1])
    t0, t1 = valid_observation_range(length, leads, horizon)
    if t0 >= t1:
        return (
            np.zeros((0, horizon, dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    if dim < ACTION_DIM:
        raise ValueError(f"Expected at least {ACTION_DIM} action dims, got {dim}")
    count = t1 - t0
    out = np.zeros((count, horizon, dim), dtype=np.float32)
    observation_index = np.arange(t0, t1, dtype=np.int64)
    if dim > ACTION_DIM:
        out[:, :, ACTION_DIM:] = actions[observation_index, None, ACTION_DIM:]
    for name, (lo, hi) in ACTION_SLICES.items():
        lead = int(leads[name])
        src0 = t0 + lead
        src = actions[src0 : src0 + count + horizon - 1]
        if src.shape[0] != count + horizon - 1:
            raise RuntimeError(
                f"Slice {name} source length {src.shape[0]} != {count + horizon - 1}"
            )
        windows = np.lib.stride_tricks.sliding_window_view(src, (horizon, dim))[:, 0]
        out[:, :, lo:hi] = np.array(windows[:, :, lo:hi], copy=True)
    return out, observation_index


def apply_joint_delta(actions: np.ndarray, states: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Subtract the observation-row state on delta joints.

    Channels where ``mask`` is false, including base velocity, stay the recorded
    command.
    """
    out = np.array(actions, dtype=np.float32, copy=True)
    mask_arr = np.asarray(mask)
    dims = int(mask_arr.shape[-1])
    state = np.asarray(states, dtype=np.float32)
    if out.ndim == 3:
        state = state[:, None, :]
    out[..., :dims] = np.where(mask_arr, out[..., :dims] - state[..., :dims], out[..., :dims])
    return out
