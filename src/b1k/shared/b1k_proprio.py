"""R1Pro proprioception layout shared between training, eval, and dataset tooling."""

import numpy as np

# Legacy proprio keys used by OpenPI training/eval. Upstream BEHAVIOR switched eval to a
# compact proprio vector (~61 dims), but OpenPI checkpoints still expect the legacy layout
# (~256 dims) when extracting the 23-dim robot state prompt.
OPENPI_R1PRO_LEGACY_PROPRIO_OBS_KEYS = (
    "joint_qpos",
    "joint_qpos_sin",
    "joint_qpos_cos",
    "joint_qvel",
    "joint_qeffort",
    "robot_pos",
    "robot_ori_cos",
    "robot_ori_sin",
    "robot_2d_ori",
    "robot_2d_ori_cos",
    "robot_2d_ori_sin",
    "robot_lin_vel",
    "robot_ang_vel",
    "arm_left_qpos",
    "arm_left_qpos_sin",
    "arm_left_qpos_cos",
    "arm_left_qvel",
    "eef_left_pos",
    "eef_left_quat",
    "gripper_left_qpos",
    "gripper_left_qvel",
    "arm_right_qpos",
    "arm_right_qpos_sin",
    "arm_right_qpos_cos",
    "arm_right_qvel",
    "eef_right_pos",
    "eef_right_quat",
    "gripper_right_qpos",
    "gripper_right_qvel",
    "trunk_qpos",
    "trunk_qvel",
    "base_qpos",
    "base_qpos_sin",
    "base_qpos_cos",
    "base_qvel",
)

R1PRO_LEGACY_PROPRIO_DIM = 256
R1PRO_COMPACT_PROPRIO_DIM = 61

R1PRO_LEGACY_PROPRIOCEPTION_INDICES = {
    "arm_left_qpos": np.s_[158:165],
    "arm_left_qvel": np.s_[179:186],
    "eef_left_pos": np.s_[186:189],
    "eef_left_quat": np.s_[189:193],
    "gripper_left_qpos": np.s_[193:195],
    "gripper_left_qvel": np.s_[195:197],
    "arm_right_qpos": np.s_[197:204],
    "arm_right_qvel": np.s_[218:225],
    "eef_right_pos": np.s_[225:228],
    "eef_right_quat": np.s_[228:232],
    "gripper_right_qpos": np.s_[232:234],
    "gripper_right_qvel": np.s_[234:236],
    "trunk_qpos": np.s_[236:240],
    "trunk_qvel": np.s_[240:244],
    "base_qpos": np.s_[244:247],
    "base_qvel": np.s_[253:256],
    "robot_pos": np.s_[140:143],
}

# Compact layout from omnigibson.eval.utils.eval_utils (post #2260).
R1PRO_COMPACT_PROPRIOCEPTION_INDICES = {
    "base_qvel": np.s_[0:3],
    "arm_left_qpos": np.s_[3:10],
    "arm_left_qvel": np.s_[10:17],
    "eef_left_pos": np.s_[17:20],
    "eef_left_quat": np.s_[20:24],
    "gripper_left_qpos": np.s_[24:26],
    "gripper_left_qvel": np.s_[26:28],
    "arm_right_qpos": np.s_[28:35],
    "arm_right_qvel": np.s_[35:42],
    "eef_right_pos": np.s_[42:45],
    "eef_right_quat": np.s_[45:49],
    "gripper_right_qpos": np.s_[49:51],
    "gripper_right_qvel": np.s_[51:53],
    "trunk_qpos": np.s_[53:57],
    "trunk_qvel": np.s_[57:61],
}

# Pack order for 256-d → 61-d. Matches 2026 eval compact proprio.
_R1PRO_COMPACT_FROM_LEGACY_KEYS = (
    "base_qvel",
    "arm_left_qpos",
    "arm_left_qvel",
    "eef_left_pos",
    "eef_left_quat",
    "gripper_left_qpos",
    "gripper_left_qvel",
    "arm_right_qpos",
    "arm_right_qvel",
    "eef_right_pos",
    "eef_right_quat",
    "gripper_right_qpos",
    "gripper_right_qvel",
    "trunk_qpos",
    "trunk_qvel",
)


def proprioception_indices_for_dim(proprio_dim: int) -> dict[str, slice]:
    if proprio_dim >= R1PRO_LEGACY_PROPRIO_DIM:
        return R1PRO_LEGACY_PROPRIOCEPTION_INDICES
    return R1PRO_COMPACT_PROPRIOCEPTION_INDICES


# Absolute action chunks in the BEHAVIOR dataset / LeRobot export use:
#   base(3) + trunk(4) + left_arm(7) + left_grip(1) + right_arm(7) + right_grip(1)
# OpenPI state / previous_action / OAT tokenizer zarr use:
#   base(3) + trunk(4) + left_arm(7) + right_arm(7) + left_grip(1) + right_grip(1)
# openpi[i] = behavior[B1K_BEHAVIOR_TO_OPENPI_ACTION_INDICES[i]]
B1K_BEHAVIOR_TO_OPENPI_ACTION_INDICES = (*range(14), *range(15, 22), 14, 22)
B1K_OPENPI_TO_BEHAVIOR_ACTION_INDICES = tuple(
    int(i) for i in np.argsort(np.asarray(B1K_BEHAVIOR_TO_OPENPI_ACTION_INDICES, dtype=np.int64))
)


def holonomic_base_qvel_to_robot_frame(base_qpos: np.ndarray, base_qvel: np.ndarray) -> np.ndarray:
    """Convert world-frame holonomic base velocity to robot-local [vx, vy, wz].

    Matches OmniGibson ``Robot._get_base_qvel_for_proprioception`` (2026 eval compact).
    ``base_qpos`` is ``[x, y, rz]`` or a 6-DoF virtual-base pose whose last yaw is ``rz``.
    """
    base_qpos = np.asarray(base_qpos)
    base_qvel = np.asarray(base_qvel)
    yaw = base_qpos[..., -1]
    c = np.cos(yaw)
    s = np.sin(yaw)
    vx, vy, wz = base_qvel[..., 0], base_qvel[..., 1], base_qvel[..., -1]
    return np.stack([c * vx + s * vy, -s * vx + c * vy, wz], axis=-1)


def legacy_proprio_to_compact(proprio_data: np.ndarray) -> np.ndarray:
    """Rewrite 256-d 2025 proprio to the 61-d 2026 compact layout.

    Leaves 61-d compact and 23-d extracted state unchanged. Converts stored
    world-frame ``base_qvel`` into the robot frame used by 2026 demos / eval.
    """
    proprio_data = np.asarray(proprio_data)
    trailing = int(proprio_data.shape[-1])
    if trailing == R1PRO_COMPACT_PROPRIO_DIM or trailing == 23:
        return proprio_data.astype(np.float32, copy=False)
    if trailing < R1PRO_LEGACY_PROPRIO_DIM:
        return proprio_data.astype(np.float32, copy=False)

    legacy = R1PRO_LEGACY_PROPRIOCEPTION_INDICES
    base_qvel = holonomic_base_qvel_to_robot_frame(
        proprio_data[..., legacy["base_qpos"]],
        proprio_data[..., legacy["base_qvel"]],
    )
    parts = []
    for key in _R1PRO_COMPACT_FROM_LEGACY_KEYS:
        parts.append(base_qvel if key == "base_qvel" else proprio_data[..., legacy[key]])
    compact = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)
    if compact.shape[-1] != R1PRO_COMPACT_PROPRIO_DIM:
        raise ValueError(f"Packed compact proprio has dim {compact.shape[-1]}, expected {R1PRO_COMPACT_PROPRIO_DIM}")
    return compact


def extract_state_from_proprio(proprio_data: np.ndarray) -> np.ndarray:
    """Map 61-d compact or 256-d legacy proprio to the 23-d PI_BEHAVIOR state.

    Layout matches BEHAVIOR actions:
    base(3) + trunk(4) + left_arm(7) + left_grip(1) + right_arm(7) + right_grip(1).
    Gripper widths are normalized from [0, 0.1] to [-1, 1].
    """
    proprio_data = np.asarray(proprio_data)
    if proprio_data.shape[-1] == 23:
        return proprio_data.astype(np.float32, copy=False)

    indices = proprioception_indices_for_dim(int(proprio_data.shape[-1]))
    base_qvel = proprio_data[..., indices["base_qvel"]]
    trunk_qpos = proprio_data[..., indices["trunk_qpos"]]
    arm_left_qpos = proprio_data[..., indices["arm_left_qpos"]]
    arm_right_qpos = proprio_data[..., indices["arm_right_qpos"]]
    left_gripper_raw = proprio_data[..., indices["gripper_left_qpos"]].sum(axis=-1, keepdims=True)
    right_gripper_raw = proprio_data[..., indices["gripper_right_qpos"]].sum(axis=-1, keepdims=True)
    max_gripper_width = 0.1
    left_gripper_width = 2.0 * (left_gripper_raw / max_gripper_width) - 1.0
    right_gripper_width = 2.0 * (right_gripper_raw / max_gripper_width) - 1.0
    return np.concatenate(
        [
            base_qvel,
            trunk_qpos,
            arm_left_qpos,
            left_gripper_width,
            arm_right_qpos,
            right_gripper_width,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def _reorder_last_axis(x: np.ndarray, indices: tuple[int, ...] | list[int]) -> np.ndarray:
    arr = np.asarray(x)
    index_arr = np.asarray(indices, dtype=np.int64)
    dims = int(index_arr.shape[0])
    if arr.shape[-1] < dims:
        raise ValueError(f"Expected trailing dim >= {dims}, got shape {arr.shape}")
    head = arr[..., index_arr]
    if arr.shape[-1] == dims:
        return head
    return np.concatenate([head, arr[..., dims:]], axis=-1)


def behavior_action_to_openpi_order(x: np.ndarray) -> np.ndarray:
    """Reorder trailing action dims from BEHAVIOR layout to OpenPI layout."""
    return _reorder_last_axis(x, B1K_BEHAVIOR_TO_OPENPI_ACTION_INDICES)


def openpi_action_to_behavior_order(x: np.ndarray) -> np.ndarray:
    """Reorder trailing action dims from OpenPI layout to BEHAVIOR layout."""
    return _reorder_last_axis(x, B1K_OPENPI_TO_BEHAVIOR_ACTION_INDICES)
