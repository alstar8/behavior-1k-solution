"""Deterministic first-step action fingerprint for the radio 95k checkpoint.

Loads frame 0 of 2026 turning_on_radio episode 0 and samples one action chunk
with frozen numpy noise. The same numbers should come out on any server that
loads this checkpoint with the same code, norm stats, and noise.
"""

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torchvision

from openpi_client.image_tools import resize_with_pad

from b1k.policies.policy_config import create_trained_policy
from b1k.training import config as _config

CKPT = Path(
    "/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior/b1k_solution/"
    "pi_behavior_2026_radio_demo0_6_comet0_4/"
    "pi_behavior_2026_radio_from100k_20260921_213403/95000"
)
DEMO = Path("/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior/2026-challenge-demos")
EPISODE_INDEX = 0
TASK_ID = 0
NUM_STEPS = 20
NOISE_SEED = 0


def _first_frame(video_path: Path) -> np.ndarray:
    torchvision.set_video_backend("pyav")
    reader = torchvision.io.VideoReader(str(video_path), "video")
    reader.seek(0.0)
    frame = next(iter(reader))["data"]  # uint8 CHW
    image = frame.permute(1, 2, 0).numpy()
    return resize_with_pad(image[..., :3], 224, 224)


def main() -> None:
    table = pq.read_table(
        DEMO / "data/chunk-000/file-000.parquet",
        columns=["observation.state", "episode_index", "timestamp", "frame_index"],
    )
    if int(table["episode_index"][0].as_py()) != EPISODE_INDEX:
        raise RuntimeError(f"Expected episode {EPISODE_INDEX} at row 0")
    state = np.asarray(table["observation.state"][0].as_py(), dtype=np.float32)
    timestamp = float(table["timestamp"][0].as_py())
    frame_index = int(table["frame_index"][0].as_py())

    obs = {
        "observation/egocentric_camera": _first_frame(
            DEMO / "videos/observation.rgb.zed_link_camera_0/chunk-000/file-000.mp4"
        ),
        "observation/wrist_image_left": _first_frame(
            DEMO / "videos/observation.rgb.left_realsense_link_camera_0/chunk-000/file-000.mp4"
        ),
        "observation/wrist_image_right": _first_frame(
            DEMO / "videos/observation.rgb.right_realsense_link_camera_0/chunk-000/file-000.mp4"
        ),
        "observation/state": state,
        "tokenized_prompt": np.array([TASK_ID, 0], dtype=np.int32),
        "tokenized_prompt_mask": np.array([True, True], dtype=bool),
        "subtask_state": np.array(0, dtype=np.int32),
    }
    noise = np.random.RandomState(NOISE_SEED).randn(30, 32).astype(np.float32)

    policy = create_trained_policy(
        _config.get_config("pi_behavior_2026_radio_demo0_6_comet0_4"),
        CKPT,
        sample_kwargs={"num_steps": NUM_STEPS},
    )
    output = policy.infer(obs, noise=noise)
    actions = np.asarray(output["actions"], dtype=np.float32)
    first = actions[0]
    payload = {
        "checkpoint": str(CKPT),
        "dataset": str(DEMO),
        "episode_index": EPISODE_INDEX,
        "frame_index": frame_index,
        "timestamp_s": timestamp,
        "task_id": TASK_ID,
        "subtask_state": 0,
        "num_flow_steps": NUM_STEPS,
        "noise": "numpy.RandomState(0).randn(30, 32).astype(float32)",
        "action_layout": "base(3), trunk(4), left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)",
        "predicted_stage": int(output["predicted_stage"]),
        "first_action": [float(f"{x:.8f}") for x in first],
        "action_chunk": [[float(f"{x:.8f}") for x in row] for row in actions],
    }
    out_path = Path(__file__).resolve().parents[1] / "docs" / "radio_95k_step0_fingerprint.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in payload if k != "action_chunk"}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
