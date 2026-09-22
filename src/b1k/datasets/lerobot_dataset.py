import bisect
from collections import defaultdict
from collections.abc import Callable, Iterable
import json
import os
from pathlib import Path
import random

import datasets
from datasets import load_dataset
from huggingface_hub import snapshot_download
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import EPISODES_PATH
from lerobot.datasets.utils import EPISODES_STATS_PATH
from lerobot.datasets.utils import STATS_PATH
from lerobot.datasets.utils import TASKS_PATH
from lerobot.datasets.utils import backward_compatible_episodes_stats
from lerobot.datasets.utils import cast_stats_to_numpy
from lerobot.datasets.utils import check_delta_timestamps
from lerobot.datasets.utils import check_timestamps_sync
from lerobot.datasets.utils import check_version_compatibility
from lerobot.datasets.utils import get_delta_indices
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.utils import get_safe_version
from lerobot.datasets.utils import hf_transform_to_torch
from lerobot.datasets.utils import is_valid_version
from lerobot.datasets.utils import load_info
from lerobot.datasets.utils import load_json
from lerobot.datasets.utils import load_jsonlines
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.datasets.video_utils import get_safe_default_codec
import numpy as np
import packaging.version
import torch as th
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info
import torchvision

try:
    from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_INDICES
    from omnigibson.eval.utils.eval_utils import get_robot_camera_names
    from omnigibson.eval.utils.obs_utils import instance_id_to_instance
    from omnigibson.utils.ui_utils import create_module_logger

    _HAS_OMNIGIBSON = True
except Exception:
    import csv
    import logging

    _HAS_OMNIGIBSON = False

    def _load_task_names_to_indices_fallback() -> dict[str, int]:
        data_root = os.environ.get("OMNIGIBSON_DATA_PATH") or os.environ.get("OPENPI_BEHAVIOR_DATA_ROOT")
        if not data_root:
            return {}
        candidates = [
            Path(data_root) / "2026-challenge-task-instances" / "metadata" / "B100_task_misc.csv",
            Path(data_root) / "metadata" / "B100_task_misc.csv",
        ]
        for csv_path in candidates:
            if not csv_path.is_file():
                continue
            task_ids: dict[str, int] = {}
            with csv_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    task_ids[row["Task"]] = int(row["Task ID"])
            return task_ids
        return {}

    TASK_NAMES_TO_INDICES = _load_task_names_to_indices_fallback()

    def get_robot_camera_names(robot_name, robot_eval_config):  # type: ignore[no-redef]
        return dict(robot_eval_config["camera_sensor_names"])

    def instance_id_to_instance(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError("omnigibson is required for segmentation instance decoding")

    def create_module_logger(name: str):  # type: ignore[no-redef]
        return logging.getLogger(name)


_R1PRO_ROBOT_EVAL_CONFIG = {
    "camera_sensor_names": {
        "left_wrist": "robot_r1:left_realsense_link:Camera:0",
        "right_wrist": "robot_r1:right_realsense_link:Camera:0",
        "head": "robot_r1:zed_link:Camera:0",
    }
}
ROBOT_CAMERA_NAMES = {"R1Pro": get_robot_camera_names("robot_r1", _R1PRO_ROBOT_EVAL_CONFIG)}
from b1k.shared.lerobot_compat import ensure_legacy_lerobot_api

ensure_legacy_lerobot_api()

if "turning_on_radio" not in TASK_NAMES_TO_INDICES:
    TASK_NAMES_TO_INDICES = {**TASK_NAMES_TO_INDICES, "turning_on_radio": 0}

ANNOTATIONS_PATH = "annotations"
ORCHESTRATORS_PATH = "orchestrators"
B1K_VIDEO_GOP_FRAMES = 250
B1K_LEROBOT_V3_CAMERA_SOURCES = {
    "head": "observation.rgb.zed_link_camera_0",
    "left_wrist": "observation.rgb.left_realsense_link_camera_0",
    "right_wrist": "observation.rgb.right_realsense_link_camera_0",
}
B1K_CANONICAL_RGB_KEYS = {
    camera: f"observation.images.rgb.{camera}" for camera in B1K_LEROBOT_V3_CAMERA_SOURCES
}
logger = create_module_logger("BehaviorLeRobotDataset")


def b1k_dataset_layout(data_root: Path) -> str:
    """Return ``legacy_jsonl`` (2025) or ``lerobot_parquet`` (2026+)."""
    if (data_root / "meta" / "episodes.jsonl").is_file():
        return "legacy_jsonl"
    if (data_root / "meta" / "episodes").is_dir():
        return "lerobot_parquet"
    raise FileNotFoundError(
        f"Unsupported B1K dataset layout under {data_root / 'meta'}. "
        "Expected meta/episodes.jsonl (2025) or meta/episodes/*.parquet (2026+)."
    )


def _load_tasks_parquet(data_root: Path) -> list[dict]:
    import pyarrow.parquet as pq

    tasks_path = data_root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Missing task metadata: {tasks_path}")
    table = pq.read_table(tasks_path)
    return [{col: table[col][i].as_py() for col in table.column_names} for i in range(table.num_rows)]


def _load_tasks_jsonl_if_present(data_root: Path) -> list[dict] | None:
    """Prefer meta/tasks.jsonl when present (NL prompts + task_name slugs)."""
    path = data_root / "meta" / "tasks.jsonl"
    if not path.is_file():
        # LeRobot TASKS_PATH is typically meta/tasks.jsonl
        path = data_root / TASKS_PATH
    if not path.is_file():
        return None
    return load_jsonlines(path)


def _load_episodes_parquet(data_root: Path) -> list[dict]:
    import pyarrow.parquet as pq

    episodes_dir = data_root / "meta" / "episodes"
    records: list[dict] = []
    for path in sorted(episodes_dir.rglob("*.parquet")):
        table = pq.read_table(path)
        for i in range(table.num_rows):
            records.append({col: table[col][i].as_py() for col in table.column_names})
    records.sort(key=lambda rec: rec["episode_index"])
    return records


def _is_video_tolerance_error(exc: AssertionError) -> bool:
    message = str(exc)
    return "violate the tolerance" in message and "loaded timestamps" in message


def _resolve_cached_video_path(video_path: Path, dataset_root: Path) -> Path:
    """Prefer a local SSD/NVMe copy of a video when OPENPI_VIDEO_CACHE_DIR is set."""
    cache_root = os.environ.get("OPENPI_VIDEO_CACHE_DIR")
    if not cache_root:
        return video_path
    try:
        rel = video_path.resolve().relative_to(Path(dataset_root).resolve())
    except ValueError:
        return video_path
    cached = Path(cache_root) / Path(dataset_root).name / rel
    if cached.is_file():
        return cached
    return video_path


def _decode_video_frames_with_preroll(
    video_path: Path,
    timestamps: list[float],
    tolerance_s: float,
    fps: int,
    backend: str,
) -> th.Tensor:
    """Decode B1K frames when PyAV seeks to the next keyframe instead of the previous one."""
    if backend != "pyav":
        return decode_video_frames(video_path, timestamps, tolerance_s, backend)

    first_ts = min(timestamps)
    last_ts = max(timestamps)
    preroll_s = B1K_VIDEO_GOP_FRAMES / fps + tolerance_s

    torchvision.set_video_backend("pyav")
    reader = torchvision.io.VideoReader(str(video_path), "video")
    reader.seek(max(first_ts - preroll_s, 0.0), keyframes_only=True)

    loaded_frames = []
    loaded_ts = []
    try:
        for frame in reader:
            current_ts = frame["pts"]
            loaded_frames.append(frame["data"])
            loaded_ts.append(current_ts)
            if current_ts >= last_ts + tolerance_s:
                break
    finally:
        reader.container.close()

    query_ts = th.tensor(timestamps)
    assert loaded_ts, (
        f"One or several query timestamps unexpectedly violate the tolerance ([] > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        "\nloaded timestamps: []"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )
    loaded_ts = th.tensor(loaded_ts)
    dist = th.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)
    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    closest_frames = th.stack([loaded_frames[idx] for idx in argmin_])
    return closest_frames.type(th.float32) / 255


class BehaviorLeRobotDataset(LeRobotDataset):
    """
    BehaviorLeRobotDataset is a customized dataset class for loading and managing LeRobot datasets,
    with additional filtering and loading options tailored for the BEHAVIOR-1K benchmark.
    This class extends LeRobotDataset and introduces the following customizations:
        - Task-based filtering: Load only episodes corresponding to specific tasks.
        - Modality and camera selection: Load only specified modalities (e.g., "rgb", "depth", "seg_instance_id")
          and cameras (e.g., "left_wrist", "right_wrist", "head").
        - Ability to download and use additional annotation and metainfo files.
        - Local-only mode: Optionally restrict dataset usage to local files, disabling downloads.
        - Optional batch streaming using keyframe for faster access.
    These customizations allow for more efficient and targeted dataset usage in the context of B1K tasks
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = "pyav",
        batch_encoding_size: int = 1,
        # === Customized arguments for BehaviorLeRobotDataset ===
        tasks: Iterable[str] = None,
        modalities: Iterable[str] = None,
        cameras: Iterable[str] = None,
        local_only: bool = False,
        check_timestamp_sync: bool = True,
        chunk_streaming_using_keyframe: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        fine_grained_level: int = 0,  # 0, 1, 2, 3
        train_rgb_type: str = "regular",  # regular | bbox | point
        return_seg_instance: bool = False,
        skill_list: list[str] = ["all"],
    ):
        """
        Custom args:
            episodes (List[int]): list of episodes to use PER TASK.
                NOTE: This is different from the actual episode indices in the dataset.
                Rather, this is meant to be used for train/val split, or loading a specific amount of partial data.
                If set to None, all episodes will be loaded for a given task.
            tasks (List[str]): list of task names to load. If None, all tasks will be loaded.
            modalities (List[str]): list of modality names to load. If None, all modalities will be loaded.
                must be a subset of ["rgb", "depth", "seg_instance_id"]
            cameras (List[str]): list of camera names to load. If None, all cameras will be loaded.
                must be a subset of ["left_wrist", "right_wrist", "head"]
            local_only (bool): whether to only use local data (not download from HuggingFace).
                NOTE: set this to False and force_cache_sync to True if you want to force re-syncing the local cache with the remote dataset.
                For more details, please refer to the `force_cache_sync` argument in the base class.
            check_timestamp_sync (bool): whether to check timestamp synchronization between different modalities and the state/action data.
                While it is set to True in the original LeRobotDataset and is set to True here by default, it can be set to False to skip the check for faster loading.
                This will especially save time if you are loading the complete challenge demo dataset.
            chunk_streaming_using_keyframe (bool): whether to use chunk streaming mode for loading the dataset using keyframes.
                When this is enabled, the dataset will pseudo-randomly load data in chunks based on keyframes, allowing for faster access to the data.
                NOTE: As B1K challenge demos has GOP size of 250 frames for efficient storage, it is STRONGLY recommended to set this to True if you don't need true frame-level random access.
                When this is enabled, it is recommended to set shuffle to True for better randomness in chunk selection.
                We also enforce that segmentation instance ID videos can only be loaded in chunk_streaming_using_keyframe mode for faster access.
            shuffle (bool): whether to shuffle the chunks after loading. This ONLY applies in chunk streaming mode. Recommended to be set to True for better randomness in chunk selection.
            seed (int): random seed for shuffling chunks.
            fine_grained_level (int): fine-grained level of orchestrators to use for training.
            train_rgb_type (str): type of rgb to use for training.
            return_seg_instance (bool): whether to return seg instance.
            skill_list (list[str]): ["all", "move_to:0.5"] etc.
        """
        Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = Path(os.path.expanduser(str(root))) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.revision = revision or CODEBASE_VERSION
        self.video_backend = video_backend or get_safe_default_codec()
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0
        self.return_seg_instance = return_seg_instance
        self.train_rgb_type = train_rgb_type
        self.skill_list = skill_list

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # ========== Customizations ==========
        self.seed = seed
        if modalities is None:
            modalities = ["rgb", "depth", "seg_instance_id"]
        if "seg_instance_id" in modalities:
            assert chunk_streaming_using_keyframe, "For the sake of data loading speed, please use chunk_streaming_using_keyframe=True when loading segmentation instance ID videos."
        if "depth" in modalities:
            assert self.video_backend == "pyav", (
                "Depth videos can only be decoded with the 'pyav' backend. "
                "Please set video_backend='pyav' when initializing the dataset."
            )
        if cameras is None:
            cameras = ["head", "left_wrist", "right_wrist"]
        self.task_names = set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        self.task_indices = [TASK_NAMES_TO_INDICES.get(task, 0) for task in self.task_names]
        # Load metadata
        self.meta = BehaviorLerobotDatasetMetadata(
            repo_id=self.repo_id,
            root=self.root,
            revision=self.revision,
            force_cache_sync=force_cache_sync,
            tasks=self.task_names,
            modalities=modalities,
            cameras=cameras,
        )
        self.episodes = self._select_episodes(episodes)
        self.episodes = self._keep_complete_episodes(self.episodes)
        # handle streaming mode and shuffling of episodes
        self._chunk_streaming_using_keyframe = chunk_streaming_using_keyframe
        if self._chunk_streaming_using_keyframe:
            if not shuffle:
                logger.warning(
                    "chunk_streaming_using_keyframe mode is enabled but shuffle is set to False. This may lead to less randomness in chunk selection."
                )
            self.chunks = self._get_keyframe_chunk_indices()
            # Now, we randomly permute the episodes if shuffle is True
            if shuffle:
                self.current_streaming_chunk_idx = None
                self.current_streaming_frame_idx = None
            else:
                self.current_streaming_chunk_idx = 0
                self.current_streaming_frame_idx = self.chunks[self.current_streaming_chunk_idx][0]
            self.obs_loaders = dict()
            self._should_obs_loaders_reload = True
        # record the positional index of each episode index within self.episodes
        self.episode_data_index_pos = {ep_idx: i for i, ep_idx in enumerate(self.episodes)}
        logger.info(f"Total episodes: {len(self.episodes)}")
        # ====================================

        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            if getattr(self.meta, "_layout", None) == "lerobot_parquet":
                self.stats = self.meta.stats
            else:
                episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
                self.stats = aggregate_stats(episodes_stats)

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            for fpath in self.get_episodes_file_paths():
                assert (self.root / fpath).is_file(), f"Missing file: {self.root / fpath}"
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError) as e:
            if local_only:
                raise e
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps
        if check_timestamp_sync:
            timestamps = th.stack(self.hf_dataset["timestamp"]).numpy()
            episode_indices = th.stack(self.hf_dataset["episode_index"]).numpy()
            ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
            check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        self.prepare_task(fine_grained_level)

        self.omnigibson_mapping = {ep_idx: defaultdict(dict) for ep_idx in self.episodes}

    def _select_episodes(self, episodes: list[int] | None) -> list[int]:
        layout = b1k_dataset_layout(self.root)
        epi_by_task: dict[int, list[int]] = defaultdict(list)
        if layout == "lerobot_parquet":
            for ep_idx, ep_data in sorted(self.meta.episodes.items()):
                task_names = ep_data.get("tasks", [])
                matching = [task_name for task_name in task_names if task_name in self.task_names]
                if not matching:
                    continue
                task_id = self.meta.task_to_task_index[matching[0]]
                epi_by_task[task_id].append(ep_idx)
        else:
            all_episodes = load_jsonlines(self.root / EPISODES_PATH)
            for item in all_episodes:
                if item["episode_index"] // 1e4 in self.meta.tasks:
                    epi_by_task[item["episode_index"] // 1e4].append(item["episode_index"])
        for task_id, ep_indices in epi_by_task.items():
            epi_by_task[task_id] = sorted(ep_indices)
            if episodes is not None:
                epi_by_task[task_id] = [
                    ep_indices[i] for i in episodes if i < len(ep_indices)
                ]
        return sorted(ep for eps in epi_by_task.values() for ep in eps)

    def _keep_complete_episodes(self, episodes: list[int]) -> list[int]:
        """Drop episodes whose parquet or camera videos are missing on disk."""
        kept: list[int] = []
        for ep_idx in episodes:
            data_path = self.root / self.meta.get_data_file_path(ep_idx)
            if not data_path.is_file():
                continue
            if any(
                not (self.root / self.meta.get_video_file_path(ep_idx, vid_key)).is_file()
                for vid_key in self.meta.video_keys
            ):
                continue
            kept.append(ep_idx)
        dropped = len(episodes) - len(kept)
        if dropped:
            logger.warning(
                "Dropped %s/%s episodes with missing parquet or video files under %s",
                dropped,
                len(episodes),
                self.root,
            )
        if not kept:
            raise FileNotFoundError(
                f"No complete episodes under {self.root} for tasks {sorted(self.task_names)}"
            )
        return kept

    def _alias_v3_rgb_keys(self, item: dict) -> dict:
        if b1k_dataset_layout(self.root) != "lerobot_parquet":
            return item
        for camera, v3_key in B1K_LEROBOT_V3_CAMERA_SOURCES.items():
            canonical_key = B1K_CANONICAL_RGB_KEYS[camera]
            if v3_key in item:
                item[canonical_key] = item[v3_key]
        return item

    def prepare_task(self, fine_grained_level: int):
        """set train subtask mode for lerobot dataset"""
        self.fine_grained_level = fine_grained_level

        # calculate the start and end indices of each episode
        self.task_sizes = {}
        try:
            for ep_id, ep_orch in self.meta.orchestrators.items():
                self.task_sizes[ep_id] = [task_info["end_frame"] for task_info in ep_orch[fine_grained_level]]
        except Exception as e:
            print(f"[warn] {self.repo_id} failed to calculate episode subtask cumulate: {e}")

        print(f"prepare task with fine_grained_level {self.fine_grained_level} for {self.root}")

    def get_episodes_file_paths(self) -> list[str]:
        """
        Overwrite the original method to use the episodes indices instead of range(self.meta.total_episodes)
        """
        episodes = self.episodes if self.episodes is not None else list(self.meta.episodes.keys())
        if b1k_dataset_layout(self.root) == "lerobot_parquet":
            fpaths = sorted(
                {str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes}
            )
            if len(self.meta.video_keys) > 0:
                fpaths += sorted(
                    {
                        str(self.root / self.meta.get_video_file_path(ep_idx, vid_key))
                        for vid_key in self.meta.video_keys
                        for ep_idx in episodes
                    }
                )
            return fpaths

        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        # append metainfo and language annotations
        fpaths += [str(self.meta.get_metainfo_path(ep_idx)) for ep_idx in episodes]
        # TODO: add this back once we have all the language annotations
        # fpaths += [str(self.meta.get_annotation_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths

    def download_episodes(self, download_videos: bool = True) -> None:
        """
        Overwrite base method to allow more flexible pattern matching.
        Here, we do coarse filtering based on tasks, cameras, and modalities.
        We do this instead of filename patterns to speed up pattern checking and download speed.
        """
        allow_patterns = []
        if set(self.task_indices) != set(TASK_NAMES_TO_INDICES.values()):
            for task in self.task_indices:
                allow_patterns.append(f"**/task-{task:04d}/**")
        if len(self.meta.modalities) != 3:
            for modality in self.meta.modalities:
                if len(self.meta.camera_names) != 3:
                    for camera in self.meta.camera_names:
                        allow_patterns.append(f"**/observation.images.{modality}.{camera}/**")
                else:
                    allow_patterns.append(f"**/observation.images.{modality}.*/**")
        elif len(self.meta.camera_names) != 3:
            for camera in self.meta.camera_names:
                allow_patterns.append(f"**/observation.images.*.{camera}/**")
        ignore_patterns = []
        if not download_videos:
            ignore_patterns.append("videos/")
        if set(self.task_indices) != set(TASK_NAMES_TO_INDICES.values()):
            for task in set(TASK_NAMES_TO_INDICES.values()).difference(self.task_indices):
                ignore_patterns.append(f"**/task-{task:04d}/**")

        allow_patterns = None if allow_patterns == [] else allow_patterns
        ignore_patterns = None if ignore_patterns == [] else ignore_patterns
        self.pull_from_repo(allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        """
        Overwrite base class to increase max workers to num of CPUs - 2
        """
        logger.info(f"Pulling dataset {self.repo_id} from HuggingFace hub...")
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=os.cpu_count() - 2,
        )

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if b1k_dataset_layout(self.root) == "lerobot_parquet":
            files = sorted({str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes})
            hf_dataset = load_dataset("parquet", data_files=files, split="train")
            hf_dataset.set_transform(hf_transform_to_torch)
            return hf_dataset

        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def __getitem__(self, idx) -> dict:
        if not self._chunk_streaming_using_keyframe:
            max_attempts = min(32, len(self))
            for attempt in range(max_attempts):
                candidate_idx = (int(idx) + attempt) % len(self)
                try:
                    item = super().__getitem__(candidate_idx)
                    item["task"] = self._get_fine_grained_task(item)
                    return self._alias_v3_rgb_keys(item)
                except AssertionError as exc:
                    if not _is_video_tolerance_error(exc):
                        raise
                    logger.warning(
                        "Skipping sample %s due to video timestamp mismatch on attempt %s/%s.",
                        candidate_idx,
                        attempt + 1,
                        max_attempts,
                    )
            raise RuntimeError(f"Failed to load a valid sample after {max_attempts} attempts starting from {idx}.")

        # Streaming mode: we will load the episode at the current streaming index, and then increment the index for next call
        # Randomize chunk index on first call
        if self.current_streaming_chunk_idx is None:
            worker_info = get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id
            num_workers = 1 if worker_info is None else worker_info.num_workers
            if not hasattr(self, "_active_chunks") or self._active_chunks is None:
                indices = list(range(worker_id, len(self.chunks), num_workers))
                worker_chunks = [self.chunks[i] for i in indices]
                rng = np.random.default_rng(self.seed + worker_id)
                rng.shuffle(worker_chunks)
                self._active_chunks = worker_chunks
            rng = np.random.default_rng(self.seed + worker_id)
            self.current_streaming_chunk_idx = rng.integers(0, len(self._active_chunks)).item()
            self.current_streaming_frame_idx = self._active_chunks[self.current_streaming_chunk_idx][0]
        # Current chunk iterated, move to next chunk
        if self.current_streaming_frame_idx >= self._active_chunks[self.current_streaming_chunk_idx][1]:
            self.current_streaming_chunk_idx += 1
            # All data iterated, restart from beginning
            if self.current_streaming_chunk_idx >= len(self._active_chunks):
                self.current_streaming_chunk_idx = 0
            self.current_streaming_frame_idx = self._active_chunks[self.current_streaming_chunk_idx][0]
            self._should_obs_loaders_reload = True
        item = self.hf_dataset[self.current_streaming_frame_idx]
        item.pop("observation.task_info", None)
        ep_idx = item["episode_index"].item()

        if self._should_obs_loaders_reload:
            from omnigibson.eval.utils.obs_utils import OBS_LOADER_MAP

            for loader in self.obs_loaders.values():
                loader.close()
            self.obs_loaders = dict()
            # reload video loaders for new episode
            self.current_streaming_episode_idx = ep_idx
            for vid_key in self.meta.video_keys:
                kwargs = {}
                task_id = item["task_index"].item()
                if "seg_instance_id" in vid_key:
                    # load id list
                    with open(
                        self.root / "meta/episodes" / f"task-{task_id:04d}" / f"episode_{ep_idx:08d}.json",
                    ) as f:
                        meta = json.load(f)
                        instance_id_mapping = json.loads(meta["ins_id_mapping"])
                        instance_id_mapping = {int(k): v for k, v in instance_id_mapping.items()}
                        self.omnigibson_mapping[ep_idx]["instance_id_mapping"] = instance_id_mapping
                        self.omnigibson_mapping[ep_idx]["unique_ins_ids"][vid_key.split(".")[-1]] = meta[
                            f"{ROBOT_CAMERA_NAMES['R1Pro'][vid_key.split('.')[-1]]}::unique_ins_ids"
                        ]
                        kwargs["id_list"] = th.tensor(
                            self.omnigibson_mapping[ep_idx]["unique_ins_ids"][vid_key.split(".")[-1]]
                        )
                if "rgb" in vid_key:
                    kwargs["train_rgb_type"] = self.train_rgb_type
                self.obs_loaders[vid_key] = iter(
                    OBS_LOADER_MAP[vid_key.split(".")[2]](
                        data_path=self.root,
                        task_id=task_id,
                        camera_id=vid_key.split(".")[-1],
                        demo_id=f"{ep_idx:08d}",
                        start_idx=self._active_chunks[self.current_streaming_chunk_idx][2],
                        start_idx_is_keyframe=False,
                        batch_size=1,
                        stride=1,
                        **kwargs,
                    )
                )
            self._should_obs_loaders_reload = False

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(self.current_streaming_frame_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        task_skill = self._get_current_task_skill(item)
        weight = skill_weight(task_skill, self.skill_list)
        if not random.choices([True, False], weights=[weight, 1 - weight])[0]:
            self.current_streaming_frame_idx += 1
            for key in self.meta.video_keys:
                next(self.obs_loaders[key])[0]
            return self.__getitem__(idx)

        # load visual observations
        for key in self.meta.video_keys:
            item[key] = next(self.obs_loaders[key])[0]

            if self.return_seg_instance and "seg_instance_id" in key:
                seg_instance, instance_mapping = instance_id_to_instance(
                    obs=item[key],
                    instance_id_mapping=self.omnigibson_mapping[ep_idx]["instance_id_mapping"],
                    unique_ins_ids=np.array(self.omnigibson_mapping[ep_idx]["unique_ins_ids"][key.split(".")[-1]]),
                )
                instance_mapping = {instance_name: id for id, instance_name in instance_mapping.items()}

                frame_index = round(item["timestamp"].item() * self.fps)
                sub_idx = bisect.bisect_right(self.task_sizes[ep_idx], frame_index, hi=len(self.task_sizes[ep_idx]) - 1)
                skill_annotation = self.meta.annotations[ep_idx]["skill_annotation"]
                relative_obj_names = skill_annotation[sub_idx]["object_id"][0]
                for i, relative_obj_name in enumerate(relative_obj_names):
                    instance_id = instance_mapping[relative_obj_name]
                    seg_instance[seg_instance == instance_id] = -(i + 1)
                seg_instance[seg_instance > 0] = 0
                seg_instance *= -1
                item[key.replace("seg_instance_id", "seg_instance")] = seg_instance

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        item["task"] = self._get_fine_grained_task(item)
        self.current_streaming_frame_idx += 1

        return self._alias_v3_rgb_keys(item)

    def _get_current_task_skill(self, item: dict) -> str:
        ep_idx = item["episode_index"].item()
        frame_index = round(item["timestamp"].item() * self.fps)
        sub_idx = bisect.bisect_right(self.task_sizes[ep_idx], frame_index, hi=len(self.task_sizes[ep_idx]) - 1)
        task_skill = self.meta.orchestrators[ep_idx][1][sub_idx]["task"]
        return task_skill

    def _get_fine_grained_task(self, item: dict) -> str:
        ep_idx = item["episode_index"].item()
        task_idx = item["task_index"].item()
        # Prefer natural-language task prompts from meta/tasks for global (level-0) instructions.
        if self.fine_grained_level == 0 and task_idx in self.meta.tasks:
            return self.meta.tasks[task_idx]
        frame_index = round(item["timestamp"].item() * self.fps)
        try:
            sub_idx = bisect.bisect_right(self.task_sizes[ep_idx], frame_index, hi=len(self.task_sizes[ep_idx]) - 1)
            task_text = self.meta.orchestrators[ep_idx][self.fine_grained_level][sub_idx]["task"]

        except Exception as e:
            print(f"[warn] {self.repo_id} failed to get subtask {item}: {e}")
            task_text = self.meta.tasks[task_idx]
        return task_text

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_idx = self.episode_data_index_pos[ep_idx]
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": th.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, th.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = _resolve_cached_video_path(
                self.root / self.meta.get_video_file_path(ep_idx, vid_key),
                self.root,
            )
            try:
                frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            except AssertionError as exc:
                if not _is_video_tolerance_error(exc):
                    raise
                frames = _decode_video_frames_with_preroll(
                    video_path,
                    query_ts,
                    self.tolerance_s,
                    self.fps,
                    self.video_backend,
                )
            item[vid_key] = frames.squeeze(0)

        return item

    def _get_keyframe_chunk_indices(self, chunk_size=250) -> list[tuple[int, int, int]]:
        """
        Divide each episode into chunks of data based on GOP of the data (here for B1K, GOP size is 250 frames).
        Args:
            chunk_size (int): size of each chunk in number of frames. Default is 250 for B1K. Should be the GOP size of the video data.
        Returns:
            List of tuples, where each tuple contains (start_index, end_index, local_start_index) for each chunk.
        """
        episode_lengths = {ep_idx: ep_dict["length"] for ep_idx, ep_dict in self.meta.episodes.items()}
        episode_lengths = [episode_lengths[ep_idx] for ep_idx in self.episodes]
        chunks = []
        offset = 0
        for L in episode_lengths:
            local_starts = list(range(0, L, chunk_size))
            local_ends = local_starts[1:] + [L]
            for ls, le in zip(local_starts, local_ends):
                chunks.append((offset + ls, offset + le, ls))
            offset += L
        return chunks


class BehaviorLerobotDatasetMetadata(LeRobotDatasetMetadata):
    """
    BehaviorLerobotDatasetMetadata extends LeRobotDatasetMetadata with the following customizations:
        1. Restricts the set of allowed modalities to {"rgb", "depth", "seg_instance_id"}.
        2. Restricts the set of allowed camera names to those defined in ROBOT_CAMERA_NAMES["R1Pro"].
        3. Provides a filtered view of dataset features, including only those corresponding to the selected modalities and camera names.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        # === Customized arguments for BehaviorLeRobotDataset ===
        tasks: Iterable[str] = None,
        modalities: Iterable[str] = None,
        cameras: Iterable[str] = None,
    ):
        # ========== Customizations ==========
        self.task_name_candidates = set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        self.modalities = set(modalities)
        self.camera_names = set(cameras)
        assert self.modalities.issubset(
            {"rgb", "depth", "seg_instance_id"}
        ), f"Modalities must be a subset of ['rgb', 'depth', 'seg_instance_id'], but got {self.modalities}"
        assert self.camera_names.issubset(
            ROBOT_CAMERA_NAMES["R1Pro"]
        ), f"Camera names must be a subset of {ROBOT_CAMERA_NAMES['R1Pro']}, but got {self.camera_names}"
        # ===================================

        self.repo_id = repo_id
        self.revision = revision or CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/**", ignore_patterns="meta/episodes/**")
            self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        self._layout = b1k_dataset_layout(self.root)
        if self._layout != "lerobot_parquet":
            check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index, self.task_names = self.load_tasks(self.root)
        # filter based on self.task_name_candidates
        valid_task_indices = [idx for idx, name in self.task_names.items() if name in self.task_name_candidates]
        # Keep slug -> index for episode/task lookups; prompts live in self.tasks.
        self.task_to_task_index = {self.task_names[idx]: idx for idx in valid_task_indices}
        self.tasks = {idx: self.tasks[idx] for idx in valid_task_indices}
        self.task_names = set(self.task_names[idx] for idx in valid_task_indices)

        self.episodes = self.load_episodes(self.root)
        self.annotations = self.load_annotations(self.root)
        self.orchestrators = self.load_orchestrators(self.root)
        if self._layout == "lerobot_parquet":
            self.stats = self.load_stats(self.root)
            if self.stats is None:
                raise FileNotFoundError(f"Missing dataset stats at {self.root / STATS_PATH}")
            self.episodes_stats = {}
        elif self._version < packaging.version.parse("v2.1"):
            self.stats = self.load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        else:
            self.episodes_stats = self.load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))
        logger.info(f"Loaded metadata for {len(self.episodes)} episodes.")

    def get_data_file_path(self, ep_index: int) -> Path:
        if getattr(self, "_layout", None) == "lerobot_parquet":
            ep = self.episodes[ep_index]
            return Path(
                self.data_path.format(
                    chunk_index=ep["data/chunk_index"],
                    file_index=ep["data/file_index"],
                )
            )
        return super().get_data_file_path(ep_index)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        if getattr(self, "_layout", None) == "lerobot_parquet":
            ep = self.episodes[ep_index]
            chunk_key = f"videos/{vid_key}/chunk_index"
            file_key = f"videos/{vid_key}/file_index"
            return Path(
                self.video_path.format(
                    video_key=vid_key,
                    chunk_index=ep[chunk_key],
                    file_index=ep[file_key],
                )
            )
        return super().get_video_file_path(ep_index, vid_key)

    def load_tasks(self, local_dir: Path) -> tuple[dict, dict, dict]:
        if b1k_dataset_layout(local_dir) == "lerobot_parquet":
            jsonl_tasks = _load_tasks_jsonl_if_present(local_dir)
            if jsonl_tasks is not None and all("task_name" in item and "task" in item for item in jsonl_tasks):
                # tasks.jsonl: task_name=slug, task=natural-language prompt
                task_names = {
                    item["task_index"]: item["task_name"] for item in sorted(jsonl_tasks, key=lambda x: x["task_index"])
                }
                task_map = {item["task_index"]: item["task"] for item in sorted(jsonl_tasks, key=lambda x: x["task_index"])}
            else:
                tasks = _load_tasks_parquet(local_dir)
                # tasks.parquet currently stores the slug in the `task` column
                task_names = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
                task_map = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
            task_to_task_index = {task: task_index for task_index, task in task_names.items()}
            return task_map, task_to_task_index, task_names

        tasks = load_jsonlines(local_dir / TASKS_PATH)
        task_names = {item["task_index"]: item["task_name"] for item in sorted(tasks, key=lambda x: x["task_index"])}
        tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
        task_to_task_index = {task: task_index for task_index, task in task_names.items()}
        return tasks, task_to_task_index, task_names

    def load_episodes(self, local_dir: Path) -> dict:
        if getattr(self, "_layout", None) == "lerobot_parquet" or b1k_dataset_layout(local_dir) == "lerobot_parquet":
            return {
                rec["episode_index"]: rec
                for rec in _load_episodes_parquet(local_dir)
                if any(task_name in self.task_name_candidates for task_name in rec.get("tasks", []))
            }

        episodes = load_jsonlines(local_dir / EPISODES_PATH)
        return {
            item["episode_index"]: item
            for item in sorted(episodes, key=lambda x: x["episode_index"])
            if item["episode_index"] // 1e4 in self.tasks
        }

    def load_stats(self, local_dir: Path) -> dict[str, dict[str, np.ndarray]]:
        if not (local_dir / STATS_PATH).exists():
            return None
        stats = load_json(local_dir / STATS_PATH)
        return cast_stats_to_numpy(stats)

    def load_episodes_stats(self, local_dir: Path) -> dict:
        episodes_stats = load_jsonlines(local_dir / EPISODES_STATS_PATH)
        return {
            item["episode_index"]: cast_stats_to_numpy(item["stats"])
            for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
            if item["episode_index"] in self.episodes
        }

    def load_annotations(self, local_dir: Path) -> dict:
        annotations = local_dir / ANNOTATIONS_PATH
        if not annotations.is_dir():
            return {}
        task_list = [task_id for task_id in annotations.iterdir() if task_id.is_dir()]
        return {
            int(episode.stem[8:]): load_json(episode)
            for task_id in task_list
            if int(task_id.name[5:]) in self.tasks
            for episode in sorted(task_id.iterdir())
        }

    def load_orchestrators(self, local_dir: Path) -> dict:
        orchestrators_path = local_dir / ORCHESTRATORS_PATH
        orchestrators = {
            episode_key: load_orchestrators_data(episode_data["tasks"][0], episode_data["length"])
            for episode_key, episode_data in sorted(self.episodes.items())
        }
        if orchestrators_path.exists():
            for task in self.tasks:
                if (orchestrators_path / f"task-{task:04d}").exists():
                    orchestrators.update(
                        {
                            int(episode.stem[8:]): load_orchestrators_data(
                                episode, self.episodes[int(episode.stem[8:])]["length"]
                            )
                            for episode in sorted((orchestrators_path / f"task-{task:04d}").iterdir())
                        }
                    )
        return orchestrators

    def get_annotation_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.annotation_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_metainfo_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.metainfo_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    @property
    def annotation_path(self) -> str | None:
        """Formattable string for the annotation files."""
        return self.info["annotation_path"]

    @property
    def metainfo_path(self) -> str | None:
        """Formattable string for the metainfo files."""
        return self.info["metainfo_path"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        if getattr(self, "_layout", None) == "lerobot_parquet":
            features = {}
            if "rgb" in self.modalities:
                for camera in self.camera_names:
                    v3_key = B1K_LEROBOT_V3_CAMERA_SOURCES[camera]
                    if v3_key in self.info["features"]:
                        features[v3_key] = self.info["features"][v3_key]
            return features

        features = dict()
        # pop not required features
        for name in self.info["features"].keys():
            if (
                name.startswith("observation.images.")
                and name.split(".")[-1] in self.camera_names
                and name.split(".")[-2] in self.modalities
            ):
                features[name] = self.info["features"][name]
        return features


def load_orchestrators_data(episode_path_or_level_0_task, episode_len):
    output_data = defaultdict(list)
    if type(episode_path_or_level_0_task) == str:
        for i in range(4):
            output_data[i] = [
                {
                    "task": episode_path_or_level_0_task,
                    "start_frame": 0,
                    "end_frame": episode_len - 1,
                }
            ]
        return output_data
    episode_path = episode_path_or_level_0_task
    task_annotated_data = load_json(episode_path / "task_annotated.json")
    level_0_task = task_annotated_data["cot_task_description"]
    output_data[0].append(
        {
            "task": level_0_task,
            "start_frame": 0,
            "end_frame": episode_len - 1,
        }
    )
    try:
        num_level1_tasks = len(task_annotated_data["cot_subtask_description_list"])
        for i in range(num_level1_tasks):
            subtask_data = load_json(episode_path / f"subtask_{i}_annotated.json")
            subtask = subtask_data["cot_subtask_description"]
            start_frame, end_frame = subtask_data["start_frame"], subtask_data["end_frame"] - 1
            skill = subtask_data["skill_description"]
            output_data[1].append(
                {
                    "task": skill,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            )
            output_data[2].append(
                {
                    "task": subtask,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            )
            for event_data_path in sorted(episode_path.glob(f"event_{i}_*_annotated.json")):
                event_data = load_json(event_data_path)
                event_task = event_data["subtask_answer_detailed"]
                start_frame, end_frame = event_data["start_frame"], event_data["end_frame"] - 1
                output_data[3].append(
                    {
                        "task": event_task,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                    }
                )
    except Exception as e:
        print(f"[warn] {episode_path} failed to load orchestrators data: {e}, falling back to default task.")
        for i in range(len(output_data)):
            output_data[i] = output_data[0]
    return output_data


def skill_weight(cur_skill, skill_list: list[str]) -> float:
    if "all" in skill_list:
        skill_list = [skill for skill in skill_list if skill != "all"]
        for skill_item in skill_list:
            skill, weight = skill_item.split(":")
            if skill == cur_skill:
                return float(weight)
        return 1.0
    for skill_item in skill_list:
        skill, weight = skill_item.split(":")
        if skill == cur_skill:
            return float(weight)
    return 0.0


class MultiBehaviorLeRobotDataset:
    def __init__(self, datasets: list[BehaviorLeRobotDataset], sample_weights: list[float] | None = None):
        if sample_weights is None:
            sample_weights = [1.0 / len(datasets)] * len(datasets)
        assert len(datasets) == len(sample_weights), "Length of datasets and sample weights must be the same"
        if sum(sample_weights) != 1.0:
            sample_weights = [weight / sum(sample_weights) for weight in sample_weights]

        self.datasets = datasets
        self.sample_weights = sample_weights

    def __len__(self):
        return max(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        index = np.random.choice(range(len(self.datasets)), p=self.sample_weights)
        dataset = self.datasets[index]
        return dataset[int(idx) % len(dataset)]
