"""Data loading for BEHAVIOR-1K dataset.

Reference: https://github.com/wensi-ai/openpi/tree/behavior
"""

import logging
import time

# Import all base data loading from OpenPI
from openpi.training.data_loader import (
    Dataset,
    IterableDataset,
    DataLoader,
    TransformedDataset,
    IterableTransformedDataset,
    FakeDataset,
    TorchDataLoader,
)

from b1k.training import config as _config
import openpi.transforms as _transforms

from b1k.models.observation import Observation
from b1k.transforms_normalize import NormalizeWithPerTimestamp


class DataLoaderImpl(DataLoader):
    """Custom DataLoader using our Observation with fast_tokens."""
    
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield Observation.from_dict(batch), batch["actions"]


def _episode_lengths_from_meta(dataset) -> dict[int, float]:
    lengths: dict[int, float] = {}
    meta = getattr(dataset, "meta", None)
    episodes = getattr(meta, "episodes", None) if meta is not None else None
    if not episodes:
        return lengths
    for episode_index, info in episodes.items():
        length = info.get("length", info.get("episode_length", 0))
        lengths[int(episode_index)] = float(length or 0)
    return lengths


def create_behavior_dataset(data_config: _config.DataConfig, action_horizon: int, seed: int | None = None) -> Dataset:
    """Create a BEHAVIOR-1K dataset for training.

    Uses the 2026-capable BehaviorLeRobotDataset (LeRobot v3 parquet + 2025 jsonl).
    """
    from b1k.datasets.lerobot_dataset import BehaviorLeRobotDataset
    from b1k import transforms as b1k_transforms

    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed for BehaviorLeRobotDataset: {seed}")

    tasks = data_config.tasks

    dataset = BehaviorLeRobotDataset(
        repo_id=data_config.repo_id,
        root=data_config.behavior_dataset_root,
        tasks=tasks,
        modalities=list(data_config.modalities) if data_config.modalities else ["rgb"],
        local_only=True,
        delta_timestamps={
            key: [t / 30.0 for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        episodes=data_config.episodes_index,
        chunk_streaming_using_keyframe=False,
        shuffle=True,
        seed=seed,
        tolerance_s=getattr(data_config, "tolerance_s", 1.0 / 30.0),
        check_timestamp_sync=False,
        fine_grained_level=0,
    )

    pre_transforms = []
    if getattr(data_config, "align_legacy_rft_to_2026", False):
        pre_transforms.append(b1k_transforms.AlignLegacyRftTo2026())
    pre_transforms.append(
        b1k_transforms.AttachEpisodeMetadata(
            episode_lengths=_episode_lengths_from_meta(dataset),
            force_task_index=getattr(data_config, "force_task_index", None),
        )
    )
    dataset = TransformedDataset(dataset, pre_transforms)

    if data_config.prompt_from_task:
        inner = dataset._dataset if hasattr(dataset, "_dataset") else dataset
        if hasattr(inner, "meta") and hasattr(inner.meta, "tasks"):
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(inner.meta.tasks)])

    return dataset


def create_multi_behavior_dataset(
    data_configs: list, sample_weights: list[float] | None, action_horizon: int, seed: int | None = None
) -> Dataset:
    from b1k.datasets.lerobot_dataset import MultiBehaviorLeRobotDataset

    datasets = [create_behavior_dataset(data_config, action_horizon, seed=seed) for data_config in data_configs]
    return MultiBehaviorLeRobotDataset(datasets, sample_weights=sample_weights)


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform dataset with B1K-specific per-timestamp normalization support.
    
    CRITICAL: This overrides wensi-ai's transform_dataset to pass use_per_timestamp to Normalize.
    wensi-ai's version doesn't support per-timestamp normalization which causes huge action losses!
    """
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    # Build transform list
    transforms_list = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        # Use custom Normalize with per-timestamp support (wensi-ai's doesn't have it!)
        NormalizeWithPerTimestamp(
            norm_stats, 
            use_quantiles=data_config.use_quantile_norm,
            use_per_timestamp=data_config.use_per_timestamp_norm  # CRITICAL: Per-timestamp normalization!
        ),
    ]
    
    # Add subtask state computation for PI_BEHAVIOR models
    model_transforms = []
    for transform in data_config.model_transforms.inputs:
        if hasattr(transform, "__class__") and transform.__class__.__name__ == "ComputeSubtaskStateFromMeta":
            from b1k import transforms as b1k_transforms
            model_transforms.append(b1k_transforms.ComputeSubtaskStateFromMeta(dataset=dataset))
        else:
            model_transforms.append(transform)
    
    transforms_list.extend(model_transforms)

    return TransformedDataset(dataset, transforms_list)


def extract_episode_lengths_from_dataset(dataset) -> dict[int, float]:
    """Extract episode lengths from B1K dataset metadata.
    
    Args:
        dataset: BehaviorLeRobotDataset instance
        
    Returns:
        Dictionary mapping episode_index to episode_length (in frames)
        
    Raises:
        ValueError: If dataset doesn't have required metadata
    """
    if not hasattr(dataset, 'episode_data_index'):
        raise ValueError("Dataset must have episode_data_index attribute")
    
    episode_data_index = dataset.episode_data_index
    if 'to' not in episode_data_index or 'from' not in episode_data_index:
        raise ValueError("episode_data_index must have 'to' and 'from' keys")
    
    episode_to = episode_data_index['to'] 
    episode_from = episode_data_index['from']
    episodes = dataset.episodes
    
    episode_lengths = {}
    for i, episode_index in enumerate(episodes):
        if i < len(episode_to) and i < len(episode_from):
            episode_length = episode_to[i] - episode_from[i]
            episode_lengths[episode_index] = float(episode_length)
    
    logging.info(f"Extracted {len(episode_lengths)} episode lengths from dataset")
    return episode_lengths


def create_behavior_data_loader(
    config: _config.TrainConfig,
    *,
    sharding=None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader:
    """Create a data loader for BEHAVIOR-1K training."""
    import jax
    import time
    
    factories = _config.get_data_factories(config)
    data_configs = [factory.create(config.assets_dirs, config.model) for factory in factories]
    data_config = data_configs[0]

    seed = config.seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed: {seed}")

    if len(data_configs) > 1:
        dataset = create_multi_behavior_dataset(
            data_configs,
            sample_weights=getattr(config, "sample_weights", None),
            action_horizon=config.model.action_horizon,
            seed=seed,
        )
    else:
        dataset = create_behavior_dataset(data_config, action_horizon=config.model.action_horizon, seed=seed)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=seed,
    )
    
    return DataLoaderImpl(data_config, data_loader)
