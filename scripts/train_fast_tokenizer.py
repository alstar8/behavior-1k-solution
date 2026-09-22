"""Train FAST tokenizer for action encoding.

This script:
1. Loads action chunks from B1K dataset (with sampling)
2. Applies delta transforms and per-timestamp normalization
3. Trains FAST tokenizer on specified action dimensions
4. Saves tokenizer to assets directory
5. Reports compression statistics
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
import numpy as np
import pandas as pd
import tyro
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoProcessor

import openpi.transforms as transforms

# Import B1K-specific modules
from b1k.training import config as _config
from b1k.shared.b1k_proprio import extract_state_from_proprio, legacy_proprio_to_compact


def get_delta_transform_from_config(config_name: str):
    """Get the delta action transform from config."""
    config = _config.get_config(config_name)
    factory = _config.get_data_factories(config)[0]
    data_config = factory.create(config.assets_dirs, config.model)
    
    delta_transform = None
    for transform in data_config.data_transforms.inputs:
        if isinstance(transform, transforms.DeltaActions):
            delta_transform = transform
            break
    
    if delta_transform is None:
        raise ValueError("No DeltaActions transform found in config")
    
    return delta_transform


def apply_delta_transform(state: np.ndarray, actions: np.ndarray, mask) -> np.ndarray:
    """Apply delta transform using the mask from config."""
    if mask is None:
        return actions
    
    delta_actions = actions.copy()
    mask = np.asarray(mask)
    dims = mask.shape[-1]
    delta_actions[:dims] = np.where(mask, actions[:dims] - state[:dims], actions[:dims])
    
    return delta_actions


def process_episode_file(args):
    """Process episode file and return action chunks."""
    episode_file, delta_mask, action_horizon, sample_fraction, align_legacy = args
    
    try:
        df = pd.read_parquet(episode_file)
        if "episode_index" in df.columns:
            groups = list(df.groupby("episode_index", sort=True))
        else:
            groups = [(0, df)]

        all_chunks = []
        for _, ep_df in groups:
            raw_state = np.asarray(ep_df["observation.state"].tolist(), dtype=np.float32)
            raw_actions = np.asarray(ep_df["action"].tolist(), dtype=np.float32)
            if align_legacy:
                raw_state = legacy_proprio_to_compact(raw_state)
            states = extract_state_from_proprio(raw_state)
            if states.ndim == 1:
                states = states[None, :]
                raw_actions = raw_actions[None, :]

            if len(raw_actions) < action_horizon:
                continue

            mask = np.asarray(delta_mask)
            dims = mask.shape[-1]
            windows = np.lib.stride_tricks.sliding_window_view(raw_actions, (action_horizon, raw_actions.shape[1]))[:, 0]
            state_exp = states[: windows.shape[0], None, :]
            action_chunks = windows.copy()
            action_chunks[..., :dims] = np.where(
                mask, windows[..., :dims] - state_exp[..., :dims], windows[..., :dims]
            )
            if sample_fraction < 1.0:
                n_chunks = len(action_chunks)
                n_samples = max(1, int(n_chunks * sample_fraction))
                episode_seed = hash(str(episode_file)) % (2**31)
                rng = np.random.RandomState(episode_seed)
                indices = rng.choice(n_chunks, size=n_samples, replace=False)
                action_chunks = action_chunks[indices]
            all_chunks.append(action_chunks)

        if not all_chunks:
            return None
        return np.concatenate(all_chunks, axis=0)
        
    except Exception as e:
        print(f"Error processing {episode_file}: {e}")
        return None


def train_fast_tokenizer(
    action_chunks: np.ndarray,
    vocab_size: int = 1024,
    scale: float = 10.0,
    num_epochs: int = 10,
    batch_size: int = 256,
) -> AutoProcessor:
    """
    Train FAST tokenizer (BPE on DCT coefficients) on action chunks.
    
    Uses the .fit() method to train a new tokenizer on the provided data.
    """
    print(f"Training FAST tokenizer on {len(action_chunks)} action chunks...")
    print(f"Action chunk shape: {action_chunks.shape}")
    print(f"Vocab size: {vocab_size}")
    print(f"DCT scale: {scale}")
    
    # Download the tokenizer source code (not pretrained weights)
    # We'll train a new tokenizer on our own data
    base_tokenizer = AutoProcessor.from_pretrained(
        "physical-intelligence/fast",
        trust_remote_code=True
    )
    
    # Convert action_chunks array to list of arrays (expected by .fit())
    action_data_list = [action_chunks[i] for i in range(len(action_chunks))]
    
    # Train the new tokenizer on our action data using .fit()
    # This trains the BPE tokenizer on DCT coefficients
    print("Training new tokenizer (this may take a few minutes)...")
    tokenizer = base_tokenizer.fit(
        action_data_list,
        scale=scale,
        vocab_size=vocab_size,
        time_horizon=action_chunks.shape[1],  # action_horizon
        action_dim=action_chunks.shape[2],     # encoded dimensions
    )
    print("✓ Tokenizer training complete!")
    
    # Validate it works
    sample_chunk = action_chunks[0]
    encoded = tokenizer(sample_chunk[None])[0]
    if isinstance(encoded, list):
        encoded = np.array(encoded)
    print(f"Sample encoding: {len(encoded)} tokens for chunk shape {sample_chunk.shape}")
    
    return tokenizer


def compute_compression_stats(tokenizer, action_chunks: np.ndarray):
    """Compute compression statistics."""
    print("\nComputing compression statistics...")
    
    # Sample for stats (use max 1000 chunks for speed)
    sample_size = min(1000, len(action_chunks))
    sample_indices = np.random.RandomState(42).choice(len(action_chunks), size=sample_size, replace=False)
    sample_chunks = action_chunks[sample_indices]
    
    token_lengths = []
    for chunk in sample_chunks:
        encoded = tokenizer(chunk[None])[0]
        if isinstance(encoded, list):
            token_lengths.append(len(encoded))
        else:
            token_lengths.append(encoded.shape[0] if hasattr(encoded, 'shape') else len(encoded))
    
    token_lengths = np.array(token_lengths)
    
    # Compression ratio: (H * D) / avg_tokens
    input_size = action_chunks.shape[1] * action_chunks.shape[2]
    avg_tokens = np.mean(token_lengths)
    compression_ratio = input_size / avg_tokens
    
    stats = {
        'compression_ratio': float(compression_ratio),
        'mean_token_length': float(np.mean(token_lengths)),
        'p99_token_length': float(np.percentile(token_lengths, 99)),
        'min_token_length': float(np.min(token_lengths)),
        'max_token_length': float(np.max(token_lengths)),
    }
    
    print(f"Compression Statistics:")
    print(f"  Average compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Mean token length: {stats['mean_token_length']:.1f}")
    print(f"  P99 token length: {stats['p99_token_length']:.0f}")
    print(f"  Min token length: {stats['min_token_length']:.0f}")
    print(f"  Max token length: {stats['max_token_length']:.0f}")
    
    return stats


def main(
    config_name: str,
    max_episodes: int | None = None,
    num_workers: int | None = None,
    sample_fraction: float = 0.1,
    encoded_dims: str = "0:6,7:23",
    vocab_size: int = 1024,
    scale: float = 10.0,
    num_epochs: int = 10,
):
    """
    Train FAST tokenizer for action encoding.
    
    Args:
        config_name: Training config name
        max_episodes: Max episodes to use (None = all episodes in dataset)
        num_workers: Number of parallel workers
        sample_fraction: Fraction of chunks to sample per episode
        encoded_dims: Comma-separated dimension ranges (e.g., "0:6,7:23")
        vocab_size: FAST vocabulary size (BPE vocab size)
        scale: DCT scaling factor (default: 10.0)
        num_epochs: Training epochs (not used by FAST)
    """
    # Load config
    config = _config.get_config(config_name)
    factory = _config.get_data_factories(config)[0]
    data_config = factory.create(config.assets_dirs, config.model)
    
    if not data_config.behavior_dataset_root:
        raise ValueError("This script only works with B1K behavior datasets")
    
    # Parse encoded dimensions
    encoded_dim_ranges = []
    for range_str in encoded_dims.split(','):
        start, end = map(int, range_str.strip().split(':'))
        encoded_dim_ranges.append((start, end))
    
    total_encoded_dims = sum(end - start for start, end in encoded_dim_ranges)
    print(f"Encoding {total_encoded_dims} dimensions: {encoded_dims}")
    
    # Get delta transform
    delta_transform = get_delta_transform_from_config(config_name)
    delta_mask = delta_transform.mask
    action_horizon = config.model.action_horizon
    
    print(f"Action horizon: {action_horizon}")
    print(f"Delta mask: {delta_mask}")
    
    jobs = []
    for src_factory in _config.get_data_factories(config):
        base = src_factory.base_config or _config.DataConfig()
        root = Path(str(base.behavior_dataset_root)).expanduser()
        tasks = getattr(base, "tasks", None)
        align_legacy = bool(getattr(base, "align_legacy_rft_to_2026", False))
        chunk_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
        episode_files = sorted(root.glob("data/task-*/episode_*.parquet"))
        if tasks == ["turning_on_radio"]:
            chunk_files = [path for path in chunk_files if "chunk-000" in str(path)]
            episode_files = [path for path in episode_files if "task-0000" in str(path)]
        files = chunk_files + episode_files
        print(f"Found {len(files)} parquet files under {root} (align_legacy={align_legacy})")
        for path in files:
            jobs.append((path, delta_mask, action_horizon, sample_fraction, align_legacy))

    if max_episodes is not None:
        jobs = jobs[:max_episodes]
        print(f"Limited to {len(jobs)} files")

    if not jobs:
        raise ValueError("No episode files found")

    import multiprocessing as mp

    num_workers = num_workers or min(16, len(jobs))
    num_workers = max(1, min(num_workers, 16, len(jobs)))
    print(f"\nProcessing {len(jobs)} files with {num_workers} workers...")
    all_chunks = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn")) as executor:
        for chunks in executor.map(process_episode_file, jobs):
            if chunks is not None:
                all_chunks.append(chunks)
    
    # Concatenate all chunks
    all_chunks = np.concatenate(all_chunks, axis=0)
    print(f"Collected {len(all_chunks)} action chunks")
    max_fast_chunks = 80_000
    if len(all_chunks) > max_fast_chunks:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(all_chunks), size=max_fast_chunks, replace=False)
        all_chunks = all_chunks[indices]
        print(f"Subsampled FAST tokenizer data to {len(all_chunks)} chunks")
    
    # Extract only encoded dimensions FIRST (before denorm/renorm)
    encoded_chunks = []
    for start, end in encoded_dim_ranges:
        encoded_chunks.append(all_chunks[:, :, start:end])
    encoded_chunks = np.concatenate(encoded_chunks, axis=-1)  # [N, H, D_encoded]
    print(f"Extracted {encoded_chunks.shape[-1]} encoded dimensions")
    
    # Apply normalization to encoded dimensions only
    # NOTE: For FAST, we ALWAYS use QUANTILE normalization (no per-timestamp)
    # This clips outliers and provides consistent [-1, 1] range for DCT compression
    # At this point, encoded_chunks contains RAW DELTA ACTIONS (not normalized yet)
    print(f"\nBefore normalization - overall stats:")
    print(f"  Min: {np.min(encoded_chunks):.4f}, Max: {np.max(encoded_chunks):.4f}")
    print(f"  Mean: {np.mean(encoded_chunks):.4f}, Std: {np.std(encoded_chunks):.4f}")
    
    norm_stats = data_config.norm_stats
    if norm_stats is not None:
        action_stats = norm_stats['actions']
        
        # Build encoded dimension indices
        encoded_dim_indices = []
        for start, end in encoded_dim_ranges:
            encoded_dim_indices.extend(range(start, end))
        encoded_dim_indices = np.array(encoded_dim_indices)
        
        # Use QUANTILE normalization: clip to [q01, q99] and map to [-1, 1]
        q01 = action_stats.q01[encoded_dim_indices]  # [D_encoded]
        q99 = action_stats.q99[encoded_dim_indices]  # [D_encoded]
        
        print(f"\nNormalization stats (q01, q99) for encoded dimensions:")
        for i, dim_idx in enumerate(encoded_dim_indices):
            print(f"  Orig dim {dim_idx}: q01={q01[i]:7.4f}, q99={q99[i]:7.4f}, range={q99[i]-q01[i]:7.4f}")
        
        # Clip to quantile range and normalize to [-1, 1]
        encoded_chunks = np.clip(encoded_chunks, q01, q99)
        encoded_chunks = 2.0 * (encoded_chunks - q01) / np.maximum(q99 - q01, 1e-6) - 1.0
        print(f"\nApplied quantile normalization [q01, q99] → [-1, 1]")
        
        print(f"\nAfter normalization - overall stats:")
        print(f"  Min: {np.min(encoded_chunks):.4f}, Max: {np.max(encoded_chunks):.4f}")
        print(f"  Mean: {np.mean(encoded_chunks):.4f}, Std: {np.std(encoded_chunks):.4f}")
        
        print(f"\nPer-dimension stats (after normalization):")
        for d in range(encoded_chunks.shape[-1]):
            dim_data = encoded_chunks[:, :, d]
            print(f"  Dim {d}: min={np.min(dim_data):7.4f}, max={np.max(dim_data):7.4f}, "
                  f"mean={np.mean(dim_data):7.4f}, std={np.std(dim_data):7.4f}")
    else:
        print("No normalization stats found, using raw delta actions")
    
    print(f"Encoded chunks shape: {encoded_chunks.shape}")
    
    # Train FAST tokenizer
    tokenizer = train_fast_tokenizer(
        encoded_chunks,
        vocab_size=vocab_size,
        scale=scale,
        num_epochs=num_epochs,
    )
    
    # Compute compression statistics
    compression_stats = compute_compression_stats(tokenizer, encoded_chunks)
    
    # Save tokenizer to assets directory
    output_dir = config.assets_dirs / data_config.asset_id / "fast_tokenizer"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata
    metadata = {
        'vocab_size': vocab_size,
        'scale': scale,
        'encoded_dims': encoded_dims,
        'encoded_dim_ranges': encoded_dim_ranges,
        'total_encoded_dims': total_encoded_dims,
        'action_horizon': action_horizon,
        'num_training_chunks': len(encoded_chunks),
        'compression_stats': compression_stats,
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Saved FAST tokenizer to {output_dir}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")


if __name__ == "__main__":
    tyro.cli(main)

