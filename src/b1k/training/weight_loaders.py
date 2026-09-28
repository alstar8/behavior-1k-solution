"""Weight loaders for PI_BEHAVIOR model initialization from Pi05 checkpoints.

Reference: https://github.com/Physical-Intelligence
"""

import dataclasses
import logging
import re

import flax.traverse_util
import jax
import numpy as np
import orbax.checkpoint as ocp

import openpi.shared.array_typing as at
import openpi.shared.download as download

# Re-export base loaders from OpenPI
from openpi.training.weight_loaders import (
    WeightLoader,
    NoOpWeightLoader,
    CheckpointWeightLoader,
    _merge_params,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PiBehaviorWeightLoader(WeightLoader):
    """Loads checkpoints for PI_BEHAVIOR model.
    
    Automatically detects:
    - Pi05 checkpoint: Loads weights, preserves new PI_BEHAVIOR parameters
    - PI_BEHAVIOR checkpoint: Loads all weights directly
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # Load checkpoint. Saved shardings may come from a different device count,
        # so restore every array onto the devices that are available now.
        params_path = download.maybe_download(self.params_path)
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        with ocp.PyTreeCheckpointer() as ckptr:
            metadata = ckptr.metadata(params_path)
            restored = ckptr.restore(
                params_path,
                ocp.args.PyTreeRestore(
                    item=metadata,
                    restore_args=jax.tree.map(
                        lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=np.ndarray),
                        metadata,
                    ),
                ),
            )
        
        # Handle nested 'params' key (from some checkpoint formats)
        if isinstance(restored, dict) and "params" in restored:
            loaded_params = restored["params"]
        else:
            loaded_params = restored
        
        # Remove 'value' suffixes (from nnx.State format)
        flat_params = flax.traverse_util.flatten_dict(loaded_params)
        if all(kp[-1] == "value" for kp in flat_params if len(kp) > 0):
            flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
            loaded_params = flax.traverse_util.unflatten_dict(flat_params)
        
        # Detect checkpoint type
        has_task_embeddings = 'task_embeddings' in loaded_params
        
        if has_task_embeddings:
            # Loading PI_BEHAVIOR checkpoint - load ALL weights from checkpoint.
            # optimality_embed is new for CFGRL and stays at its zero init.
            logging.info("Loading PI_BEHAVIOR checkpoint (all weights will be loaded)")
            return _merge_params(loaded_params, params, missing_regex=".*optimality_embed.*")
        else:
            # Loading Pi05 checkpoint - preserve new PI_BEHAVIOR-specific parameters
            logging.info("Loading Pi05 checkpoint (new PI_BEHAVIOR parameters will use random init)")
            
            # These parameters are NEW in PI_BEHAVIOR (not in Pi05), so keep them from params (random init)
            missing_regex = (
                ".*task_embeddings.*|"
                ".*stage_pred_from_vlm.*|"
                ".*task_stage_embeddings.*|"
                ".*gate_sincos.*|"
                ".*gate_task_stage.*|"
                ".*gate_task.*|"
                ".*fusion_layer.*|"
                ".*stage_projection.*|"
                ".*task_subtask_fusion.*|"
                ".*fast_token_embedding.*|"
                ".*fast_token_proj.*|"
                ".*kv_transform.*"
            )
            return _merge_params(loaded_params, params, missing_regex=missing_regex)
