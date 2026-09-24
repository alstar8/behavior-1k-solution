# Evaluating `pi_behavior_2026_radio_from100k` step 95000

Specialist checkpoint for `turning_on_radio` (task id 0). It was finetuned from the 100-task generalist at step 100000. Task embeddings and System 2 stay at those generalist weights. The policy head was finetuned on all 200 2026 challenge demos and all 120 Comet RFT radio trajectories.

## Checkpoints

Best saved radio weights (step 95000). Use this directory, not the `params` folder alone. The loader reads `params/` and `assets/behavior-1k/2026-challenge-demos/norm_stats.json` next to it.

```text
/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior/b1k_solution/pi_behavior_2026_radio_demo0_6_comet0_4/pi_behavior_2026_radio_from100k_20260921_213403_step95000
```

The training run also writes a rolling `95000` directory under the experiment folder. That copy is deleted when step 100000 is saved. The path above is a hardlinked snapshot of the same files.

Task embeddings and System 2 (frozen, trained on all 100 tasks) live in the generalist checkpoint pytree: `task_embeddings`, `task_stage_embeddings`, `stage_pred_from_vlm`, `gate_sincos`, `gate_task`, `gate_task_stage`, `fusion_layer1`, `fusion_layer2`, `stage_projection`.

```text
/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior/b1k_solution/pi_behavior_2026_all100_demo0_6_comet0_4/pi_behavior_2026_all100_bs256_w80_20260919_004055/100000/params
```

Those tensors are already inside the radio checkpoint. Do not load the generalist `params` directory as the policy.

## Online evaluation

Serve the policy on a GPU that is not already taken by training. The server speaks the 2026 BEHAVIOR websocket protocol (`omnigibson.eval.utils.network_utils`).

```bash
ROOT=/workspace-SR008.nfs2/users/staroverov/B1K/behavior-1k-solution
OG=/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/BEHAVIOR-1K
source /workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/openpi_comet/.venv/bin/activate
export PYTHONPATH="$ROOT/src:$ROOT/openpi/src:$OG/OmniGibson:${PYTHONPATH:-}"
export OMNIGIBSON_DATA_PATH=/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$ROOT"
python scripts/serve_b1k.py \
  --task-id 0 \
  --actions-to-execute 26 \
  --actions-to-keep 4 \
  --execute-in-n-steps 20 \
  --num-steps 20 \
  --apply-eval-tricks \
  policy:checkpoint \
  --policy.config pi_behavior_2026_radio_demo0_6_comet0_4 \
  --policy.dir /workspace-SR008.nfs2/datasets/staroverov_b1k/behavior/b1k_solution/pi_behavior_2026_radio_demo0_6_comet0_4/pi_behavior_2026_radio_from100k_20260921_213403_step95000
```

`--task-id 0` selects the radio embedding. The wrapper keeps the last 4 of every 30 predicted actions for correlation-aware inpainting, executes 26 predictions over 20 control steps (1.3x), and holds the stage until two of the last three predictions agree. `--apply-eval-tricks` is the competition correction rule (open the gripper after a failed grasp, plus the legacy radio rule). Turn it off only for an ablation.

In a second process, with the OmniGibson environment that matches `B1K_AIRI/BEHAVIOR-1K`, run the public test split. Instance index 0 is the first public-test instance (simulator id 301), not demo episode 0.

```bash
cd /workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/BEHAVIOR-1K/OmniGibson
export OMNIGIBSON_DATA_PATH=/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior
python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --robot-config omnigibson/eval/r1pro.yaml \
  --mode public_test \
  --host 127.0.0.1 --port 8000 \
  --instance-indices 0 \
  --write-video \
  --output-dir /workspace-SR008.nfs2/datasets/staroverov_b1k/behavior/b1k_solution/eval_radio_95k
```

Each rollout writes `json/turning_on_radio_<instance>_<rollout>.json` with `success` and `q_score`. A full public number is the mean over instances 0–19. Hidden-test instances are `--mode hidden_test`.

## Same prediction on another server

This check does not start the simulator. It runs the policy once on frame 0 of 2026 challenge demo episode 0 (`turning_on_radio`, `timestamp=0`, `frame_index=0`) and compares the action chunk.

Fixed inputs, so the result does not depend on the JAX RNG:

- task id 0, stage 0 (`tokenized_prompt = [0, 0]`)
- 20 flow steps
- no inpainting and no eval tricks
- noise `numpy.random.RandomState(0).randn(30, 32).astype(np.float32)`

Images are the first frame of the three episode-0 RGB videos, resized to 224 with `resize_with_pad`. Proprioception is the 61-d `observation.state` at row 0 of `data/chunk-000/file-000.parquet`.

Reproduce with:

```bash
JAX_PLATFORMS=cpu python scripts/fingerprint_radio_step0.py
```

The full 30×23 chunk is in `docs/radio_95k_step0_fingerprint.json`. The first action (the vector at chunk index 0) is below. Layout is base (3), trunk (4), left arm (7), left gripper (1), right arm (7), right gripper (1). Predicted stage was 0.

```text
-0.00017936 -0.00027047  0.00079258
 1.02531469 -1.44968617 -0.47021994 -0.00000000
-0.00025504  0.00035996  0.00142660 -0.01004299  0.00880431 -0.00162328 -0.00116465
 1.00568199
 0.00385501  0.00098415  0.00868926 -0.00554582 -0.00095821 -0.01325336  0.00053768
 1.00707126
```

Compare with `np.allclose(..., atol=1e-5)`. bf16 matmul can differ slightly across GPU types; a mismatch larger than that means the checkpoint, norm stats, noise, or frame is not the same.
