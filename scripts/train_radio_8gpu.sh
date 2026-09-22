#!/usr/bin/env bash
# 8-GPU PI_BEHAVIOR specialist on turning_on_radio:
# 60% 2026-challenge-demos + 40% Comet RFT, 72h wall clock.
set -euo pipefail

ROOT="/workspace-SR008.nfs2/users/staroverov/B1K/behavior-1k-solution"
COMET_ENV="/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/openpi_comet/.venv"
OG_ROOT="/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/BEHAVIOR-1K"
CONFIG_NAME="pi_behavior_2026_radio_demo0_6_comet0_4"
EXP_NAME="${OPENPI_EXP_NAME:-pi_behavior_2026_radio_demo0_6_comet0_4_$(date +%Y%m%d_%H%M%S)}"
CKPT_ROOT="/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior/b1k_solution"
LOG_DIR="${CKPT_ROOT}/logs"
TRAIN_WALL_CLOCK="${TRAIN_WALL_CLOCK:-72h}"
mkdir -p "${LOG_DIR}" "${CKPT_ROOT}" "${ROOT}/outputs/assets"

export PATH="${COMET_ENV}/bin:${PATH}"
export VIRTUAL_ENV="${COMET_ENV}"
export PYTHONPATH="${ROOT}/src:${ROOT}/openpi/src:${OG_ROOT}/OmniGibson:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${CKPT_ROOT}/wandb"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.82}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/workspace-SR008.nfs2/users/staroverov/.cache/openpi}"
export HF_HOME="${HF_HOME:-/workspace-SR008.nfs2/users/staroverov/.cache/huggingface}"
export OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior}"
export TOKENIZERS_PARALLELISM=false

cd "${ROOT}"
python -c "import b1k, openpi; print('b1k', b1k.__file__); print('openpi', openpi.__file__)"

ASSETS_DIR="${ROOT}/outputs/assets/${CONFIG_NAME}/behavior-1k/2026-challenge-demos"
if [ ! -f "${ASSETS_DIR}/norm_stats.json" ]; then
  echo "Computing norm stats + correlation matrix..."
  JAX_PLATFORMS=cpu python scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}" --correlation --per-timestamp
fi
if [ ! -f "${ASSETS_DIR}/fast_tokenizer/processor_config.json" ] && [ ! -f "${ASSETS_DIR}/fast_tokenizer/tokenizer.json" ]; then
  echo "Training FAST tokenizer..."
  JAX_PLATFORMS=cpu python scripts/train_fast_tokenizer.py \
    --config-name "${CONFIG_NAME}" \
    --encoded-dims="0:6,7:23" \
    --vocab-size=1024
fi

echo "Starting 8-GPU radio finetune exp=${EXP_NAME} for up to ${TRAIN_WALL_CLOCK}"
echo "Init weights: 100-task ckpt step 100000; task embeddings + System 2 frozen"
exec timeout --signal=TERM --kill-after=15m "${TRAIN_WALL_CLOCK}" \
  python scripts/train.py "${CONFIG_NAME}" \
  --exp_name="${EXP_NAME}" \
  --overwrite \
  --fsdp_devices=8 \
  --batch_size=256 \
  --num_workers=80 \
  --num_train_steps=2000000 \
  --save_interval=5000 \
  --keep_period=50000 \
  --log_interval=25
