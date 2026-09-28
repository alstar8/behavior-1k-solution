#!/usr/bin/env bash
# 4-GPU CFGRL finetune of the 13-task PI_BEHAVIOR checkpoint (step 20000).
# Forward actions are optimality 2, true rewinds are optimality 1, and 10% drop the bit.
set -euo pipefail

ROOT="/workspace-SR008.nfs2/users/staroverov/B1K/behavior-1k-solution"
COMET_ENV="/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/openpi_comet/.venv"
OG_ROOT="/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/BEHAVIOR-1K"
CONFIG_NAME="pi_behavior_2026_ckpt3_13tasks_cfgrl_rewind"
EXP_NAME="${OPENPI_EXP_NAME:-pi_behavior_2026_ckpt3_cfgrl_rewind_$(date +%Y%m%d_%H%M%S)}"
CKPT_ROOT="/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior/b1k_solution"
LOG_DIR="${CKPT_ROOT}/logs"
TRAIN_WALL_CLOCK="${TRAIN_WALL_CLOCK:-48h}"
ASSETS_DIR="${ROOT}/outputs/assets/pi_behavior_2026_all100_demo0_6_comet0_4/behavior-1k/2026-challenge-demos"
mkdir -p "${LOG_DIR}" "${CKPT_ROOT}"

export PATH="${COMET_ENV}/bin:${PATH}"
export VIRTUAL_ENV="${COMET_ENV}"
export PYTHONPATH="${ROOT}/src:${ROOT}/openpi/src:${OG_ROOT}/OmniGibson:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${CKPT_ROOT}/wandb"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/workspace-SR008.nfs2/users/staroverov/.cache/openpi}"
export HF_HOME="${HF_HOME:-/workspace-SR008.nfs2/users/staroverov/.cache/huggingface}"
export OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior}"
export TOKENIZERS_PARALLELISM=false
export OPENPI_DATALOADER_PREFETCH_FACTOR="${OPENPI_DATALOADER_PREFETCH_FACTOR:-4}"
export OPENPI_BATCH_PREFETCH="${OPENPI_BATCH_PREFETCH:-4}"

cd "${ROOT}"
python -c "import b1k, openpi; print('b1k', b1k.__file__); print('openpi', openpi.__file__)"

if [ ! -f "${ASSETS_DIR}/norm_stats.json" ] || [ ! -f "${ASSETS_DIR}/fast_tokenizer/tokenizer.json" ]; then
  echo "Missing 100-task assets at ${ASSETS_DIR}" >&2
  exit 1
fi

echo "Starting 4-GPU CFGRL rewind finetune exp=${EXP_NAME} for up to ${TRAIN_WALL_CLOCK}"
echo "Init weights: 13-task ckpt step 20000"
exec timeout --signal=TERM --kill-after=15m "${TRAIN_WALL_CLOCK}" \
  python scripts/train.py "${CONFIG_NAME}" \
  --exp_name="${EXP_NAME}" \
  --overwrite \
  --fsdp_devices=4 \
  --batch_size=512 \
  --num_workers=64 \
  --num_train_steps=20000 \
  --save_interval=2000 \
  --keep_period=10000 \
  --log_interval=25
