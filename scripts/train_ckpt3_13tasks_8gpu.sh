#!/usr/bin/env bash
# 8-GPU PI_BEHAVIOR finetune on the 2025 checkpoint-3 task subset (13 tasks):
# 60% 2026-challenge-demos + 40% Comet RFT, 72h wall clock.
# Reuses the 100-task norm stats and FAST tokenizer. Task embeddings and
# System 2 stay frozen at the 100-task generalist (step 100000).
set -euo pipefail

ROOT="/workspace-SR008.nfs2/users/staroverov/B1K/behavior-1k-solution"
COMET_ENV="/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/openpi_comet/.venv"
OG_ROOT="/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/BEHAVIOR-1K"
CONFIG_NAME="pi_behavior_2026_ckpt3_13tasks_demo0_6_comet0_4"
EXP_NAME="${OPENPI_EXP_NAME:-pi_behavior_2026_ckpt3_13tasks_$(date +%Y%m%d_%H%M%S)}"
CKPT_ROOT="/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior/b1k_solution"
LOG_DIR="${CKPT_ROOT}/logs"
TRAIN_WALL_CLOCK="${TRAIN_WALL_CLOCK:-72h}"
ASSETS_DIR="${ROOT}/outputs/assets/pi_behavior_2026_all100_demo0_6_comet0_4/behavior-1k/2026-challenge-demos"
mkdir -p "${LOG_DIR}" "${CKPT_ROOT}"

export PATH="${COMET_ENV}/bin:${PATH}"
export VIRTUAL_ENV="${COMET_ENV}"
export PYTHONPATH="${ROOT}/src:${ROOT}/openpi/src:${OG_ROOT}/OmniGibson:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${CKPT_ROOT}/wandb"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
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

if [ ! -f "${ASSETS_DIR}/norm_stats.json" ] || [ ! -f "${ASSETS_DIR}/fast_tokenizer/tokenizer.json" ]; then
  echo "Missing 100-task assets at ${ASSETS_DIR}" >&2
  exit 1
fi

echo "Starting 8-GPU 13-task finetune exp=${EXP_NAME} for up to ${TRAIN_WALL_CLOCK}"
echo "Init weights: 100-task ckpt step 100000; task embeddings + System 2 frozen"
exec timeout --signal=TERM --kill-after=15m "${TRAIN_WALL_CLOCK}" \
  python scripts/train.py "${CONFIG_NAME}" \
  --exp_name="${EXP_NAME}" \
  --overwrite \
  --fsdp_devices=8 \
  --batch_size=1024 \
  --num_workers=80 \
  --num_train_steps=2000000 \
  --save_interval=5000 \
  --keep_period=50000 \
  --log_interval=25
