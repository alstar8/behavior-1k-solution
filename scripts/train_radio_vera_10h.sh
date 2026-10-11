#!/usr/bin/env bash
# 10h forward-only radio finetune.
# 60% 2026-challenge-demos turning_on_radio, 40% VERA-VLA/b1k_turn_on_radio.
# Loads radio CFGRL step 24000 params into a new directory. Does not write 24000 or 66000.
#
# Usage:
#   bash scripts/train_radio_vera_10h.sh
#   bash scripts/train_radio_vera_10h.sh rank
set -euo pipefail

ROOT="/workspace-SR008.nfs2/users/staroverov/B1K/behavior-1k-solution"
B1K_ROOT="/workspace-SR008.nfs2/users/staroverov/B1K"
COMET_ENV="/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/openpi_comet/.venv"
OG_ROOT="/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/BEHAVIOR-1K"
CONFIG_NAME="pi_behavior_2026_radio_vera40_nocfgrl"
EXP_NAME="${OPENPI_EXP_NAME:-pi_behavior_2026_radio_vera40_$(date +%Y%m%d_%H%M%S)}"
CKPT_ROOT="/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior/b1k_solution"
LOG_DIR="${CKPT_ROOT}/logs"
CKPT_DIR="${CKPT_ROOT}/${CONFIG_NAME}/${EXP_NAME}"
SRC_CKPT="${CKPT_ROOT}/pi_behavior_2026_ckpt2_radio_cfgrl_rewind/pi_behavior_2026_ckpt2_radio_cfgrl_20261006_172030/24000"
PROTECTED_66000="${CKPT_ROOT}/pi_behavior_2026_ckpt2_16tasks_cfgrl_rewind/pi_behavior_2026_ckpt2_cfgrl_rewind_20260929_031128"
TRAIN_WALL_CLOCK="${TRAIN_WALL_CLOCK:-10h}"
ASSETS_DIR="${ROOT}/outputs/assets/pi_behavior_2026_all100_demo0_6_comet0_4/behavior-1k/2026-challenge-demos"
VERA_ROOT="/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior/b1k_turn_on_radio"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p 2222 -i /home/user/.ssh/id_rsa"

MASTER_HOST="$(tr -d '[:space:]' < /etc/volcano/mpimaster.host)"
WORKER_HOSTS=()
if [[ -f /etc/volcano/mpiworker.host ]]; then
  mapfile -t WORKER_HOSTS < /etc/volcano/mpiworker.host
fi
HOSTS=("${MASTER_HOST}" "${WORKER_HOSTS[@]}")

mkdir -p "${LOG_DIR}"

guard_paths() {
  if [[ ! -f "${SRC_CKPT}/_CHECKPOINT_METADATA" || ! -d "${SRC_CKPT}/params" || ! -d "${SRC_CKPT}/train_state" ]]; then
    echo "Missing source checkpoint ${SRC_CKPT}" >&2
    exit 1
  fi
  if [[ ! -f "${VERA_ROOT}/meta/info.json" || ! -f "${VERA_ROOT}/meta/tasks.jsonl" ]]; then
    echo "Missing VERA dataset at ${VERA_ROOT}" >&2
    exit 1
  fi
  case "${CKPT_DIR}" in
    "${SRC_CKPT}"|"${SRC_CKPT}"/*|*20261006_172030*|*20260929_031128*|*"${PROTECTED_66000}"*)
      echo "Refusing to write into an existing checkpoint run: ${CKPT_DIR}" >&2
      exit 1
      ;;
  esac
}

run_rank() {
  guard_paths
  export PATH="${COMET_ENV}/bin:${PATH}"
  export VIRTUAL_ENV="${COMET_ENV}"
  export PYTHONPATH="${ROOT}/src:${ROOT}/openpi/src:${OG_ROOT}/OmniGibson:${PYTHONPATH:-}"
  export PYTHONUNBUFFERED=1
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export WANDB_DIR="${CKPT_ROOT}/wandb"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
  export NCCL_IB_HCA="${NCCL_IB_HCA:-^mlx5_bond}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
  export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/workspace-SR008.nfs2/users/staroverov/.cache/jax}"
  mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
  export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/workspace-SR008.nfs2/users/staroverov/.cache/openpi}"
  export HF_HOME="${HF_HOME:-/workspace-SR008.nfs2/users/staroverov/.cache/huggingface}"
  export OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior}"
  export TOKENIZERS_PARALLELISM=false
  export OPENPI_DATALOADER_IN_ORDER="${OPENPI_DATALOADER_IN_ORDER:-1}"
  export OPENPI_DATALOADER_PREFETCH_FACTOR="${OPENPI_DATALOADER_PREFETCH_FACTOR:-4}"
  export OPENPI_BATCH_PREFETCH="${OPENPI_BATCH_PREFETCH:-4}"
  export JAX_COORDINATION_TIMEOUT="${JAX_COORDINATION_TIMEOUT:-600}"

  if [[ -n "${XLA_FLAGS:-}" ]]; then
    filtered_xla_flags=()
    for flag in ${XLA_FLAGS}; do
      case "${flag}" in
        --xla_gpu_enable_async_all_gather=true|--xla_gpu_enable_async_reduce_scatter=true) ;;
        *) filtered_xla_flags+=("${flag}") ;;
      esac
    done
    if ((${#filtered_xla_flags[@]})); then
      export XLA_FLAGS="${filtered_xla_flags[*]}"
    else
      unset XLA_FLAGS
    fi
  fi
  autotune_level="${XLA_GPU_AUTOTUNE_LEVEL:-1}"
  case " ${XLA_FLAGS:-} " in
    *"xla_gpu_autotune_level"*) ;;
    *) export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=${autotune_level}" ;;
  esac

  cd "${ROOT}"
  echo "rank=${WORLD_RANK}/${WORLD_SIZE} host=$(hostname) master=${MASTER_ADDR}:${MASTER_PORT}"
  echo "exp=${EXP_NAME} batch=${OPENPI_BATCH_SIZE:-1536} fsdp=${OPENPI_FSDP_DEVICES:-8} wall=${TRAIN_WALL_CLOCK}"
  echo "source=${SRC_CKPT}"
  echo "output=${CKPT_DIR}"
  python -c "import b1k, openpi; print('b1k', b1k.__file__); print('openpi', openpi.__file__)"

  # Empty directory plus --resume avoids the two-host FileExistsError race.
  # No committed step means resume is aborted and 24000/params is loaded.
  exec timeout --signal=TERM --kill-after=20m "${TRAIN_WALL_CLOCK}" \
    python scripts/train.py "${CONFIG_NAME}" \
    --exp_name="${EXP_NAME}" \
    --fsdp_devices="${OPENPI_FSDP_DEVICES:-8}" \
    --batch_size="${OPENPI_BATCH_SIZE:-1536}" \
    --num_workers="${OPENPI_NUM_WORKERS:-64}" \
    --num_train_steps="${OPENPI_NUM_TRAIN_STEPS:-20000}" \
    --save_interval=2000 \
    --keep_period=2000 \
    --log_interval=25 \
    --resume
}

launch_all() {
  guard_paths
  if [[ ! -f "${ASSETS_DIR}/norm_stats.json" ]]; then
    echo "Missing assets at ${ASSETS_DIR}" >&2
    exit 1
  fi
  if [[ "${#HOSTS[@]}" -ne 2 ]]; then
    echo "Expected 2 hosts, found ${#HOSTS[@]}: ${HOSTS[*]}" >&2
    exit 1
  fi
  if [[ -e "${CKPT_DIR}" ]]; then
    echo "Output directory already exists: ${CKPT_DIR}" >&2
    exit 1
  fi
  mkdir -p "${CKPT_DIR}"
  if [[ -n "$(ls -A "${CKPT_DIR}")" ]]; then
    echo "Output directory is not empty: ${CKPT_DIR}" >&2
    exit 1
  fi

  local master_addr master_port rank host log pidfile
  master_addr="$(hostname -i | awk '{print $1}')"
  master_port="${MASTER_PORT:-12371}"
  echo "Launching ${EXP_NAME}"
  echo "MASTER_ADDR=${master_addr} MASTER_PORT=${master_port}"
  echo "hosts: ${HOSTS[*]}"
  echo "read-only source: ${SRC_CKPT}"
  echo "new checkpoints: ${CKPT_DIR}"
  echo "wall=${TRAIN_WALL_CLOCK} batch=1536 weights=0.6/0.4"

  for rank in "${!HOSTS[@]}"; do
    host="${HOSTS[$rank]}"
    log="${LOG_DIR}/train_${EXP_NAME}_16gpu_rank${rank}.log"
    pidfile="${LOG_DIR}/train_${EXP_NAME}_16gpu_rank${rank}.pid"
    echo "starting rank ${rank} on ${host} -> ${log}"
    ssh ${SSH_OPTS} -f "${host}" \
      "cd '${B1K_ROOT}' && setsid env WORLD_RANK=${rank} WORLD_SIZE=2 MASTER_ADDR=${master_addr} MASTER_PORT=${master_port} EXP_NAME='${EXP_NAME}' OPENPI_EXP_NAME='${EXP_NAME}' CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OPENPI_BATCH_SIZE=1536 OPENPI_FSDP_DEVICES=8 OPENPI_NUM_WORKERS=64 OPENPI_DATALOADER_IN_ORDER=1 OPENPI_DATALOADER_PREFETCH_FACTOR=4 OPENPI_BATCH_PREFETCH=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 NCCL_NVLS_ENABLE=0 XLA_GPU_AUTOTUNE_LEVEL=1 JAX_COMPILATION_CACHE_DIR=/workspace-SR008.nfs2/users/staroverov/.cache/jax TRAIN_WALL_CLOCK='${TRAIN_WALL_CLOCK}' bash '${ROOT}/scripts/train_radio_vera_10h.sh' rank > '${log}' 2>&1 < /dev/null & echo \$! > '${pidfile}'; exit 0" \
      </dev/null
  done
  sleep 2
  echo "Detached. Rank logs:"
  echo "  ${LOG_DIR}/train_${EXP_NAME}_16gpu_rank0.log"
  echo "  ${LOG_DIR}/train_${EXP_NAME}_16gpu_rank1.log"
  echo "EXP_NAME=${EXP_NAME}"
}

mode="${1:-launch}"
case "${mode}" in
  rank) run_rank ;;
  launch) launch_all ;;
  *)
    echo "Unknown mode: ${mode}" >&2
    exit 1
    ;;
esac
