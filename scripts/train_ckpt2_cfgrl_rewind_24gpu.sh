#!/usr/bin/env bash
# Resume the checkpoint-2 CFGRL rewind run on 3 nodes x 8 GPUs.
#
# Usage:
#   bash scripts/train_ckpt2_cfgrl_rewind_24gpu.sh          # start all ranks
#   bash scripts/train_ckpt2_cfgrl_rewind_24gpu.sh rank     # one JAX process
set -euo pipefail

ROOT="/workspace-SR008.nfs2/users/staroverov/B1K/behavior-1k-solution"
B1K_ROOT="/workspace-SR008.nfs2/users/staroverov/B1K"
COMET_ENV="/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/openpi_comet/.venv"
OG_ROOT="/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/BEHAVIOR-1K"
CONFIG_NAME="pi_behavior_2026_ckpt2_16tasks_cfgrl_rewind"
EXP_NAME="${OPENPI_EXP_NAME:-pi_behavior_2026_ckpt2_cfgrl_rewind_20260929_031128}"
CKPT_ROOT="/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior/b1k_solution"
LOG_DIR="${CKPT_ROOT}/logs"
CKPT_DIR="${CKPT_ROOT}/${CONFIG_NAME}/${EXP_NAME}"
TRAIN_WALL_CLOCK="${TRAIN_WALL_CLOCK:-96h}"
ASSETS_DIR="${ROOT}/outputs/assets/pi_behavior_2026_all100_demo0_6_comet0_4/behavior-1k/2026-challenge-demos"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p 2222 -i /home/user/.ssh/id_rsa"

MASTER_HOST="$(tr -d '[:space:]' < /etc/volcano/mpimaster.host)"
WORKER_HOSTS=()
if [[ -f /etc/volcano/mpiworker.host ]]; then
  mapfile -t WORKER_HOSTS < /etc/volcano/mpiworker.host
fi
HOSTS=("${MASTER_HOST}" "${WORKER_HOSTS[@]}")
if [[ -n "${OPENPI_HOSTS:-}" ]]; then
  read -r -a HOSTS <<< "${OPENPI_HOSTS}"
fi

mkdir -p "${LOG_DIR}"

run_rank() {
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
  # Prefetch depth that sustained the batch-3072 run. In-order keeps the first batch deterministic.
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
  # Level 0 skips the multi-host GEMM search that deadlocked clique init.
  # Level 1 turns the search on without the correctness check.
  autotune_level="${XLA_GPU_AUTOTUNE_LEVEL:-0}"
  case " ${XLA_FLAGS:-} " in
    *"xla_gpu_autotune_level"*) ;;
    *) export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=${autotune_level}" ;;
  esac

  cd "${ROOT}"
  echo "rank=${WORLD_RANK}/${WORLD_SIZE} host=$(hostname) master=${MASTER_ADDR}:${MASTER_PORT}"
  echo "exp=${EXP_NAME} batch=${OPENPI_BATCH_SIZE:-3072} fsdp=${OPENPI_FSDP_DEVICES:-8} wall=${TRAIN_WALL_CLOCK}"
  echo "dataloader in_order=${OPENPI_DATALOADER_IN_ORDER} prefetch=${OPENPI_DATALOADER_PREFETCH_FACTOR} batch_prefetch=${OPENPI_BATCH_PREFETCH} xla_flags=${XLA_FLAGS:-<unset>}"
  echo "jax_cache=${JAX_COMPILATION_CACHE_DIR}"
  python -c "import b1k, openpi; print('b1k', b1k.__file__); print('openpi', openpi.__file__)"

  exec timeout --signal=TERM --kill-after=20m "${TRAIN_WALL_CLOCK}" \
    python scripts/train.py "${CONFIG_NAME}" \
    --exp_name="${EXP_NAME}" \
    --resume \
    --fsdp_devices="${OPENPI_FSDP_DEVICES:-8}" \
    --batch_size="${OPENPI_BATCH_SIZE:-3072}" \
    --num_workers="${OPENPI_NUM_WORKERS:-64}" \
    --num_train_steps="${OPENPI_NUM_TRAIN_STEPS:-66840}" \
    --save_interval=2000 \
    --keep_period=10000 \
    --log_interval=25
}

launch_all() {
  if [[ ! -f "${ASSETS_DIR}/norm_stats.json" ]]; then
    echo "Missing assets at ${ASSETS_DIR}" >&2
    exit 1
  fi
  if ! compgen -G "${CKPT_DIR}/*/train_state" > /dev/null; then
    echo "Missing resume checkpoint under ${CKPT_DIR}" >&2
    exit 1
  fi

  local master_addr master_port world_size rank host log pidfile
  master_addr="$(hostname -i | awk '{print $1}')"
  master_port="${MASTER_PORT:-12361}"
  world_size="${#HOSTS[@]}"
  if [[ "${world_size}" -ne 3 ]]; then
    echo "Expected 3 hosts, found ${world_size}: ${HOSTS[*]}" >&2
    exit 1
  fi

  echo "Launching ${world_size} ranks for ${EXP_NAME}"
  echo "MASTER_ADDR=${master_addr} MASTER_PORT=${master_port}"
  echo "hosts: ${HOSTS[*]}"
    echo "resume checkpoint dir: ${CKPT_DIR}"

  for rank in "${!HOSTS[@]}"; do
    host="${HOSTS[$rank]}"
    log="${LOG_DIR}/train_${EXP_NAME}_24gpu_${OPENPI_RUN_TAG:-pf6}_rank${rank}.log"
    pidfile="${LOG_DIR}/train_${EXP_NAME}_24gpu_${OPENPI_RUN_TAG:-pf6}_rank${rank}.pid"
    echo "starting rank ${rank} on ${host} -> ${log}"
    ssh ${SSH_OPTS} -f "${host}" \
      "cd '${B1K_ROOT}' && setsid env WORLD_RANK=${rank} WORLD_SIZE=${world_size} MASTER_ADDR=${master_addr} MASTER_PORT=${master_port} EXP_NAME='${EXP_NAME}' OPENPI_EXP_NAME='${EXP_NAME}' CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OPENPI_BATCH_SIZE='${OPENPI_BATCH_SIZE:-3072}' OPENPI_FSDP_DEVICES='${OPENPI_FSDP_DEVICES:-8}' OPENPI_NUM_WORKERS='${OPENPI_NUM_WORKERS:-64}' OPENPI_DATALOADER_IN_ORDER='${OPENPI_DATALOADER_IN_ORDER:-1}' OPENPI_DATALOADER_PREFETCH_FACTOR='${OPENPI_DATALOADER_PREFETCH_FACTOR:-4}' OPENPI_BATCH_PREFETCH='${OPENPI_BATCH_PREFETCH:-4}' XLA_PYTHON_CLIENT_MEM_FRACTION='${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}' NCCL_NVLS_ENABLE='${NCCL_NVLS_ENABLE:-0}' XLA_GPU_AUTOTUNE_LEVEL='${XLA_GPU_AUTOTUNE_LEVEL:-0}' JAX_COMPILATION_CACHE_DIR='${JAX_COMPILATION_CACHE_DIR:-/workspace-SR008.nfs2/users/staroverov/.cache/jax}' TRAIN_WALL_CLOCK='${TRAIN_WALL_CLOCK}' bash '${ROOT}/scripts/train_ckpt2_cfgrl_rewind_24gpu.sh' rank > '${log}' 2>&1 < /dev/null & echo \$! > '${pidfile}'; exit 0" \
      </dev/null
  done
  sleep 2
  echo "Detached. Rank logs:"
  for rank in "${!HOSTS[@]}"; do
    echo "  ${LOG_DIR}/train_${EXP_NAME}_24gpu_${OPENPI_RUN_TAG:-pf6}_rank${rank}.log"
  done
}

mode="${1:-launch}"
case "${mode}" in
  rank) run_rank ;;
  launch) launch_all ;;
  *)
    echo "Unknown mode: ${mode} (use launch|rank)" >&2
    exit 1
    ;;
esac
