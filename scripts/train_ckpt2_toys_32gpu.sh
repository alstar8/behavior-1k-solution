#!/usr/bin/env bash
# Checkpoint 2, picking_up_toys, no CFGRL, 4 nodes x 8 GPUs.
# Loads behavior_submission/checkpoint_2 and writes a new directory.
#
# Usage:
#   bash scripts/train_ckpt2_toys_32gpu.sh
#   bash scripts/train_ckpt2_toys_32gpu.sh rank
set -euo pipefail

ROOT="/workspace-SR008.nfs2/users/staroverov/B1K/behavior-1k-solution"
B1K_ROOT="/workspace-SR008.nfs2/users/staroverov/B1K"
COMET_ENV="/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/openpi_comet/.venv"
OG_ROOT="/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/BEHAVIOR-1K"
CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi_behavior_2026_ckpt2_toys_nocfgrl}"
EXP_NAME="${OPENPI_EXP_NAME:-pi_behavior_2026_ckpt2_toys_nocfgrl_$(date +%Y%m%d_%H%M%S)}"
CKPT_ROOT="/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior/b1k_solution"
LOG_DIR="${CKPT_ROOT}/logs"
CKPT_DIR="${CKPT_ROOT}/${CONFIG_NAME}/${EXP_NAME}"
INIT_PARAMS="/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior/pretrained/behavior_submission/checkpoint_2/params"
TRAIN_WALL_CLOCK="${TRAIN_WALL_CLOCK:-48h}"
ASSETS_DIR="${ROOT}/outputs/assets/pi_behavior_2026_all100_demo0_6_comet0_4/behavior-1k/2026-challenge-demos"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p 2222 -i /home/user/.ssh/id_rsa"

MASTER_HOST="$(tr -d '[:space:]' < /etc/volcano/mpimaster.host)"
WORKER_HOSTS=()
if [[ -f /etc/volcano/mpiworker.host ]]; then
  mapfile -t WORKER_HOSTS < /etc/volcano/mpiworker.host
fi
HOSTS=("${MASTER_HOST}" "${WORKER_HOSTS[@]}")

mkdir -p "${LOG_DIR}"

guard_paths() {
  if [[ ! -d "${INIT_PARAMS}" ]]; then
    echo "Missing checkpoint 2 params at ${INIT_PARAMS}" >&2
    exit 1
  fi
  case "${CKPT_DIR}" in
    *behavior_submission*|*20260929_031128*|*20261006_172030*|*vera40*|*cfgrl_rewind*)
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
  # train.py defaults to the platform allocator, which frees ~40 GB of step buffers back
  # to the driver and mallocs them again every step. BFC keeps them in one pool.
  export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-bfc}"
  # Active InfiniBand ports only; the 100G Ethernet ports are left out. Names differ
  # between nodes, so the list is built per host. "=" makes the match exact, so
  # mlx5_1 is not taken as a prefix of mlx5_10 / mlx5_11.
  if [[ -z "${NCCL_IB_HCA:-}" ]]; then
    local ib_ports=()
    for dev in /sys/class/infiniband/*; do
      if [[ "$(cat "${dev}/ports/1/link_layer" 2>/dev/null)" == "InfiniBand" && "$(cat "${dev}/ports/1/state" 2>/dev/null)" == *ACTIVE* ]]; then
        ib_ports+=("$(basename "${dev}")")
      fi
    done
    if ((${#ib_ports[@]})); then
      NCCL_IB_HCA="=$(IFS=,; echo "${ib_ports[*]}")"
      export NCCL_IB_HCA
    fi
  fi
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
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
  # Level 0 left GEMMs on untuned default kernels and ran at about half the per-GPU
  # speed of the 2-node runs. Shard autotuning off makes every host tune on its own,
  # without the cross-host exchange that deadlocked clique init.
  autotune_level="${XLA_GPU_AUTOTUNE_LEVEL:-1}"
  case " ${XLA_FLAGS:-} " in
    *"xla_gpu_autotune_level"*) ;;
    *) export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=${autotune_level}" ;;
  esac
  # Level 0 writes untuned default GEMM configs into the shared per-fusion autotune
  # cache, and later level-1 compiles reuse them instead of tuning.
  if [[ "${autotune_level}" == "0" ]]; then
    export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="${JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES:-none}"
  fi
  case " ${XLA_FLAGS:-} " in
    *"xla_gpu_shard_autotuning"*) ;;
    *) export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_shard_autotuning=false" ;;
  esac

  cd "${ROOT}"
  echo "rank=${WORLD_RANK}/${WORLD_SIZE} host=$(hostname) master=${MASTER_ADDR}:${MASTER_PORT}"
  echo "exp=${EXP_NAME} batch=${OPENPI_BATCH_SIZE:-3072} fsdp=${OPENPI_FSDP_DEVICES:-8} wall=${TRAIN_WALL_CLOCK}"
  echo "init=${INIT_PARAMS}"
  echo "output=${CKPT_DIR}"
  echo "NCCL_IB_HCA=${NCCL_IB_HCA:-} XLA_FLAGS=${XLA_FLAGS} allocator=${XLA_PYTHON_CLIENT_ALLOCATOR}"
  python -c "import b1k, openpi; print('b1k', b1k.__file__); print('openpi', openpi.__file__)"

  # Empty directory plus --resume avoids the two-host FileExistsError race.
  # No committed step means resume is aborted and checkpoint 2 params are loaded.
  exec timeout --signal=TERM --kill-after=20m "${TRAIN_WALL_CLOCK}" \
    python scripts/train.py "${CONFIG_NAME}" \
    --exp_name="${EXP_NAME}" \
    --fsdp_devices="${OPENPI_FSDP_DEVICES:-8}" \
    --batch_size="${OPENPI_BATCH_SIZE:-3072}" \
    --num_workers="${OPENPI_NUM_WORKERS:-64}" \
    --num_train_steps="${OPENPI_NUM_TRAIN_STEPS:-100000}" \
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
  if [[ "${#HOSTS[@]}" -ne 4 ]]; then
    echo "Expected 4 hosts, found ${#HOSTS[@]}: ${HOSTS[*]}" >&2
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
  master_port="${MASTER_PORT:-12381}"
  echo "Launching ${EXP_NAME}"
  echo "MASTER_ADDR=${master_addr} MASTER_PORT=${master_port}"
  echo "hosts: ${HOSTS[*]}"
  echo "init: ${INIT_PARAMS}"
  echo "new checkpoints: ${CKPT_DIR}"
  echo "wall=${TRAIN_WALL_CLOCK} batch=3072 fsdp=8"

  for rank in "${!HOSTS[@]}"; do
    host="${HOSTS[$rank]}"
    log="${LOG_DIR}/train_${EXP_NAME}_32gpu_rank${rank}.log"
    pidfile="${LOG_DIR}/train_${EXP_NAME}_32gpu_rank${rank}.pid"
    echo "starting rank ${rank} on ${host} -> ${log}"
    ssh ${SSH_OPTS} -f "${host}" \
      "cd '${B1K_ROOT}' && setsid env WORLD_RANK=${rank} WORLD_SIZE=4 MASTER_ADDR=${master_addr} MASTER_PORT=${master_port} EXP_NAME='${EXP_NAME}' OPENPI_EXP_NAME='${EXP_NAME}' OPENPI_CONFIG_NAME='${CONFIG_NAME}' CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OPENPI_BATCH_SIZE=3072 OPENPI_FSDP_DEVICES=8 OPENPI_NUM_WORKERS=64 OPENPI_DATALOADER_IN_ORDER=1 OPENPI_DATALOADER_PREFETCH_FACTOR=4 OPENPI_BATCH_PREFETCH=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 NCCL_NVLS_ENABLE=0 XLA_GPU_AUTOTUNE_LEVEL='${XLA_GPU_AUTOTUNE_LEVEL:-1}' JAX_COMPILATION_CACHE_DIR=/workspace-SR008.nfs2/users/staroverov/.cache/jax TRAIN_WALL_CLOCK='${TRAIN_WALL_CLOCK}' bash '${ROOT}/scripts/train_ckpt2_toys_32gpu.sh' rank > '${log}' 2>&1 < /dev/null & echo \$! > '${pidfile}'; exit 0" \
      </dev/null
  done
  sleep 2
  echo "Detached. Rank logs:"
  for rank in "${!HOSTS[@]}"; do
    echo "  ${LOG_DIR}/train_${EXP_NAME}_32gpu_rank${rank}.log"
  done
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
