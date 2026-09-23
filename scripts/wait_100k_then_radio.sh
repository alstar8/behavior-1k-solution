#!/usr/bin/env bash
# After the 100-task generalist finalizes step 100000, stop it and start
# the turning_on_radio finetune on the freed 8 GPUs.
set -euo pipefail

ROOT="/workspace-SR008.nfs2/users/staroverov/B1K/behavior-1k-solution"
CKPT_ROOT="/workspace-SR008.nfs2/datasets/staroverov_b1k/behavior/b1k_solution"
GEN_DIR="${CKPT_ROOT}/pi_behavior_2026_all100_demo0_6_comet0_4/pi_behavior_2026_all100_bs256_w80_20260919_004055"
GEN_LOG="${CKPT_ROOT}/logs/train_all100_20260919_004055.log"
STEP_DIR="${GEN_DIR}/100000"
MARKER="Finished saving checkpoint (finalized tmp dir) to \`${STEP_DIR}\`"
STAMP="$(date +%Y%m%d_%H%M%S)"
WAIT_LOG="${CKPT_ROOT}/logs/wait_100k_then_radio_${STAMP}.log"

exec > >(tee -a "${WAIT_LOG}") 2>&1
echo "Waiting for finalized checkpoint ${STEP_DIR}"

while true; do
  if grep -F -q "${MARKER}" "${GEN_LOG}" 2>/dev/null && [[ -d "${STEP_DIR}/params" ]]; then
    echo "Checkpoint 100000 finalized"
    break
  fi
  if ! pgrep -f "pi_behavior_2026_all100_bs256_w80_20260919_004055" >/dev/null; then
    if [[ -d "${STEP_DIR}/params" ]]; then
      echo "Generalist already exited and 100000/params exists"
      break
    fi
    echo "Generalist exited before checkpoint 100000 was written" >&2
    exit 1
  fi
  sleep 30
done

echo "Stopping 100-task generalist"
pkill -TERM -f "python scripts/train.py pi_behavior_2026_all100_demo0_6_comet0_4 --exp_name=pi_behavior_2026_all100_bs256_w80_20260919_004055" || true
for _ in $(seq 1 90); do
  if ! pgrep -f "pi_behavior_2026_all100_bs256_w80_20260919_004055" >/dev/null; then
    break
  fi
  sleep 10
done
if pgrep -f "pi_behavior_2026_all100_bs256_w80_20260919_004055" >/dev/null; then
  echo "Generalist still alive; sending KILL"
  pkill -KILL -f "python scripts/train.py pi_behavior_2026_all100_demo0_6_comet0_4 --exp_name=pi_behavior_2026_all100_bs256_w80_20260919_004055" || true
  sleep 5
fi

# Let GPU memory drain.
sleep 20
if [[ ! -d "${STEP_DIR}/params" ]]; then
  echo "Missing ${STEP_DIR}/params after shutdown" >&2
  exit 1
fi

export OPENPI_EXP_NAME="pi_behavior_2026_radio_from100k_${STAMP}"
RADIO_LOG="${CKPT_ROOT}/logs/train_radio_${STAMP}.log"
echo "Launching radio finetune exp=${OPENPI_EXP_NAME} log=${RADIO_LOG}"
cd "${ROOT}"
setsid nohup bash scripts/train_radio_8gpu.sh > "${RADIO_LOG}" 2>&1 < /dev/null &
echo $! > "${CKPT_ROOT}/logs/train_radio_${STAMP}.pid"
echo "radio pid=$(cat "${CKPT_ROOT}/logs/train_radio_${STAMP}.pid")"
