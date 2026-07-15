#!/bin/bash
# Profile DINO-WM PushT model latency, throughput, and GPU memory on an RTX A5000.
#
# Submit:
#   sbatch deployment/slurm/profile_pusht.sbatch
#
#   # override checkpoint / batch size / rollout length:
#   CKPT_DIR=... BATCH_SIZE=4 ROLLOUT_HORIZON=50 sbatch --export=ALL deployment/slurm/profile_pusht.sbatch

#SBATCH --job-name=dino_profile_pusht
#SBATCH --output=logs/profile_pusht_%j.out
#SBATCH --error=logs/profile_pusht_%j.err
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -e

# --- repo + env ---
ZLI_PATH="${ZLI_PATH:-/mnt/cluster/workspaces/lelismart/negative_latency/zli}"
cd "$ZLI_PATH"
export PATH="/mnt/cluster/environments/lelismart/uv:$PATH"

# --- knobs ---
CKPT_DIR="${CKPT_DIR:-$ZLI_PATH/dino_wm/outputs/outputs/2026-07-06/18-46-23/checkpoints}"
CKPT_SUFFIX="${CKPT_SUFFIX:-_latest}"
BATCH_SIZE="${BATCH_SIZE:-1}"
N_WARMUP="${N_WARMUP:-10}"
N_TIMED="${N_TIMED:-50}"
ROLLOUT_HORIZON="${ROLLOUT_HORIZON:-20}"

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "ckpt=$CKPT_DIR$CKPT_SUFFIX  batch=$BATCH_SIZE  rollout_horizon=$ROLLOUT_HORIZON"

# Strip distributed env vars
run() { CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
          env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u SLURM_PROCID -u MASTER_ADDR \
          .venv/bin/python "$@"; }

mkdir -p logs

run deployment/profile.py \
    --ckpt "$CKPT_DIR" \
    --ckpt_suffix "$CKPT_SUFFIX" \
    --batch_size "$BATCH_SIZE" \
    --n_warmup "$N_WARMUP" \
    --n_timed "$N_TIMED" \
    --rollout_horizon "$ROLLOUT_HORIZON"

echo "DONE"
