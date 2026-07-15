#!/bin/bash
# Generate autoregressive rollout videos from a trained DINO-WM PushT checkpoint.
#
# Loads the world model, drives the ManiSkill PushT environment with the trained
# PPO policy to collect ground-truth trajectories, runs autoregressive rollouts,
# and writes side-by-side MP4 videos:
#
#   GT-RGB | GT-PCA | Pred-PCA
#
# With --pixel-decoder: GT-RGB | Pred-RGB appended as a second row.
#
# Submit from a node with the SLURM client:
#   sbatch deployment/slurm/viz_pusht_rollout.sbatch
#
#   # override checkpoint / episode count:
#   CKPT_DIR=outputs/2026-07-06/15-38-40/checkpoints MAX_EPISODES=10 \
#     sbatch --export=ALL deployment/slurm/viz_pusht_rollout.sbatch
#
# Outputs land in $OUT (default logs/viz_pusht_rollout/):
#   pusht_rollout_00.mp4  …  pusht_rollout_0N.mp4
# Job stdout/err: logs/viz_pusht_rollout_<jobid>.{out,err}

#SBATCH --job-name=dino_viz_pusht
#SBATCH --output=logs/viz_pusht_rollout_%j.out
#SBATCH --error=logs/viz_pusht_rollout_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -e

# --- repo + env ---------------------------------------------------------------
ZLI_PATH="${ZLI_PATH:-/mnt/cluster/workspaces/lelismart/negative_latency/zli}"
ZERO_LATENCY_ROOT="${ZERO_LATENCY_ROOT:-/mnt/cluster/workspaces/lelismart/negative_latency/zero_latency_interface}"
cd "$ZLI_PATH"

# --- knobs --------------------------------------------------------------------
CKPT_DIR="${CKPT_DIR:-}"
CKPT_SUFFIX="${CKPT_SUFFIX:-_latest}"
OUT="${OUT:-$ZLI_PATH/logs/viz_pusht_rollout}"
MAX_EPISODES="${MAX_EPISODES:-5}"
FPS="${FPS:-20}"
NUM_ROLLOUT_STEPS="${NUM_ROLLOUT_STEPS:-100}"
PIXEL_DECODER="${PIXEL_DECODER:-1}"   # set to 0 to disable pixel-decoder row
SEED="${SEED:-42}"

if [[ -z "${CKPT_DIR}" ]]; then
    # Auto-detect the latest checkpoint directory.
    CKPT_DIR=$(ls -dt "$ZLI_PATH"/dino_wm/outputs/outputs/*/*/checkpoints 2>/dev/null | head -1)
    if [[ -z "${CKPT_DIR}" ]]; then
        echo "ERROR: no checkpoint found and CKPT_DIR not set"
        exit 1
    fi
    echo "Auto-detected checkpoint: ${CKPT_DIR}"
fi

echo "============================================"
echo "DINO-WM — PushT rollout visualisation"
echo "============================================"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "ckpt=${CKPT_DIR}  suffix=${CKPT_SUFFIX}"
echo "max_episodes=${MAX_EPISODES}  rollout_steps=${NUM_ROLLOUT_STEPS}"
echo "pixel_decoder=${PIXEL_DECODER}  fps=${FPS}  out=${OUT}"
echo "============================================"

# Strip stray distributed env vars so nothing forces dist init.
run() {
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u SLURM_PROCID -u MASTER_ADDR \
        .venv/bin/python "$@"
}

mkdir -p "$OUT"
mkdir -p logs

PIXEL_FLAG=""
if [[ "${PIXEL_DECODER}" != "0" ]]; then
    PIXEL_FLAG="--pixel-decoder"
fi

run deployment/viz_rollout.py \
    --ckpt "$CKPT_DIR" \
    --ckpt-suffix "$CKPT_SUFFIX" \
    --zero-latency-root "$ZERO_LATENCY_ROOT" \
    --max-episodes "$MAX_EPISODES" \
    --num-rollout-steps "$NUM_ROLLOUT_STEPS" \
    --fps "$FPS" \
    --seed "$SEED" \
    $PIXEL_FLAG \
    --out "$OUT"

echo "DONE -> $OUT"
