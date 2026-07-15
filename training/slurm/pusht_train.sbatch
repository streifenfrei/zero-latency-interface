#!/bin/bash
#SBATCH --job-name=dino_wm_pusht
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# DINO-WM — World Model training on PushT with ONLINE data generation.
#
# Instead of loading pre-recorded episodes from disk, this launches a
# ManiSkill PushT environment ON THE TRAINING GPU that continuously rolls
# out a trained PPO policy across 64 parallel envs with a noise ladder
# (env 0 = pure policy, last env = pure noise). Data is generated in-process
# on a background thread and served through a thread-safe queue — no
# offline dataset needed.
#
# Prerequisites:
#   1. Trained PPO policy at $ZERO_LATENCY_ROOT/checkpoints/policy.pt
#   2. GPU with Vulkan driver (SAPIEN/physx_cuda backend)
#   3. uv environment with mani-skill + gymnasium (uv sync --group sim)
#
# Overridable via env:
#   ZERO_LATENCY_ROOT — path to zero_latency_interface repo (required)
#   CKPT_BASE_PATH    — where to save model checkpoints
#   EPOCHS            — training epochs (default: 100)
#   BATCH_SIZE        — global batch size (default: 32)
#   FRAMESKIP         — env steps per stored frame (default: 5)
#   NUM_HIST          — context frames (default: 3)
#   NUM_ENVS          — parallel envs / noise-ladder rungs (default: 64)
#   SEED              — random seed (default: 0)
#   SAMPLES_PER_EPOCH — samples per epoch (default: 50000)
# ---------------------------------------------------------------------------

ZLI_PATH="${ZLI_PATH:-/mnt/cluster/workspaces/lelismart/negative_latency/zli}"
ZERO_LATENCY_ROOT="${ZERO_LATENCY_ROOT:-/mnt/cluster/workspaces/lelismart/negative_latency/zero_latency_interface}"
CKPT_BASE_PATH="${CKPT_BASE_PATH:-${ZLI_PATH}/outputs}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
FRAMESKIP="${FRAMESKIP:-5}"
NUM_HIST="${NUM_HIST:-3}"
NUM_ENVS="${NUM_ENVS:-64}"
SEED="${SEED:-0}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-50000}"

# --- auto-resume ---------------------------------------------------------
# If RESUME_CKPT is not set but a latest checkpoint exists, resume from it.
# Override by exporting RESUME_CKPT=<path> (or RESUME_CKPT= to force fresh).
RESUME_CKPT="${RESUME_CKPT-__AUTO__}"
if [[ "${RESUME_CKPT}" == "__AUTO__" ]]; then
    LATEST=$(ls -t "${CKPT_BASE_PATH}"/*/checkpoints/model_latest.pth 2>/dev/null | head -1)
    if [[ -n "${LATEST}" ]]; then
        RESUME_CKPT="${LATEST}"
        echo "Auto-resuming from ${RESUME_CKPT}"
    else
        RESUME_CKPT=""
        echo "No checkpoint found — starting fresh"
    fi
fi

# --- env setup -----------------------------------------------------------
source /mnt/cluster/environments/lelismart/uv/env
cd "${ZLI_PATH}"
mkdir -p logs

export ZERO_LATENCY_ROOT="${ZERO_LATENCY_ROOT}"

echo "============================================"
echo "DINO-WM — PushT online training"
echo "============================================"
echo "config:  env=pusht_online frameskip=${FRAMESKIP} num_hist=${NUM_HIST}"
echo "         epochs=${EPOCHS} batch_size=${BATCH_SIZE} seed=${SEED}"
echo "         num_envs=${NUM_ENVS} samples_per_epoch=${SAMPLES_PER_EPOCH}"
echo "policy:  ${ZERO_LATENCY_ROOT}/checkpoints/policy.pt"
echo "ckpts:   ${CKPT_BASE_PATH}"
echo "============================================"

# Ensure zli repo root is on PYTHONPATH so training.pusht_online is importable.
export PYTHONPATH="${ZLI_PATH}:${PYTHONPATH}"

srun python training/train_pusht.py \
    --online \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --frameskip "${FRAMESKIP}" \
    --num-hist "${NUM_HIST}" \
    --num-envs "${NUM_ENVS}" \
    --seed "${SEED}" \
    --samples-per-epoch "${SAMPLES_PER_EPOCH}" \
    --ckpt-base-path "${CKPT_BASE_PATH}" \
    ${RESUME_CKPT:+--resume "${RESUME_CKPT}"}
