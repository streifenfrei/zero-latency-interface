#!/bin/bash
#SBATCH --job-name=dino_wm_pusht_hr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# DINO-WM — HIGH-RESOLUTION LATENT training on PushT (28×28 latent grid).
#
# Same online data generation as pusht_train.sbatch, but with img_size=448:
#   - ManiSkill base_camera is overridden from 128×128 → 448×448
#     (sensor_configs in training/pusht_online.py), so the renders carry
#     genuinely new detail instead of being upscaled.
#   - The model derives everything from img_size // decoder_scale(=16):
#       encoder input  = (448//16) × patch14 = 392²  (DINOv2-S/14)
#       latent grid    = 28×28 = 784 tokens   (4× the default 14×14)
#       decoded output = 28×28 × 16× upscale  = 448²  = img_size ✓
#   - The T-block then spans ~8×8 latent cells instead of ~4×4.
#
# Costs vs the default run (expect several × longer epochs):
#   - predictor attention over 4×784 = 3136 tokens  (16× attention work)
#   - encoder forward at 392² (frozen), decoder at 448² (4× pixels)
#   - producer renders 448² instead of 128² (~12× render pixels)
#   => BATCH_SIZE defaults to 8 (batch 32 will OOM the 40GB A100) and
#      NUM_ENVS to 32 to keep render throughput sane.
#
# Fresh start by default: the 14×14 checkpoints are a different latent
# geometry (weights are resolution-agnostic, so a warm start is possible:
# set RESUME_CKPT=/path/to/model_latest.pth to resume from epoch N+1).
#
# Overridable via env: same as pusht_train.sbatch, plus:
#   IMG_SIZE   — model image size / latent grid driver (default: 448)
#   CAMERA_RES — sim camera resolution (default: IMG_SIZE)
#   WITH_BLOCK_POSE — 1 = append the T-block pose (xy + cos/sin) to the WM
#                     proprio (4 -> 8 dims; default 0).  Must be enabled at
#                     inference too (--with-block-pose on the deployment
#                     runners / viz_rollout.py).
# ---------------------------------------------------------------------------

ZLI_PATH="${ZLI_PATH:-/mnt/cluster/workspaces/lelismart/negative_latency/zero-latency-interface}"
ZERO_LATENCY_ROOT="${ZERO_LATENCY_ROOT:-/mnt/cluster/workspaces/lelismart/negative_latency/zero_latency_interface}"
CKPT_BASE_PATH="${CKPT_BASE_PATH:-${ZLI_PATH}/outputs}"

IMG_SIZE="${IMG_SIZE:-448}"
CAMERA_RES="${CAMERA_RES:-${IMG_SIZE}}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FRAMESKIP="${FRAMESKIP:-1}"
NUM_HIST="${NUM_HIST:-1}"
NUM_ENVS="${NUM_ENVS:-32}"
SEED="${SEED:-0}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-50000}"
WITH_BLOCK_POSE="${WITH_BLOCK_POSE:-0}"

if [[ "${WITH_BLOCK_POSE}" == "1" ]]; then
    BLOCK_POSE_HYDRA=(--extra-hydra "env.dataset.with_block_pose=true")
else
    BLOCK_POSE_HYDRA=()
fi

# Fresh start by default (new latent geometry). Set RESUME_CKPT=<path> to
# warm-start from a 14×14 checkpoint (weights transfer, geometry changes).
RESUME_CKPT="${RESUME_CKPT-}"

# --- env setup -----------------------------------------------------------
set -e
source /mnt/cluster/environments/lelismart/uv/env
cd "${ZLI_PATH}"
mkdir -p logs

# Cluster-proven interpreter: zli/.venv (torch 2.13.0+cu130, full cuDNN).
# The repo's own .venv is the WORKSTATION build (torch 2.5.1+cu121) — it
# crashes on the cluster ("libcudnn.so.9: cannot open shared object file")
# and must not be uv-synced from cluster jobs (that also breaks it on the
# workstation). To update sim deps: cd zli && uv sync --extra sim.
VENV_PATH="${VENV_PATH:-/mnt/cluster/workspaces/lelismart/negative_latency/zli/.venv}"

source ~/.wandb_key           # W&B API key for logging

export ZERO_LATENCY_ROOT="${ZERO_LATENCY_ROOT}"

echo "============================================"
echo "DINO-WM — PushT online training (HIGH-RES)"
echo "============================================"
echo "config:  env=pusht_online img_size=${IMG_SIZE} camera_res=${CAMERA_RES}"
echo "         latent_grid=$((IMG_SIZE / 16))x$((IMG_SIZE / 16))"
echo "         frameskip=${FRAMESKIP} num_hist=${NUM_HIST}"
echo "         epochs=${EPOCHS} batch_size=${BATCH_SIZE} seed=${SEED}"
echo "         num_envs=${NUM_ENVS} samples_per_epoch=${SAMPLES_PER_EPOCH}"
echo "         with_block_pose=${WITH_BLOCK_POSE}"
echo "policy:  ${ZERO_LATENCY_ROOT}/checkpoints/policy.pt"
echo "ckpts:   ${CKPT_BASE_PATH}"
echo "============================================"

# Ensure repo root AND dino_wm/ are on PYTHONPATH so both
# training.pusht_online and dino_wm.* modules are importable.
export PYTHONPATH="${ZLI_PATH}:${ZLI_PATH}/dino_wm:${PYTHONPATH}"

# Strip distributed env vars — we run single-GPU training.
run() {
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        env -u RANK -u WORLD_SIZE -u LOCAL_RANK \
            -u SLURM_PROCID -u MASTER_ADDR -u MASTER_PORT \
        "${VENV_PATH}/bin/python" "$@"
}

run training/train_pusht.py \
    --online \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --frameskip "${FRAMESKIP}" \
    --num-hist "${NUM_HIST}" \
    --num-envs "${NUM_ENVS}" \
    --seed "${SEED}" \
    --samples-per-epoch "${SAMPLES_PER_EPOCH}" \
    --ckpt-base-path "${CKPT_BASE_PATH}" \
    --extra-hydra "img_size=${IMG_SIZE}" \
    --extra-hydra "env.dataset.camera_res=${CAMERA_RES}" \
    "${BLOCK_POSE_HYDRA[@]}" \
    ${RESUME_CKPT:+--resume "${RESUME_CKPT}"}

echo "DONE"
