#!/bin/bash
# Launch the DINO-WM PushT zero-latency interface on a GPU node.
#
# This is an *interactive* GUI application — run it with:
#   sbatch deployment/slurm/pusht_zli.sbatch
#
# Then SSH to the allocated node with X11 forwarding to see the display:
#   ssh -X <node>
#   # The OpenCV window will appear on your local machine.
#
# Overridable via env:
#   MODE          — "zli" (default), "wm_only", or "passthrough"
#   CKPT_DIR      — path to checkpoints/ directory
#   DELAY_STEPS   — simulated delay in raw timesteps (ZLI mode, default: 10)

#SBATCH --job-name=dino_pusht_zli
#SBATCH --output=logs/pusht_zli_%j.out
#SBATCH --error=logs/pusht_zli_%j.err
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -e

ZLI_PATH="${ZLI_PATH:-/mnt/cluster/workspaces/lelismart/negative_latency/zli}"
cd "$ZLI_PATH"
export PATH="/mnt/cluster/environments/lelismart/uv:$PATH"

MODE="${MODE:-zli}"
CKPT_DIR="${CKPT_DIR:-$ZLI_PATH/dino_wm/outputs/outputs/2026-07-06/18-46-23/checkpoints}"
CKPT_SUFFIX="${CKPT_SUFFIX:-_latest}"
DELAY_STEPS="${DELAY_STEPS:-10}"
DISPLAY_SIZE="${DISPLAY_SIZE:-840}"
FPS="${FPS:-20}"

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "mode=$MODE  ckpt=$CKPT_DIR$CKPT_SUFFIX  delay=$DELAY_STEPS steps"

# Strip distributed env vars.
run() { CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
          env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u SLURM_PROCID -u MASTER_ADDR \
          .venv/bin/python "$@"; }

mkdir -p logs

if [ "$MODE" = "wm_only" ]; then
    run deployment/run_wm_only.py \
        --ckpt "$CKPT_DIR" \
        --ckpt_suffix "$CKPT_SUFFIX" \
        --display_size "$DISPLAY_SIZE" \
        --fps "$FPS"
elif [ "$MODE" = "passthrough" ]; then
    run -c "
import sys
sys.path.insert(0, '.')
from deployment import get_interface
get_interface(mode='passthrough', display_size=$DISPLAY_SIZE, fps=$FPS).run()
"
else
    run deployment/run_zli.py \
        --ckpt "$CKPT_DIR" \
        --ckpt_suffix "$CKPT_SUFFIX" \
        --delay_steps "$DELAY_STEPS" \
        --display_size "$DISPLAY_SIZE" \
        --fps "$FPS"
fi

echo "DONE"
