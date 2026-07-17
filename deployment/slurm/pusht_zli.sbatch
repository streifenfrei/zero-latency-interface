#!/bin/bash
# Launch the DINO-WM PushT zero-latency interface on a GPU node (ManiSkill 3D).
#
# This is an *interactive* GUI application — run it with:
#   sbatch deployment/slurm/pusht_zli.sbatch
#
# Then SSH to the allocated node with X11 forwarding to see the display:
#   ssh -X <node>
#   # The OpenCV window will appear on your local machine.
#
# Overridable via env:
#   MODE                — "zli" (default), "wm_only", or "passthrough"
#   CKPT_DIR            — path to checkpoints/ directory
#   DELAY_STEPS         — simulated delay in raw timesteps (ZLI mode, default: 10)
#   ZERO_LATENCY_ROOT   — path to zero_latency_interface repo
#   POLICY_CHECKPOINT   — path to PPO policy checkpoint (args only)
#   CONTROL_MODE        — "pd_ee_delta_pose" (default) or "pd_ee_pose"
#   STEP_SIZE           — metres/step at full deflection (default: 0.02)
#   SIM_BACKEND         — "physx_cuda" (default) or "physx_cpu"

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

# ManiSkill environment config.
ZERO_LATENCY_ROOT="${ZERO_LATENCY_ROOT:-$ZLI_PATH/../zero_latency_interface}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-$ZERO_LATENCY_ROOT/checkpoints/policy.pt}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_delta_pose}"
STEP_SIZE="${STEP_SIZE:-0.02}"
ACTION_SCALE="${ACTION_SCALE:-0.1}"
SIM_BACKEND="${SIM_BACKEND:-physx_cuda}"

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "mode=$MODE  ckpt=$CKPT_DIR$CKPT_SUFFIX  delay=$DELAY_STEPS steps"
echo "mani: control=$CONTROL_MODE  step=$STEP_SIZE  sim=$SIM_BACKEND"
echo "zl_root=$ZERO_LATENCY_ROOT  policy_ckpt=$POLICY_CHECKPOINT"

# Strip distributed env vars.
run() { CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
          env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u SLURM_PROCID -u MASTER_ADDR \
          .venv/bin/python "$@"; }

mkdir -p logs

# Gamepad (set GAMEPAD=1 to enable).
GAMEPAD_FLAG=""
if [[ "${GAMEPAD:-0}" == "1" ]]; then
    GAMEPAD_FLAG="--gamepad"
    echo "gamepad: enabled"
fi

# Shared ManiSkill args.
MANI_ARGS=(
    --zero-latency-root "$ZERO_LATENCY_ROOT"
    --control-mode "$CONTROL_MODE"
    --step-size "$STEP_SIZE"
    --action-scale "$ACTION_SCALE"
    --sim-backend "$SIM_BACKEND"
)
if [[ -n "${POLICY_CHECKPOINT}" && -f "${POLICY_CHECKPOINT}" ]]; then
    MANI_ARGS+=(--policy-checkpoint "$POLICY_CHECKPOINT")
fi

if [ "$MODE" = "wm_only" ]; then
    run deployment/run_wm_only.py \
        --ckpt "$CKPT_DIR" \
        --ckpt_suffix "$CKPT_SUFFIX" \
        --display_size "$DISPLAY_SIZE" \
        --fps "$FPS" \
        $GAMEPAD_FLAG \
        "${MANI_ARGS[@]}"
elif [ "$MODE" = "passthrough" ]; then
    run -c "
import sys
sys.path.insert(0, '.')
from deployment import get_interface
env_kw = dict(
    zero_latency_root='$ZERO_LATENCY_ROOT',
    control_mode='$CONTROL_MODE',
    step_size=$STEP_SIZE,
    action_scale=$ACTION_SCALE,
    sim_backend='$SIM_BACKEND',
)
if '$POLICY_CHECKPOINT' and __import__('os').path.isfile('$POLICY_CHECKPOINT'):
    env_kw['policy_checkpoint'] = '$POLICY_CHECKPOINT'
get_interface(mode='passthrough', display_size=$DISPLAY_SIZE, fps=$FPS, env_kwargs=env_kw).run()
"
else
    run deployment/run_zli.py \
        --ckpt "$CKPT_DIR" \
        --ckpt_suffix "$CKPT_SUFFIX" \
        --delay_steps "$DELAY_STEPS" \
        --display_size "$DISPLAY_SIZE" \
        --fps "$FPS" \
        $GAMEPAD_FLAG \
        "${MANI_ARGS[@]}"
fi

echo "DONE"
