#!/usr/bin/env bash
# Launcher for run_overlay.py.
#
# Same X11/xcb workarounds as run_wm_only.sh / run_zli.sh (uv-managed CPython
# + NVIDIA X11): the uv CPython exports embedded X11/xcb symbols that
# interpose onto glfw/OpenCV (segfault) and the NVIDIA Vulkan ICD (SAPIEN
# Device() deadlock).  Preloading the real system libs fixes both;
# LIBGL_DRI3_DISABLE=1 alone is NOT sufficient.
#
# Note: the current working directory is preserved, so relative --ckpt /
# --policy-checkpoint paths resolve from where you invoke the script.
#
# usage: ./run_overlay.sh [args for run_overlay.py...]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libX11.so.6:/usr/lib/x86_64-linux-gnu/libxcb.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
export LIBGL_DRI3_DISABLE=1

exec "$SCRIPT_DIR/../.venv/bin/python" -u "$SCRIPT_DIR/run_overlay.py" "$@"
