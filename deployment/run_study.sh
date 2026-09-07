#!/usr/bin/env bash
# Launcher for run_study.py.
#
# Same X11/xcb workarounds as run_zli.sh/run_wm_only.sh (uv-managed CPython
# + NVIDIA X11): the uv CPython statically embeds X11 and xcb and exports
# their symbols from libpython3.10.so, which interposes onto other
# libraries' calls — segfaulting glfw/OpenCV and deadlocking the NVIDIA
# Vulkan ICD during SAPIEN Device() creation.  Preloading the real system
# libs fixes both; LIBGL_DRI3_DISABLE=1 alone is NOT sufficient.
#
# Note: the current working directory is preserved, so a relative
# --results-dir / --ckpt / --policy-checkpoint resolves from where you
# invoke the script.
#
# usage: ./run_study.sh [args for run_study.py...]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libX11.so.6:/usr/lib/x86_64-linux-gnu/libxcb.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
export LIBGL_DRI3_DISABLE=1

exec "$SCRIPT_DIR/../.venv/bin/python" -u "$SCRIPT_DIR/run_study.py" "$@"
