#!/usr/bin/env bash
# Launcher for run_wm_only.py.
#
# Workarounds needed on machines using the uv-managed CPython + NVIDIA X11:
#   The uv CPython statically embeds X11 and xcb and exports their symbols
#   from libpython3.10.so, which interposes onto other libraries' calls:
#   1. glfw/OpenCV segfault (XCreateGC crash in XOpenDisplay) — preloading
#      the real libX11 wins the symbol resolution.
#   2. SAPIEN Device() hang — the NVIDIA Vulkan ICD's xcb calls (DRI3
#      extension query during vkEnumeratePhysicalDevices) bind to libpython's
#      embedded libxcb and deadlock (futex spin). Preloading the real libxcb
#      fixes it; LIBGL_DRI3_DISABLE=1 alone is NOT sufficient.
#
# Note: the current working directory is preserved, so relative --ckpt /
# --policy-checkpoint paths resolve from where you invoke the script.
#
# usage: ./run_wm_only.sh [args for run_wm_only.py...]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libX11.so.6:/usr/lib/x86_64-linux-gnu/libxcb.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
export LIBGL_DRI3_DISABLE=1

exec "$SCRIPT_DIR/../.venv/bin/python" -u "$SCRIPT_DIR/run_wm_only.py" "$@"
