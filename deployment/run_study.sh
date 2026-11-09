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

# Venv guard: this shared venv's torch must be the workstation-pinned
# 2.5.1+cu121 build.  It has been broken repeatedly (uv sync -> nccl symbol
# errors, an interrupted install -> missing torch/__init__.py), so a REAL
# import test runs every launch and fails fast with the actual error and
# the re-pin command.  (Install new deps with `uv pip install --python
# .venv/bin/python <pkg>`, never `uv sync`/`uv run`.)
if ! "$SCRIPT_DIR/../.venv/bin/python" -c "import torch" >/dev/null 2>&1; then
    echo "[launcher] WARNING: the venv's torch is broken — import failed with:" >&2
    "$SCRIPT_DIR/../.venv/bin/python" -c "import torch" 2>&1 | tail -5 >&2 || true
    echo "[launcher] Re-pin with:" >&2
    echo "  uv pip install --python .venv/bin/python --reinstall --no-deps --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121" >&2
    exit 1
fi

export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libX11.so.6:/usr/lib/x86_64-linux-gnu/libxcb.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
export LIBGL_DRI3_DISABLE=1
# Ubuntu 20.04's system Vulkan loader (libvulkan.so.1, 1.2.131) segfaults in
# vkEnumerateInstance* (seen on G27WS0011, driver 535.230).  SAPIEN's glfw
# dlopens the UNVERSIONED libvulkan.so, which LD_PRELOAD cannot intercept, so
# put the vendored Khronos loader v1.3.296 first on the dlopen search path
# (see vendor/vulkan/README.md).
export LD_LIBRARY_PATH="$SCRIPT_DIR/vendor/vulkan${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# Real window for the study UI (gamepad.py pins SDL_VIDEODRIVER=dummy for
# its headless pad reader; set this first so the x11 driver wins).
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-x11}"

exec "$SCRIPT_DIR/../.venv/bin/python" -u "$SCRIPT_DIR/run_study.py" "$@"
