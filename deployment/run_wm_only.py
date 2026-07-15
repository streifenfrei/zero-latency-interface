#!/usr/bin/env python
"""World-model-only PushT interaction.

Runs the DINO-WM autoregressively from an initial context — user actions
drive the world model forward with no ground-truth feedback.  The PushT
environment runs silently alongside for comparison (shown as a small inset).

Usage:
    python deployment/run_wm_only.py \
        --ckpt outputs/outputs/2026-07-06/18-46-23/checkpoints
"""

import argparse
import os
import sys

# Ensure repo root is on sys.path so that dino_wm (submodule),
# deployment, and training are all importable.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from deployment import get_interface


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DINO-WM PushT world-model-only interaction")
    parser.add_argument(
        "--ckpt", required=True,
        help="path to checkpoints/ directory")
    parser.add_argument(
        "--ckpt_suffix", default="_latest",
        help="checkpoint suffix")
    parser.add_argument(
        "--display_size", type=int, default=840,
        help="max display dimension (default: 840)")
    parser.add_argument(
        "--fps", type=int, default=20,
        help="target frames per second (default: 20)")
    args = parser.parse_args()

    iface = get_interface(
        mode="wm_only",
        wm_ckpt_dir=args.ckpt,
        wm_ckpt_suffix=args.ckpt_suffix,
        display_size=args.display_size,
        fps=args.fps,
    )
    iface.run()


if __name__ == "__main__":
    main()
