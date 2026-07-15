#!/usr/bin/env python
"""Zero-latency PushT teleoperation with simulated network delay.

A constant-delay buffer holds back ground-truth frames, simulating network
latency.  The DINO-WM world model compensates by predicting ahead from the
last arrived (delayed) GT frame to the current timestep, giving the operator
a near-zero-latency view.

Display: [ WM-compensated view | delayed GT (what you'd normally see) ].

Usage:
    python deployment/run_zli.py \
        --ckpt outputs/outputs/2026-07-06/18-46-23/checkpoints \
        --delay_steps 10
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
        description="DINO-WM PushT zero-latency interface with simulated delay")
    parser.add_argument(
        "--ckpt", required=True,
        help="path to checkpoints/ directory")
    parser.add_argument(
        "--ckpt_suffix", default="_latest",
        help="checkpoint suffix")
    parser.add_argument(
        "--delay_steps", type=int, default=10,
        help="simulated network delay in raw timesteps (default: 10)")
    parser.add_argument(
        "--display_size", type=int, default=840,
        help="max display dimension (default: 840)")
    parser.add_argument(
        "--fps", type=int, default=20,
        help="target frames per second (default: 20)")
    args = parser.parse_args()

    iface = get_interface(
        mode="zli",
        wm_ckpt_dir=args.ckpt,
        wm_ckpt_suffix=args.ckpt_suffix,
        delay_steps=args.delay_steps,
        display_size=args.display_size,
        fps=args.fps,
    )
    iface.run()


if __name__ == "__main__":
    main()
