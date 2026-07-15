#!/usr/bin/env python
"""Zero-latency PushT teleoperation with simulated network delay.

A constant-delay buffer holds back ground-truth frames, simulating network
latency.  The DINO-WM world model compensates by predicting ahead from the
last arrived (delayed) GT frame to the current timestep, giving the operator
a near-zero-latency view.

Display: [ WM-compensated view | delayed GT (what you'd normally see) ].

Input: Keyboard WASD always works.  When a PS4/Xbox gamepad is connected, the
left analog stick drives the agent (matching the zero_latency_interface teleop).

Usage:
    # Keyboard only:
    python deployment/run_zli.py \
        --ckpt outputs/.../checkpoints --delay_steps 10

    # With gamepad:
    python deployment/run_zli.py \
        --ckpt outputs/.../checkpoints --delay_steps 10 --gamepad

    # Gamepad with custom speed:
    python deployment/run_zli.py \
        --ckpt outputs/.../checkpoints --gamepad --gamepad-speed 80
"""

import argparse
import os
import sys

# Ensure repo root is on sys.path.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from deployment import get_interface
from deployment.gamepad import GamepadInput


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

    # ── gamepad options ────────────────────────────────────────────────────
    gp = parser.add_argument_group("gamepad (PS4 / Xbox controller)")
    gp.add_argument("--gamepad", action="store_true",
                    help="Use a gamepad/joystick for teleoperation. "
                         "Blocks until a controller is connected.")
    gp.add_argument("--gamepad-speed", type=float, default=60.0,
                    help="Pixel-space push magnitude per step at full stick "
                         "deflection (default: 60 — matches keyboard WASD).")
    gp.add_argument("--gamepad-deadzone", type=float, default=0.15,
                    help="Stick deadzone fraction (default: 0.15).")
    gp.add_argument("--gamepad-no-fixed-magnitude", action="store_true",
                    help="Disable fixed-magnitude mode (variable-speed stick).")
    gp.add_argument("--gamepad-precision-scale", type=float, default=0.3,
                    help="Speed multiplier while LB is held (default: 0.3).")
    gp.add_argument("--gamepad-boost-scale", type=float, default=2.0,
                    help="Speed multiplier while RB is held (default: 2.0).")
    gp.add_argument("--gamepad-invert-x", action="store_true",
                    help="Invert the x-axis of the stick.")
    gp.add_argument("--gamepad-invert-y", action="store_true",
                    help="Invert the y-axis of the stick.")
    gp.add_argument("--gamepad-swap-xy", action="store_true",
                    help="Swap stick x and y axes.")
    gp.add_argument("--gamepad-debug", action="store_true",
                    help="Print live gamepad axis/button values to stdout.")
    args = parser.parse_args()

    # ── gamepad setup ──────────────────────────────────────────────────────
    gamepad = None
    if args.gamepad:
        gamepad = GamepadInput(
            speed=args.gamepad_speed,
            deadzone=args.gamepad_deadzone,
            fixed_magnitude=not args.gamepad_no_fixed_magnitude,
            precision_scale=args.gamepad_precision_scale,
            boost_scale=args.gamepad_boost_scale,
            invert_x=args.gamepad_invert_x,
            invert_y=args.gamepad_invert_y,
            swap_xy=args.gamepad_swap_xy,
            debug=args.gamepad_debug,
        )
        gamepad.setup()

    iface = get_interface(
        mode="zli",
        wm_ckpt_dir=args.ckpt,
        wm_ckpt_suffix=args.ckpt_suffix,
        delay_steps=args.delay_steps,
        display_size=args.display_size,
        fps=args.fps,
        gamepad=gamepad,
    )
    iface.run()


if __name__ == "__main__":
    main()
