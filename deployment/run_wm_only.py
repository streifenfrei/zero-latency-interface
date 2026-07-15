#!/usr/bin/env python
"""World-model-only PushT interaction.

Runs the DINO-WM autoregressively from an initial context — user actions
drive the world model forward with no ground-truth feedback.  The PushT
environment runs silently alongside for comparison (shown as a small inset).

Input: Keyboard WASD always works.  When a PS4/Xbox gamepad is connected, the
left analog stick drives the agent.

Usage:
    # Keyboard only:
    python deployment/run_wm_only.py \
        --ckpt outputs/.../checkpoints

    # With gamepad:
    python deployment/run_wm_only.py \
        --ckpt outputs/.../checkpoints --gamepad
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

    # ── gamepad options ────────────────────────────────────────────────────
    gp = parser.add_argument_group("gamepad (PS4 / Xbox controller)")
    gp.add_argument("--gamepad", action="store_true",
                    help="Use a gamepad/joystick for teleoperation.")
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
        mode="wm_only",
        wm_ckpt_dir=args.ckpt,
        wm_ckpt_suffix=args.ckpt_suffix,
        display_size=args.display_size,
        fps=args.fps,
        gamepad=gamepad,
    )
    iface.run()


if __name__ == "__main__":
    main()
