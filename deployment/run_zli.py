#!/usr/bin/env python
"""Zero-latency PushT teleoperation with simulated network delay (ManiSkill 3D).

A constant-delay buffer holds back ground-truth frames, simulating network
latency.  The DINO-WM world model compensates by predicting ahead from the
last arrived (delayed) GT frame to the current timestep, giving the operator
a near-zero-latency view.

Display: [ WM-compensated view | delayed GT (what you'd normally see) ].

The environment is ManiSkill PushT (3D PandaStick robot).  Requires:
  - A GPU with a working Vulkan driver (SAPIEN / physx_cuda).
  - The sibling ``zero_latency_interface`` repo for the env wrapper.
  - (optional) A trained PPO policy checkpoint so the wrapper config matches
    what the world model was trained with.

Input: Keyboard WASD always works.  When a PS4/Xbox gamepad is connected, the
left analog stick drives the agent (matching the zero_latency_interface teleop).

Usage:
    # Keyboard only (basic):
    python deployment/run_zli.py \\
        --ckpt outputs/.../checkpoints --delay-steps 10

    # With gamepad and explicit env config:
    python deployment/run_zli.py \\
        --ckpt outputs/.../checkpoints --delay-steps 10 --gamepad \\
        --zero-latency-root ../zero_latency_interface \\
        --step-size 0.02

    # Match a specific training run's policy checkpoint:
    python deployment/run_zli.py \\
        --ckpt outputs/.../checkpoints --gamepad \\
        --policy-checkpoint ../zero_latency_interface/checkpoints/policy.pt
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
        "--delay-steps", type=int, default=10,
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

    # ── ManiSkill environment options ──────────────────────────────────────
    ms = parser.add_argument_group("ManiSkill environment")
    ms.add_argument("--env-id", default="PushT-XYExplore-v1",
                    help="Gymnasium env id (default: PushT-XYExplore-v1). "
                         "Use PushT-v1 for the standard task.")
    ms.add_argument("--zero-latency-root", default=None,
                    help="Path to zero_latency_interface repo. "
                         "Defaults to ../zero_latency_interface relative to zli.")
    ms.add_argument("--policy-checkpoint", default=None,
                    help="Path to a trained PPO policy checkpoint (.pt). "
                         "Only its args are read to match the training-time "
                         "wrapper config (action_scale, push_height, etc.).")
    ms.add_argument("--control-mode", default="pd_ee_delta_pose",
                    choices=["pd_ee_delta_pose", "pd_ee_pose"],
                    help="EE controller (default: pd_ee_delta_pose).")
    ms.add_argument("--action-scale", type=float, default=0.1,
                    help="Maniskill action_scale: metres/step at full "
                         "normalised action (default: 0.1 → ±0.1 m).")
    ms.add_argument("--step-size", type=float, default=0.02,
                    help="Metres per step at full keyboard/gamepad deflection "
                         "(default: 0.02 — matches gamepad_teleop_pusht.py).")
    ms.add_argument("--fixed-magnitude", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Unit-circle projection on commands (training-policy "
                         "style: every non-idle move is action_scale m/step). "
                         "Default: follow the policy checkpoint's setting when "
                         "--policy-checkpoint is given, else off.  With the "
                         "projection off, full deflection moves step_size "
                         "m/step (teleop style, --speed in the other repo).")
    ms.add_argument("--level-ee", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Actively level the end-effector back to vertical "
                         "each step (delta mode), as in the training data. "
                         "Default: follow the policy checkpoint's setting when "
                         "--policy-checkpoint is given, else off (drot=0 — the "
                         "stick stays tilted after collisions).")
    ms.add_argument("--push-height", type=float, default=0.015,
                    help="Fixed stick-tip height (m) above the table "
                         "(default: 0.015).")
    ms.add_argument("--sim-backend", default="physx_cuda",
                    choices=["physx_cuda", "physx_cpu"],
                    help="ManiSkill simulation backend (default: physx_cuda).")
    ms.add_argument("--max-episode-steps", type=int, default=100_000,
                    help="Per-episode horizon (default: 100000).")
    ms.add_argument("--no-velocity", action="store_true",
                    help="Exclude EE velocity from proprio (default: include).")
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

    # ── build env kwargs ───────────────────────────────────────────────────
    env_kwargs = dict(
        env_id=args.env_id,
        zero_latency_root=args.zero_latency_root,
        policy_checkpoint=args.policy_checkpoint,
        control_mode=args.control_mode,
        action_scale=args.action_scale,
        step_size=args.step_size,
        max_input=args.gamepad_speed if args.gamepad else 60.0,
        fixed_magnitude=args.fixed_magnitude,
        level_ee=args.level_ee,
        push_height=args.push_height,
        sim_backend=args.sim_backend,
        max_episode_steps=args.max_episode_steps,
        with_velocity=not args.no_velocity,
    )

    iface = get_interface(
        mode="zli",
        wm_ckpt_dir=args.ckpt,
        wm_ckpt_suffix=args.ckpt_suffix,
        delay_steps=args.delay_steps,
        display_size=args.display_size,
        fps=args.fps,
        gamepad=gamepad,
        env_kwargs=env_kwargs,
    )
    iface.run()


if __name__ == "__main__":
    main()
