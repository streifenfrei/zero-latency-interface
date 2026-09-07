#!/usr/bin/env python
"""World-model-only PushT interaction (ManiSkill 3D).

Runs the DINO-WM autoregressively from an initial context — user actions
drive the world model forward with no ground-truth feedback.  The PushT
environment runs silently alongside for comparison (shown as a small inset).

The environment is ManiSkill PushT (3D PandaStick robot).  Requires:
  - A GPU with a working Vulkan driver (SAPIEN / physx_cuda).
  - The sibling ``zero_latency_interface`` repo for the env wrapper.

Input: Keyboard WASD always works.  When a PS4/Xbox gamepad is connected, the
left analog stick drives the agent.

Usage:
    # Keyboard only:
    python deployment/run_wm_only.py \\
        --ckpt outputs/.../checkpoints

    # With gamepad:
    python deployment/run_wm_only.py \\
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
        "--ckpt", default=None,
        help="path to checkpoints/ directory (required unless --gt-only)")
    parser.add_argument(
        "--gt-only", action="store_true",
        help="Run the ground-truth env only: no world model is loaded or "
             "rolled out (the checkpoint is not needed).")
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

    # ── ManiSkill environment options ──────────────────────────────────────
    ms = parser.add_argument_group("ManiSkill environment")
    ms.add_argument("--env-id", default="PushT-XYExplore-v1",
                    help="Gymnasium env id (default: PushT-XYExplore-v1).")
    ms.add_argument("--zero-latency-root", default=None,
                    help="Path to zero_latency_interface repo.")
    ms.add_argument("--policy-checkpoint", default=None,
                    help="Path to a trained PPO policy checkpoint (.pt).")
    ms.add_argument("--control-mode", default="pd_ee_delta_pose",
                    choices=["pd_ee_delta_pose", "pd_ee_pose"],
                    help="EE controller (default: pd_ee_delta_pose).")
    ms.add_argument("--action-scale", type=float, default=0.1,
                    help="Maniskill action_scale (default: 0.1).")
    ms.add_argument("--step-size", type=float, default=0.02,
                    help="Metres per step at full deflection (default: 0.02 — "
                         "matches gamepad_teleop_pusht.py --speed 0.020).")
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
                    help="Fixed stick-tip height (m) (default: 0.015).")
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

    # GT-only runs skip the world model entirely (no checkpoint needed);
    # otherwise the checkpoint directory is required for the WM adapter.
    if args.gt_only:
        mode = "passthrough"
    else:
        if args.ckpt is None:
            parser.error("--ckpt is required unless --gt-only is given")
        mode = "wm_only"

    iface = get_interface(
        mode=mode,
        wm_ckpt_dir=args.ckpt or "",
        wm_ckpt_suffix=args.ckpt_suffix,
        display_size=args.display_size,
        fps=args.fps,
        gamepad=gamepad,
        env_kwargs=env_kwargs,
    )
    iface.run()


if __name__ == "__main__":
    main()
