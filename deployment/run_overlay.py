#!/usr/bin/env python
"""Overlay comparison: GT environment vs open-loop WM rollout.

The world model is grounded once on the initial context and then runs purely
autoregressively from the user's actions — the same open-loop scheme as
``run_wm_only.py``.  Its prediction is blended ON TOP of the live GT frame
as a translucent overlay, so drift between the two is directly visible.

Keys:
    WASD            move (same pixel-space mapping as run_wm_only.py)
    [ / ]           shift the blend toward GT / WM (GT weight shown on screen)
    o               toggle the WM overlay on/off (pure GT view)
    r               reset episode and re-ground the WM
    ESC             quit

Input: keyboard WASD always works.  With --gamepad, the left analog stick
drives the agent (same mapping as run_wm_only.py / run_zli.py).

Usage:
    python deployment/run_overlay.py --ckpt outputs/.../checkpoints
    python deployment/run_overlay.py --ckpt outputs/.../checkpoints --gamepad \\
        --policy-checkpoint ../zero_latency_interface/checkpoints/policy.pt
"""

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import cv2
import numpy as np

from deployment.gamepad import GamepadInput
from deployment.interface import KEY_ACTIONS, _resize_to_fit
from deployment.mani_skill_env import ManiSkillPushTEnv
from deployment.wm_adapter import DinoWMPushtAdapter

WINDOW_NAME = "DINO-WM vs GT (overlay)"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GT env vs open-loop WM rollout, overlaid")
    parser.add_argument("--ckpt", required=True,
                        help="path to checkpoints/ directory")
    parser.add_argument("--ckpt_suffix", default="_latest",
                        help="checkpoint suffix")
    parser.add_argument("--display_size", type=int, default=840,
                        help="max display dimension (default: 840)")
    parser.add_argument("--fps", type=int, default=20,
                        help="target frames per second (default: 20)")
    parser.add_argument("--blend", type=float, default=0.5,
                        help="initial GT weight in the blend (0=WM, 1=GT; "
                             "default 0.5)")

    # ── gamepad options (same as run_wm_only.py) ────────────────────────────
    gp = parser.add_argument_group("gamepad (PS4 / Xbox controller)")
    gp.add_argument("--gamepad", action="store_true",
                    help="Use a gamepad/joystick for teleoperation.")
    gp.add_argument("--gamepad-speed", type=float, default=60.0,
                    help="Pixel-space push magnitude per step at full stick "
                         "deflection (default: 60).")
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

    # ── ManiSkill environment options (same defaults as run_wm_only.py) ─────
    ms = parser.add_argument_group("ManiSkill environment")
    ms.add_argument("--env-id", default="PushT-XYExplore-v1",
                    help="Gymnasium env id (default: PushT-XYExplore-v1).")
    ms.add_argument("--zero-latency-root", default=None,
                    help="Path to zero_latency_interface repo.")
    ms.add_argument("--policy-checkpoint", default=None,
                    help="Path to a trained PPO policy checkpoint (.pt); "
                         "only its args are read for the wrapper config.")
    ms.add_argument("--control-mode", default="pd_ee_delta_pose",
                    choices=["pd_ee_delta_pose", "pd_ee_pose"],
                    help="EE controller (default: pd_ee_delta_pose).")
    ms.add_argument("--action-scale", type=float, default=0.1,
                    help="Maniskill action_scale (default: 0.1).")
    ms.add_argument("--step-size", type=float, default=0.02,
                    help="Metres per step at full deflection (default: 0.02).")
    ms.add_argument("--fixed-magnitude", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Wrapper unit-circle projection; default follows the "
                         "policy checkpoint if given, else off.")
    ms.add_argument("--level-ee", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Actively level the EE each step; default follows "
                         "the policy checkpoint if given, else off.")
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

    # ── gamepad setup ────────────────────────────────────────────────────────
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

    # ── world model first (its img_size sets the env render resolution) ─────
    wm = DinoWMPushtAdapter(ckpt_dir=args.ckpt,
                            ckpt_suffix=args.ckpt_suffix)

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
        render_size=wm.img_size,
    )
    env = ManiSkillPushTEnv(**env_kwargs)
    env.seed(0)

    # ── grounding helper ─────────────────────────────────────────────────────
    def ground() -> None:
        """Reset the env and ground the WM on the initial context."""
        obs, _ = env.reset()
        frames = [obs["visual"]]
        props = [obs["proprio"]]
        for _ in range(wm.num_hist * wm.frameskip):
            obs, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
            frames.append(obs["visual"])
            props.append(obs["proprio"])
        aligned_vis = frames[::wm.frameskip]
        aligned_prop = props[::wm.frameskip]
        wm.reset(aligned_vis[-wm.num_hist:], aligned_prop[-wm.num_hist:])

    ground()
    if gamepad is not None and gamepad.connected:
        gamepad.reset_button_state()

    blend = float(np.clip(args.blend, 0.0, 1.0))
    show_wm = True
    timestep = 0
    print(f"[OVERLAY] ready — '[' / ']' blend, 'o' toggle WM, 'r' reset, "
          f"ESC quit  (GT weight={blend:.1f})", flush=True)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.display_size,
                     int(args.display_size * 0.6))

    while True:
        t0 = time.perf_counter()

        # ── input ────────────────────────────────────────────────────────────
        if gamepad is not None and gamepad.connected:
            gamepad.pump()
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == ord("r"):
            ground()
            timestep = 0
        elif key == ord("o"):
            show_wm = not show_wm
            print(f"[OVERLAY] WM overlay: {'on' if show_wm else 'off'}")
        elif key == ord("["):
            blend = max(0.0, blend - 0.1)
        elif key == ord("]"):
            blend = min(1.0, blend + 0.1)

        gp_dx, gp_dy, gp_buttons = (0.0, 0.0, {})
        if gamepad is not None and gamepad.connected:
            gp_dx, gp_dy, gp_buttons = gamepad.read()
            if gp_buttons.get("quit", False):
                break

        if abs(gp_dx) > 0.5 or abs(gp_dy) > 0.5:
            action = np.array([gp_dx, gp_dy], dtype=np.float32)
        elif key in KEY_ACTIONS:
            action = np.array(KEY_ACTIONS[key], dtype=np.float32)
        else:
            action = np.array([0, 0], dtype=np.float32)

        # ── env step + WM open-loop step ─────────────────────────────────────
        obs, _, done, info = env.step(action)
        gt = obs["visual"]
        eff = info.get("effective_action",
                       np.array(action, dtype=np.float32) / 100.0)
        wm_frame = wm.predict(np.asarray(eff, dtype=np.float32))

        # ── composite display ────────────────────────────────────────────────
        if show_wm:
            display = cv2.addWeighted(gt, blend, wm_frame, 1.0 - blend, 0.0)
        else:
            display = gt
        display = _resize_to_fit(display, args.display_size,
                                 int(args.display_size * 0.6))

        h = display.shape[0]
        cv2.putText(display, f"t={timestep}", (4, h - 48), FONT, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(
            display,
            f"GT weight={blend:.1f} ([/])  WM={'on' if show_wm else 'off'} (o)",
            (4, h - 30), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(display, "WASD/left-stick=move  r=reset  ESC=quit",
                    (4, h - 12), FONT, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, display)

        # ── pacing + episode end ─────────────────────────────────────────────
        dt = time.perf_counter() - t0
        target = 1.0 / args.fps
        if dt < target:
            time.sleep(target - dt)
        timestep += 1
        if done:
            print("[OVERLAY] episode finished — resetting...")
            ground()
            timestep = 0

    if gamepad is not None and gamepad.connected:
        gamepad.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
