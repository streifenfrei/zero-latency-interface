#!/usr/bin/env python
"""User-study UI: baseline / delayed / delayed+ZLI PushT teleoperation.

A single long-running kiosk-style app for running a two-group user study:

  Home screen
      Two buttons start a new trial: "Delay only" (group 1) or
      "Delay + WM / ZLI" (group 2).  Whichever is clicked decides the
      condition for THIS trial's phase 2 — there is no separate
      participant-ID entry; each completed trial gets a sequential
      ``trial_id`` (continuing from the results CSV, if one already exists).
  Phase 1 — practice (always no delay, no WM)
      Runs for a fixed ``--baseline-minutes``.  The episode auto-resets on
      success so the participant can practice more than once; the phase
      simply ends when the timer expires.
  Phase 2 — the measured attempt, in the trial's assigned condition
      A single uninterrupted attempt: ends on task success or
      ``--task-timeout-minutes``, whichever comes first.  No auto-reset.
  Results screen
      Shows completion time / success / timeout, with a Save button that
      appends one row to the results CSV and returns to the Home screen.

A countdown precedes both phase 1 and phase 2.  ESC at any screen other
than Home aborts the current trial back to Home without saving.  Ctrl+C in
the terminal quits the whole process — the same process should stay open
for the entire study session (env + world model are loaded once, up front,
and reused for every trial/participant).

The task view shown to the participant is always a single clean panel (no
debug insets, no live timer) — deliberately different from run_zli.py's
developer-facing multi-panel display.

Usage:
    python deployment/run_study.py --ckpt outputs/.../checkpoints \\
        --policy-checkpoint ../zero_latency_interface/checkpoints/policy.pt \\
        --gamepad --delay-steps 20 --baseline-minutes 2 --task-timeout-minutes 5
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import cv2
import numpy as np

from deployment.delay import ConstantDelay
from deployment.gamepad import GamepadInput
from deployment.interface import FONT, KEY_ACTIONS, _resize_to_fit, zli_tick
from deployment.mani_skill_env import ManiSkillPushTEnv
from deployment.wm_adapter import DinoWMPushtAdapter

WINDOW_NAME = "Zero-Latency Interface - User Study"

FIELDNAMES = [
    "trial_id", "timestamp_iso", "condition", "delay_steps",
    "baseline_minutes", "task_timeout_minutes",
    "goal_tolerance",
    "completion_time_s", "success", "timed_out",
]


# ── CSV results ──────────────────────────────────────────────────────────────

def load_next_trial_id(csv_path: Path) -> int:
    """Continue numbering from an existing results CSV, if present."""
    if not csv_path.exists():
        return 1
    max_id = 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                max_id = max(max_id, int(row["trial_id"]))
            except (KeyError, ValueError):
                continue
    return max_id + 1


def append_csv_row(csv_path: Path, row: dict) -> None:
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ── tiny cv2 button widget ───────────────────────────────────────────────────

@dataclass
class Button:
    id: str
    rect: tuple[int, int, int, int]  # x1, y1, x2, y2
    label: str
    key: Optional[int] = None  # ord() of a keyboard shortcut


def draw_buttons(canvas: np.ndarray, buttons: list[Button]) -> None:
    for b in buttons:
        x1, y1, x2, y2 = b.rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 70, 70), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (200, 200, 200), 1)
        (tw, th), _ = cv2.getTextSize(b.label, FONT, 0.55, 1)
        tx = x1 + max((x2 - x1 - tw) // 2, 4)
        ty = y1 + (y2 - y1 + th) // 2
        cv2.putText(canvas, b.label, (tx, ty), FONT, 0.55, (255, 255, 255),
                   1, cv2.LINE_AA)


@dataclass
class ClickState:
    buttons: list[Button] = field(default_factory=list)
    clicked_id: Optional[str] = None


def make_mouse_callback(state: ClickState):
    def _on_mouse(event, x, y, flags, param):
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for b in state.buttons:
            x1, y1, x2, y2 = b.rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                state.clicked_id = b.id
                return
    return _on_mouse


# ── shared helpers ───────────────────────────────────────────────────────────

def _put(canvas, text, pos, color=(255, 255, 255), scale=0.6, thickness=1):
    cv2.putText(canvas, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)


def make_frame_canvas(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Center a (possibly smaller, aspect-preserved) frame on a WxH canvas."""
    resized = _resize_to_fit(img, w, h)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y_off = (h - resized.shape[0]) // 2
    x_off = (w - resized.shape[1]) // 2
    canvas[y_off:y_off + resized.shape[0],
           x_off:x_off + resized.shape[1]] = resized
    return canvas


def poll_input(gamepad: Optional[GamepadInput]):
    """One input-poll tick. Returns (key, gp_dx, gp_dy, gp_buttons, quit_now)."""
    if gamepad is not None and gamepad.connected:
        gamepad.pump()
    key = cv2.waitKey(1) & 0xFF
    quit_now = (key == 27)  # ESC
    gp_dx, gp_dy, gp_buttons = 0.0, 0.0, {}
    if gamepad is not None and gamepad.connected:
        gp_dx, gp_dy, gp_buttons = gamepad.read()
        quit_now = quit_now or gp_buttons.get("quit", False)
    return key, gp_dx, gp_dy, gp_buttons, quit_now


def get_action(key: int, gp_dx: float, gp_dy: float) -> np.ndarray:
    if abs(gp_dx) > 0.5 or abs(gp_dy) > 0.5:
        return np.array([gp_dx, gp_dy], dtype=np.float32)
    if key in KEY_ACTIONS:
        return np.array(KEY_ACTIONS[key], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)


def is_success(info: dict) -> bool:
    s = info.get("success", False)
    if hasattr(s, "cpu"):  # torch tensor — may still live on the GPU
        return bool(s.detach().cpu().any())
    return bool(np.asarray(s).any())


def pace(t0: float, fps: int) -> None:
    dt = time.perf_counter() - t0
    target = 1.0 / fps
    if dt < target:
        time.sleep(target - dt)


def _join(thread) -> None:
    if thread is not None and thread.is_alive():
        thread.join()


# ── screens ──────────────────────────────────────────────────────────────────

def show_home_screen(state: ClickState, gamepad, W: int, H: int,
                     n_trials: int) -> str:
    """Blocks until a condition button is chosen. Returns "delayed" or
    "delayed_zli"."""
    buttons = [
        Button("delayed", (W // 2 - 280, H // 2 - 35, W // 2 - 20, H // 2 + 35),
              "[1] Start Trial: Delay Only"),
        Button("delayed_zli", (W // 2 + 20, H // 2 - 35, W // 2 + 280, H // 2 + 35),
              "[2] Start Trial: Delay + WM"),
    ]
    buttons[0].key = ord("1")
    buttons[1].key = ord("2")
    state.buttons = buttons
    state.clicked_id = None

    while True:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        _put(canvas, "Zero-Latency Interface -- User Study", (20, 40),
             (255, 255, 255), 0.8)
        _put(canvas, f"{n_trials} trial(s) recorded so far", (20, 70),
             (150, 150, 150), 0.5)
        draw_buttons(canvas, buttons)
        _put(canvas, "Click a button or press the bracketed key.",
             (20, H - 20), (150, 150, 150), 0.5)
        cv2.imshow(WINDOW_NAME, canvas)

        key, _, _, _, _ = poll_input(gamepad)
        if state.clicked_id is not None:
            return state.clicked_id
        for b in buttons:
            if b.key is not None and key == b.key:
                return b.id


def run_countdown(gamepad, W: int, H: int, seconds: int, label: str) -> bool:
    """Returns False if aborted (ESC/gamepad-quit), True otherwise."""
    for n in range(seconds, 0, -1):
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < 1.0:
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            _put(canvas, label, (W // 2 - 220, H // 2 - 100),
                 (200, 200, 200), 0.6)
            text = str(n)
            (tw, th), _ = cv2.getTextSize(text, FONT, 4.0, 6)
            cv2.putText(canvas, text, (W // 2 - tw // 2, H // 2 + th // 2),
                       FONT, 4.0, (255, 255, 255), 6, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, canvas)
            _, _, _, _, quit_now = poll_input(gamepad)
            if quit_now:
                return False
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < 0.5:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        (tw, th), _ = cv2.getTextSize("GO", FONT, 3.0, 6)
        cv2.putText(canvas, "GO", (W // 2 - tw // 2, H // 2 + th // 2),
                   FONT, 3.0, (100, 255, 100), 6, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, canvas)
        _, _, _, _, quit_now = poll_input(gamepad)
        if quit_now:
            return False
    return True


def run_baseline_phase(env, gamepad, W: int, H: int, minutes: float,
                       fps: int) -> bool:
    """No delay, no WM.  Auto-resets on success.  Ends on timer expiry.

    Returns False if aborted.
    """
    env.reset()
    duration = minutes * 60.0
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < duration:
        t0 = time.perf_counter()
        key, gp_dx, gp_dy, _, quit_now = poll_input(gamepad)
        if quit_now:
            return False
        action = get_action(key, gp_dx, gp_dy)
        obs, _, done, info = env.step(action)

        canvas = make_frame_canvas(obs["visual"], W, H)
        _put(canvas, "Practice -- get familiar with the task (no delay)",
             (10, 24), (150, 150, 150), 0.5)
        cv2.imshow(WINDOW_NAME, canvas)

        if is_success(info) or done:
            env.reset()
        pace(t0, fps)
    return True


def show_phase2_prompt(state: ClickState, gamepad, W: int, H: int) -> bool:
    """Blocks on a single Continue button. Returns False if aborted."""
    buttons = [Button("continue", (W // 2 - 160, H // 2 - 35,
                                   W // 2 + 160, H // 2 + 35),
                      "[SPACE] Continue to Phase 2", key=32)]
    state.buttons = buttons
    state.clicked_id = None
    while True:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        _put(canvas, "Practice complete.", (W // 2 - 130, H // 2 - 90),
             (255, 255, 255), 0.7)
        draw_buttons(canvas, buttons)
        cv2.imshow(WINDOW_NAME, canvas)

        key, _, _, _, quit_now = poll_input(gamepad)
        if quit_now:
            return False
        if state.clicked_id == "continue" or key == 32:
            return True


def run_phase2_task_delayed(env, gamepad, W: int, H: int, delay_steps: int,
                            timeout_minutes: float, fps: int) -> Optional[dict]:
    """Delay-only condition (group 1): single attempt, no WM.

    Returns a result dict, or None if aborted.
    """
    obs, _ = env.reset()
    delay = ConstantDelay(delay_steps)
    delayed_frame = obs["visual"]
    timestep = 0
    timeout = timeout_minutes * 60.0
    t_start = time.perf_counter()

    while True:
        t0 = time.perf_counter()
        key, gp_dx, gp_dy, _, quit_now = poll_input(gamepad)
        if quit_now:
            return None
        action = get_action(key, gp_dx, gp_dy)
        obs, _, _, info = env.step(action)

        delay.submit(obs["visual"], obs["proprio"], timestep)
        received = delay.receive(timestep)
        if received is not None:
            delayed_frame = received[0]

        cv2.imshow(WINDOW_NAME, make_frame_canvas(delayed_frame, W, H))

        elapsed = time.perf_counter() - t_start
        if is_success(info):
            return {"completion_time_s": elapsed, "success": True,
                   "timed_out": False}
        if elapsed >= timeout:
            return {"completion_time_s": elapsed, "success": False,
                   "timed_out": True}
        timestep += 1
        pace(t0, fps)


def run_phase2_task_zli(env, wm: DinoWMPushtAdapter, gamepad, W: int, H: int,
                        delay_steps: int, timeout_minutes: float,
                        fps: int) -> Optional[dict]:
    """Delay+WM condition (group 2): single attempt, WM-compensated view.

    Returns a result dict, or None if aborted.  On every exit path the
    async grounding worker is joined before returning — ``wm`` is a
    session-long shared object reused by later trials, and a stale worker
    from THIS trial must not still be running (and touching wm state) when
    the next trial calls ``wm.reset(...)``.
    """
    obs, _ = env.reset()
    fs, needed = wm.frameskip, wm.num_hist

    # Prime the WM's context exactly like PushTInterface._collect_context:
    # step with zero action, recording every raw step's effective action
    # (action_history must stay indexed by absolute raw timestep, or the
    # grounding worker's lookups desync — see interface.py's _collect_context).
    gt_frames = [obs["visual"].copy()]
    gt_proprios = [obs["proprio"].copy()]
    action_history: list = []
    frame_counter = 0
    timestep = 0
    while len(gt_frames) < needed:
        key, gp_dx, gp_dy, _, quit_now = poll_input(gamepad)
        if quit_now:
            return None
        obs, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        eff = np.asarray(info.get("effective_action",
                                  np.zeros(2, dtype=np.float32)),
                         dtype=np.float32)
        action_history.append(eff)
        frame_counter += 1
        if frame_counter % fs == 0:
            gt_frames.append(obs["visual"].copy())
            gt_proprios.append(obs["proprio"].copy())
        timestep += 1
        canvas = make_frame_canvas(obs["visual"], W, H)
        _put(canvas, "Preparing...", (10, 24), (150, 150, 150), 0.5)
        cv2.imshow(WINDOW_NAME, canvas)

    wm.reset(gt_frames[-needed:], gt_proprios[-needed:])
    display_frame = gt_frames[-1]
    delay = ConstantDelay(delay_steps)
    grounding_thread = None
    last_grounded_ts = -1
    get_timestep: Callable[[], int] = lambda: timestep
    # Debug toggle: show/hide the live GT simulation.  Hidden by default so
    # the participant only ever sees the WM-compensated view.
    show_gt = False
    prev_key = -1

    timeout = timeout_minutes * 60.0
    t_start = time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            key, gp_dx, gp_dy, _, quit_now = poll_input(gamepad)
            if quit_now:
                return None
            if key == ord("g") and prev_key != ord("g"):
                show_gt = not show_gt
                print(f"[STUDY] debug: live GT overlay "
                      f"{'shown' if show_gt else 'hidden'}")
            prev_key = key
            action = get_action(key, gp_dx, gp_dy)
            obs, _, _, info = env.step(action)
            gt_frame = obs["visual"]
            eff_action = np.asarray(
                info.get("effective_action", np.zeros(2, dtype=np.float32)),
                dtype=np.float32)
            action_history.append(eff_action)

            delay.submit(gt_frame, obs["proprio"], timestep)
            received = delay.receive(timestep)
            display_frame, _, grounding_thread, last_grounded_ts = zli_tick(
                wm, received, action_history, get_timestep, eff_action,
                gt_frame, grounding_thread, last_grounded_ts,
            )

            canvas = make_frame_canvas(display_frame, W, H)
            if show_gt:
                # Small live-GT inset (bottom-right) so the experimenter can
                # judge the compensation against the real state.
                inset_w = int(W * 0.28)
                inset_h = int(inset_w * display_frame.shape[0]
                              / max(display_frame.shape[1], 1))
                inset = _resize_to_fit(gt_frame, inset_w, inset_h)
                y0 = canvas.shape[0] - inset.shape[0]
                x0 = canvas.shape[1] - inset.shape[1]
                canvas[y0:y0 + inset.shape[0],
                       x0:x0 + inset.shape[1]] = inset
                _put(canvas, "live GT", (x0 + 2, y0 + 14), (200, 200, 200))
            cv2.imshow(WINDOW_NAME, canvas)

            elapsed = time.perf_counter() - t_start
            if is_success(info):
                return {"completion_time_s": elapsed, "success": True,
                       "timed_out": False}
            if elapsed >= timeout:
                return {"completion_time_s": elapsed, "success": False,
                       "timed_out": True}
            timestep += 1
            pace(t0, fps)
    finally:
        _join(grounding_thread)


def show_results_screen(state: ClickState, gamepad, W: int, H: int,
                        condition: str, result: dict) -> bool:
    """Blocks on a Save button. Returns True if saved, False if discarded."""
    buttons = [Button("save", (W // 2 - 130, H - 100, W // 2 + 130, H - 50),
                      "[S] Save & Return Home", key=ord("s"))]
    state.buttons = buttons
    state.clicked_id = None

    cond_label = "Delay only" if condition == "delayed" else "Delay + WM"
    if result["success"]:
        status, color = "SUCCESS", (100, 255, 100)
    elif result["timed_out"]:
        status, color = "TIMED OUT", (100, 100, 255)
    else:
        status, color = "INCOMPLETE", (100, 100, 255)

    while True:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        _put(canvas, "Trial complete", (20, 40), (255, 255, 255), 0.8)
        _put(canvas, f"Condition: {cond_label}", (20, 80), (200, 200, 200), 0.6)
        _put(canvas, f"Result: {status}", (20, 110), color, 0.6)
        _put(canvas, f"Completion time: {result['completion_time_s']:.2f} s",
             (20, 140), (255, 255, 255), 0.6)
        draw_buttons(canvas, buttons)
        cv2.imshow(WINDOW_NAME, canvas)

        key, _, _, _, quit_now = poll_input(gamepad)
        if state.clicked_id == "save" or key == ord("s"):
            return True
        if quit_now:
            return False


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="User-study UI: baseline / delayed / delayed+ZLI PushT")
    parser.add_argument("--ckpt", required=True,
                        help="path to checkpoints/ directory (world model)")
    parser.add_argument("--ckpt_suffix", default="_latest")
    parser.add_argument("--display_size", type=int, default=840)
    parser.add_argument("--fps", type=int, default=20)

    study = parser.add_argument_group("study")
    study.add_argument("--results-dir", default="results",
                       help="Directory holding study_results.csv "
                            "(default: results/).")
    study.add_argument("--delay-steps", type=int, default=20,
                       help="Simulated delay for BOTH conditions "
                            "(default: 20).")
    study.add_argument("--baseline-minutes", type=float, default=2.0,
                       help="Fixed practice duration (default: 2.0).")
    study.add_argument("--task-timeout-minutes", type=float, default=5.0,
                       help="Phase-2 timeout if not completed (default: 5.0).")
    study.add_argument("--goal-tolerance", type=float, default=0.70,
                       help="Required overlap fraction (0..1) between the "
                            "pushed T-block and the goal for success.  "
                            "LOWER = more tolerant (default: 0.90, the "
                            "stock PushT threshold).")
    study.add_argument("--countdown-seconds", type=int, default=3)

    gp = parser.add_argument_group("gamepad (PS4 / Xbox controller)")
    gp.add_argument("--gamepad", action="store_true")
    gp.add_argument("--gamepad-speed", type=float, default=60.0)
    gp.add_argument("--gamepad-deadzone", type=float, default=0.15)
    gp.add_argument("--gamepad-no-fixed-magnitude", action="store_true")
    gp.add_argument("--gamepad-precision-scale", type=float, default=0.3)
    gp.add_argument("--gamepad-boost-scale", type=float, default=2.0)
    gp.add_argument("--gamepad-invert-x", action="store_true")
    gp.add_argument("--gamepad-invert-y", action="store_true")
    gp.add_argument("--gamepad-swap-xy", action="store_true")
    gp.add_argument("--gamepad-debug", action="store_true")

    ms = parser.add_argument_group("ManiSkill environment")
    ms.add_argument("--env-id", default="PushT-XYExplore-v1")
    ms.add_argument("--zero-latency-root", default=None)
    ms.add_argument("--policy-checkpoint", default=None)
    ms.add_argument("--control-mode", default="pd_ee_delta_pose",
                    choices=["pd_ee_delta_pose", "pd_ee_pose"])
    ms.add_argument("--action-scale", type=float, default=0.1)
    ms.add_argument("--step-size", type=float, default=0.02)
    ms.add_argument("--fixed-magnitude", action=argparse.BooleanOptionalAction,
                    default=None)
    ms.add_argument("--level-ee", action=argparse.BooleanOptionalAction,
                    default=None)
    ms.add_argument("--push-height", type=float, default=0.015)
    ms.add_argument("--sim-backend", default="physx_cuda",
                    choices=["physx_cuda", "physx_cpu"])
    ms.add_argument("--max-episode-steps", type=int, default=100_000)
    ms.add_argument("--no-velocity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    # World model first (loaded once for the whole session; either
    # condition may be picked per-trial) so the env's render resolution
    # can match its training image size.
    wm = DinoWMPushtAdapter(ckpt_dir=args.ckpt, ckpt_suffix=args.ckpt_suffix)

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
        goal_tolerance=args.goal_tolerance,
        render_size=wm.img_size,
    )
    env = ManiSkillPushTEnv(**env_kwargs)
    env.seed(0)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "study_results.csv"
    next_trial_id = load_next_trial_id(csv_path)
    print(f"[STUDY] results file: {csv_path}  (next trial_id={next_trial_id})")

    W, H = args.display_size, int(args.display_size * 0.6)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, W, H)
    # The Qt highgui backend doesn't create the window's actual handle until
    # the first imshow() — setMouseCallback() before that fails with a NULL
    # window pointer.  A few imshow+waitKey pumps give its event loop room
    # to actually realize the window before we look it up by name.
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    for _ in range(5):
        cv2.imshow(WINDOW_NAME, blank)
        cv2.waitKey(30)
    state = ClickState()
    try:
        cv2.setMouseCallback(WINDOW_NAME, make_mouse_callback(state))
    except cv2.error as exc:
        # Every button also has a keyboard shortcut, so mouse support is a
        # nice-to-have — don't let an OpenCV/Qt quirk crash the whole
        # session over it.
        print(f"[STUDY] WARNING: mouse click support unavailable ({exc}); "
             f"use the bracketed keyboard shortcuts instead.")

    try:
        while True:
            n_trials = next_trial_id - 1
            condition = show_home_screen(state, gamepad, W, H, n_trials)

            if not run_countdown(gamepad, W, H, args.countdown_seconds,
                                 "Practice starting..."):
                continue
            if not run_baseline_phase(env, gamepad, W, H,
                                      args.baseline_minutes, args.fps):
                continue
            if not show_phase2_prompt(state, gamepad, W, H):
                continue
            if not run_countdown(gamepad, W, H, args.countdown_seconds,
                                 "Timed task starting..."):
                continue

            if condition == "delayed":
                result = run_phase2_task_delayed(
                    env, gamepad, W, H, args.delay_steps,
                    args.task_timeout_minutes, args.fps)
            else:
                result = run_phase2_task_zli(
                    env, wm, gamepad, W, H, args.delay_steps,
                    args.task_timeout_minutes, args.fps)
            if result is None:
                continue

            if show_results_screen(state, gamepad, W, H, condition, result):
                append_csv_row(csv_path, {
                    "trial_id": next_trial_id,
                    "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "condition": condition,
                    "delay_steps": args.delay_steps,
                    "baseline_minutes": args.baseline_minutes,
                    "task_timeout_minutes": args.task_timeout_minutes,
                    "goal_tolerance": args.goal_tolerance,
                    "completion_time_s": f"{result['completion_time_s']:.3f}",
                    "success": result["success"],
                    "timed_out": result["timed_out"],
                })
                print(f"[STUDY] saved trial_id={next_trial_id} "
                     f"condition={condition} "
                     f"completion_time_s={result['completion_time_s']:.2f} "
                     f"success={result['success']}")
                next_trial_id += 1
    finally:
        if gamepad is not None and gamepad.connected:
            gamepad.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
