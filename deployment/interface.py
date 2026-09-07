"""Interactive PushT zero-latency interface.

Wires together the ManiSkill 3D PushT environment, the DINO-WM world model
adapter, a configurable delay simulator, and an OpenCV-based display.  Two
modes are supported:

``wm_only``
    The world model runs open-loop from an initial context.  User actions
    drive the WM autoregressively; the environment runs silently alongside
    for comparison.  Display: [ WM prediction | env GT (small) ].

``zli``
    Simulated network delay on the GT frame stream.  The WM compensates
    by predicting ahead from the last arrived (delayed) GT frame to the
    current timestep.  Display: [ WM prediction | delayed GT ].

Input
    Keyboard (WASD) is always available as a fallback.  When a gamepad is
    connected via :class:`deployment.gamepad.GamepadInput`, the left analog
    stick drives the PushT agent — stick direction controls push direction,
    and the stick maps to a fixed-speed move (like the RL policy).  LB/RB
    act as precision/boost modifiers.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from deployment.delay import ConstantDelay, DelayModel, NoDelay
from deployment.gamepad import GamepadInput
from deployment.wm_adapter import DinoWMPushtAdapter

# ── constants ────────────────────────────────────────────────────────────────
WINDOW_NAME = "DINO-WM Zero-Latency Interface"
GT_INSET_SCALE = 0.3        # GT inset size relative to main display
FONT = cv2.FONT_HERSHEY_SIMPLEX
TICK_RATE = 20              # target loop Hz

# Keyboard mappings — produce pixel-space (dx, dy), normalised by max_input
# inside the ManiSkill adapter.  60 = full-speed step.  Directions are
# on-screen, mapped to world axes for PushT's camera (eye=+x, looking toward
# -x): screen up = world -x, right = +y (same as gamepad_teleop_pusht.py).
KEY_ACTIONS: dict[int, tuple[float, float]] = {
    ord("w"): (-60, 0),     # up (away from camera)
    ord("s"): (60, 0),      # down (toward camera)
    ord("a"): (0, -60),     # left
    ord("d"): (0, 60),      # right
    ord("q"): (-60, -60),   # up-left
    ord("e"): (-60, 60),    # up-right
    ord("z"): (60, -60),    # down-left
    ord("c"): (60, 60),     # down-right
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _put_text(img: np.ndarray, text: str, pos: tuple[int, int],
              color=(255, 255, 255), scale=0.5) -> None:
    cv2.putText(img, text, pos, FONT, scale, color, 1, cv2.LINE_AA)


def _resize_to_fit(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_NEAREST)
    return img


def zli_tick(
    wm: DinoWMPushtAdapter,
    delay_received: tuple | None,
    action_history: list,
    get_timestep,
    raw_action: np.ndarray,
    gt_frame: np.ndarray,
    grounding_thread: Optional[threading.Thread],
    last_grounded_ts: int,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[threading.Thread], int]:
    """One tick of delay+WM display update (the ``zli`` mode's core logic).

    Factored out of :meth:`PushTInterface.run` so it can be reused by other
    drivers (e.g. a study UI) without duplicating the async-grounding launch
    rule and ``get_timestep``-closure pattern.

    Parameters
    ----------
    wm:
        Already-grounded adapter.
    delay_received:
        The ``(frame, proprio, gt_ts)`` tuple from ``delay.receive(...)``,
        or ``None`` if nothing arrived this tick.
    action_history:
        The caller's growing raw-action list (indexed by absolute raw
        timestep — see ``_collect_context``'s invariant).
    get_timestep:
        Callable returning the current absolute raw timestep; passed through
        to the async worker so it can chase the advancing present.
    raw_action:
        This tick's action, fed to ``wm.predict()``.
    gt_frame:
        Current live GT frame — the fallback display before the WM is
        grounded.
    grounding_thread, last_grounded_ts:
        The caller's async-grounding bookkeeping from the previous tick.

    Returns
    -------
    (display_frame, delayed_frame, grounding_thread, last_grounded_ts)
        ``delayed_frame`` is ``None`` unless ``delay_received`` was not
        ``None`` this tick (caller should keep its own "last delayed frame"
        only when this isn't ``None``, matching the pre-extraction
        behaviour).
    """
    # ALWAYS advance the WM one raw step per tick (single-step
    # autoregression).  While an async grounding runs this advances the
    # previous chain, keeping the display responsive; on publish it jumps
    # to the grounded rollout.
    if wm.is_initialised:
        display_frame = wm.predict(raw_action)
    else:
        display_frame = gt_frame

    delayed_frame = None
    if delay_received is not None:
        delayed_frame, delayed_proprio, gt_ts = delay_received
        # Track the delayed stream's frameskip-aligned frames so the next
        # grounding's context keeps the training stride.
        wm.update_ground_truth(delayed_frame, delayed_proprio, gt_ts)
        # Launch rule: only when no worker is running AND the frame is
        # newer than the last grounding's frame.  No cancellation — the
        # worker runs to completion; the next newer frame launches the
        # next grounding.
        if (grounding_thread is None or not grounding_thread.is_alive()) \
                and gt_ts > last_grounded_ts:
            last_grounded_ts = gt_ts
            grounding_thread = threading.Thread(
                target=wm.ground_and_predict_async,
                args=(delayed_frame, delayed_proprio, gt_ts,
                      action_history, get_timestep),
                daemon=True, name="wm-grounding",
            )
            grounding_thread.start()

    return display_frame, delayed_frame, grounding_thread, last_grounded_ts


# ── main interface class ─────────────────────────────────────────────────────

class PushTInterface:
    """Interactive PushT zero-latency interface (ManiSkill 3D backend).

    Parameters
    ----------
    mode:
        ``"wm_only"`` — world-model-only prediction (no delay compensation).
        ``"zli"``    — zero-latency with simulated network delay.
        ``"passthrough"`` — direct env display (no WM, no delay) for baseline.
    wm_ckpt_dir:
        Path to the DINO-WM ``checkpoints/`` directory.
    wm_ckpt_suffix:
        Suffix for the checkpoint file.
    delay_steps:
        Number of raw timesteps of delay (ZLI mode only).
    display_size:
        Max display dimension in pixels.
    fps:
        Target display frame rate.
    gamepad:
        Optional :class:`GamepadInput` instance.  When provided (and connected),
        the left analog stick drives the PushT agent.  Keyboard WASD remains
        available as a fallback.
    env_kwargs:
        Forwarded to :class:`deployment.mani_skill_env.ManiSkillPushTEnv`.
    """

    def __init__(
        self,
        mode: str = "zli",
        wm_ckpt_dir: str = "",
        wm_ckpt_suffix: str = "_latest",
        delay_steps: int = 10,
        display_size: int = 840,
        fps: int = 20,
        gamepad: Optional[GamepadInput] = None,
        env_kwargs: dict | None = None,
    ) -> None:
        self._mode = mode
        self._display_size = display_size
        self._fps = fps
        self._gamepad = gamepad

        _env_kw = dict(env_kwargs) if env_kwargs else {}

        # --- world model ------------------------------------------------------
        # Created first so the env's render resolution can match the WM's
        # training image size (e.g. 448 for the high-res checkpoint).
        self._wm: DinoWMPushtAdapter | None = None
        if mode in ("wm_only", "zli"):
            self._wm = DinoWMPushtAdapter(
                ckpt_dir=wm_ckpt_dir,
                ckpt_suffix=wm_ckpt_suffix,
            )

        # --- environment (ManiSkill 3D) --------------------------------------
        from deployment.mani_skill_env import ManiSkillPushTEnv

        if "render_size" not in _env_kw and self._wm is not None:
            _env_kw["render_size"] = self._wm.img_size
        self._env = ManiSkillPushTEnv(**_env_kw)
        self._env.seed(0)

        # --- delay simulator --------------------------------------------------
        if mode == "zli":
            self._delay: DelayModel = ConstantDelay(delay_steps)
        else:
            self._delay = NoDelay()

        # --- state ------------------------------------------------------------
        self._obs: dict | None = None       # current env observation
        self._state: np.ndarray | None = None  # current env full state
        self._timestep: int = 0
        self._action_history: list[np.ndarray] = []  # all raw actions

        # For WM grounding: track frameskip-aligned GT frames.
        self._gt_aligned_frames: list[np.ndarray] = []
        self._gt_aligned_proprios: list[np.ndarray] = []
        self._frame_counter: int = 0  # raw frame counter (for alignment)

        # Display state.
        self._wm_frame: np.ndarray | None = None
        self._delayed_frame: np.ndarray | None = None
        self._running: bool = False

        # Async grounding (zli): worker thread + the gt_ts of the last
        # LAUNCHED grounding (main-thread-only).
        self._grounding_thread: Optional[threading.Thread] = None
        self._last_grounded_ts: int = -1

    # -- public API ------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive loop. Blocks until ESC is pressed."""
        self._running = True
        self._grounding_thread = None
        self._last_grounded_ts = -1

        # Reset environment.
        obs, state = self._env.reset()
        self._obs = obs
        self._state = state
        self._timestep = 0
        self._action_history.clear()
        self._gt_aligned_frames.clear()
        self._gt_aligned_proprios.clear()
        self._frame_counter = 0
        self._wm_frame = obs["visual"]
        self._delayed_frame = obs["visual"]

        # Ground WM with initial context frames.
        if self._wm is not None:
            self._collect_context()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self._display_size,
                         int(self._display_size * 0.6))

        # Sync gamepad button state so we don't get spurious edge triggers.
        if self._gamepad is not None and self._gamepad.connected:
            self._gamepad.reset_button_state()

        clock = deque(maxlen=30)

        while self._running:
            t0 = time.perf_counter()

            # --- pump gamepad events (if connected) --------------------------
            if self._gamepad is not None and self._gamepad.connected:
                self._gamepad.pump()

            # --- input --------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            quit_now = (key == 27)  # ESC

            # Read gamepad + keyboard; gamepad takes priority for movement.
            gp_dx, gp_dy, gp_buttons = (0.0, 0.0, {})
            if self._gamepad is not None and self._gamepad.connected:
                gp_dx, gp_dy, gp_buttons = self._gamepad.read()
                quit_now = quit_now or gp_buttons.get("quit", False)

            if quit_now:
                self._running = False
                break

            action = self._get_action(key, gp_dx, gp_dy, gp_buttons)

            # --- env step -----------------------------------------------------
            obs, reward, done, info = self._env.step(action)
            self._obs = obs
            self._state = info["state"]
            gt_frame = obs["visual"]
            gt_proprio = obs["proprio"]

            # --- track frameskip-aligned frames for WM context ---------------
            self._frame_counter += 1
            fs = self._wm.frameskip if self._wm else 5
            if self._frame_counter % fs == 0:
                self._gt_aligned_frames.append(gt_frame.copy())
                self._gt_aligned_proprios.append(gt_proprio.copy())
                cap = (self._wm.num_hist + 5) if self._wm else 10
                if len(self._gt_aligned_frames) > cap:
                    self._gt_aligned_frames = self._gt_aligned_frames[-cap:]
                    self._gt_aligned_proprios = self._gt_aligned_proprios[-cap:]

            # --- action tracking ----------------------------------------------
            # Store the effective action (after fix_magnitude + scaled_delta)
            # so the WM sees actions in the same format as training data.
            eff_action = info.get("effective_action",
                                   np.array(action, dtype=np.float32) / 100.0)
            raw_action = np.asarray(eff_action, dtype=np.float32)
            self._action_history.append(raw_action)

            # --- delay pipeline -----------------------------------------------
            self._delay.submit(gt_frame, gt_proprio, self._timestep)
            received = self._delay.receive(self._timestep)

            # --- world model update -------------------------------------------
            if self._mode == "passthrough":
                self._wm_frame = gt_frame
            elif self._wm is not None:
                if self._mode == "wm_only":
                    # Open-loop: the WM runs purely autoregressively from its
                    # initial context.  GT frames are never fed back into the
                    # WM (they only appear in the inset for comparison), so
                    # every step just advances the model one raw timestep.
                    if self._wm.is_initialised:
                        self._wm_frame = self._wm.predict(raw_action)
                    else:
                        self._wm_frame = gt_frame
                elif self._mode == "zli":
                    (self._wm_frame, new_delayed, self._grounding_thread,
                     self._last_grounded_ts) = zli_tick(
                        self._wm, received, self._action_history,
                        self._get_timestep, raw_action, gt_frame,
                        self._grounding_thread, self._last_grounded_ts,
                    )
                    if new_delayed is not None:
                        self._delayed_frame = new_delayed
                elif self._wm.is_initialised:
                    # No new GT — advance WM by one step.
                    self._wm_frame = self._wm.predict(raw_action)
                else:
                    self._wm_frame = gt_frame

            # --- render -------------------------------------------------------
            display = self._render_display()
            cv2.imshow(WINDOW_NAME, display)

            # --- timing -------------------------------------------------------
            dt = time.perf_counter() - t0
            clock.append(dt)
            self._timestep += 1

            # Pace the loop to the target fps (env is stepped once per
            # iteration) so per-second speeds are stable across machines.
            target = 1.0 / self._fps
            if dt < target:
                time.sleep(target - dt)

            # Reset if episode ended or gamepad button pressed.
            reset_requested = gp_buttons.get("reset", False) if gp_buttons else False
            next_ep = gp_buttons.get("next_episode", False) if gp_buttons else False
            if done or reset_requested or next_ep:
                self._do_reset()

        self._join_grounding_worker()
        if self._gamepad is not None and self._gamepad.connected:
            self._gamepad.close()
        cv2.destroyAllWindows()

    # -- internals -------------------------------------------------------------

    def _collect_context(self) -> None:
        """Collect enough frameskip-aligned frames to ground the WM."""
        assert self._wm is not None
        fs = self._wm.frameskip
        needed = self._wm.num_hist
        print(f"[IF] collecting {needed} context frames "
              f"(fs={fs}, {needed * fs} raw steps)...")

        # We already have the first frame (from reset).
        self._gt_aligned_frames.append(self._obs["visual"].copy())
        self._gt_aligned_proprios.append(self._obs["proprio"].copy())
        self._frame_counter = 0

        while len(self._gt_aligned_frames) < needed:
            # Step the environment with zero action to collect frames.
            obs, _, _, info = self._env.step(
                np.array([0, 0], dtype=np.float32))
            # Keep _action_history indexed by absolute raw timestep — the
            # async grounding worker looks up actions by self._timestep
            # value, so a gap here (steps counted but not recorded) would
            # desync every later index lookup by exactly this gap.
            eff_action = info.get(
                "effective_action", np.zeros(2, dtype=np.float32))
            self._action_history.append(
                np.asarray(eff_action, dtype=np.float32))
            self._frame_counter += 1
            if self._frame_counter % fs == 0:
                self._gt_aligned_frames.append(obs["visual"].copy())
                self._gt_aligned_proprios.append(obs["proprio"].copy())
            self._timestep += 1

        # Ground the WM.
        ctx_frames = self._gt_aligned_frames[-needed:]
        ctx_proprios = self._gt_aligned_proprios[-needed:]
        self._wm.reset(ctx_frames, ctx_proprios)
        self._wm_frame = ctx_frames[-1]
        print(f"[IF] WM grounded with {needed} context frames "
              f"(raw timestep={self._timestep})")

    def _get_action(
        self,
        key: int,
        gp_dx: float = 0.0,
        gp_dy: float = 0.0,
        gp_buttons: dict | None = None,
    ) -> np.ndarray:
        """Read input → (dx, dy) action in pixel space [0, 512].

        Gamepad input takes priority for movement; keyboard WASD is the
        fallback.  When the gamepad stick is idle (near zero) and a WASD
        key is pressed, keyboard control is used.
        """
        del gp_buttons  # button-actions handled directly in run()
        # Gamepad takes priority when the stick is actively deflected.
        gp_active = (abs(gp_dx) > 0.5 or abs(gp_dy) > 0.5)
        if gp_active:
            return np.array([gp_dx, gp_dy], dtype=np.float32)
        # Keyboard fallback.
        if key in KEY_ACTIONS:
            return np.array(KEY_ACTIONS[key], dtype=np.float32)
        return np.array([0, 0], dtype=np.float32)

    def _get_timestep(self) -> int:
        """GIL-atomic timestep read for the async grounding worker."""
        return self._timestep

    def _join_grounding_worker(self) -> None:
        """Wait for the async grounding worker to finish (no cancellation).

        Bounded: the worker completes its current rollout (~ceil(D/fs) model
        steps) and exits on its own.
        """
        if self._grounding_thread is not None:
            if self._grounding_thread.is_alive():
                self._grounding_thread.join()
            self._grounding_thread = None

    def _do_reset(self) -> None:
        """Reset environment and WM state."""
        print("[IF] episode finished — resetting...")
        # The worker holds references to _action_history (cleared below) and
        # reads _timestep (zeroed below) — join it before touching either.
        self._join_grounding_worker()
        self._last_grounded_ts = -1
        obs, state = self._env.reset()
        self._obs = obs
        self._state = state
        self._timestep = 0
        self._action_history.clear()
        self._gt_aligned_frames.clear()
        self._gt_aligned_proprios.clear()
        self._frame_counter = 0
        self._wm_frame = obs["visual"]
        self._delayed_frame = obs["visual"]
        self._delay.reset()
        if self._wm is not None:
            self._collect_context()

    def _render_display(self) -> np.ndarray:
        """Compose the display frame."""
        wm_frame = self._wm_frame
        if wm_frame is None:
            wm_frame = self._obs["visual"] if self._obs is not None else \
                np.zeros((224, 224, 3), dtype=np.uint8)

        main_h = int(self._display_size * 0.6)
        main = _resize_to_fit(wm_frame, self._display_size, main_h)

        # Build the display layout.
        # Main panel: WM prediction (or passthrough).
        # Right/bottom panel: GT / delayed GT for comparison.

        if self._mode == "passthrough":
            label = "Passthrough (no WM, no delay)"
        elif self._mode == "wm_only":
            label = "WM-only (open-loop prediction)"
        else:
            label = f"ZLI (WM-compensated, delay={self._delay._delay if hasattr(self._delay, '_delay') else '?'} steps)"

        # Create canvas.
        canvas_h = main.shape[0] + 80  # room for text overlay
        canvas_w = main.shape[1]
        if self._mode in ("wm_only", "zli"):
            # Add GT/delayed-GT inset on the right.
            gt_frame = self._delayed_frame if self._mode == "zli" else \
                (self._obs["visual"] if self._obs is not None else wm_frame)
            gt_inset = _resize_to_fit(
                gt_frame,
                int(self._display_size * GT_INSET_SCALE),
                int(main.shape[0] * GT_INSET_SCALE),
            )
            canvas_w = main.shape[1] + gt_inset.shape[1] + 4

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Place main panel.
        canvas[:main.shape[0], :main.shape[1]] = main

        # Place GT inset.
        if self._mode in ("wm_only", "zli"):
            gt_frame = self._delayed_frame if self._mode == "zli" else \
                (self._obs["visual"] if self._obs is not None else wm_frame)
            gt_inset = _resize_to_fit(
                gt_frame,
                int(self._display_size * GT_INSET_SCALE),
                int(main.shape[0] * GT_INSET_SCALE),
            )
            x_off = main.shape[1] + 4
            canvas[:gt_inset.shape[0], x_off:x_off + gt_inset.shape[1]] = gt_inset
            gt_label = "delayed GT" if self._mode == "zli" else "env GT"
            _put_text(canvas, gt_label, (x_off + 2, 14), (200, 200, 200))

            if self._mode == "zli":
                # Third panel: the undelayed (live) GT, below the delayed one.
                live_frame = self._obs["visual"] if self._obs is not None \
                    else wm_frame
                live_inset = _resize_to_fit(
                    live_frame,
                    int(self._display_size * GT_INSET_SCALE),
                    int(main.shape[0] * GT_INSET_SCALE),
                )
                y_off = gt_inset.shape[0] + 2
                canvas[y_off:y_off + live_inset.shape[0],
                       x_off:x_off + live_inset.shape[1]] = live_inset
                _put_text(canvas, "live GT", (x_off + 2, y_off + 14),
                          (200, 200, 200))

        # Text overlay at bottom.
        y_base = main.shape[0] + 15
        _put_text(canvas, label, (4, y_base), (100, 255, 100))
        _put_text(canvas, f"t={self._timestep}", (4, y_base + 18), (200, 200, 200))

        if self._gamepad is not None and self._gamepad.connected:
            hint = "L-stick=move  LB=precise  RB=boost  A=reset  B=quit  ESC=quit"
        else:
            hint = "WASD=move  ESC=quit  R=reset"
        _put_text(canvas, hint, (4, y_base + 36), (150, 150, 150))

        return canvas
