"""Interactive PushT zero-latency interface.

Wires together the PushT environment, the DINO-WM world model adapter,
a configurable delay simulator, and an OpenCV-based display.  Two modes
are supported:

``wm_only``
    The world model runs open-loop from an initial context.  User actions
    drive the WM autoregressively; the environment runs silently alongside
    for comparison.  Display: [ WM prediction | env GT (small) ].

``zli``
    Simulated network delay on the GT frame stream.  The WM compensates
    by predicting ahead from the last arrived (delayed) GT frame to the
    current timestep.  Display: [ WM prediction | delayed GT ].
"""

from __future__ import annotations

import importlib.util
import os
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

# Import PushTEnv directly — bypass env/__init__.py which triggers MuJoCo/d4rl.
_push_t_path = os.path.join(
    os.path.dirname(__file__), "..", "dino_wm", "env", "pusht", "pusht_env.py"
)
_push_t_spec = importlib.util.spec_from_file_location(
    "pusht_env", _push_t_path)
_push_t_module = importlib.util.module_from_spec(_push_t_spec)
_push_t_spec.loader.exec_module(_push_t_module)
PushTEnv = _push_t_module.PushTEnv
from deployment.delay import ConstantDelay, DelayModel, NoDelay
from deployment.wm_adapter import DinoWMPushtAdapter

# ── constants ────────────────────────────────────────────────────────────────
WINDOW_NAME = "DINO-WM Zero-Latency Interface"
GT_INSET_SCALE = 0.3        # GT inset size relative to main display
FONT = cv2.FONT_HERSHEY_SIMPLEX
TICK_RATE = 20              # target loop Hz

# Keyboard mappings.
KEY_ACTIONS: dict[int, tuple[float, float]] = {
    ord("w"): (0, 60),      # up
    ord("s"): (0, -60),     # down
    ord("a"): (-60, 0),     # left
    ord("d"): (60, 0),      # right
    ord("q"): (-60, 60),    # up-left
    ord("e"): (60, 60),     # up-right
    ord("z"): (-60, -60),   # down-left
    ord("c"): (60, -60),    # down-right
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


# ── main interface class ─────────────────────────────────────────────────────

class PushTInterface:
    """Interactive PushT zero-latency interface.

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
    """

    def __init__(
        self,
        mode: str = "zli",
        wm_ckpt_dir: str = "",
        wm_ckpt_suffix: str = "_latest",
        delay_steps: int = 10,
        display_size: int = 840,
        fps: int = 20,
    ) -> None:
        self._mode = mode
        self._display_size = display_size
        self._fps = fps

        # --- environment ------------------------------------------------------
        self._env = PushTEnv(
            with_velocity=True,
            with_target=True,
            render_size=224,
        )
        self._env.seed(0)

        # --- world model ------------------------------------------------------
        self._wm: DinoWMPushtAdapter | None = None
        if mode in ("wm_only", "zli"):
            self._wm = DinoWMPushtAdapter(
                ckpt_dir=wm_ckpt_dir,
                ckpt_suffix=wm_ckpt_suffix,
            )

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

    # -- public API ------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive loop. Blocks until ESC is pressed."""
        self._running = True

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

        clock = deque(maxlen=30)

        while self._running:
            t0 = time.perf_counter()

            # --- input --------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                self._running = False
                break
            action = self._get_action(key)

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
            raw_action = np.array(action, dtype=np.float32) / 100.0
            self._action_history.append(raw_action)

            # --- delay pipeline -----------------------------------------------
            self._delay.submit(gt_frame, self._timestep)
            received = self._delay.receive(self._timestep)

            # --- world model update -------------------------------------------
            if self._mode == "passthrough":
                self._wm_frame = gt_frame
            elif self._wm is not None:
                if received is not None:
                    # A delayed GT frame arrived — ground & predict ahead.
                    delayed_frame, gt_ts = received
                    self._delayed_frame = delayed_frame
                    actions_since = self._action_history[
                        gt_ts + 1 : self._timestep + 1
                    ]
                    if actions_since:
                        self._wm_frame = self._wm.ground_and_predict(
                            delayed_frame,
                            self._obs["proprio"],  # current proprio as approximation
                            actions_since,
                        )
                    else:
                        self._wm_frame = delayed_frame
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

            # Reset if episode ended.
            if done:
                self._do_reset()

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
            obs, _, _, _ = self._env.step(np.array([0, 0]))
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

    def _get_action(self, key: int) -> np.ndarray:
        """Read keyboard input → (dx, dy) action in pixel space [0, 512]."""
        if key in KEY_ACTIONS:
            return np.array(KEY_ACTIONS[key], dtype=np.float32)
        return np.array([0, 0], dtype=np.float32)

    def _do_reset(self) -> None:
        """Reset environment and WM state."""
        print("[IF] episode finished — resetting...")
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

        # Text overlay at bottom.
        y_base = main.shape[0] + 15
        _put_text(canvas, label, (4, y_base), (100, 255, 100))
        _put_text(canvas, f"t={self._timestep}", (4, y_base + 18), (200, 200, 200))
        _put_text(canvas, "WASD=move  ESC=quit  R=reset",
                  (4, y_base + 36), (150, 150, 150))

        return canvas
