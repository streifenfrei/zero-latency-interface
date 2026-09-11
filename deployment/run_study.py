#!/usr/bin/env python
"""Schedule-driven user-study UI for PushT teleoperation (kiosk app).

The whole study session runs in one long-lived process:

  * At startup a schedule CSV is loaded (columns: participant_id,
    schedule_row, block_index, delay_n, phase, setting, trial_in_block,
    session_trial, is_spare, seed).  NOTHING about the schedule's shape is
    hardcoded — any number of participants/blocks/rows works.
  * Fully anonymized: no names are ever recorded — only a sequential
    trial id.  A stale session-state file (e.g. the UI crashed mid-run)
    makes the home screen offer Resume (continue at that run) or Restart
    (start the trial's sequence from the first run).
  * The participant then works through their schedule rows in CSV order:
    each row is preceded by a run prompt (no countdown — the run starts
    immediately when the supervisor clicks Start) and ends with a results
    screen offering Next Run / Retry.  'spare' rows are skipped;
    'practice' rows are undelayed GT; 'main' rows use the row's delay_n
    with the row's setting (no_wm = delayed view, wm = WM-compensated).
  * Every completed run is auto-saved to the results CSV the moment it
    ends (crash-proof); Retry re-runs the same row as a new attempt.

Per-run metrics (see deployment/run_metrics.py): completion time, T-block
and EE path length, contact/push onsets (proxy), input direction
reversals, SPARC input smoothness, idle fraction / duty cycle, and — for
the wm setting — per-frame prediction error (SSIM + LPIPS) computed in a
background thread so the simulation tick rate is unaffected.  The tick
rate is strictly paced to --fps for ALL conditions so wall-clock
completion times are comparable, and the achieved rate is recorded per
run.

Keys: mouse or bracketed keyboard shortcuts on buttons; 'g' (wm condition)
toggles a debug live-GT inset; ESC aborts the current run back to its
prompt without saving; Ctrl+C quits the process.

Usage:
    python deployment/run_study.py --ckpt outputs/.../checkpoints \\
        --policy-checkpoint ../zero_latency_interface/checkpoints/policy.pt \\
        --gamepad --schedule deployment/schedule.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import cv2
import numpy as np

# Must be set BEFORE importing deployment.gamepad: that module sets
# SDL_VIDEODRIVER=dummy at import time (for its headless pad reader), so
# the real x11 driver has to be pinned first or set_mode() opens an
# invisible window.
os.environ.setdefault("SDL_VIDEODRIVER", "x11")

from deployment.delay import ConstantDelay
from deployment.gamepad import GamepadInput
from deployment.interface import FONT, _resize_to_fit, zli_tick
from deployment.mani_skill_env import ManiSkillPushTEnv
from deployment.run_metrics import RunMetrics
from deployment.wm_adapter import DinoWMPushtAdapter

WINDOW_NAME = "Zero Latency Interface - User Study"

# ── UI palette / typography ──────────────────────────────────────────────────
BG = (20, 20, 19)              # dark neutral canvas
TEXT = (232, 231, 227)         # primary text
MUTED = (160, 159, 153)        # secondary text
ACCENT = (42, 120, 214)        # button fill (blue #2a78d6)
BORDER = (110, 109, 105)       # secondary outlines
FONT_HEAD = cv2.FONT_HERSHEY_DUPLEX

# CSV schemas live in study_schema.py so analysis tools can import them
# without pulling in the WM/torch stack.
from deployment.study_schema import (  # noqa: E402
    FIELDNAMES, METRIC_FIELDS,
)


# ── schedule / CSV helpers ───────────────────────────────────────────────────

def load_schedule(path: Path) -> dict[str, list[dict]]:
    """participant_id -> rows in CSV order (spare rows dropped)."""
    rows: dict[str, list[dict]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("is_spare", "0")).strip() in ("1", "true", "True"):
                continue
            pid = row["participant_id"]
            row["delay_n"] = int(row["delay_n"])
            row["do_tlx"] = str(row.get("do_tlx", "0")).strip() in (
                "1", "true", "True")
            rows.setdefault(pid, []).append(row)
    return rows


def load_results(path: Path) -> tuple[set, dict]:
    """(completed (pid, session_trial) pairs, max attempt per pair).

    ``session_trial`` is the unique per-run id within a participant's
    schedule (``schedule_row`` is constant per participant — it is the
    participant's row in the counterbalancing design, NOT a run id).
    Rows missing the expected columns are skipped (e.g. a legacy-format
    file) rather than crashing the session.
    """
    completed: set = set()
    attempts: dict = {}
    max_trial_id = 0
    if not path.exists():
        return completed, attempts, 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if "participant_id" not in r or "session_trial" not in r:
                continue
            try:
                key = (r["participant_id"], int(r["session_trial"]))
            except (ValueError, KeyError):
                continue
            completed.add(key)
            attempts[key] = max(attempts.get(key, 0), int(r.get("attempt", 1)))
            try:
                max_trial_id = max(max_trial_id, int(r.get("trial_id", 0)))
            except (TypeError, ValueError):
                pass
    return completed, attempts, max_trial_id


def shelve_incompatible_results(path: Path) -> bool:
    """Rename aside a results CSV whose header isn't the current schema.

    Returns True if the file was shelved (caller should treat results as
    empty).  Preserves the old data instead of mixing schemas in one file.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path, newline="") as f:
            header = f.readline().strip()
    except OSError:
        return False
    if header == ",".join(FIELDNAMES):
        return False
    new_path = path.with_name(
        f"{path.stem}_legacy_{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}")
    path.rename(new_path)
    print(f"[STUDY] results file has an incompatible schema — shelved as "
          f"{new_path.name} (old data preserved)")
    return True


def append_result_row(path: Path, row: dict) -> None:
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            w.writeheader()
        w.writerow(row)


def load_session_state(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def save_session_state(path: Path, pid: str, session_trial: int,
                       attempt: int, trial_id: int = 0) -> None:
    with open(path, "w") as f:
        json.dump({"participant_id": pid, "session_trial": session_trial,
                   "attempt": attempt, "trial_id": trial_id,
                   "started_ts": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)


def clear_session_state(path: Path) -> None:
    if path.exists():
        path.unlink()


def delete_trial_rows(path: Path, trial_id: int) -> None:
    """Remove every results row belonging to a trial (restart = fresh
    progress; otherwise a later Resume would skip to the OLD frontier)."""
    if not path.exists():
        return
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("trial_id", "").strip() != str(trial_id):
                rows.append(r)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


# ── tiny cv2 button widget ───────────────────────────────────────────────────

@dataclass
class Button:
    id: str
    rect: tuple[int, int, int, int]  # x1, y1, x2, y2
    label: str
    key: Optional[int] = None
    icon: Optional[str] = None  # "ps_x": PS cross-button pictogram


def _rounded_rect(canvas: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                  radius: int, color: tuple, thickness: int = -1) -> None:
    """Rounded rectangle (ellipse-capped) — cv2 has no native one."""
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r <= 0:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        return
    # NOTE: two 2-tuples — a single 4-tuple is interpreted as
    # (x, y, width, height) by opencv-python, which silently fills a
    # huge rect instead of the intended one.
    cv2.rectangle(canvas, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(canvas, (x1, y1 + r), (x2, y2 - r), color, thickness)
    # Corner arcs (OpenCV sweeps clockwise): TL 180→270, TR 270→360,
    # BL 90→180, BR 0→90.
    cv2.ellipse(canvas, (x1 + r, y1 + r), (r, r), 0, 180, 270, color,
                thickness)
    cv2.ellipse(canvas, (x2 - r, y1 + r), (r, r), 0, 270, 360, color,
                thickness)
    cv2.ellipse(canvas, (x1 + r, y2 - r), (r, r), 0, 90, 180, color, thickness)
    cv2.ellipse(canvas, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def _draw_ps_x(canvas: np.ndarray, cx: int, cy: int, r: int = 9) -> None:
    """PS cross-button pictogram: a circle with an X inside."""
    cv2.circle(canvas, (cx, cy), r, (255, 255, 255), 1, cv2.LINE_AA)
    d = int(r * 2 / 3)
    cv2.line(canvas, (cx - d, cy - d), (cx + d, cy + d), (255, 255, 255),
             1, cv2.LINE_AA)
    cv2.line(canvas, (cx - d, cy + d), (cx + d, cy - d), (255, 255, 255),
             1, cv2.LINE_AA)


def draw_buttons(canvas: np.ndarray, buttons: list[Button]) -> None:
    for b in buttons:
        x1, y1, x2, y2 = b.rect
        bh = y2 - y1
        _rounded_rect(canvas, x1, y1, x2, y2, max(bh // 3, 8), ACCENT, -1)
        cy = (y1 + y2) // 2
        (tw, th), _ = cv2.getTextSize(b.label, FONT, 0.55, 1)
        ty = y1 + (y2 - y1 + th) // 2
        if b.icon == "ps_x" and "{X}" in b.label:
            # Inline pictogram: the label is "Press {X} to continue" and the
            # icon is drawn in the {X} slot, centered as a whole.
            left, right = b.label.split("{X}", 1)
            (lw, _), _ = cv2.getTextSize(left, FONT, 0.55, 1)
            (rw, _), _ = cv2.getTextSize(right, FONT, 0.55, 1)
            icon_w = 2 * 9 + 14
            total = lw + icon_w + rw
            start = x1 + max((x2 - x1 - total) // 2, 4)
            cv2.putText(canvas, left, (start, ty), FONT, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            _draw_ps_x(canvas, start + lw + 9 + 7, cy)
            cv2.putText(canvas, right, (start + lw + icon_w, ty), FONT, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            continue
        tx = x1 + max((x2 - x1 - tw) // 2, 4)
        cv2.putText(canvas, b.label, (tx, ty), FONT, 0.55, (255, 255, 255),
                   1, cv2.LINE_AA)


# ── pygame window (replaces cv2's Qt highgui: no hover toolbar, no
# pan/zoom, a plain title bar) ────────────────────────────────────────────────
_win = None  # set in main()
_show_fps = False          # F1 toggle; hidden by default
_fps_hist = deque(maxlen=60)


def _draw_fps(canvas: np.ndarray, W: int) -> None:
    """Small FPS counter, top-right (actual achieved tick rate)."""
    if not _show_fps or len(_fps_hist) < 2:
        return
    fps = 1.0 / (sum(_fps_hist) / len(_fps_hist))
    text = f"fps {fps:4.1f}"
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    x = W - tw - 14
    _put(canvas, text, (x, 22), MUTED, 0.5)


class _StudyWindow:
    def __init__(self, title: str, w: int, h: int) -> None:
        import pygame
        print("[UI] pygame import ok", flush=True)
        self._pg = pygame
        pygame.display.set_caption(title)
        print("[UI] set_caption ok", flush=True)
        self._surface = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        print("[UI] set_mode ok", flush=True)
        self._w, self._h = w, h  # logical canvas size
        self._scale = 1.0
        self._off = (0, 0)
        self.quit_requested = False

    def show(self, bgr_canvas: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr_canvas, cv2.COLOR_BGR2RGB)
        surf = self._pg.image.frombuffer(
            rgb.tobytes(), (self._w, self._h), "RGB")
        win_w, win_h = self._surface.get_size()
        scale = min(win_w / self._w, win_h / self._h)
        if scale != 1.0:
            surf = self._pg.transform.smoothscale(
                surf, (int(self._w * scale), int(self._h * scale)))
        self._surface.fill((20, 20, 19))
        ox = (win_w - surf.get_width()) // 2
        oy = (win_h - surf.get_height()) // 2
        self._surface.blit(surf, (ox, oy))
        self._scale = scale
        self._off = (ox, oy)
        self._pg.display.flip()

    def to_logical(self, px: int, py: int) -> tuple[int, int]:
        """Window pixel coords → logical canvas coords."""
        ox, oy = self._off
        return (int((px - ox) / self._scale), int((py - oy) / self._scale))

    def close(self) -> None:
        self._pg.display.quit()


# ── shared helpers ───────────────────────────────────────────────────────────

def _put(canvas, text, pos, color=TEXT, scale=0.6, thickness=1,
         font=FONT):
    cv2.putText(canvas, text, pos, font, scale, color, thickness,
                cv2.LINE_AA)


def make_frame_canvas(img: np.ndarray, w: int, h: int) -> np.ndarray:
    resized = _resize_to_fit(img, w, h)
    canvas = np.full((h, w, 3), BG, dtype=np.uint8)
    y_off = (h - resized.shape[0]) // 2
    x_off = (w - resized.shape[1]) // 2
    canvas[y_off:y_off + resized.shape[0],
           x_off:x_off + resized.shape[1]] = resized
    return canvas


def poll_input(gamepad):
    """One input-poll tick over the pygame window + gamepad.

    Returns ``(key, gp_dx, gp_dy, gp_buttons, quit_now, clicks)`` where
    ``clicks`` is a list of (x, y) left-button press positions this tick.
    """
    import pygame

    key = 0
    clicks: list = []
    quit_now = False
    for ev in pygame.event.get():
        if ev.type == pygame.KEYDOWN:
            key = ev.key
            if key == pygame.K_F1:
                global _show_fps
                _show_fps = not _show_fps
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            clicks.append(_win.to_logical(*ev.pos))
        elif ev.type == pygame.QUIT:
            quit_now = True
            _win.quit_requested = True
    if key == 27:  # ESC quits the app (session state stays -> resumable)
        quit_now = True
        _win.quit_requested = True
    gp_dx, gp_dy, gp_buttons = 0.0, 0.0, {}
    if gamepad is not None and gamepad.connected:
        gamepad.pump()
        gp_dx, gp_dy, gp_buttons = gamepad.read()
        # NOTE: gp_buttons["quit"] (B/circle) is deliberately NOT used —
        # B is one of the four continue face buttons in the study UI;
        # the only abort/quit actions are ESC and the window close button.
    return key, gp_dx, gp_dy, gp_buttons, quit_now, clicks


def stick_action(gp_dx: float, gp_dy: float) -> np.ndarray:
    """Joystick-only steering: the participant never steers with the
    keyboard (supervisor keys only)."""
    if abs(gp_dx) > 0.5 or abs(gp_dy) > 0.5:
        return np.array([gp_dx, gp_dy], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)


def is_success(info: dict) -> bool:
    s = info.get("success", False)
    if hasattr(s, "cpu"):
        return bool(s.detach().cpu().any())
    return bool(np.asarray(s).any())


def _join(thread) -> None:
    if thread is not None and thread.is_alive():
        thread.join()


# ── screens ──────────────────────────────────────────────────────────────────

def wait_for_button(gamepad, W: int, H: int,
                    buttons: list[Button], title: str, subtitle: str = "",
                    lines: list = None, primary_id: Optional[str] = None,
                    gamepad_continue: bool = False,
                    hidden_keys: dict = None,
                    lines_above: bool = False,
                    ) -> tuple[Optional[str], bool]:
    """Show a prompt screen until a button is clicked or ESC is pressed.

    Returns (button_id, False) on click and (None, True) on ESC.  With
    ``primary_id`` set, Enter and (when ``gamepad_continue``) any of the
    four right-hand gamepad face buttons return that id.  ``hidden_keys``
    maps keycodes to button ids for keyboard-only actions that have NO
    visible button (e.g. a supervisor-only retry).  With ``lines_above``,
    the ``lines`` are anchored just above the topmost button instead of a
    fixed position.
    """
    hidden_keys = hidden_keys or {}
    while True:
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        _put(canvas, title, (32, 48), TEXT, 0.85, font=FONT_HEAD)
        if subtitle:
            _put(canvas, subtitle, (32, 88), MUTED, 0.55)
        if lines:
            if lines_above:
                y_end = min(b.rect[1] for b in buttons) - 16
                y_start = y_end - 24 * (len(lines) - 1)
                for i, ln in enumerate(lines):
                    _put(canvas, ln, (32, y_start + i * 24), MUTED, 0.5)
            else:
                for i, ln in enumerate(lines):
                    _put(canvas, ln, (32, 122 + i * 24), MUTED, 0.5)
        draw_buttons(canvas, buttons)
        _win.show(canvas)

        key, _, _, _, quit_now, clicks = poll_input(gamepad)
        if quit_now:
            return None, True
        if primary_id is not None and key == 13:  # Enter
            return primary_id, False
        if gamepad_continue and primary_id is not None and gamepad is not None \
                and gamepad.connected and gamepad.any_right_button_pressed():
            return primary_id, False
        for b in buttons:
            x1, y1, x2, y2 = b.rect
            if any(x1 <= x <= x2 and y1 <= y <= y2 for x, y in clicks):
                return b.id, False
        for b in buttons:
            if b.key is not None and key == b.key:
                return b.id, False
        if key in hidden_keys:
            return hidden_keys[key], False


def _row_label(row: dict) -> str:
    """Human-readable condition label for one schedule row."""
    if row is None:
        return ""
    if row["phase"] == "practice":
        return "Practice (no delay)"
    return (f"Main · delay {row['delay_n']} · "
            + ("WM" if row["setting"] == "wm" else "no WM"))


def inter_run_prompt(gamepad, W: int, H: int, run_label: str,
                     next_label: str, completed_run: bool) -> Optional[str]:
    """THE single prompt between runs — blinded and metric-free.

    ``run_label``/``next_label`` are the practice/main counters
    ("Practice - X/10" / "Trial - X/24") — they carry no condition info.
    Retry is keyboard-only (R, invisible) — no button, no hint.

    ``completed_run=False``: start prompt before a run (first run, or
    after an abort).  ``completed_run=True``: concludes the run that just
    finished.  Returns 'advance', 'retry', or None (ESC → home).
    """
    if not completed_run:
        buttons = [Button("advance", (W // 2 - 130, int(H * 0.62) - 28,
                                      W // 2 + 130, int(H * 0.62) + 28),
                          "Press {X} to start run", key=32, icon="ps_x")]
        bid, esc = wait_for_button(
            gamepad, W, H, buttons, "Get ready",
            run_label,[],
            primary_id="advance", gamepad_continue=True,
            lines_above=True)
        return "advance" if bid == "advance" else None

    buttons = [
        Button("advance", (W // 2 - 130, int(H * 0.62) - 28, W // 2 + 130,
                           int(H * 0.62) + 28), "Press {X} to continue",
               key=ord("n"), icon="ps_x"),
    ]
    bid, esc = wait_for_button(
        gamepad, W, H, buttons,
        f"{run_label} complete",
        f"",
        [],
        primary_id="advance", gamepad_continue=True,
        hidden_keys={ord("r"): "retry"})
    if esc:
        return None
    return bid if bid in ("advance", "retry") else "advance"


# ── intake questionnaire (once per trial, anonymized) ────────────────────────

INTAKE_QUESTIONS = [
    ("Age band", ["18-24", "25-34", "35-44", "45+"]),
    ("Gender", ["Female", "Male", "Non-binary", "Prefer not to say"]),
    ("Handedness", ["Left", "Right", "Ambidextrous"]),
    ("Gaming (hours per week)", ["Less than 1", "1-5", "5-10", "10-20",
                                 "More than 20"]),
    ("Dual-analog gamepad (hours per week)", ["Less than 1", "1-5", "5-10",
                                              "10-20", "More than 20"]),
]

INTAKE_ANSWER_FIELDS = ["age_band", "gender", "handedness",
                          "gaming_hours_week", "gamepad_hours_week"]


def load_intake(results_path: Path, trial_id: int) -> Optional[dict]:
    """Intake answers for a trial, scanned from its saved results rows.

    NOTE: accepted limitation — the answers only exist once the first run
    row is written, so a crash BEFORE the first run save loses them and
    resume re-asks the intake.
    """
    if not results_path.exists():
        return None
    with open(results_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("trial_id", "").strip() == str(trial_id) \
                    and r.get("age_band", "").strip():
                return {k: r.get(k, "") for k in INTAKE_ANSWER_FIELDS}
    return None


def _intake_canvas(title: str, options: list, sel: int, idx: int,
                   W: int, H: int) -> np.ndarray:
    """One intake question: options as a row of selectable cards."""
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)
    _put(canvas, title, (32, 48), TEXT, 0.85, font=FONT_HEAD)
    _put(canvas, "A few questions before the trial starts.", (32, 80),
         MUTED, 0.55)
    n = len(options)
    card_w = min(180, int(W * 0.8 / n) - 12)
    total = n * (card_w + 12) - 12
    x0 = (W - total) // 2
    y0 = int(H * 0.4)
    y1 = y0 + 56
    for i, opt in enumerate(options):
        x1 = x0 + i * (card_w + 12)
        selected = (i == sel)
        _rounded_rect(canvas, x1, y0, x1 + card_w, y1, 10,
                      ACCENT if selected else (40, 40, 39), -1)
        _rounded_rect(canvas, x1, y0, x1 + card_w, y1, 10,
                      ACCENT if selected else BORDER, 1)
        (tw, th), _ = cv2.getTextSize(opt, FONT, 0.5, 1)
        _put(canvas, opt, (x1 + (card_w - tw) // 2,
                           y0 + (56 + th) // 2 - 2), TEXT, 0.5)
    # dots progress indicator
    for d in range(len(INTAKE_QUESTIONS)):
        if d < idx:
            cv2.circle(canvas, (W - 190 + d * 22, 54), 5, MUTED, -1,
                       cv2.LINE_AA)
        else:
            cv2.circle(canvas, (W - 190 + d * 22, 54), 5, MUTED, 1,
                       cv2.LINE_AA)
    _put(canvas,
         "Left/right on the stick or the D-pad to select — X to confirm — "
         "O to go back", (32, H - 30), MUTED, 0.5)
    return canvas


def intake_questionnaire(gamepad, W: int, H: int) -> Optional[dict]:
    """Ask the intake questions.  Returns {column: answer} or None (ESC)."""
    answers = {}
    idx = 0
    sels = [0] * len(INTAKE_QUESTIONS)
    hold_at = 0.0
    prev_dir = 0
    while idx < len(INTAKE_QUESTIONS):
        title, options = INTAKE_QUESTIONS[idx]
        _win.show(_intake_canvas(title, options, sels[idx], idx, W, H))
        key, _, gp_dy, _, quit_now, _ = poll_input(gamepad)
        if quit_now:
            return None
        now = time.monotonic()
        hx = gamepad.hat_x() if (gamepad is not None
                                 and gamepad.connected) else 0
        direction = hx
        if direction == 0:
            if gp_dy > 20:
                direction = 1
            elif gp_dy < -20:
                direction = -1
        if direction and (direction != prev_dir or now >= hold_at):
            sels[idx] = max(0, min(len(options) - 1,
                                   sels[idx] + direction))
            hold_at = now + 0.15
        prev_dir = direction
        if gamepad is not None and gamepad.connected                 and gamepad.any_right_button_pressed():
            answers[title] = options[sels[idx]]
            idx += 1
            prev_dir = 0
            hold_at = 0.0
            continue
        if gamepad is not None and gamepad.connected                 and gamepad.b_button_pressed() and idx > 0:
            idx -= 1
            prev_dir = 0
            hold_at = 0.0
    return {
        "age_band": answers.get("Age band", ""),
        "gender": answers.get("Gender", ""),
        "handedness": answers.get("Handedness", ""),
        "gaming_hours_week": answers.get("Gaming (hours per week)", ""),
        "gamepad_hours_week": answers.get(
            "Dual-analog gamepad (hours per week)", ""),
    }


# ── TLX-style questionnaire ──────────────────────────────────────────────────

TLX_SCALES = [
    ("Mental demand", "How mentally demanding was the task?",
     "Very low", "Very high"),
    ("Performance",
     "How successful were you in accomplishing what you were asked to do?",
     "Perfect", "Failure"),
    ("Frustration",
     "How insecure, discouraged, irritated, stressed and annoyed were you?",
     "Very low", "Very high"),
    ("Responsiveness", "The system responded immediately to my inputs.",
     "strongly disagree", "strongly agree"),
    ("Authenticity",
     "I trusted what I saw on screen to reflect the real robot state",
     "strongly disagree", "strongly agree"),
]

TLX_TICKS = 21  # 20 segments -> values 0..100 in steps of 5


def _tlx_scale_canvas(title: str, question: str, anchors: tuple,
                      tick: int, idx: int, W: int, H: int) -> np.ndarray:
    """Render one questionnaire scale (no digits — anchors only)."""
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)
    _put(canvas, title, (32, 48), TEXT, 0.85, font=FONT_HEAD)
    _put(canvas, "Considering the last run.", (32, 80), MUTED, 0.55)
    _put(canvas, question, (32, 116), TEXT, 0.6)
    # Progress indicator as dots — no digits anywhere, per the spec.
    dot_x0 = W - 190
    for d in range(len(TLX_SCALES)):
        if d < idx:
            cv2.circle(canvas, (dot_x0 + d * 22, 54), 5, MUTED, -1,
                       cv2.LINE_AA)
        else:
            cv2.circle(canvas, (dot_x0 + d * 22, 54), 5, MUTED, 1,
                       cv2.LINE_AA)
    x1, x2 = int(W * 0.15), int(W * 0.85)
    y = int(H * 0.55)
    cv2.line(canvas, (x1, y), (x2, y), BORDER, 2, cv2.LINE_AA)
    for i in range(TLX_TICKS):
        tx = x1 + (x2 - x1) * i // (TLX_TICKS - 1)
        h = 14 if i % 5 == 0 else 8
        cv2.line(canvas, (tx, y - h), (tx, y + h), BORDER, 2, cv2.LINE_AA)
    sel_x = x1 + (x2 - x1) * tick // (TLX_TICKS - 1)
    cv2.circle(canvas, (sel_x, y), 9, ACCENT, -1, cv2.LINE_AA)
    cv2.circle(canvas, (sel_x, y), 9, TEXT, 2, cv2.LINE_AA)
    _put(canvas, anchors[0], (x1 - 8, y + 34), MUTED, 0.5)
    (aw, _), _ = cv2.getTextSize(anchors[1], FONT, 0.5, 1)
    _put(canvas, anchors[1], (x2 - aw + 8, y + 34), MUTED, 0.5)
    _put(canvas,
         "Left/right on the stick or the D-pad to select — X to confirm — "
         "O to go back", (32, H - 30), MUTED, 0.5)
    return canvas


def tlx_questionnaire(gamepad, W: int, H: int) -> Optional[dict]:
    """Ask the 5 TLX scales.  Returns {column: value} or None (ESC skip)."""
    answers = {}
    idx = 0
    ticks = [10] * len(TLX_SCALES)  # start at the middle
    hold_at = 0.0
    prev_dir = 0
    while idx < len(TLX_SCALES):
        title, question, lo, hi = TLX_SCALES[idx]
        _win.show(_tlx_scale_canvas(title, question, (lo, hi),
                                    ticks[idx], idx + 1, W, H))
        key, _, gp_dy, _, quit_now, _ = poll_input(gamepad)
        if quit_now:
            return None
        now = time.monotonic()
        hx = gamepad.hat_x() if (gamepad is not None
                                 and gamepad.connected) else 0
        direction = hx
        if direction == 0:
            # NOTE: read() maps gp_dx to the stick's up/down axis and
            # gp_dy to left/right (PushT camera convention) — use gp_dy.
            if gp_dy > 20:
                direction = 1
            elif gp_dy < -20:
                direction = -1
        if direction and (direction != prev_dir or now >= hold_at):
            ticks[idx] = max(0, min(TLX_TICKS - 1, ticks[idx] + direction))
            hold_at = now + 0.15
        prev_dir = direction
        if gamepad is not None and gamepad.connected                 and gamepad.any_right_button_pressed():
            answers[TLX_SCALES[idx][0]] = ticks[idx] * 5
            idx += 1
            prev_dir = 0
            hold_at = 0.0
            continue
        if gamepad is not None and gamepad.connected                 and gamepad.b_button_pressed() and idx > 0:
            idx -= 1
            prev_dir = 0
            hold_at = 0.0
    return {
        "tlx_mental": answers.get("Mental demand", ""),
        "tlx_performance": answers.get("Performance", ""),
        "tlx_frustration": answers.get("Frustration", ""),
        "tlx_responsiveness": answers.get("Responsiveness", ""),
        "tlx_authenticity": answers.get("Authenticity", ""),
    }


# ── run drivers (one per setting) ────────────────────────────────────────────

def _run_gt(env, gamepad, W: int, H: int, timeout_minutes: float, fps: int,
            metrics: RunMetrics, run_label: str = "") -> Optional[dict]:
    """Practice / undelayed GT run."""
    env.reset()
    timeout = timeout_minutes * 60.0
    t_start = time.perf_counter()
    deadline = t_start + 1.0 / fps
    tick_start = t_start
    while True:
        key, gp_dx, gp_dy, _, quit_now, _ = poll_input(gamepad)
        if quit_now:
            return None
        if key == ord("r"):  # supervisor retry: restart this run
            return "retry"
        action = stick_action(gp_dx, gp_dy)
        obs, _, _, info = env.step(action)
        canvas = make_frame_canvas(obs["visual"], W, H)
        if run_label:
            _put(canvas, run_label, (12, 24), MUTED, 0.5)
        _win.show(canvas)

        # Pace first so the recorded tick spans the FULL period (work +
        # wait) — otherwise the measured tick rate ignores --fps and
        # reports the raw step rate.
        sleep = deadline - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        deadline += 1.0 / fps
        now = time.perf_counter()
        metrics.record(action, info["state"], now - tick_start)
        _fps_hist.append(now - tick_start)
        tick_start = now
        if _show_fps:
            _draw_fps(canvas, W)
            _win.show(canvas)

        elapsed = now - t_start
        if is_success(info):
            return {"completion_time_s": elapsed, "success": True,
                    "timed_out": False}
        if elapsed >= timeout:
            return {"completion_time_s": elapsed, "success": False,
                    "timed_out": True}


def _run_delayed(env, gamepad, W: int, H: int, delay_steps: int,
                 timeout_minutes: float, fps: int,
                 metrics: RunMetrics, run_label: str = "") -> Optional[dict]:
    """no_wm setting: delayed GT view, single attempt."""
    obs, _ = env.reset()
    delay = ConstantDelay(delay_steps)
    delayed_frame = obs["visual"]
    timestep = 0
    timeout = timeout_minutes * 60.0
    t_start = time.perf_counter()
    deadline = t_start + 1.0 / fps
    tick_start = t_start
    while True:
        key, gp_dx, gp_dy, _, quit_now, _ = poll_input(gamepad)
        if quit_now:
            return None
        if key == ord("r"):  # supervisor retry: restart this run
            return "retry"
        action = stick_action(gp_dx, gp_dy)
        obs, _, _, info = env.step(action)

        delay.submit(obs["visual"], obs["proprio"], timestep)
        received = delay.receive(timestep)
        if received is not None:
            delayed_frame = received[0]
        canvas = make_frame_canvas(delayed_frame, W, H)
        if run_label:
            _put(canvas, run_label, (12, 24), MUTED, 0.5)
        _win.show(canvas)

        # Pace first so the recorded tick spans the FULL period (work +
        # wait) — otherwise the measured tick rate ignores --fps.
        sleep = deadline - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        deadline += 1.0 / fps
        now = time.perf_counter()
        metrics.record(action, info["state"], now - tick_start)
        _fps_hist.append(now - tick_start)
        tick_start = now
        if _show_fps:
            _draw_fps(canvas, W)
            _win.show(canvas)

        elapsed = now - t_start
        if is_success(info):
            return {"completion_time_s": elapsed, "success": True,
                    "timed_out": False}
        if elapsed >= timeout:
            return {"completion_time_s": elapsed, "success": False,
                    "timed_out": True}
        timestep += 1


def _run_wm(env, wm: DinoWMPushtAdapter, gamepad, W: int, H: int,
            delay_steps: int, timeout_minutes: float, fps: int,
            metrics: RunMetrics, run_label: str = "") -> Optional[dict]:
    """wm setting: WM-compensated view (shared zli_tick path).

    Returns a result dict or None (aborted).  The grounding worker is
    joined on every exit path — the WM is a session-long shared object.
    """
    obs, _ = env.reset()
    fs, needed = wm.frameskip, wm.num_hist

    # Prime the WM's context exactly like PushTInterface._collect_context:
    # action_history must stay indexed by absolute raw timestep (the
    # grounding worker looks actions up by timestep value).
    gt_frames = [obs["visual"].copy()]
    gt_proprios = [obs["proprio"].copy()]
    action_history: list = []
    frame_counter = 0
    timestep = 0
    while len(gt_frames) < needed:
        key, _, _, _, quit_now, _ = poll_input(gamepad)
        if quit_now:
            return None
        obs, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        action_history.append(np.asarray(
            info.get("effective_action", np.zeros(2, dtype=np.float32)),
            dtype=np.float32))
        frame_counter += 1
        if frame_counter % fs == 0:
            gt_frames.append(obs["visual"].copy())
            gt_proprios.append(obs["proprio"].copy())
        timestep += 1
        canvas = make_frame_canvas(obs["visual"], W, H)
        _put(canvas, "Preparing...", (10, 24), (150, 150, 150), 0.5)
        _win.show(canvas)

    wm.reset(gt_frames[-needed:], gt_proprios[-needed:])
    display_frame = gt_frames[-1]
    delay = ConstantDelay(delay_steps)
    grounding_thread = None
    last_grounded_ts = -1
    get_timestep: Callable[[], int] = lambda: timestep
    show_gt = False
    prev_key = -1

    timeout = timeout_minutes * 60.0
    t_start = time.perf_counter()
    deadline = t_start + 1.0 / fps
    tick_start = t_start
    try:
        while True:
            key, gp_dx, gp_dy, _, quit_now, _ = poll_input(gamepad)
            if quit_now:
                return None
            if key == ord("r"):  # supervisor retry: restart this run
                return "retry"
            if key == ord("g") and prev_key != ord("g"):
                show_gt = not show_gt
                print(f"[STUDY] debug: live GT overlay "
                      f"{'shown' if show_gt else 'hidden'}")
            prev_key = key
            action = stick_action(gp_dx, gp_dy)
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
            pair = None
            if display_frame is not gt_frame:
                pair = (display_frame, gt_frame)

            canvas = make_frame_canvas(display_frame, W, H)
            if run_label:
                _put(canvas, run_label, (12, 24), MUTED, 0.5)
            if show_gt:
                inset_w = int(W * 0.28)
                inset_h = int(inset_w * display_frame.shape[0]
                              / max(display_frame.shape[1], 1))
                inset = _resize_to_fit(gt_frame, inset_w, inset_h)
                y0 = canvas.shape[0] - inset.shape[0]
                x0 = canvas.shape[1] - inset.shape[1]
                canvas[y0:y0 + inset.shape[0],
                       x0:x0 + inset.shape[1]] = inset
                _put(canvas, "live GT", (x0 + 2, y0 + 14), (200, 200, 200))
            _win.show(canvas)

            # Pace first so the recorded tick spans the FULL period (work +
            # wait) — otherwise the measured tick rate ignores --fps.
            sleep = deadline - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            deadline += 1.0 / fps
            now = time.perf_counter()
            # Record the PIXEL-SPACE operator input, not eff_action: the
            # input-based metrics (reversals / idle / duty / SPARC) use
            # pixel-space thresholds (idle <= 9 px, reversal >= 12 px) and
            # must see the same units in every condition.  eff_action is
            # for the world model only.
            metrics.record(action, info["state"], now - tick_start,
                           frame_pair=pair)
            _fps_hist.append(now - tick_start)
            tick_start = now
            if _show_fps:
                _draw_fps(canvas, W)
                _win.show(canvas)

            elapsed = now - t_start
            if is_success(info):
                return {"completion_time_s": elapsed, "success": True,
                        "timed_out": False}
            if elapsed >= timeout:
                return {"completion_time_s": elapsed, "success": False,
                        "timed_out": True}
            timestep += 1
            # NOTE: no second pacing block here — the tick was already
            # paced above; a duplicate would advance the deadline twice
            # and cap the loop at fps/2.
    finally:
        _join(grounding_thread)


def run_row(env, wm, gamepad, W: int, H: int, row: dict,
            timeout_minutes: float, fps: int,
            run_label: str = "") -> tuple[Optional[dict], dict]:
    """Execute one schedule row.  Returns (result_or_None, metrics_dict)."""
    collect_pairs = (row["setting"] == "wm")
    metrics = RunMetrics(fps=fps, collect_pairs=collect_pairs)
    metrics.start()
    if row["phase"] == "practice":
        result = _run_gt(env, gamepad, W, H, timeout_minutes, fps, metrics,
                         run_label)
    elif row["setting"] == "wm":
        result = _run_wm(env, wm, gamepad, W, H, row["delay_n"],
                         timeout_minutes, fps, metrics, run_label)
    else:  # no_wm
        result = _run_delayed(env, gamepad, W, H, row["delay_n"],
                              timeout_minutes, fps, metrics, run_label)
    metrics_dict = metrics.finalize(discard=(result is None))
    return result, metrics_dict


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Schedule-driven user-study UI for PushT teleoperation")
    parser.add_argument("--ckpt", required=True,
                        help="path to checkpoints/ directory (world model)")
    parser.add_argument("--ckpt_suffix", default="_latest")
    parser.add_argument("--display_size", type=int, default=840)
    parser.add_argument("--fps", type=int, default=10,
                        help="tick rate for ALL conditions (comparable "
                             "wall-clock times; default 20)")

    study = parser.add_argument_group("study")
    study.add_argument("--results-dir", default="results",
                       help="directory for the results CSV / session "
                            "state (default: results/)")
    study.add_argument("--schedule", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "schedule.csv"),
        help="schedule CSV (default: deployment/schedule.csv)")
    study.add_argument("--task-timeout-minutes", type=float, default=1.0,
                       help="per-run timeout, all phases (default: 5.0)")
    study.add_argument("--goal-tolerance", type=float, default=0.80,
                       help="success overlap fraction, lower = more "
                            "tolerant (default: 0.80)")

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

    ms.add_argument("--with-block-pose", action="store_true",
                    help="Append the T-block pose (xy + cos/sin) to the "
                         "WM proprio (4 -> 8 dims).  Must match the flag "
                         "the checkpoint was trained with.")
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
        with_block_pose=args.with_block_pose,
        goal_tolerance=args.goal_tolerance,
        render_size=wm.img_size,
    )
    env = ManiSkillPushTEnv(**env_kwargs)
    env.seed(0)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    schedule = load_schedule(Path(args.schedule))
    results_path = results_dir / "study_results.csv"
    state_path = results_dir / "session_state.json"
    shelve_incompatible_results(results_path)
    completed, attempts, max_trial_id = load_results(results_path)
    print(f"[STUDY] schedule: {len(schedule)} participants, "
          f"{sum(len(v) for v in schedule.values())} non-spare runs")

    W, H = args.display_size, int(args.display_size * 0.6)
    global _win
    _win = _StudyWindow(WINDOW_NAME, W, H)

    def mk_button(id_: str, x1: int, x2: int, label: str,
                  key=None, centered: bool = False) -> Button:
        yc = H // 2 if centered else int(H * 0.62)
        return Button(id_, (x1, yc - 28, x2, yc + 28), label, key)

    try:
        while True:
            if _win.quit_requested:  # window close button
                break
            # ── home: New Trial, or resume/restart an interrupted one ─────
            stale = load_session_state(state_path)
            if stale is not None:
                r_buttons = [
                    mk_button("resume", W // 2 - 270, W // 2 - 10,
                              f"[R] Resume (continue at run "
                              f"{stale.get('session_trial', '?')})",
                              key=ord("r"), centered=True),
                    mk_button("restart", W // 2 + 10, W // 2 + 270,
                              "[S] Restart from run 1",
                              key=ord("s"), centered=True),
                ]
                bid, esc = wait_for_button(
                    gamepad, W, H, r_buttons,
                    "Unfinished trial detected",
                    "An earlier trial was interrupted.",
                    ["Resume continues from that run; Restart begins the",
                     "trial's sequence from the first run."])
                if esc:
                    continue
                pid = stale["participant_id"]
                # Keep the interrupted trial's id (stored in the state file;
                # fall back to the results-derived counter for old files).
                trial_id = int(stale.get("trial_id") or max_trial_id)
                restart = (bid == "restart")
                if restart:
                    # A true restart: erase the trial's progress and intake
                    # so it starts fresh (and a later resume continues from
                    # the NEW progress, not the old frontier).  With the
                    # purge done, completed is empty for this participant,
                    # so the normal skip/advance loop starts at run 1 and
                    # progresses as usual — restart must NOT stay set, or
                    # the completed-row skip is disabled forever and the
                    # loop re-runs the same row indefinitely.
                    clear_session_state(state_path)
                    delete_trial_rows(results_path, trial_id)
                    completed = {k for k in completed if k[0] != pid}
                    attempts = {k: v for k, v in attempts.items()
                                if k[0] != pid}
                    intake_answers = None
                    restart = False
                else:
                    intake_answers = load_intake(results_path, trial_id)
                if intake_answers is None:
                    intake = intake_questionnaire(gamepad, W, H)
                    intake_answers = intake if intake else {
                        k: "" for k in INTAKE_ANSWER_FIELDS}
            else:
                home_buttons = [
                    mk_button("new", W // 2 - 130, W // 2 + 130,
                              "New Trial", key=ord("n"), centered=True),
                ]
                bid, esc = wait_for_button(gamepad, W, H, home_buttons,
                                           "", "", [])
                if bid is None:
                    if esc:  # ESC on the home screen quits the app
                        _win.quit_requested = True
                    continue
                # Next participant in schedule order with an unfinished run.
                pid = None
                for cand in schedule:
                    if any((cand, int(r["session_trial"])) not in completed
                           for r in schedule[cand]):
                        pid = cand
                        break
                if pid is None:
                    done_buttons = [mk_button("home", W // 2 - 130,
                                              W // 2 + 130,
                                              "[H] All done", key=ord("h"))]
                    wait_for_button(gamepad, W, H, done_buttons,
                                    "All participants complete",
                                    "No unfinished runs remain.")
                    continue
                trial_id = max_trial_id + 1
                max_trial_id = trial_id
                restart = False
                print(f"[STUDY] new trial {trial_id} -> {pid}")

                # intake questionnaire (once per trial; answers are
                # written into every results row of this trial)
                intake = intake_questionnaire(gamepad, W, H)
                intake_answers = intake if intake else {
                    k: "" for k in INTAKE_ANSWER_FIELDS}
            if pid not in schedule:
                print(f"[STUDY] participant {pid} has no schedule rows")
                continue

            # ── run the participant's rows in order ──────────────────────────
            # Exactly ONE prompt between runs (the run that just finished is
            # concluded and the next one announced on the same screen; no
            # metrics are ever shown).  last_saved tracks whether the
            # preceding row was saved, which decides prompt form.
            rows = schedule[pid]
            practice_rows = [r for r in rows if r["phase"] == "practice"]
            main_rows = [r for r in rows if r["phase"] != "practice"]

            def run_label(row: dict) -> str:
                if row["phase"] == "practice":
                    idx = practice_rows.index(row) + 1
                    return f"Practice - {idx}/{len(practice_rows)}"
                idx = main_rows.index(row) + 1
                return f"Trial - {idx}/{len(main_rows)}"

            i = 0
            last_saved = False
            first_run = True  # New Trial starts the first run immediately
            while i < len(rows):
                row = rows[i]
                key_r = (pid, int(row["session_trial"]))
                if key_r in completed and not restart:
                    i += 1
                    continue  # last_saved preserved: the completed-run
                              # prompt for the just-saved row follows
                phase_label = _row_label(row)
                attempt = attempts.get(key_r, 0) + 1
                attempts[key_r] = attempt

                if last_saved:
                    # Conclude the just-saved run; 'advance' runs the
                    # current row immediately (no second prompt).
                    choice = inter_run_prompt(
                        gamepad, W, H, run_label(row),
                        run_label(rows[i + 1]) if i + 1 < len(rows) else "",
                        completed_run=True)
                    if choice is None:
                        break
                    if choice == "retry":
                        i -= 1  # re-run the just-saved row
                        row = rows[i]
                        key_r = (pid, int(row["session_trial"]))
                        phase_label = _row_label(row)
                        attempt = attempts.get(key_r, 0) + 1
                        attempts[key_r] = attempt
                    last_saved = False
                elif not first_run:
                    # Start prompt (after an abort).
                    choice = inter_run_prompt(
                        gamepad, W, H, run_label(row), "",
                        completed_run=False)
                    if choice is None:
                        break
                else:
                    choice = "advance"  # start the first run immediately
                first_run = False

                save_session_state(state_path, pid,
                                   int(row["session_trial"]), attempt,
                                   trial_id)
                result, metrics_dict = run_row(
                    env, wm, gamepad, W, H, row,
                    args.task_timeout_minutes, args.fps, run_label(row))
                if result == "retry":  # supervisor R: restart this run
                    clear_session_state(state_path)
                    last_saved = False
                    continue
                if result is None:  # aborted — no save
                    if _win.quit_requested:
                        # The app is quitting (window X) — KEEP the session
                        # state so the trial can be resumed next launch.
                        break
                    clear_session_state(state_path)
                    last_saved = False
                    continue

                clear_session_state(state_path)
                tlx_answers = None
                if row.get("do_tlx"):
                    tlx_answers = tlx_questionnaire(gamepad, W, H)
                out_row = {
                    "participant_id": pid,
                    "schedule_row": row["schedule_row"],
                    "block_index": row["block_index"],
                    "delay_n": row["delay_n"],
                    "phase": row["phase"],
                    "setting": row["setting"],
                    "trial_in_block": row["trial_in_block"],
                    "session_trial": row["session_trial"],
                    "attempt": attempt,
                    "trial_id": trial_id,
                    "completion_time_s": f"{result['completion_time_s']:.3f}",
                    "success": result["success"],
                    "timed_out": result["timed_out"],
                    "goal_tolerance": args.goal_tolerance,
                    "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                out_row.update({k: f"{v:.6f}" if isinstance(v, float)
                                else v for k, v in metrics_dict.items()})
                out_row.update(intake_answers)
                if tlx_answers:
                    out_row.update(tlx_answers)
                append_result_row(results_path, out_row)
                completed.add(key_r)
                last_saved = True
                print(f"[STUDY] trial {trial_id} saved {pid} run {row['session_trial']} "
                      f"attempt {attempt} "
                      f"t={result['completion_time_s']:.2f}s "
                      f"success={result['success']}")

            if i >= len(rows):
                d_buttons = [mk_button("home", W // 2 - 140, W // 2 + 140,
                                     "[H] Back to Home", key=ord("h"))]
                wait_for_button(gamepad, W, H, d_buttons,
                                "Trial complete",
                                f"Trial {trial_id} finished all "
                                f"{len(rows)} scheduled runs.",
                                primary_id="home", gamepad_continue=True)
    finally:
        if gamepad is not None and gamepad.connected:
            gamepad.close()
        _win.close()


if __name__ == "__main__":
    main()
