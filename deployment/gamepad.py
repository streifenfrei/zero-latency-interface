"""PS4/Xbox gamepad input for PushT teleoperation.

Uses pygame with the SDL dummy video driver so we can read gamepad events
without opening a pygame window (the OpenCV display is the only GUI window
and uses a different backend, so there is no conflict).

Mirrors the gamepad handling in zero_latency_interface/gamepad_teleop_pusht.py.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Use SDL dummy driver so pygame doesn't open a window.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")


# ── helpers ──────────────────────────────────────────────────────────────────

def apply_deadzone(value: float, deadzone: float = 0.15) -> float:
    """Deadzone with smooth rescaling so motion starts gently past the threshold.

    Parameters
    ----------
    value : float
        Raw axis value in ``[-1, 1]``.
    deadzone : float
        Fraction of the range to treat as zero (default 0.15).

    Returns
    -------
    float
        Rescaled value in ``[-1, 1]``, or 0.0 within the deadzone.
    """
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


# ── button / axis indices (Xbox-style; works for PS4 controllers too) ───────

@dataclass
class GamepadMapping:
    """Button and axis indices for the controller.

    These are the default Xbox-style indices.  PS4 controllers connected via
    USB or Bluetooth report the same layout.  If your controller uses different
    indices, pass a custom ``GamepadMapping`` or use ``--gamepad-debug`` to
    discover the actual values.
    """
    # Buttons
    A: int = 0       # cross (PS4)
    B: int = 1       # circle (PS4)
    Y: int = 3       # triangle (PS4)
    LB: int = 4      # L1
    RB: int = 5      # R1
    BACK: int = 6    # share / select

    # Axes (left stick)
    LX: int = 0      # horizontal: +1 = right
    LY: int = 1      # vertical:   +1 = down

    def to_dict(self) -> dict:
        return {
            "A": self.A, "B": self.B, "Y": self.Y,
            "LB": self.LB, "RB": self.RB, "BACK": self.BACK,
            "LX": self.LX, "LY": self.LY,
        }


# ── gamepad input class ──────────────────────────────────────────────────────

class GamepadInput:
    """PS4 / Xbox gamepad reader for PushT teleoperation.

    Parameters
    ----------
    speed : float
        Pixel-space push magnitude per step at full stick deflection
        (default 60 — matches the keyboard WASD magnitude).
    deadzone : float
        Stick deadzone fraction (default 0.15).
    fixed_magnitude : bool
        If True, the stick direction is projected onto the unit circle so the
        move is always at full ``speed`` regardless of how far the stick is
        pushed — the stick only steers.  If False, the magnitude scales with
        deflection (classic variable-speed stick).
    precision_scale : float
        Speed multiplier while the left bumper (LB / L1) is held.
    boost_scale : float
        Speed multiplier while the right bumper (RB / R1) is held.
    invert_x : bool
        Invert the x axis (forward/back).
    invert_y : bool
        Invert the y axis (left/right).
    swap_xy : bool
        Swap which stick axis maps to x vs y.
    mapping : GamepadMapping or None
        Button / axis indices.  Pass a custom mapping if your controller
        differs from the Xbox default.
    debug : bool
        If True, print live axis and button values to stdout.
    """

    def __init__(
        self,
        speed: float = 60.0,
        deadzone: float = 0.15,
        fixed_magnitude: bool = True,
        precision_scale: float = 0.3,
        boost_scale: float = 2.0,
        invert_x: bool = False,
        invert_y: bool = False,
        swap_xy: bool = False,
        mapping: Optional[GamepadMapping] = None,
        debug: bool = False,
    ) -> None:
        self._speed = speed
        self._deadzone = deadzone
        self._fixed_magnitude = fixed_magnitude
        self._precision_scale = precision_scale
        self._boost_scale = boost_scale
        self._invert_x = invert_x
        self._invert_y = invert_y
        self._swap_xy = swap_xy
        self._mapping = mapping or GamepadMapping()
        self._debug = debug

        self._pygame = None
        self._joystick = None
        self._prev_buttons: dict[int, bool] = {}
        self._connected: bool = False

    # ── lifecycle ─────────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    def setup(self) -> bool:
        """Initialise pygame and connect to the first available gamepad.

        Blocks until a gamepad is detected.  Returns True on success.
        """
        import pygame

        self._pygame = pygame
        pygame.init()
        pygame.joystick.init()

        waited = False
        while pygame.joystick.get_count() == 0:
            if not waited:
                print("[GP] No gamepad detected. Plug one in... (waiting)")
                waited = True
            pygame.event.pump()
            time.sleep(0.5)
            pygame.joystick.quit()
            pygame.joystick.init()

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
        self._connected = True

        print(
            f"[GP] Using gamepad: {self._joystick.get_name()}  "
            f"(axes={self._joystick.get_numaxes()}, "
            f"buttons={self._joystick.get_numbuttons()})"
        )
        if self._debug:
            print(f"[GP] mapping={self._mapping.to_dict()}")
        return True

    def close(self) -> None:
        """Shut down pygame."""
        self._connected = False
        if self._pygame is not None:
            try:
                self._pygame.quit()
            except Exception:
                pass
            self._pygame = None
            self._joystick = None

    # ── input reading ─────────────────────────────────────────────────────

    def pump(self) -> None:
        """Pump the pygame event queue (must be called once per loop iteration)."""
        if self._pygame is not None:
            self._pygame.event.pump()

    def read(self) -> tuple[float, float, dict[str, bool]]:
        """Read the current gamepad state.

        Returns
        -------
        dx : float
            Stick-derived x displacement in pixel-space units (not normalised).
        dy : float
            Stick-derived y displacement in pixel-space units (not normalised).
        buttons : dict
            Button state dictionary with keys ``reset``, ``next_episode``,
            ``quit``, ``precision``, ``boost``.
        """
        if not self._connected:
            return 0.0, 0.0, {}

        js = self._joystick
        m = self._mapping

        # ── left stick → planar deltas ────────────────────────────────────
        raw_x = apply_deadzone(js.get_axis(m.LX), self._deadzone)  # +1 = right
        raw_y = apply_deadzone(js.get_axis(m.LY), self._deadzone)  # +1 = down

        # Default intuitive mapping for PushT's camera (eye=+x, looking toward
        # -x), matching gamepad_teleop_pusht.py:
        #   stick down (+LY)  → world +x (toward camera, bottom of screen)
        #   stick right (+LX) → world +y (right of screen)
        dx, dy = raw_y, raw_x

        if self._swap_xy:
            dx, dy = dy, dx
        if self._invert_x:
            dx = -dx
        if self._invert_y:
            dy = -dy

        # Fixed-magnitude: project to unit circle.
        if self._fixed_magnitude:
            mag = (dx * dx + dy * dy) ** 0.5
            if mag > 1e-6:
                dx, dy = dx / mag, dy / mag

        # Speed modifiers via shoulder buttons.
        scale = self._speed
        if self._held(m.LB):
            scale *= self._precision_scale
        if self._held(m.RB):
            scale *= self._boost_scale

        dx *= scale
        dy *= scale

        # ── buttons ───────────────────────────────────────────────────────
        buttons = {
            "reset": self._pressed(m.A),
            "next_episode": self._pressed(m.Y),
            "quit": self._pressed(m.B) or self._pressed(m.BACK),
            "precision": self._held(m.LB),
            "boost": self._held(m.RB),
        }

        if self._debug:
            axes = [round(js.get_axis(i), 2) for i in range(js.get_numaxes())]
            bdown = [i for i in range(js.get_numbuttons()) if js.get_button(i)]
            print(
                f"[GP] axes={axes} buttons_down={bdown}  "
                f"dx={dx:.1f} dy={dy:.1f}  buttons={buttons}",
                end="\r", flush=True,
            )

        return dx, dy, buttons

    # ── internals ─────────────────────────────────────────────────────────

    def _held(self, idx: int) -> bool:
        """Level-triggered: True while button *idx* is held down."""
        if self._joystick is None:
            return False
        if idx >= self._joystick.get_numbuttons():
            return False
        return bool(self._joystick.get_button(idx))

    def _pressed(self, idx: int) -> bool:
        """Edge-triggered: True only on the frame the button goes down."""
        cur = self._held(idx)
        was = self._prev_buttons.get(idx, False)
        self._prev_buttons[idx] = cur
        return cur and not was

    def reset_button_state(self) -> None:
        """Clear edge-trigger state (call after handling button events)."""
        # Sync state without generating spurious edges next tick.
        if self._joystick is not None:
            for i in range(self._joystick.get_numbuttons()):
                self._prev_buttons[i] = bool(self._joystick.get_button(i))
