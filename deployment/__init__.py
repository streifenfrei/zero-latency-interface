"""Zero-latency interface for DINO-WM PushT world model.

Provides pluggable delay simulation, a WM adapter for interactive
frame-by-frame prediction, and gamepad (PS4/Xbox) input support.

Note: PushTInterface + DinoWMPushtAdapter require torch + pymunk + pygame +
shapely.  Install in a venv with those deps before using the full interface.
"""

from deployment.delay import ConstantDelay, DelayModel, NoDelay
from deployment.gamepad import GamepadInput, GamepadMapping, apply_deadzone

# DinoWMPushtAdapter is imported lazily — it depends on torch + dino_wm.
# PushTInterface is imported lazily — it depends on pymunk/pygame/shapely.


def get_adapter(*args, **kwargs):
    """Lazy factory for DinoWMPushtAdapter (avoids eager torch import)."""
    from deployment.wm_adapter import DinoWMPushtAdapter
    return DinoWMPushtAdapter(*args, **kwargs)


def get_interface(*args, **kwargs):
    """Lazy factory for PushTInterface (avoids eager pymunk import)."""
    from deployment.interface import PushTInterface
    return PushTInterface(*args, **kwargs)


__all__ = [
    "ConstantDelay",
    "DelayModel",
    "GamepadInput",
    "GamepadMapping",
    "NoDelay",
    "apply_deadzone",
    "get_adapter",
    "get_interface",
]
