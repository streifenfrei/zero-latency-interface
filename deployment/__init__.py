"""Zero-latency interface for DINO-WM PushT world model.

Provides pluggable delay simulation and a WM adapter for interactive,
frame-by-frame prediction compensating for network latency.

Note: PushTInterface requires pymunk + pygame + shapely (for the PushT env).
Install with:  pip install pymunk pygame shapely scikit-image
"""

from deployment.delay import ConstantDelay, DelayModel, NoDelay
from deployment.wm_adapter import DinoWMPushtAdapter

# PushTInterface is imported lazily — it depends on pymunk/pygame/shapely
# which may not be installed on headless machines.


def get_interface(*args, **kwargs):
    """Lazy factory for PushTInterface (avoids eager pymunk import)."""
    from deployment.interface import PushTInterface
    return PushTInterface(*args, **kwargs)


__all__ = [
    "ConstantDelay",
    "DelayModel",
    "DinoWMPushtAdapter",
    "NoDelay",
    "get_interface",
]
