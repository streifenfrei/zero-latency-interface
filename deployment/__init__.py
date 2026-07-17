"""Zero-latency interface for DINO-WM PushT world model.

Provides pluggable delay simulation, a WM adapter for interactive
frame-by-frame prediction, and gamepad (PS4/Xbox) input support.

The environment is ManiSkill PushT (3D PandaStick robot).  Requires:
  - A GPU with a working Vulkan driver (SAPIEN / physx_cuda).
  - The sibling ``zero_latency_interface`` repo for ``PushTXYWrapper``.
  - ManiSkill + gymnasium.
"""

from deployment.delay import ConstantDelay, DelayModel, NoDelay
from deployment.gamepad import GamepadInput, GamepadMapping, apply_deadzone

# DinoWMPushtAdapter is imported lazily — it depends on torch + dino_wm.
# PushTInterface is imported lazily — it depends on ManiSkill + torch.
# ManiSkillPushTEnv is imported lazily — it depends on ManiSkill + gymnasium.


def get_adapter(*args, **kwargs):
    """Lazy factory for DinoWMPushtAdapter (avoids eager torch import)."""
    from deployment.wm_adapter import DinoWMPushtAdapter
    return DinoWMPushtAdapter(*args, **kwargs)


def get_interface(*args, **kwargs):
    """Lazy factory for PushTInterface (avoids eager ManiSkill import)."""
    from deployment.interface import PushTInterface
    return PushTInterface(*args, **kwargs)


def get_mani_skill_env(*args, **kwargs):
    """Lazy factory for ManiSkillPushTEnv."""
    from deployment.mani_skill_env import ManiSkillPushTEnv
    return ManiSkillPushTEnv(*args, **kwargs)


__all__ = [
    "ConstantDelay",
    "DelayModel",
    "GamepadInput",
    "GamepadMapping",
    "NoDelay",
    "apply_deadzone",
    "get_adapter",
    "get_interface",
    "get_mani_skill_env",
]
