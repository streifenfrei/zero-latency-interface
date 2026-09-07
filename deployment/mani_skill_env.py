"""ManiSkill 3D PushT environment adapter for the zero-latency deployment.

Provides the same ``(obs_dict, state)`` interface as the 2D pygame
``PushTEnv`` so that ``PushTInterface`` can swap between them without
changes to the display/WM pipeline.

Requires the sibling ``zero_latency_interface`` repo for
``PushTXYWrapper`` / ``pusht_explore`` / ``rl_train_pusht``.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class ManiSkillPushTEnv:
    """ManiSkill PushT environment wrapped to match the ``PushTEnv`` interface.

    Uses ``PushT-XYExplore-v1`` (exploring starts) with a single env,
    ``PushTXYWrapper`` for action processing / proprio extraction, and
    ``ManiSkillVectorEnv`` for the batch interface.

    Parameters
    ----------
    env_id:
        Gymnasium env id.  Default ``PushT-XYExplore-v1``.
    zero_latency_root:
        Path to the sibling ``zero_latency_interface`` repo.  Required for
        importing ``rl_train_pusht`` / ``pusht_explore``.
    policy_checkpoint:
        Path to a trained PPO policy ``.pt`` checkpoint.  Only its ``args``
        are read (to mirror the training-time wrapper config).  If ``None``,
        sensible defaults are used.
    control_mode:
        ``"pd_ee_delta_pose"`` (default) or ``"pd_ee_pose"``.
    action_scale:
        Metres per step at full-normalised action (``[-1,1] → ±action_scale``
        m).  Default 0.1 for delta mode.
    step_size:
        Metres per step at full keyboard/gamepad deflection.  Default 0.02.
    max_input:
        The keyboard/gamepad value that maps to full deflection.  Default 60
        (matches the existing WASD magnitude and gamepad default speed).
    fixed_magnitude:
        ``None`` (default) follows the policy checkpoint's setting if one is
        given, else ``False``.  With the unit-circle projection on, every
        non-idle move is ``action_scale`` m/step (how the training policy
        acted — and how the WM's action encoder saw the data); note this
        makes the env move faster than the original gamepad teleop when
        ``action_scale`` > ``step_size``.  ``False`` preserves the teleop
        command magnitude so full deflection moves exactly ``step_size``
        m/step (``gamepad_teleop_pusht.py --speed`` style) — with the
        checkpoint's ``action_scale`` cancelling out of the conversion, the
        effective actions are ``(dx/max_input) * step_size/0.1`` regardless.
    level_ee:
        ``None`` (default) follows the policy checkpoint's setting if one is
        given, else ``False``.  When on (delta mode), each step commands a
        rotation correction back to the upright orientation captured at
        reset, capped at ``level_ee_rot_bound`` rad/step — this is why the
        training data (and the WM rollout) always shows a vertical stick.
        Off, delta mode sends drot=0 and contact torques tilt the stick
        permanently.
    push_height:
        Fixed stick-tip height (m) above the table.  Default 0.015.
    render_size:
        Output frame side length in pixels.  Default 224 (matches the WM).
    max_episode_steps:
        Per-episode horizon.  Default 100_000 (effectively unlimited for a
        human operator).
    sim_backend:
        ManiSkill simulation backend.  Default ``"physx_cuda"``.
    with_velocity:
        Include EE velocity in proprio.  Default ``True``.
    goal_tolerance:
        Required overlap fraction (0..1) between the pushed T-block and the
        goal T for task success — ManiSkill's ``intersection_thresh``.
        LOWER = more tolerant of position/orientation error.  Default 0.90
        (the stock PushT threshold).
    seed:
        Random seed.
    """

    def __init__(
        self,
        *,
        env_id: str = "PushT-XYExplore-v1",
        zero_latency_root: Optional[str] = None,
        policy_checkpoint: Optional[str] = None,
        control_mode: str = "pd_ee_delta_pose",
        action_scale: float = 0.1,
        step_size: float = 0.02,
        max_input: float = 60.0,
        fixed_magnitude: Optional[bool] = None,
        level_ee: Optional[bool] = None,
        push_height: float = 0.015,
        render_size: int = 224,
        max_episode_steps: int = 100_000,
        sim_backend: str = "physx_cuda",
        with_velocity: bool = True,
        goal_tolerance: float = 0.90,
        seed: int = 0,
    ) -> None:
        # Resolve and register the sibling repo.
        if zero_latency_root is None:
            zero_latency_root = os.path.join(
                os.path.dirname(__file__), "..", "..", "zero_latency_interface"
            )
        zl_root = os.path.abspath(zero_latency_root)
        if zl_root not in sys.path:
            sys.path.insert(0, zl_root)

        import gymnasium as gym
        import mani_skill  # noqa: F401
        import pusht_explore  # noqa: F401
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
        from rl_train_pusht import PushTXYWrapper

        # --- load training-time config if available --------------------------
        targs: dict = {}
        if policy_checkpoint is not None and os.path.isfile(policy_checkpoint):
            ckpt = torch.load(
                policy_checkpoint, map_location="cpu", weights_only=False
            )
            targs = ckpt.get("args", {})
        elif policy_checkpoint is not None:
            print(
                f"[MS-Env] WARNING: policy checkpoint not found at "
                f"{policy_checkpoint!r} — falling back to CLI defaults for the "
                f"wrapper config (action_scale, level_ee, ... may differ from "
                f"the training run)."
            )

        _ctrl = targs.get("control_mode", control_mode)
        _ascale = targs.get("action_scale", action_scale)
        _push_h = targs.get("push_height", push_height)
        _full_state = targs.get("full_state", False)
        _obs_mode = "state" if _full_state else "none"
        _fix_push_height = targs.get("fix_push_height", True)
        _fix_orientation = targs.get("fix_orientation", True)
        _no_ee_vel = targs.get("no_ee_vel", False)
        # Default: follow the checkpoint when given; otherwise teleop-style
        # (False) — the configuration where the WM rollout empirically tracks
        # GT and the speed matches gamepad_teleop_pusht.py.  Explicitly
        # passing fixed_magnitude (True/False) overrides both.
        _fixed_magnitude = (
            targs.get("fixed_magnitude", False)
            if fixed_magnitude is None else fixed_magnitude
        )
        _level_ee = targs.get("level_ee", False) if level_ee is None else level_ee
        _level_ee_gain = targs.get("level_ee_gain", 1.0)
        _level_ee_rot_bound = targs.get("level_ee_rot_bound", 0.1)
        _xy_absolute_target = targs.get("xy_absolute_target", False)
        _xy_windup = targs.get("xy_windup", 0.05)
        _ee_init_radius = targs.get("ee_init_radius", 0.0)
        _ee_init_radius_min = targs.get("ee_init_radius_min", 0.12)

        self._step_size = step_size
        self._max_input = max_input
        self._action_scale = _ascale
        self._with_velocity = with_velocity or (not _no_ee_vel)
        self._render_size = render_size
        self._seed = seed

        # --- build env -------------------------------------------------------
        base = gym.make(
            env_id,
            num_envs=1,
            obs_mode=_obs_mode,
            control_mode=_ctrl,
            reward_mode="none",
            render_mode="rgb_array",
            robot_uids="panda_stick",
            ee_init_radius=_ee_init_radius,
            ee_init_radius_min=_ee_init_radius_min,
            sim_backend=sim_backend,
            max_episode_steps=max_episode_steps,
        )
        # Success tolerance: PushTEnv.evaluate() reads intersection_thresh as
        # an instance attribute at call time, so a runtime override changes
        # the goal condition without touching the task class.
        base.unwrapped.intersection_thresh = float(goal_tolerance)
        self._wrapper = PushTXYWrapper(
            base,
            action_scale=_ascale,
            push_height=_push_h,
            include_ee_vel=not _no_ee_vel,
            include_ee_rot=targs.get("ee_rot", False),
            include_goal=not targs.get("no_goal", True),
            fix_push_height=_fix_push_height,
            fix_orientation=_fix_orientation,
            full_state=_full_state,
            control_mode=_ctrl,
            fixed_magnitude=_fixed_magnitude,
            level_ee=_level_ee,
            level_ee_gain=_level_ee_gain,
            level_ee_rot_bound=_level_ee_rot_bound,
            xy_absolute_target=_xy_absolute_target,
            xy_windup=_xy_windup,
        )
        self._venv = ManiSkillVectorEnv(
            self._wrapper,
            auto_reset=False,
            ignore_terminations=False,
            record_metrics=False,
        )

        self.proprio_dim: int = 2 if _no_ee_vel else 4
        self.state_dim: int = self._wrapper.obs_dim

        self._device = self._venv.device
        print(
            f"[MS-Env] PushT-XYExplore-v1 ready — control={_ctrl}  "
            f"action_scale={_ascale:.3f}  step_size={step_size:.3f}m  "
            f"proprio_dim={self.proprio_dim}  sim={sim_backend}"
        )

    # -- PushTEnv-compatible public API ---------------------------------------

    def reset(self):
        """Reset the environment.

        Returns
        -------
        (obs_dict, state)
            ``obs_dict`` has keys ``"visual"`` (HWC uint8 RGB) and
            ``"proprio"`` (D,) float32.
        """
        obs, _info = self._venv.reset(seed=self._seed)
        return self._build_ret(obs)

    def step(self, action: np.ndarray):
        """Step the environment with a *pixel-space* action.

        Parameters
        ----------
        action:
            ``(dx, dy)`` in the existing keyboard/gamepad pixel space
            (max magnitude ≈ ``max_input``, default 60).

        Returns
        -------
        (obs_dict, reward, done, info)
            ``info["state"]`` is the full compact observation.
            ``info["effective_action"]`` is the action actually applied to the
            simulator (in ManiSkill-normalised space), suitable for the WM
            action history.
        """
        ms_action = self._pixel_to_maniskill(action)
        obs, reward, term, trunc, info = self._venv.step(ms_action)
        done = bool(term.item()) if isinstance(term, torch.Tensor) else bool(term)
        done = done or (
            bool(trunc.item()) if isinstance(trunc, torch.Tensor) else bool(trunc)
        )
        obs_dict, _state = self._build_ret(obs)
        info = dict(info)
        info["state"] = obs.squeeze(0).cpu().numpy().astype(np.float32)
        # The effective action (after fix_magnitude + scaled_delta) is what
        # the WM's action encoder was trained on.
        eff = self._wrapper.effective_action(
            torch.as_tensor(ms_action, device=self._device)
        )
        info["effective_action"] = (
            eff.squeeze(0).cpu().numpy().astype(np.float32)
        )
        return obs_dict, float(reward), done, info

    def render(self) -> np.ndarray:
        """Return the current RGB frame (HWC uint8)."""
        return self._render_frame()

    def close(self) -> None:
        self._venv.close()

    def seed(self, seed: int) -> None:
        self._seed = seed

    # -- internals ------------------------------------------------------------

    def _render_frame(self) -> np.ndarray:
        """Render → (H, W, 3) uint8 at ``render_size``."""
        rgb = self._venv.render()  # (1, H, W, 3) or (H, W, 3)
        rgb = torch.as_tensor(rgb, device="cpu")
        if rgb.ndim == 4:
            rgb = rgb[0]  # strip batch
        if rgb.shape[-1] == 3:
            pass  # HWC as expected
        elif rgb.shape[0] == 3:
            rgb = rgb.permute(1, 2, 0)  # CHW → HWC
        h, w = rgb.shape[:2]
        if h != self._render_size or w != self._render_size:
            rgb = rgb.float().permute(2, 0, 1).unsqueeze(0)  # 1CHW
            rgb = F.interpolate(
                rgb, size=(self._render_size, self._render_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0).permute(1, 2, 0)  # back to HWC
        return rgb.clamp(0, 255).byte().numpy()

    def _build_ret(self, obs):
        """Build (obs_dict, state) from the wrapper observation."""
        rgb = self._render_frame()
        obs_np = obs.squeeze(0).cpu().numpy().astype(np.float32)
        proprio = obs_np[:self.proprio_dim].copy()
        obs_dict = {"visual": rgb, "proprio": proprio}
        return obs_dict, obs_np.copy()

    def _pixel_to_maniskill(self, action: np.ndarray) -> np.ndarray:
        """Convert pixel-space ``(dx, dy)`` → batch-normalised ManiSkill action.

        Keyboard/gamepad produce ``(dx, dy)`` where full deflection ≈
        ``max_input`` (60).  We normalise by that, clamp to the unit disc,
        then scale to the ManiSkill normalised range.

        Formula (mirrors ``gamepad_teleop_pusht.py``):
            action = clip((dx / max_input) * (step_size / action_scale), -1, 1)
        """
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 2:
            raise ValueError(f"action must be 2D, got shape {a.shape}")
        dx, dy = a[0], a[1]

        dx_n = dx / self._max_input
        dy_n = dy / self._max_input
        # Clamp to unit disc (matching teleop fixed-magnitude behaviour).
        mag = np.sqrt(dx_n * dx_n + dy_n * dy_n)
        if mag > 1.0:
            dx_n /= mag
            dy_n /= mag

        scale = self._step_size / self._action_scale
        ms_dx = np.clip(dx_n * scale, -1.0, 1.0)
        ms_dy = np.clip(dy_n * scale, -1.0, 1.0)
        return np.array([ms_dx, ms_dy], dtype=np.float32).reshape(1, 2)
