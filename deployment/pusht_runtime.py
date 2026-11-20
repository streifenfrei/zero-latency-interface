"""PushT runtime for the ManiSkill PushT deployment (vendored).

Trimmed runtime extraction of the sibling repo's ``rl_train_pusht.py``
(vendored 2026-11): contains only the shared deployment interface —
``PushTXYWrapper`` (the EE wrapper the study/ZLI/viz scripts drive,
including the ``ee_bounds`` workspace clamp), ``Agent`` (the PPO actor-
critic whose ``actor_mean`` head generates policy rollouts), and
``layer_init``.  The deprecated training loop, ``Args`` and the tyro/
wandb machinery are deliberately NOT included.

The original module header (mirrors gamepad_teleop_pusht.py):
    * Action space = 2D ``[dx, dy]`` in [-1, 1] — no z, no rotation;
      z and orientation are held by the ``pd_ee_pose`` controller.
    * Observation = compact proprioception + T-block state (ee xy,
      [ee vel], block xy, block cos/sin, [goal xy + cos/sin]).
    * The policy is a single-step MLP conditioned on the current obs.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.vector.utils import batch_space
from mani_skill.utils.geometry.rotation_conversions import (
    matrix_to_euler_angles, quaternion_invert, quaternion_multiply,
    quaternion_to_matrix)
from torch.distributions.normal import Normal

import deployment.pusht_explore  # noqa: F401  registers PushT-XYExplore-v1

class PushTXYWrapper(gym.Wrapper):
    def __init__(self, env, action_scale, push_height,
                 include_ee_vel, include_ee_rot, include_goal,
                 fix_push_height=True, fix_orientation=True, full_state=False,
                 control_mode="pd_ee_pose", fixed_magnitude=False, level_ee=False,
                 level_ee_gain=1.0, level_ee_rot_bound=0.1,
                 xy_absolute_target=False, xy_windup=0.05,
                 ee_bounds=None):
        super().__init__(env)
        self.base = env.unwrapped
        self.num_envs = self.base.num_envs
        self.device = self.base.device
        self.action_scale = action_scale
        self.fixed_magnitude = fixed_magnitude
        self.push_height = push_height
        self.include_ee_vel = include_ee_vel
        self.include_ee_rot = include_ee_rot
        self.include_goal = include_goal
        self.fix_push_height = fix_push_height
        self.fix_orientation = fix_orientation
        self.full_state = full_state
        self.control_mode = control_mode
        self.level_ee = level_ee
        self.level_ee_gain = level_ee_gain
        self.level_ee_rot_bound = level_ee_rot_bound
        # Closed-loop xy: accumulate the policy's relative deltas into an ABSOLUTE xy target the
        # controller tracks each step (pd_ee_pose). Position errors (e.g. orientation coupling)
        # then get corrected instead of accumulating -> no drift. xy_windup caps how far the target
        # may lead the measured tip (anti-windup for a blocked arm). pd_ee_pose mode only.
        self.xy_absolute_target = xy_absolute_target
        self.xy_windup = xy_windup
        self._target_xy = None   # (B,2) absolute xy target, lazily seeded at reset
        # EE workspace bounds in WORLD table coordinates (xmin, xmax, ymin,
        # ymax): the commanded tip position is clamped to this rectangle each
        # step.  The controllers work in the robot BASE frame — root at world
        # (-0.615, 0, 0), axes aligned — so store the corners translated once.
        self.ee_bounds = ee_bounds
        self._ee_lo = self._ee_hi = None
        if ee_bounds is not None:
            xmin, xmax, ymin, ymax = (float(v) for v in ee_bounds)
            if not (xmin < xmax and ymin < ymax):
                raise ValueError(
                    f"ee_bounds must satisfy xmin < xmax and ymin < ymax, "
                    f"got {ee_bounds!r}")
            self._ee_lo = torch.tensor([xmin + 0.615, ymin],
                                       device=self.device)
            self._ee_hi = torch.tensor([xmax + 0.615, ymax],
                                       device=self.device)
        self.arm = self.base.agent.controller.controllers["arm"]
        # Make the rotation channel an ABSOLUTE hold while xy stays a relative delta: widen the
        # delta controller's rotation bound from its default +-0.1 rad/step (rot_lower) to
        # +-level_ee_rot_bound, so the per-step orientation correction isn't clamped before it can
        # cancel the tilt the arm picks up while reaching. The controller reads rot_lower LIVE in
        # _clip_and_scale_action (rot_action *= config.rot_lower), so this mutation takes effect
        # immediately; pos bounds (+-0.1 m) and the wrapper's 2-D action space are untouched, so xy
        # behaviour and the learned policy interface are unchanged. Only in delta mode w/ leveling.
        if level_ee and "delta" in control_mode:
            self.arm.config.rot_lower = -abs(level_ee_rot_bound)
            self.arm.config.rot_upper = abs(level_ee_rot_bound)
        self._hold_euler = None  # (1,3) upright orientation, captured on first reset
        self._hold_quat = None   # (1,4) same, as a quaternion (for delta-mode leveling)
        self._last_tilt_rad = 0.0    # diagnostic: mean orientation error at the last step
        self._last_tilt_max_rad = 0.0
        self._last_sat_frac = 0.0
        self._level_ee_logged = False
        self._level_ee_steps = 0
        # Per-step rotation budget of the delta controller: |rot| <= |rot_lower| rad. Used to
        # cap the leveling correction and to convert it to normalized action units. Scalar for
        # panda_stick (rot_lower=-0.1); fall back to 0.1 if the controller lacks the field.
        self._rot_lower = float(np.ravel(getattr(self.arm.config, "rot_lower", -0.1))[0])

        # 2D normalized action space (dx, dy)
        self.single_action_space = gym.spaces.Box(
            low=-np.ones(2, np.float32), high=np.ones(2, np.float32), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        # observation space (flat vector)
        if full_state:
            # pass the env's full privileged state observation straight through
            self.single_observation_space = self.base.single_observation_space
            self.obs_dim = int(np.prod(self.single_observation_space.shape))
        else:
            dim = 2  # ee xy
            if include_ee_vel:
                dim += 2
            if include_ee_rot:
                dim += 2
            dim += 2  # block xy
            dim += 2  # block cos,sin
            if include_goal:
                dim += 4  # goal xy + cos,sin
            self.obs_dim = dim
            self.single_observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def _capture_orientation(self):
        if self._hold_euler is None:
            # Re-fetch the arm controller (a reconfigure can swap the object out from under us).
            self.arm = self.base.agent.controller.controllers["arm"]
            q = self.arm.ee_pose_at_base.q  # (B,4)
            eul = matrix_to_euler_angles(quaternion_to_matrix(q), "XYZ")  # (B,3)
            # all envs share the same rest orientation; keep one row, broadcast later
            self._hold_euler = eul[0:1].detach().to(self.device)
            self._hold_quat = q[0:1].detach().to(self.device)  # (1,4) upright target

    def _leveling_rot(self):
        """Normalized rotation-delta command (B,3) for ``full[:, 3:6]`` in delta mode that
        drives the ee back to its captured upright orientation ``_hold_quat``.

        In the GPU delta path the controller turns ``full[:, 3:6]`` into a root-frame euler-XYZ
        delta ``e`` (radians) via ``e = full[:, 3:6] * rot_lower`` (norm-clipped to <= |rot_lower|),
        applied as ``q_target ~ exp(e) (x) q_cur``. So the delta that re-levels the stick is
        ``e = log(q_hold (x) q_cur^-1)``, scaled by ``level_ee_gain`` then capped to the per-step
        budget; we return ``e / rot_lower`` so the controller's own scaling reproduces it exactly.

        ``level_ee_gain`` is a P-gain on the measured orientation error. gain=1 commands exactly
        the error each step; a single first-order IK step + PD only *partially* executes that
        against contact, so the stick still drifts. gain>1 over-drives the correction so it
        SATURATES the controller's +-|rot_lower| rad/step budget whenever there is meaningful
        tilt -- i.e. drive hard toward perpendicular every step. The cap + per-step re-measurement
        keep it stable (it can never command more than the budget, and reads the true tilt each
        step). The hard ceiling is |rot_lower| rad/step of correction: if contact tilts the stick
        faster than that, no gain can hold it and an absolute-orientation controller is needed.

        Sets ``self._last_tilt_rad`` (mean measured error) for the one-time diagnostic log."""
        self.arm = self.base.agent.controller.controllers["arm"]   # refresh (reconfigure-safe)
        q_cur = self.arm.ee_pose_at_base.q                         # (B,4)
        q_err = quaternion_multiply(self._hold_quat,               # (1,4) broadcast
                                    quaternion_invert(q_cur))      # -> (B,4)
        e = matrix_to_euler_angles(quaternion_to_matrix(q_err), "XYZ")  # (B,3) measured error
        err_norm = e.norm(dim=-1)                                  # (B,) per-env error (rad)
        self._last_tilt_rad = float(err_norm.mean())               # diagnostic (pre-gain, mean)
        self._last_tilt_max_rad = float(err_norm.max())            # diagnostic (pre-gain, max)
        e = e * self.level_ee_gain                                 # P-gain (over-drive vs contact)
        # Cap the correction to the controller's per-step rotation budget (norm-preserving).
        budget = abs(self._rot_lower)
        n = e.norm(dim=-1, keepdim=True)
        # Fraction of envs whose desired correction EXCEEDS the per-step budget (i.e. saturated:
        # the cap, not the gain, is binding -- raising the gain there changes nothing).
        self._last_sat_frac = float((n.squeeze(-1) > budget).float().mean())
        e = e * torch.clamp(budget / n.clamp_min(1e-9), max=1.0)
        return e / self._rot_lower  # normalized command; controller re-multiplies by rot_lower

    def _log_level_ee(self):
        """Verifiable, throttled diagnostic for the leveling correction. Confirms it is ACTIVE
        (not silently falling back to drot=0) and -- the key question for 'gain does nothing' --
        reports the live orientation error and what fraction of envs are SATURATED (desired
        correction > the per-step cap, where the gain is irrelevant). Grep the .out for 'level_ee'.

          - mean/max tilt high + sat~100%  -> hit the delta-mode ceiling; need absolute-orientation.
          - tilt low                        -> leveling is holding; the visible drift is elsewhere.
        """
        deg = 180.0 / np.pi
        if not getattr(self, "_level_ee_logged", False):
            self._level_ee_logged = True
            print(f"[PushTXYWrapper] level_ee ACTIVE gain={self.level_ee_gain} "
                  f"rot_lower={self._rot_lower} initial_mean_tilt={self._last_tilt_rad*deg:.2f}deg",
                  flush=True)
        self._level_ee_steps += 1
        if self._level_ee_steps % 200 == 0:
            print(f"[PushTXYWrapper] level_ee step={self._level_ee_steps} "
                  f"mean_tilt={self._last_tilt_rad*deg:6.2f}deg "
                  f"max_tilt={self._last_tilt_max_rad*deg:6.2f}deg "
                  f"saturated={self._last_sat_frac*100:5.1f}% (gain={self.level_ee_gain})",
                  flush=True)

    def _build_obs(self):
        b = self.base
        tcp = b.agent.tcp
        parts = [tcp.pose.p[:, :2]]
        if self.include_ee_vel:
            parts.append(tcp.linear_velocity[:, :2])
        if self.include_ee_rot:
            ze = b.quat_to_z_euler(tcp.pose.q)
            parts.append(torch.stack([ze.cos(), ze.sin()], dim=-1))
        parts.append(b.tee.pose.p[:, :2])
        zt = b.quat_to_z_euler(b.tee.pose.q)
        parts.append(torch.stack([zt.cos(), zt.sin()], dim=-1))
        if self.include_goal:
            parts.append(b.goal_tee.pose.p[:, :2])
            zg = b.quat_to_z_euler(b.goal_tee.pose.q)
            parts.append(torch.stack([zg.cos(), zg.sin()], dim=-1))
        return torch.cat(parts, dim=-1).float()

    def reset(self, *, seed=None, options=None):
        raw, info = self.env.reset(seed=seed, options=options)
        self._capture_orientation()
        if self.xy_absolute_target:
            # Seed/re-seed the absolute xy target to the (re)spawned tip so each fresh episode
            # starts tracking from its actual start pose. Honour partial auto-resets (env_idx) so
            # only the just-reset envs snap; the others keep their accumulated target.
            cur_xy = self.arm.ee_pose_at_base.p[:, :2]
            idx = options.get("env_idx") if isinstance(options, dict) else None
            if self._target_xy is None or idx is None:
                self._target_xy = cur_xy.clone()
            else:
                self._target_xy[idx] = cur_xy[idx]
        return (raw if self.full_state else self._build_obs()), info

    def _reward_components(self):
        """The 3 dense-reward terms (unnormalized), for LOGGING only -- the actual
        training reward is the env's stock normalized_dense (see step()). Mirrors
        ManiSkill's PushT dense reward (push_t.py:516-530) exactly so the logged parts
        sum (/3) to the training reward."""
        b = self.base
        zt = b.quat_to_z_euler(b.tee.pose.q)
        rot = (((zt - b.goal_z_rot).cos() + 1) / 2) ** 2 / 2          # [0, 0.5]
        d_xy = torch.linalg.norm(
            b.tee.pose.p[:, :2] - b.goal_tee.pose.p[:, :2], dim=1)
        pos = (1 - torch.tanh(5 * d_xy)) ** 2 / 2                     # [0, 0.5]
        d_tcp = torch.linalg.norm(b.tee.pose.p - b.agent.tcp.pose.p, dim=1)
        prox = (1 - torch.tanh(5 * d_tcp)).sqrt() / 20               # [0, 0.05]
        return {"orientation": rot, "position": pos,
                "proximity": prox, "total": rot + pos + prox}

    def fix_magnitude(self, action):
        """If ``fixed_magnitude``, project each command onto the unit circle so the policy
        controls only DIRECTION at a constant speed (||a||=1 -> action_scale m/step). Zero/
        near-zero commands are left at zero, preserving the idle/no-move atom (a unit vector
        has no well-defined direction there anyway). A no-op when ``fixed_magnitude`` is off."""
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if not self.fixed_magnitude:
            return action
        norm = action.norm(dim=-1, keepdim=True)
        # 1/norm where the command is non-trivial, else 0 -> idle stays idle (no div-by-zero).
        inv = torch.where(norm > 1e-6, 1.0 / norm, torch.zeros_like(norm))
        return action * inv

    def effective_action(self, action):
        """The action that actually drives a transition: direction-fix then speed-scale,
        ``scaled_delta(fix_magnitude(command))``. This is the single source of truth used by
        ``step()`` and the teleop-comparable quantity to STORE for a transition (data provider /
        measure script). With ``fixed_magnitude`` on, every non-idle action has |.|=action_scale/0.1
        per the unit direction; with it off this is just ``scaled_delta``."""
        return self.scaled_delta(self.fix_magnitude(action))

    def _clamp_ee_xy(self, xy):
        """Clamp a commanded EE xy (base frame, metres) to the workspace
        bounds (world frame); pass-through when no bounds are set."""
        if self._ee_lo is None:
            return xy
        return xy.clamp(self._ee_lo, self._ee_hi)

    def scaled_delta(self, action):
        """Map a normalized [-1,1] xy *command* to the controller's EFFECTIVE delta input.

        action_scale = meters/step at full deflection (== the gamepad teleop --speed): the
        controller maps its [-1,1] input to +-0.1 m, so the effective action is
        clip(action * action_scale/0.1, -1, 1) -- exactly the teleop formula
        clip(stick*speed/0.1, -1, 1). action_scale=0.1 is the identity (full authority).

        This is the teleop-comparable quantity: a policy emits commands in [-1,1] regardless
        of speed, but the EFFECTIVE action it applies is gated to +-action_scale/0.1. Use this
        (not the raw command) to compare a policy's action range against teleop or to store
        the action that actually drove a transition."""
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        return (action * (self.action_scale / 0.1)).clamp(-1.0, 1.0)

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action = action.clamp(-1.0, 1.0)
        full = torch.zeros((self.num_envs, 6), device=self.device)
        if self.control_mode == "pd_ee_delta_pose":
            # local Jacobian delta-IK: dz/drot = 0 hold z + orientation RELATIVE (the mode
            # that learned, baseline D). The command is scaled to the effective delta by
            # scaled_delta() (action_scale = meters/step at full deflection == teleop --speed);
            # action_scale=0.1 reproduces the canonical full-authority delta (old default), so
            # existing checkpoints are unchanged. push_height / fix_* unused in delta mode.
            full[:, 0:2] = self.effective_action(action)
            # level_ee: replace the drift-prone drot=0 with an active correction back to the
            # captured upright orientation, so the stick stays perpendicular to the table.
            if self.level_ee:
                if self._hold_quat is None:
                    # Lazy capture: reset() may not have propagated through the vector-env /
                    # RecordEpisode wrapper chain (which silently leaves drot=0 -> the stick
                    # drifts and the gain looks like it does nothing). Capture on first step.
                    self._capture_orientation()
                full[:, 3:6] = self._leveling_rot()
                self._log_level_ee()
            # Clamp the commanded tip position to the workspace: the delta
            # controller maps a normalized ±1 xy to ±0.1 m, so convert
            # target → clamp → back to normalized delta.
            if self._ee_lo is not None:
                cur_xy = self.arm.ee_pose_at_base.p[:, :2]
                full[:, 0:2] = (self._clamp_ee_xy(
                    cur_xy + full[:, 0:2] * 0.1) - cur_xy) / 0.1
        else:  # pd_ee_pose: absolute target pose
            cur = self.arm.ee_pose_at_base
            delta = self.fix_magnitude(action) * self.action_scale  # (B,2) m, relative intent
            if self.xy_absolute_target:
                # Closed-loop: accumulate deltas into an absolute target the controller tracks, so
                # coupling/disturbance is corrected each step instead of accumulating (no drift).
                if self._target_xy is None:
                    self._target_xy = cur.p[:, :2].clone()
                self._target_xy = self._clamp_ee_xy(self._target_xy + delta)
                # anti-windup: don't let the target lead the measured tip by more than xy_windup
                self._target_xy = cur.p[:, :2] + (self._target_xy - cur.p[:, :2]).clamp(
                    -self.xy_windup, self.xy_windup)
                full[:, 0:2] = self._target_xy
            else:
                full[:, 0:2] = self._clamp_ee_xy(
                    cur.p[:, :2] + delta)  # relative-to-measured increment
            # z: fixed absolute push height, or hold the current measured height (relative)
            full[:, 2] = self.push_height if self.fix_push_height else cur.p[:, 2]
            # orientation: fixed captured upright pose, or hold the current measured orientation
            if self.fix_orientation:
                full[:, 3:6] = self._hold_euler  # broadcast (1,3) -> (B,3)
            else:
                full[:, 3:6] = matrix_to_euler_angles(quaternion_to_matrix(cur.q), "XYZ")
        # Use the env's stock normalized_dense reward (reward_mode set in make_env) --
        # the proven PushT signal. _reward_components is recomputed only for logging.
        raw, reward, term, trunc, info = self.env.step(full)
        info = dict(info)
        info["reward_components"] = self._reward_components()
        return (raw if self.full_state else self._build_obs()), reward, term, trunc, info


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------
def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    """MLP actor-critic. Conditions only on the current observation (single-step)."""

    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        # 3 hidden layers, matching ManiSkill's proven PushT baseline (a 2-hidden-layer
        # net plateaued ~31 deg short of the goal orientation; the extra depth + ent_coef=0
        # is what let the baseline trainer solve it -- see the A..E diagnostic).
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        mean = torch.nan_to_num(self.actor_mean(x))
        std = self.actor_logstd.expand_as(mean).exp()
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, self.critic(x).squeeze(-1)


# --------------------------------------------------------------------------------------
# Train
