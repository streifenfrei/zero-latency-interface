"""PushT-v1 with task-space (Cartesian) randomized EE start -- "exploring starts".

The frozen-policy runs showed a zero-advantage deadlock: from the single fixed EE
start the stick never reaches the T, so the only action-dependent reward term is
tiny and the value function predicts the (block-pose-determined) return perfectly,
collapsing advantages to ~0. Injecting reward-relevant *contact* states at reset
breaks that. The obvious knob -- ManiSkill's ``robot_init_qpos_noise`` -- adds
Gaussian noise in *joint* space, which drives the stick into table/self penetration
-> contact-buffer overflow -> NaN. Wrong tool.

This subclass instead samples the stick tip on a ring around the T and solves IK to
place it there, contact-ready from every direction.

Two GPU pitfalls this implementation is careful about (both bit earlier versions):
  1. Reads of live poses (``self.tee.pose``, ``end_link.pose``, ``get_qpos()``)
     inside ``_initialize_episode`` return stale/uninitialised buffers -- the GPU
     sim does not ``_gpu_apply_all``/``_gpu_fetch_all`` until *after* this method
     returns (sapien_env.py:958). On the first reset that segfaults. So we read
     NOTHING from the sim here: we mirror the base init to keep the tee xy as a
     tensor, and derive the rest EE pose from pure-torch forward kinematics.
  2. The pd_ee_pose controller drives IK in the *robot base* frame
     (``ee_pose_at_base``), not world -- despite the kinematics docstring. The
     robot root sits at world (-0.615, 0, 0), so a world-frame target lands the
     stick ~0.6 m off. We build the target in the base frame.

Registered as ``PushT-XYExplore-v1``. ``ee_init_radius=0`` delegates to the base
task verbatim, so it is a safe drop-in default.
"""

import numpy as np
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.push_t import PushTEnv
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

# panda_stick rest keyframe (push_t WhiteTableSceneBuilder.initialize) and the robot
# root's world pose (set right after). Both are fixed task constants, so we use them
# directly instead of reading them back from the sim during a reset.
_SEED_QPOS = [0.662, 0.212, 0.086, -2.685, -0.115, 2.898, 1.673]
_BASE_OFFSET = (-0.615, 0.0, 0.0)


@register_env("PushT-XYExplore-v1", max_episode_steps=100)
class PushTExploreEnv(PushTEnv):
    def __init__(self, *args,
                 ee_init_radius: float = 0.0,
                 ee_init_radius_min: float = 0.12,
                 ee_init_height: float = 0.04,
                 ik_iters: int = 50,
                 ik_alpha: float = 0.8,
                 ik_tol: float = 0.03,
                 **kwargs):
        """
        Args:
            ee_init_radius: outer radius (m) of the ring around the T on which the
                stick tip is sampled. 0 disables randomization (fixed start = base task).
            ee_init_radius_min: inner radius (m). Keep above the T's footprint
                (~0.1 m half-extent) so the stick never spawns inside the block.
            ee_init_height: world z (m) the stick tip starts at -- a touch above the
                table so init never penetrates; the policy's push_height lowers it.
            ik_iters / ik_alpha: iterated Levenberg-Marquardt IK budget / step size.
            ik_tol: max EE position error (m) for a solve to count as converged.
                Non-converged envs (finite but wrong -- e.g. unreachable ring angle)
                revert to the fixed start instead of spawning a bad/below-table pose.
        """
        self.ee_init_radius = ee_init_radius
        self.ee_init_radius_min = ee_init_radius_min
        self.ee_init_height = ee_init_height
        self.ik_iters = ik_iters
        self.ik_alpha = ik_alpha
        self.ik_tol = ik_tol
        self._qlim = None
        super().__init__(*args, **kwargs)

    @property
    def _default_sim_config(self):
        # Exploring starts put the stick on/near the block+table across all envs, so
        # total contacts blow past SAPIEN's default max_rigid_contact_count (2**19);
        # observed overflow needed >=602910 (IK) / >=882102 (the old joint noise).
        # Bump to 2**21 for headroom while keeping the base task's other GPU-mem dims.
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**18,
                max_rigid_contact_count=2**21,
            )
        )

    def _solve_ik(self, kin, target, q0):
        """Joint positions that put the controlled link at ``target`` (base-frame
        pose), seeded from ``q0`` (b, dof). GPU IK is one LM step per call so we
        iterate; CPU IK (pinocchio) solves in one call but is per-env / may be None.
        ``current_pose`` is left None so each call recomputes FK from the running q
        (base frame), matching the target frame."""
        if kin.use_gpu_ik:
            cfg = dict(type="levenberg_marquardt", solver_iterations=1,
                       alpha=self.ik_alpha)
            q = q0
            for _ in range(self.ik_iters):
                q = kin.compute_ik(target, q, is_delta_pose=False, solver_config=cfg)
            return q
        out = []
        for i in range(q0.shape[0]):
            sol = kin.compute_ik(target[i], q0[i:i + 1])
            out.append(sol[0] if sol is not None else q0[i])
        return torch.stack(out)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self.ee_init_radius <= 0:
            return super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)

            # --- mirror PushTEnv._initialize_episode (no sim reads; we keep tee xy) ---
            self.table_scene.initialize(env_idx)

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] += self.goal_offset[0]
            goal_xyz[:, 1] += self.goal_offset[1]
            goal_xyz[:, 2] = 1e-3
            self.goal_tee.set_pose(Pose.create_from_pq(
                p=goal_xyz, q=euler2quat(0, 0, self.goal_z_rot)))

            tee_xyz = goal_xyz.clone()
            tee_xyz[:, 0] += torch.rand(b) * self.tee_spawnbox_xlength + self.tee_spawnbox_xoffset
            tee_xyz[:, 1] += torch.rand(b) * self.tee_spawnbox_ylength + self.tee_spawnbox_yoffset
            tee_xyz[:, 2] = 0.04 / 2 + 1e-3
            ang = torch.rand(b) * (2 * torch.pi)
            tee_q = torch.zeros((b, 4))
            tee_q[:, 0] = (ang / 2).cos()
            tee_q[:, -1] = (ang / 2).sin()
            self.tee.set_pose(Pose.create_from_pq(p=tee_xyz, q=tee_q))

            marker = torch.zeros((b, 3))
            marker[:] = self.ee_starting_pos2D
            self.ee_goal_pos.set_pose(Pose.create_from_pq(
                p=marker, q=euler2quat(0, np.pi / 2, 0)))

            # --- place the stick on a ring around the tee, in the robot base frame ---
            kin = self.agent.controller.controllers["arm"].kinematics
            seed = torch.tensor(_SEED_QPOS, dtype=torch.float32).repeat(b, 1)
            # rest EE orientation (base frame) from pure-torch FK on the seed qpos
            rest_mat = kin.pk_chain.forward_kinematics(seed).get_matrix()
            rest_q = matrix_to_quaternion(rest_mat[:, :3, :3])

            theta = torch.rand(b) * (2 * torch.pi)
            r = self.ee_init_radius_min + torch.rand(b) * (
                self.ee_init_radius - self.ee_init_radius_min)
            tgt = torch.zeros((b, 3))
            tgt[:, 0] = tee_xyz[:, 0] - _BASE_OFFSET[0] + r * theta.cos()
            tgt[:, 1] = tee_xyz[:, 1] - _BASE_OFFSET[1] + r * theta.sin()
            tgt[:, 2] = self.ee_init_height - _BASE_OFFSET[2]

            q = self._solve_ik(kin, Pose.create_from_pq(tgt, rest_q), seed)

            # Safety net: revert any env whose solve is non-finite OR did not converge
            # to the fixed seed. Check convergence with pure-torch FK on the solved q
            # (no sim read) -- the isfinite test alone misses "finite but wrong" solves
            # (unreachable ring angle -> arm flung far / below the table). Those are the
            # ~2-4% off-ring envs; reverting them keeps every start either a clean ring
            # placement or the safe fixed start, never a penetrating one.
            ach_p = kin.pk_chain.forward_kinematics(q).get_matrix()[:, :3, 3]
            err = torch.linalg.norm(ach_p - tgt, dim=1)
            bad = (~torch.isfinite(q).all(dim=1)) | (err > self.ik_tol)
            q = torch.where(bad[:, None], seed, q)
            ferr = err[torch.isfinite(err)]
            self._dbg = dict(
                bad_frac=bad.float().mean().item(),
                err_mean=(ferr.mean().item() if ferr.numel() else float("nan")),
                err_max=(ferr.max().item() if ferr.numel() else float("nan")),
                fk_z_min=ach_p[:, 2].min().item(),
            )

            if self._qlim is None:  # static joint limits; cache once
                lim = self.agent.robot.get_qlimits()
                self._qlim = (lim[0] if lim.ndim == 3 else lim)
            q = torch.clamp(q, self._qlim[:, 0], self._qlim[:, 1])

            self.agent.reset(q)
