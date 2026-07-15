"""On-device online data provision for world-model training on PushT.

Generates training data **in-process, on the training device** by rolling out a
trained PPO policy across parallel ManiSkill PushT environments with a noise
ladder. Each environment is assigned a corruption probability on a linear ramp
from pure-policy (env 0) to pure-noise (last env), creating a dataset at graded
distances from the policy distribution.

The core architecture is a producer-consumer pattern: a background thread
continuously rolls out trajectory windows and pushes chunks to a thread-safe
queue; the main training thread pulls from the queue via an IterableDataset
that the Trainer's DataLoader wraps.

Design mirrors ``zero_latency_interface/model/data.py`` (PushTNoiseLadderProvider)
but adapted for the dino_wm data format: each yielded sample is a (T, ...) chunk
(not a pre-assembled batch), actions are concatenated across ``frameskip``
sub-steps matching ``TrajSlicerDataset``, and images are normalized to [-1, 1].

The policy checkpoint, env wrapper (PushTXYWrapper), Agent class, and
exploring-starts env registration are imported at runtime from the sibling
``zero_latency_interface`` repository via ``sys.path`` insertion — no code
duplication.
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import sys
import threading
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from dino_wm.datasets.traj_dset import TrajDataset

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Producer-consumer infrastructure: provider, iterable dataset, empty traj stub
# ---------------------------------------------------------------------------

class PushTOnlineProvider:
    """Roll out a trained PushT policy across ``num_envs`` parallel envs with a
    per-env noise probability that increases linearly across the batch.

    Env *k* (of *N*) gets corruption probability
    ``p_k = lerp(noise_p_min, noise_p_max, k/(N-1))``. At each sub-step, with
    probability ``p_k`` the policy action is replaced by a corruption action
    (random direction or idle, per ``noise_kind``). Env 0 is pure policy, the
    last env is pure noise, and the intermediate envs span the spectrum.

    The policy and env are reconstructed from the checkpoint's saved ``args``
    using the ``PushTXYWrapper`` / ``Agent`` from the sibling
    ``zero_latency_interface`` repository.
    """

    def __init__(
        self,
        policy_checkpoint: str,
        zero_latency_root: str,
        num_envs: int = 64,
        chunk_length: int = 4,
        frameskip: int = 5,
        img_size: int = 224,
        val_ratio: float = 0.1,
        noise_kind: str = "random",
        idle_fraction: float = 0.5,
        noise_p_min: float = 0.0,
        noise_p_max: float = 1.0,
        max_episode_steps: int = 100,
        device: str = "cuda",
        sim_backend: str = "physx_cuda",
        seed: int = 0,
        buffer_size: int | None = None,
    ) -> None:
        if noise_kind not in ("random", "idle", "mixed"):
            raise ValueError(
                f"noise_kind must be 'random', 'idle' or 'mixed', got {noise_kind!r}"
            )

        self.policy_checkpoint = policy_checkpoint
        self.zero_latency_root = zero_latency_root
        self.num_envs = num_envs
        self.chunk_length = chunk_length
        self.frameskip = frameskip
        self.img_size = img_size
        self.val_ratio = val_ratio
        self.noise_kind = noise_kind
        self.idle_fraction = idle_fraction
        self.noise_p_min = noise_p_min
        self.noise_p_max = noise_p_max
        self.max_episode_steps = max_episode_steps
        self.device = torch.device(device)
        self.sim_backend = sim_backend
        self.seed = seed
        self.buffer_size = (
            buffer_size if buffer_size is not None else max(num_envs * 4, 256)
        )

        # Lazily built in setup().
        self._venv = None
        self._wrapper = None
        self._agent = None
        self._obs = None
        self._p: Tensor | None = None  # (N,) per-env corruption probability

        # Dimension attributes (set during setup, read by iterable dataset).
        self.action_dim: int = 0
        self.proprio_dim: int = 0
        self.state_dim: int = 0

        # Producer-consumer plumbing.
        self._queue: queue.Queue | None = None
        self._stop_event: threading.Event | None = None
        self._producer_thread: threading.Thread | None = None
        self._producer_ready: threading.Event | None = None

        # Consumer-local buffers + routing (main thread only).
        self._train_buf: list[tuple[Tensor, Tensor, Tensor]] = []
        self._val_buf: list[tuple[Tensor, Tensor, Tensor]] = []
        self._emit: int = 0
        self._val_every = round(1.0 / val_ratio) if val_ratio and val_ratio > 0 else 0
        self._max_val_buf = 64  # cap val buffer so undrained GPU memory is bounded
        self._max_train_buf = max(self.buffer_size, 512)

        # RNG (producer-thread-owned — created in _producer_loop).
        self._gen: torch.Generator | None = None

    # -- lifecycle ---------------------------------------------------------

    def setup(self) -> None:
        """Build the ManiSkill env, load the policy, start the producer thread."""
        if self._venv is not None:
            return

        # Ensure the zero_latency_interface repo root is importable so we can
        # import its top-level scripts (rl_train_pusht, pusht_explore).
        zl_root = os.path.abspath(self.zero_latency_root)
        if zl_root not in sys.path:
            sys.path.insert(0, zl_root)

        # Lazy heavy imports: keep the dino_wm package importable without
        # ManiSkill when the offline pusht config is used.
        import gymnasium as gym
        import mani_skill  # noqa: F401  registers PushT-v1
        import pusht_explore  # noqa: F401  registers PushT-XYExplore-v1
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
        from rl_train_pusht import Agent, PushTXYWrapper

        ckpt = torch.load(
            self.policy_checkpoint, map_location=self.device, weights_only=False
        )
        targs = ckpt["args"]  # training-time Args dict

        full_state = targs.get("full_state", False)
        if full_state:
            obs_mode = "state"
        else:
            obs_mode = "none"  # PushTXYWrapper builds compact policy obs

        control_mode = targs.get("control_mode", "pd_ee_delta_pose")
        action_scale = targs.get("action_scale", 0.1)

        env = gym.make(
            "PushT-XYExplore-v1",
            num_envs=self.num_envs,
            obs_mode=obs_mode,
            control_mode=control_mode,
            reward_mode="none",
            render_mode="rgb_array",
            robot_uids="panda_stick",
            ee_init_radius=targs.get("ee_init_radius", 0.0),
            ee_init_radius_min=targs.get("ee_init_radius_min", 0.12),
            sim_backend=self.sim_backend,
            max_episode_steps=self.max_episode_steps,
        )
        env = PushTXYWrapper(
            env,
            action_scale=action_scale,
            push_height=targs["push_height"],
            include_ee_vel=not targs["no_ee_vel"],
            include_ee_rot=targs["ee_rot"],
            include_goal=not targs["no_goal"],
            fix_push_height=targs.get("fix_push_height", True),
            fix_orientation=targs.get("fix_orientation", True),
            full_state=full_state,
            control_mode=control_mode,
            fixed_magnitude=targs.get("fixed_magnitude", False),
            level_ee=targs.get("level_ee", False),
            level_ee_gain=targs.get("level_ee_gain", 1.0),
            level_ee_rot_bound=targs.get("level_ee_rot_bound", 0.1),
            xy_absolute_target=targs.get("xy_absolute_target", False),
            xy_windup=targs.get("xy_windup", 0.05),
        )
        venv = ManiSkillVectorEnv(
            env, auto_reset=True, ignore_terminations=False, record_metrics=False
        )

        agent = Agent(ckpt["obs_dim"], ckpt["act_dim"], targs["hidden_size"]).to(
            self.device
        )
        agent.load_state_dict(ckpt["agent"])
        agent.eval()

        self._venv, self._wrapper, self._agent = venv, env, agent

        # Linear corruption-probability ladder: env k → lerp(p_min, p_max, k/(N-1)).
        self._p = torch.linspace(
            self.noise_p_min, self.noise_p_max, self.num_envs, device=self.device
        )

        # Set dimension attributes that the Trainer reads for encoder init.
        # action_dim: raw dim × frameskip (matches TrajSlicerDataset concat).
        self.action_dim = ckpt["act_dim"] * self.frameskip
        # proprio: ee_xy + (optional) ee_vel — first N dims of the compact obs.
        self.proprio_dim = 2 if targs.get("no_ee_vel", False) else 4
        self.state_dim = ckpt["obs_dim"]

        # Initial reset.
        obs, _ = venv.reset(seed=self.seed)
        self._obs = obs

        log.info(
            "PushTOnlineProvider: %d envs, p ∈ [%.3f, %.3f], noise=%s, "
            "control_mode=%s, T=%d, fk=%d, action_dim=%d, proprio_dim=%d, "
            "state_dim=%d, action_scale=%.4f",
            self.num_envs,
            self.noise_p_min,
            self.noise_p_max,
            self.noise_kind,
            control_mode,
            self.chunk_length,
            self.frameskip,
            self.action_dim,
            self.proprio_dim,
            self.state_dim,
            action_scale,
        )

        # Start producer thread.
        self._queue = queue.Queue(maxsize=self.buffer_size)
        self._stop_event = threading.Event()
        self._producer_ready = threading.Event()
        self._stop_event.clear()
        self._producer_ready.clear()
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            name="data-producer",
            daemon=True,
        )
        self._producer_thread.start()
        self._producer_ready.wait()  # block until CUDA context is bound
        log.info("Producer thread started (buffer_size=%d)", self.buffer_size)

    def close(self) -> None:
        """Signal producer to stop, join, drain queue, release env."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._producer_thread is not None and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=30)
            if self._producer_thread.is_alive():
                log.warning(
                    "Data producer thread did not exit within 30 s — "
                    "continuing with cleanup"
                )
        drained = 0
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    drained += 1
                except queue.Empty:
                    break
        if drained:
            log.debug("Drained %d leftover chunks from producer queue", drained)
        if self._venv is not None:
            self._venv.close()
            self._venv = self._wrapper = self._agent = self._obs = None

    # -- producer thread --------------------------------------------------

    def _producer_loop(self) -> None:
        """Background thread: continuously roll out windows, push chunks."""
        try:
            if self.device.type == "cuda":
                torch.cuda.set_device(
                    self.device.index if self.device.index is not None else 0
                )
            self._gen = torch.Generator(device=self.device).manual_seed(self.seed)
        except Exception:
            log.exception("Producer thread failed to bind CUDA context")
            self._push_sentinel()
            self._producer_ready.set()
            return

        self._producer_ready.set()
        try:
            while not self._stop_event.is_set():
                self._rollout_window()
        except Exception:
            log.exception("Producer thread crashed — pushing sentinel")
            self._push_sentinel()
            raise

    def _push_sentinel(self) -> None:
        """Push a ``None`` sentinel so a blocking ``get_chunk`` unblocks."""
        try:
            if self._queue is not None:
                self._queue.put(None, timeout=5)
        except Exception:
            pass

    # -- rollout + noise injection ---------------------------------------

    def _random_directions(self, shape: tuple[int, ...]) -> Tensor:
        """Uniform random unit vectors of shape ``(..., A)``."""
        g = torch.randn(shape, device=self.device, generator=self._gen)
        return g / g.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _policy_action(self) -> Tensor:
        """Deterministic policy action for the current obs."""
        obs_t = torch.nan_to_num(
            torch.as_tensor(self._obs, dtype=torch.float32, device=self.device)
        )
        return self._agent.actor_mean(obs_t).clamp(-1.0, 1.0)

    def _apply_noise(self, a: Tensor) -> Tensor:
        """Per-env: with prob ``p_k`` replace policy action with corruption."""
        n = self.num_envs
        replace = torch.rand(n, device=self.device, generator=self._gen) < self._p
        if self.noise_kind == "idle":
            noise = torch.zeros_like(a)
        else:
            noise = self._random_directions(a.shape)
            if self.noise_kind == "mixed":
                idle = (
                    torch.rand(n, device=self.device, generator=self._gen)
                    < self.idle_fraction
                )
                noise[idle] = 0.0
        return torch.where(replace.unsqueeze(-1), noise, a)

    def _render_frames(self) -> Tensor:
        """Current scene render as ``(N, 3, H, W)`` uint8 on device."""
        f = torch.as_tensor(self._venv.render(), device=self.device)
        if f.ndim == 4 and f.shape[-1] == 3:  # (N, H, W, 3) → (N, 3, H, W)
            f = f.permute(0, 3, 1, 2)
        f = f.contiguous()
        # Resize to target resolution.
        f = (
            torch.nn.functional.interpolate(
                f.float(),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            .round_()
            .clamp_(0, 255)
            .to(torch.uint8)
        )
        return f

    def _extract_state(self) -> Tensor:
        """Current compact observation as ``(N, S)`` float32."""
        return torch.as_tensor(self._obs, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _rollout_window(self) -> None:
        """Step the envs for ``chunk_length`` macro frames, each spanning
        ``frameskip`` sub-steps. Push valid (non-boundary-crossing) chunks to
        the shared queue.

        Each chunk is ``(visual, action, state)`` where:
        - visual: ``(T_macro, 3, H, W)`` uint8
        - action: ``(T_macro, A * frameskip)`` float32  (sub-actions concatenated)
        - state:  ``(T_macro, S)`` float32  (compact policy observation)
        """
        T = self.chunk_length
        fk = self.frameskip
        total_steps = T * fk  # total env steps in this window

        frames: list[Tensor] = []
        actions: list[Tensor] = []
        states: list[Tensor] = []
        done_acc = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for m in range(T):
            # Record observation BEFORE stepping.
            frames.append(self._render_frames())   # (N, 3, H, W) uint8
            states.append(self._extract_state())    # (N, S) float32

            macro_action: list[Tensor] = []
            for s in range(fk):
                step_idx = m * fk + s
                cmd = self._apply_noise(self._policy_action())  # (N, 2)
                eff = self._wrapper.effective_action(cmd)       # (N, 2) in [-1,1]
                macro_action.append(eff)
                self._obs, _, term, trunc, _ = self._venv.step(cmd)
                # Track dones on all sub-steps except the absolute last one.
                if step_idx < total_steps - 1:
                    d = torch.as_tensor(term, device=self.device, dtype=torch.bool) \
                        | torch.as_tensor(trunc, device=self.device, dtype=torch.bool)
                    done_acc |= d

            # Concatenate fk sub-actions → (N, A * fk)
            actions.append(torch.cat(macro_action, dim=-1))

        fchunk = torch.stack(frames, dim=1)    # (N, T, 3, H, W) uint8
        achunk = torch.stack(actions, dim=1)   # (N, T, A*fk) float32
        schunk = torch.stack(states, dim=1)    # (N, T, S) float32

        # Push individual (non-done) env chunks to the queue.
        keep = (~done_acc).nonzero(as_tuple=False).flatten().tolist()
        for i in keep:
            self._queue.put(
                (
                    fchunk[i].contiguous(),
                    achunk[i].contiguous(),
                    schunk[i].contiguous(),
                )
            )

    # -- consumer side (main thread) -------------------------------------

    def get_chunk(
        self, validation: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return a single trajectory chunk ``(visual, action, state)`` from the
        producer queue, routing entries to train/val buffers per ``val_ratio``."""
        if self._venv is None:
            raise RuntimeError("setup() must be called before get_chunk().")
        buf = self._val_buf if validation else self._train_buf

        # Drain the shared queue into consumer-local buffers until the
        # requested split has an entry to return.
        while not buf:
            entry = self._queue.get()  # blocks until producer yields a chunk
            if entry is None:
                raise RuntimeError(
                    "Data producer thread crashed — see log above for details."
                )
            if self._val_every and self._emit % self._val_every == 0:
                self._val_buf.append(entry)
                if len(self._val_buf) > self._max_val_buf:
                    # Overflow oldest val entry to train to bound GPU memory.
                    self._train_buf.append(self._val_buf.pop(0))
            else:
                self._train_buf.append(entry)
                if len(self._train_buf) > self._max_train_buf:
                    self._train_buf.pop(0)
            self._emit += 1

        return buf.pop(0)


# ---------------------------------------------------------------------------
# Iterable dataset adapter
# ---------------------------------------------------------------------------

class PushTOnlineIterableDataset(IterableDataset):
    """Wraps a :class:`PushTOnlineProvider` as a torch ``IterableDataset``
    whose ``__iter__`` yields individual training samples compatible with the
    dino_wm :class:`~train.Trainer`.

    Each yielded item is ``(obs_dict, action, state)`` where:
    - ``obs_dict`` = ``{'visual': (T, 3, H, W) float [-1,1],
                         'proprio': (T, P) float}``
    - ``action`` = ``(T, A * frameskip)`` float
    - ``state``  = ``(T, S)`` float
    """

    def __init__(
        self,
        provider: PushTOnlineProvider,
        validation: bool = False,
        samples_per_epoch: int = 50000,
    ) -> None:
        self._provider = provider
        self._validation = validation
        self._samples_per_epoch = samples_per_epoch

        # Copy dimension attributes so the Trainer can read them.
        self.action_dim = provider.action_dim
        self.proprio_dim = provider.proprio_dim
        self.state_dim = provider.state_dim

    def __len__(self) -> int:
        return self._samples_per_epoch

    def __iter__(self) -> Iterator[tuple[dict[str, Tensor], Tensor, Tensor]]:
        for _ in range(self._samples_per_epoch):
            visual, action, state = self._provider.get_chunk(
                validation=self._validation
            )
            # Normalize images: uint8 [0,255] → float [-1,1]
            # Matches defaults_transform: Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            visual = visual.float() / 255.0  # [0, 1]
            visual = (visual - 0.5) / 0.5    # [-1, 1]

            # Extract proprio from the first proprio_dim dimensions of state.
            proprio = state[:, : self.proprio_dim].float()

            obs = {"visual": visual, "proprio": proprio}
            yield obs, action.float(), state.float()


# ---------------------------------------------------------------------------
# Empty trajectory dataset (satisfies traj_dsets interface)
# ---------------------------------------------------------------------------

class PushTOnlineTrajDataset(TrajDataset):
    """Empty trajectory dataset — openloop rollout evaluation is skipped when
    ``len(traj_dset) == 0`` (the Trainer already guards this)."""

    def __init__(self) -> None:
        self.proprio_dim = 0
        self.action_dim = 0
        self.state_dim = 0

    def get_seq_length(self, idx: int) -> int:
        raise IndexError("Empty dataset")

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int):
        raise IndexError("Empty dataset")


# ---------------------------------------------------------------------------
# Hydra factory function
# ---------------------------------------------------------------------------

# Module-level registry for atexit cleanup.
_providers: list[PushTOnlineProvider] = []


def _cleanup_providers() -> None:
    for p in _providers:
        try:
            p.close()
        except Exception:
            log.exception("Error during provider cleanup")


atexit.register(_cleanup_providers)


def load_pusht_online_train_val(
    num_hist: int = 3,
    num_pred: int = 1,
    frameskip: int = 5,
    samples_per_epoch: int = 50000,
    **kwargs,
) -> tuple[dict[str, PushTOnlineIterableDataset], dict[str, PushTOnlineTrajDataset]]:
    """Hydra-compatible factory that creates the online PushT data pipeline.

    Called by the Trainer as::

        hydra.utils.call(cfg.env.dataset, num_hist=..., num_pred=..., frameskip=...)

    Returns ``(datasets_dict, traj_dsets_dict)`` matching the existing
    ``load_pusht_slice_train_val`` interface.

    Args:
        num_hist:  Number of context frames (from Trainer config).
        num_pred:  Number of prediction frames (from Trainer config).
        frameskip: Temporal subsampling factor (from Trainer config).
        samples_per_epoch: How many individual samples the iterable dataset
            yields before raising StopIteration (one "epoch").
        **kwargs:  All remaining kwargs from the env config, forwarded to
            ``PushTOnlineProvider`` (policy_checkpoint, zero_latency_root,
            num_envs, img_size, val_ratio, noise_kind, noise_p_min/max, ...).

    Returns:
        A 2-tuple of ``(datasets, traj_dsets)``, each a dict with keys
        ``"train"`` and ``"valid"``.
    """
    chunk_length = num_hist + num_pred

    # Strip Hydra config keys that aren't provider constructor params.
    kwargs.pop("data_path", None)

    provider = PushTOnlineProvider(
        chunk_length=chunk_length,
        frameskip=frameskip,
        **kwargs,
    )
    provider.setup()
    _providers.append(provider)

    train_ds = PushTOnlineIterableDataset(
        provider, validation=False, samples_per_epoch=samples_per_epoch
    )
    val_ds = PushTOnlineIterableDataset(
        provider, validation=True,
        samples_per_epoch=max(1, int(samples_per_epoch * provider.val_ratio)),
    )

    empty_traj = PushTOnlineTrajDataset()

    datasets = {"train": train_ds, "valid": val_ds}
    traj_dset = {"train": empty_traj, "valid": empty_traj}
    return datasets, traj_dset
