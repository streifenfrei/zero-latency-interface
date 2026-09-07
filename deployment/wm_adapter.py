"""DINO-WM PushT world model adapter.

Wraps the trained ``VWorldModel`` for interactive, frame-by-frame use.
The model was trained with ``frameskip=5`` — each prediction step covers
5 raw environment steps.  This adapter accumulates actions across the
frameskip window and advances the world model at that granularity.
Between model updates the most recent prediction is held.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

from dino_wm.models.visual_world_model import VWorldModel
from dino_wm.models.dino import DinoV2Encoder


def _install_legacy_model_aliases():
    """Make checkpoints trained with the old ``models.*`` package importable.

    The training run that produced the PushT checkpoints was executed from a
    dino_wm checkout where ``models/`` (and ``distributed_fn``) lived at the
    repo root. The current checkout namespaces these modules under
    ``dino_wm.models``, so ``torch.load`` fails with ``No module named
    'models'`` unless we alias the old import paths. Call before loading any
    checkpoint that still references them.
    """
    import importlib
    import sys
    import types

    import dino_wm

    if "models" in sys.modules:
        return

    # Let ``import distributed_fn`` (used by dino_wm.models.vqvae) resolve to
    # the dino_wm repo root, like it did during training.
    dino_wm_root = os.path.dirname(dino_wm.__file__) if dino_wm.__file__ else None
    if dino_wm_root is None:  # namespace package fallback
        dino_wm_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dino_wm")
    if dino_wm_root not in sys.path:
        sys.path.insert(0, dino_wm_root)

    import dino_wm.models as _real

    pkg = types.ModuleType("models")
    pkg.__path__ = _real.__path__
    sys.modules["models"] = pkg
    for sub in ("vit", "visual_world_model", "dino", "proprio", "vqvae",
                "decoder", "encoder"):
        try:
            sys.modules[f"models.{sub}"] = importlib.import_module(
                f"dino_wm.models.{sub}")
        except ImportError:
            pass

# ── helpers ──────────────────────────────────────────────────────────────────

def _np_to_model_tensor(img: np.ndarray, device: torch.device,
                        img_size: int = 224) -> torch.Tensor:
    """(H, W, 3) uint8 → (1, 1, 3, 224, 224) float in [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0
    x = rearrange(x, "h w c -> 1 1 c h w")
    if x.shape[-1] != img_size:
        x = F.interpolate(x.squeeze(0), size=(img_size, img_size),
                          mode="bilinear", align_corners=False).unsqueeze(0)
    x = (x - 0.5) / 0.5
    return x.to(device)


def _tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) float in [-1, 1] → (H, W, 3) uint8."""
    x = t.detach().cpu().float()
    x = x * 0.5 + 0.5
    x = x.clamp(0, 1).permute(1, 2, 0).mul(255).byte().numpy()
    return x


# ── adapter ─────────────────────────────────────────────────────────────────

class DinoWMPushtAdapter:
    """DINO-WM PushT world model for interactive zero-latency prediction.

    Parameters
    ----------
    ckpt_dir:
        Path to the ``checkpoints/`` directory containing the trained model.
    ckpt_suffix:
        Suffix for the checkpoint file (e.g. ``_latest``).
    device:
        Torch device string.  Defaults to ``cuda`` if available.

    Key attributes (matching the telemanipulation_client WorldModel interface):
      - ``context_length`` : number of subsampled context frames (= num_hist)
      - ``frameskip``      : raw frames per model step (5)
    """

    def __init__(
        self,
        ckpt_dir: str,
        ckpt_suffix: str = "_latest",
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        # --- load checkpoint & config -----------------------------------------
        _install_legacy_model_aliases()
        ckpt_path = os.path.join(ckpt_dir, f"model{ckpt_suffix}.pth")
        print(f"[WM] loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)

        hydra_yaml = os.path.join(
            os.path.dirname(ckpt_dir.rstrip("/")), "hydra.yaml")
        cfg = OmegaConf.load(hydra_yaml)

        self.num_hist: int = cfg.num_hist       # e.g. 3
        self.num_pred: int = cfg.num_pred       # e.g. 1
        self.frameskip: int = cfg.frameskip     # e.g. 5
        self.img_size: int = cfg.img_size       # 224
        self.context_length: int = self.num_hist

        # Proprio dim from the loaded encoder (4 for PushT w/ velocity)
        self._proprio_dim: int = ckpt["proprio_encoder"].in_chans
        self._action_dim: int = ckpt["action_encoder"].in_chans  # 2*5=10
        self._raw_action_dim: int = self._action_dim // self.frameskip  # 2

        # --- build model ------------------------------------------------------
        encoder = DinoV2Encoder(
            name=cfg.encoder.name, feature_key=cfg.encoder.feature_key)
        for p in encoder.parameters():
            p.requires_grad = False

        self._model = VWorldModel(
            image_size=self.img_size,
            num_hist=self.num_hist,
            num_pred=self.num_pred,
            encoder=encoder,
            proprio_encoder=ckpt["proprio_encoder"],
            action_encoder=ckpt["action_encoder"],
            decoder=ckpt["decoder"],
            predictor=ckpt["predictor"],
            proprio_dim=ckpt["proprio_encoder"].emb_dim,
            action_dim=ckpt["action_encoder"].emb_dim,
            concat_dim=cfg.concat_dim,
            num_action_repeat=cfg.num_action_repeat,
            num_proprio_repeat=cfg.num_proprio_repeat,
            train_encoder=False, train_predictor=False, train_decoder=False,
        ).to(self._device)
        self._model.eval()
        print(f"[WM] ready — num_hist={self.num_hist}  frameskip={self.frameskip}  "
              f"action_dim={self._action_dim}  proprio_dim={self._proprio_dim}")

        # --- state ------------------------------------------------------------
        # Sliding window of the most recent subsampled frames + proprio (for
        # context). Each entry is a (visual_tensor, proprio_tensor) pair at
        # frameskip-aligned indices.
        self._ctx_visual: list[torch.Tensor] = []   # each (1, 1, 3, H, W)
        self._ctx_proprio: list[torch.Tensor] = []  # each (1, 1, D)
        # Action chunk applied AFTER each context frame, parallel to
        # _ctx_visual (each (action_dim,)).  Training encoded every frame
        # together with the actions that follow it, so the chain must too —
        # zero-padding these makes the model predict "nothing happened".
        self._ctx_actions: list[np.ndarray] = []

        # Delayed-GT history keyed by the timestep each frame was produced
        # at, used ONLY to build grounding contexts.  Keyed by timestep (not
        # an alignment counter) so a grounding can pick frames at EXACTLY
        # gt_ts - k*frameskip — the stride the action chunks assume.  Kept
        # separate from _ctx_visual so delayed frames never leak into the
        # autoregressive chain's prediction window between groundings.
        self._gt_hist: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        # Action buffer: accumulate frameskip raw actions → one concatenated
        # action. Index i of this buffer corresponds to raw action at step i
        # within the current frameskip window.
        self._action_buf: list[np.ndarray] = []     # each (raw_action_dim,)

        # Previous prediction (held between model updates).
        self._current_prediction: np.ndarray | None = None
        self._current_proprio: np.ndarray | None = None

        self._model_initialised: bool = False
        self._step_counter: int = 0

        # Thread-safety: the ZLI grounding runs in a worker thread while the
        # main loop calls predict() every tick.  torch releases the GIL during
        # CUDA kernels, so ALL adapter state mutation and EVERY model forward
        # must be serialized through this lock.
        self._lock = threading.RLock()
        # gt_ts of the last completed async grounding (for tests/debug).
        self._last_grounding_ts: int = -1
        # Per-step chain logging is noisy (every frameskip ticks) — opt in.
        self._log_chain: bool = os.environ.get("ZLI_LOG_CHAIN", "0") != "0"

    # -- public API ------------------------------------------------------------

    @property
    def is_initialised(self) -> bool:
        with self._lock:
            return self._model_initialised

    @property
    def last_grounding_ts(self) -> int:
        """gt_ts of the last completed async grounding (-1 if none)."""
        with self._lock:
            return self._last_grounding_ts

    def reset(self, frames: list[np.ndarray],
              proprios: list[np.ndarray]) -> None:
        """Ground the world model on a chunk of consecutive GT frames.

        Parameters
        ----------
        frames:
            List of ``context_length`` RGB frames (HWC uint8) at *subsampled*
            indices.  ``frames[-1]`` is the frame the WM is grounded at.
        proprios:
            List of proprio vectors aligned with *frames*, each of shape
            ``(proprio_dim,)``.
        """
        assert len(frames) == self.num_hist
        assert len(proprios) == self.num_hist

        with self._lock:
            self._ctx_visual = [
                _np_to_model_tensor(f, self._device, self.img_size)
                for f in frames
            ]
            self._ctx_proprio = [
                torch.from_numpy(p.astype(np.float32)).view(1, 1, -1)
                .to(self._device)
                for p in proprios
            ]
            # No action history at episode start — zeros match the training
            # data's head padding.
            self._ctx_actions = [
                np.zeros(self._action_dim, dtype=np.float32)
                for _ in range(self.num_hist)
            ]
            self._gt_hist = {}
            self._action_buf.clear()
            self._current_prediction = frames[-1]
            self._current_proprio = proprios[-1]
            self._model_initialised = True
            self._step_counter = 0
            self._last_grounding_ts = -1

    def predict(self, action: np.ndarray) -> np.ndarray:
        """Advance the world model by one *raw* timestep.

        The model predicts at frameskip granularity.  Actions are accumulated
        in an internal buffer; when the buffer fills to ``frameskip`` entries,
        one autoregressive prediction step is taken and the buffer is cleared.
        Between model steps the most recent prediction is returned unchanged.

        Parameters
        ----------
        action:
            Raw action vector of shape ``(raw_action_dim,)`` — e.g. ``[dx, dy]``
            for PushT.

        Returns
        -------
        np.ndarray
            The current best-guess RGB frame (HWC uint8).
        """
        if not self._model_initialised:
            raise RuntimeError("reset() must be called before predict()")

        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != self._raw_action_dim:
            raise ValueError(
                f"action has shape {a.shape}, expected ({self._raw_action_dim},)"
            )
        with self._lock:
            self._action_buf.append(a)
            self._step_counter += 1

            # If the buffer is full, advance the model one step.
            if len(self._action_buf) >= self.frameskip:
                self._advance_model()
                self._action_buf.clear()

            return self._current_prediction

    def update_ground_truth(self, frame: np.ndarray, proprio: np.ndarray,
                            ts: int) -> None:
        """Register a newly-arrived (delayed) GT frame, keyed by its timestep.

        Called once per arrived frame in the ZLI (before ground_and_predict).
        EVERY arrival is stored, keyed by the timestep the frame was produced
        at, so a grounding can select context frames at exactly the training
        stride (``gt_ts - k*frameskip``) instead of whatever phase an
        alignment counter happened to land on — a frame at the wrong time
        would be paired with the wrong action chunk.  Frames too old to be
        used as context are dropped.  Never touches ``_ctx_visual``, whose
        entries are the autoregressive chain's prediction window.
        """
        ts = int(ts)
        with self._lock:
            self._gt_hist[ts] = (
                _np_to_model_tensor(frame, self._device, self.img_size),
                torch.from_numpy(proprio.astype(np.float32)).view(1, 1, -1)
                .to(self._device),
            )
            span = (self.num_hist - 1) * self.frameskip
            if len(self._gt_hist) > span + 2 * self.frameskip:
                cutoff = ts - span
                for old in [k for k in self._gt_hist if k < cutoff]:
                    del self._gt_hist[old]

    def ground_and_predict(
        self,
        gt_frame: np.ndarray,
        gt_proprio: np.ndarray,
        gt_ts: int,
        actions: list[np.ndarray],
    ) -> np.ndarray:
        """Ground the WM on a GT frame and predict ahead to the current time.

        Used when a delayed GT frame arrives in the ZLI.

        Parameters
        ----------
        gt_frame, gt_proprio:
            The newly-arrived delayed GT observation.
        actions:
            Raw actions covering ``(num_hist-1)*frameskip`` steps before
            ``gt_timestep+1`` up to the current timestep (newest last).  The
            first ``(num_hist-1)*frameskip`` entries are the chunks encoded
            with the context frames; the remainder are the actions applied
            since the GT frame, used for the rollout.

        Returns
        -------
        np.ndarray
            Predicted RGB frame for the current timestep.
        """
        with self._lock:
            # Ground: the context ends with the newest GT frame.  Built from
            # the timestamped GT history (never from the chain's context).
            ctx_vis, ctx_prop = self._build_grounding_context(
                gt_ts, gt_frame, gt_proprio)
            self._ctx_visual = ctx_vis
            self._ctx_proprio = ctx_prop
            self._model_initialised = True

            # Build observations for rollout.
            obs_0 = {
                "visual": torch.cat(self._ctx_visual, dim=1),
                "proprio": torch.cat(self._ctx_proprio, dim=1),
            }

            # Build the action tensor the way training chunks did: every
            # frame — context and predicted — is encoded together with the
            # frameskip raw actions applied AFTER it.  Zero context actions
            # (the old approximation) made the model under-predict motion,
            # most visibly at small delays where the context dominates.
            H = self.num_hist
            fs = self.frameskip
            n_raw = max(len(actions) - (H - 1) * fs, 0)   # actions since gt_ts+1
            n_wm_steps = (n_raw + fs - 1) // fs
            # Left-pad with zeros if the episode start cut the context short,
            # right-pad by repeating the last action to a whole chunk count.
            head_pad = max((H - 1) * fs - (len(actions) - n_raw), 0)
            padded = [np.zeros(self._raw_action_dim, dtype=np.float32)] * head_pad \
                + list(actions)
            tail = n_wm_steps * fs - n_raw
            if tail > 0 and padded:
                padded = padded + [np.asarray(padded[-1], dtype=np.float32)] * tail

            n_chunks = H + n_wm_steps
            act_chunks = []
            for k in range(n_chunks):
                if k < n_chunks - 1:
                    chunk = np.stack(padded[k * fs:(k + 1) * fs], axis=0)
                    act_chunks.append(chunk.reshape(-1))
                else:
                    # The chunk applied after the current state is not known
                    # yet; it only affects predictions beyond the display.
                    act_chunks.append(np.zeros(self._action_dim, dtype=np.float32))
            all_actions = np.stack(act_chunks, axis=0)  # (n_chunks, action_dim)
            act_tensor = torch.from_numpy(all_actions).unsqueeze(0).to(self._device)
            # Keep the chunks that belong to the frames left as context.
            self._ctx_actions = [np.asarray(c, dtype=np.float32)
                                 for c in act_chunks[-H:]]

            # Run rollout.  rollout returns num_hist + n_wm_steps + 1 frames:
            # the context encoding, n_wm_steps predictions, and one extra
            # lookahead frame we never use.  The prediction after the last
            # applied action is frame num_hist + n_wm_steps - 1 — decode only
            # that frame (the decoder dominates cost at large delays).
            with torch.no_grad():
                z_obs, _ = self._model.rollout(obs_0, act_tensor)

            if n_wm_steps > 0:
                idx = self.num_hist + n_wm_steps - 1
                self._current_prediction = _tensor_to_np(
                    self._decode_frame(z_obs, idx)[0])

            # The rollout above already accounts for actions_since_gt, so
            # restart the incremental buffers at the current timestep —
            # otherwise the next predict() calls would replay the
            # pre-grounding tail of the action buffer (double-counting it)
            # when the model advances again.  (_gt_hist is deliberately left
            # alone: it is the delayed stream's own record, independent of
            # groundings.)
            self._action_buf.clear()
            self._step_counter = 0

            return self._current_prediction

    def ground_and_predict_async(
        self,
        gt_frame: np.ndarray,
        gt_proprio: np.ndarray,
        gt_ts: int,
        action_history: list,
        get_timestep: Callable[[], int],
    ) -> None:
        """Async ZLI grounding — runs in a worker thread.

        Grounds the WM on the newly-arrived delayed frame and rolls forward
        autoregressively, chasing the advancing present (the main loop keeps
        appending actions while this runs), then atomically publishes the new
        context + prediction.  Never cancels: callers join instead.  The
        main thread's in-flight chain (predict() calls during the rollout) is
        discarded at publish — the grounding is authoritative.
        """
        H = self.num_hist
        fs = self.frameskip

        t_start = time.perf_counter()
        print(f"[WM-grounding] start: gt_ts={gt_ts}  "
              f"delay={get_timestep() - gt_ts} raw steps", flush=True)

        with self._lock:
            ctx_vis, ctx_prop = self._build_grounding_context(
                gt_ts, gt_frame, gt_proprio)
            target0 = get_timestep()

        obs_0 = {
            "visual": torch.cat(ctx_vis, dim=1),
            "proprio": torch.cat(ctx_prop, dim=1),
        }
        # Context chunks are stable forever (history is append-only).
        act_0 = np.stack([
            self._action_chunk_for_step(action_history, gt_ts, k, target0)
            for k in range(H)
        ], axis=0)
        act_0_t = torch.from_numpy(act_0).unsqueeze(0).to(self._device)
        # Chunk per frame in z (context first, then one per rolled frame) —
        # kept so the published context carries its real actions.
        all_chunks: list[np.ndarray] = [np.asarray(c, dtype=np.float32)
                                        for c in act_0]
        print(f"[WM-grounding]   ctx actions: {self._fmt_chunks(act_0)}",
              flush=True)

        # Encode the context ONCE (DINO encoder + proprio/action tiling).
        with self._lock:
            with torch.no_grad():
                z = self._model.encode(obs_0, act_0_t)

        # Chase loop: roll one WM step at a time while the NEXT model frame
        # lands on or before the present — never past it (rolling past the
        # frameskip grid would extrapolate with unknown future actions).
        # The present advances while we roll, so re-check until the snapshot
        # is stable or the roll comes within 2*frameskip raw steps of it.
        # Per-step locking lets main-thread predict() interleave.
        # A step cap guarantees termination even when the model is slower
        # than real time (e.g. frameskip=1 @448): publish the best available
        # rollout instead of chasing forever; the remainder carry keeps the
        # chain aligned with the (small) lag.
        D0 = max(target0 - gt_ts, 0)
        max_steps = (D0 + fs - 1) // fs + 2 * fs
        rolled = 0
        while True:
            with self._lock:
                target = get_timestep()
            while gt_ts + (rolled + 1) * fs <= target \
                    and rolled < max_steps:
                # The whole chunk for this step is already known (its end
                # equals the frame's time, which is <= the present).
                chunk = self._action_chunk_for_step(
                    action_history, gt_ts, H + rolled, target)
                act_t = torch.from_numpy(chunk).view(1, 1, -1).to(self._device)
                with self._lock:
                    with torch.no_grad():
                        z_pred = self._model.predict(z[:, -H:])
                        z_new = z_pred[:, -1:]
                        z_new = self._model.replace_actions_from_z(z_new, act_t)
                        z = torch.cat([z, z_new], dim=1)
                all_chunks.append(np.asarray(chunk, dtype=np.float32))
                rolled += 1
            with self._lock:
                target2 = get_timestep()
            if target2 <= target or rolled >= max_steps or \
                    target2 - (gt_ts + rolled * fs) <= 2 * fs:
                break

        # Publish atomically: decode the last H frames in one batched pass,
        # install them as the new context (chain continuity post-publish) and
        # the last one as the displayed prediction.  The actions applied since
        # the last rolled frame (the partial next chunk) are carried over into
        # the action buffer so the chain stays exactly aligned with the
        # present instead of dropping them.
        with self._lock:
            with torch.no_grad():
                z_obses, _ = self._model.separate_emb(z)
                tail = {
                    "visual": z_obses["visual"][:, -H:],
                    "proprio": z_obses["proprio"][:, -H:],
                }
                decoded, _ = self._model.decode_obs(tail)
                vis = decoded["visual"]  # (1, H, 3, 224, 224)
                self._ctx_visual = [vis[:, i:i + 1] for i in range(H)]
                # Carry each published frame's own action chunk so the chain
                # continues with real actions instead of zeros.
                self._ctx_actions = [np.asarray(c, dtype=np.float32)
                                     for c in all_chunks[-H:]]
                gt_prop_tensor = torch.from_numpy(
                    gt_proprio.astype(np.float32)).view(1, 1, -1) \
                    .to(self._device)
                # No real proprio measurements for the rolled frames — keep
                # the grounding proprio (matches _advance_model's approach).
                self._ctx_proprio = [gt_prop_tensor.clone() for _ in range(H)]
                self._current_prediction = _tensor_to_np(vis[0, -1])
                self._current_proprio = np.asarray(
                    gt_proprio, dtype=np.float32).copy()
                # Carry over the actions applied after the last rolled frame
                # (0..fs-1 of them; the rest of that chunk arrives via
                # predict() in the next ticks).
                remainder = action_history[
                    gt_ts + rolled * fs + 1 : target + 1]
                self._action_buf = [
                    np.asarray(a, dtype=np.float32) for a in remainder]
                self._step_counter = len(self._action_buf)
                self._last_grounding_ts = gt_ts
                self._model_initialised = True

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        n_carry = len(action_history[gt_ts + rolled * fs + 1:target + 1])
        print(f"[WM-grounding] done: gt_ts={gt_ts} → frame@{gt_ts + rolled * fs}"
              f" (present={target}, {rolled} WM steps, {dt_ms:.0f} ms)\n"
              f"[WM-grounding]   roll actions: "
              f"{self._fmt_chunks(all_chunks[H:])}  "
              f"carried={n_carry} raw",
              flush=True)

    # -- internals -------------------------------------------------------------

    def _build_grounding_context(
        self,
        gt_ts: int,
        gt_frame: np.ndarray,
        gt_proprio: np.ndarray,
    ) -> tuple[list, list]:
        """Context frames at exactly ``gt_ts - k*frameskip`` (newest last).

        The newest entry is the just-arrived GT frame; older entries come
        from the timestamped delayed-GT history, so every context frame sits
        on the training stride AND pairs with the action chunk
        ``_action_chunk_for_step`` computes for its slot.  Entries missing
        (episode start) repeat the nearest available frame.  Caller must hold
        the lock.
        """
        gt_tensor = _np_to_model_tensor(gt_frame, self._device, self.img_size)
        gt_prop_tensor = torch.from_numpy(
            gt_proprio.astype(np.float32)).view(1, 1, -1).to(self._device)

        H, fs = self.num_hist, self.frameskip
        vis: list = []
        prop: list = []
        for k in range(H):
            ts = gt_ts - (H - 1 - k) * fs
            if ts >= gt_ts:
                vis.append(gt_tensor)
                prop.append(gt_prop_tensor)
            elif ts in self._gt_hist:
                v, p = self._gt_hist[ts]
                vis.append(v)
                prop.append(p)
            else:
                vis.append(None)
                prop.append(None)

        first = next((i for i, v in enumerate(vis) if v is not None), None)
        if first is None:  # nothing stored yet — repeat the GT frame
            return [gt_tensor] * H, [gt_prop_tensor] * H
        for i in range(H):
            if vis[i] is None:
                j = first if i < first else i - 1
                vis[i] = vis[j]
                prop[i] = prop[j]
        return vis, prop

    def _fmt_chunks(self, chunks: list) -> str:
        """Compact per-chunk action summary: the SUM of each frameskip chunk.

        The sum is the net displacement the chunk commands, so a stalled
        rollout (all-zero chunks) is obvious at a glance.
        """
        parts = []
        for c in chunks:
            rows = np.asarray(c, dtype=np.float32).reshape(-1,
                                                           self._raw_action_dim)
            s = rows.sum(axis=0)
            parts.append("(" + ",".join(f"{v:+.2f}" for v in s) + ")")
        return " ".join(parts) if parts else "-"

    def _action_chunk_for_step(
        self,
        action_history: list,
        gt_ts: int,
        chunk_idx: int,
        target_ts: int,
    ) -> np.ndarray:
        """Flat ``(action_dim,)`` chunk for absolute chunk index *chunk_idx*.

        Chunk ``k`` covers raw steps
        ``[gt_ts + 1 - (H-1)*fs + k*fs, +fs)``; context chunks are 0..H-1,
        rollout chunks H..H+n-1.  Steps before 0 are zero (episode-start head
        pad); known steps come from ``action_history[step]``; steps beyond
        ``target_ts`` repeat the last known action (matching the sync
        convention's tail padding — the caller zeroes the final chunk
        explicitly).
        """
        fs = self.frameskip
        start = gt_ts + 1 - (self.num_hist - 1) * fs + chunk_idx * fs
        rows = []
        for step in range(start, start + fs):
            if step < 0:
                rows.append(np.zeros(self._raw_action_dim, dtype=np.float32))
            elif step <= target_ts:
                if step < len(action_history):
                    rows.append(np.asarray(action_history[step],
                                           dtype=np.float32))
                else:
                    rows.append(np.zeros(self._raw_action_dim, dtype=np.float32))
            elif action_history:
                rows.append(np.asarray(action_history[-1], dtype=np.float32))
            else:
                rows.append(np.zeros(self._raw_action_dim, dtype=np.float32))
        return np.stack(rows, axis=0).reshape(-1)

    def _decode_frame(self, z_obs: dict, idx: int) -> torch.Tensor:
        """Decode a single rollout frame (``idx`` along the time dim).

        The VQVAE decoder dominates rollout cost at large delays, so only the
        one frame we display is decoded instead of the whole rollout.

        Returns the visual as ``(1, 3, H, W)`` in [-1, 1].
        """
        one = {
            "visual": z_obs["visual"][:, idx:idx + 1],
            "proprio": z_obs["proprio"][:, idx:idx + 1],
        }
        decoded, _ = self._model.decode_obs(one)
        return decoded["visual"][0]

    def _advance_model(self) -> None:
        """Run one autoregressive model step using the buffered actions."""
        if len(self._ctx_visual) < self.num_hist:
            return  # not enough context yet

        # Build context observations.
        obs_0 = {
            "visual": torch.cat(self._ctx_visual[-self.num_hist:], dim=1),
            "proprio": torch.cat(self._ctx_proprio[-self.num_hist:], dim=1),
        }

        # Like training chunks, EVERY context frame is encoded together with
        # the actions applied after it.  The buffered actions are the ones
        # applied after the current newest context frame; the older frames'
        # chunks come from _ctx_actions (zero-padding them made the model
        # predict "nothing happened" and stalled the chain).  The predicted
        # frame's own chunk isn't known yet — it stays zero.
        concat_action = np.concatenate(
            self._action_buf[-self.frameskip:], axis=0).astype(np.float32)
        while len(self._ctx_actions) < len(self._ctx_visual):
            self._ctx_actions.insert(
                0, np.zeros(self._action_dim, dtype=np.float32))
        self._ctx_actions[-1] = concat_action
        ctx_chunks = np.stack(self._ctx_actions[-self.num_hist:], axis=0)
        wm_action = np.concatenate([
            ctx_chunks,
            np.zeros((1, self._action_dim), dtype=np.float32),  # predicted frame
        ], axis=0)
        act_tensor = torch.from_numpy(wm_action).unsqueeze(0).to(self._device)

        if self._log_chain:
            print(f"[WM-chain] step: ctx actions "
                  f"{self._fmt_chunks(self._ctx_actions[-self.num_hist:])}",
                  flush=True)

        # Manual single step (encode → predict → append), skipping the extra
        # lookahead frame that model.rollout always appends — with frameskip=1
        # that wasted predictor forward runs EVERY tick.
        with torch.no_grad():
            z = self._model.encode(obs_0, act_tensor[:, :self.num_hist])
            z_pred = self._model.predict(z[:, -self.num_hist:])
            z_new = z_pred[:, -1:]
            z_new = self._model.replace_actions_from_z(
                z_new, act_tensor[:, self.num_hist:self.num_hist + 1])
            z = torch.cat([z, z_new], dim=1)
            z_obs, _ = self._model.separate_emb(z)

        idx = self.num_hist  # the appended prediction frame
        pred_vis = self._decode_frame(z_obs, idx)  # (1, 3, H, W) in [-1, 1]

        self._current_prediction = _tensor_to_np(pred_vis[0])

        # Update context: add the new prediction as the latest "frame".
        # Context entries are (1, 1, 3, H, W).
        new_vis = pred_vis.unsqueeze(1)  # (1, 1, 3, H, W)
        # For proprio, we don't have a real measurement; keep the last one.
        new_prop = self._ctx_proprio[-1].clone()
        self._ctx_visual.append(new_vis)
        self._ctx_proprio.append(new_prop)
        # The new frame's own chunk is unknown until the next 5 actions
        # arrive; predict() fills it in when the buffer next fills.
        self._ctx_actions.append(np.zeros(self._action_dim, dtype=np.float32))

        # Trim context window (all three lists stay parallel).
        if len(self._ctx_visual) > self.num_hist + 5:
            self._ctx_visual = self._ctx_visual[-self.num_hist:]
            self._ctx_proprio = self._ctx_proprio[-self.num_hist:]
            self._ctx_actions = self._ctx_actions[-self.num_hist:]
