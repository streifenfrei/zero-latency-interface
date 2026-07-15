"""DINO-WM PushT world model adapter.

Wraps the trained ``VWorldModel`` for interactive, frame-by-frame use.
The model was trained with ``frameskip=5`` — each prediction step covers
5 raw environment steps.  This adapter accumulates actions across the
frameskip window and advances the world model at that granularity.
Between model updates the most recent prediction is held.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

from dino_wm.models.visual_world_model import VWorldModel
from dino_wm.models.dino import DinoV2Encoder

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

        # Action buffer: accumulate frameskip raw actions → one concatenated
        # action. Index i of this buffer corresponds to raw action at step i
        # within the current frameskip window.
        self._action_buf: list[np.ndarray] = []     # each (raw_action_dim,)

        # Previous prediction (held between model updates).
        self._current_prediction: np.ndarray | None = None
        self._current_proprio: np.ndarray | None = None

        self._model_initialised: bool = False
        self._step_counter: int = 0
        self._sub_step_counter: int = 0  # position within current frameskip window

    # -- public API ------------------------------------------------------------

    @property
    def is_initialised(self) -> bool:
        return self._model_initialised

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

        self._ctx_visual = [
            _np_to_model_tensor(f, self._device, self.img_size)
            for f in frames
        ]
        self._ctx_proprio = [
            torch.from_numpy(p.astype(np.float32)).view(1, 1, -1).to(self._device)
            for p in proprios
        ]
        self._action_buf.clear()
        self._current_prediction = frames[-1]
        self._current_proprio = proprios[-1]
        self._model_initialised = True
        self._step_counter = 0
        self._sub_step_counter = 0

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
        self._action_buf.append(a)
        self._step_counter += 1

        # If the buffer is full, advance the model one step.
        if len(self._action_buf) >= self.frameskip:
            self._advance_model()
            self._action_buf.clear()

        return self._current_prediction

    def update_ground_truth(self, frame: np.ndarray,
                            proprio: np.ndarray) -> None:
        """Register a newly-arrived GT frame for context tracking.

        Called every raw timestep (even when the WM doesn't step).
        The frame/proprio at frameskip-aligned indices are stored as
        potential context for grounding.
        """
        self._sub_step_counter += 1
        if self._sub_step_counter >= self.frameskip:
            self._sub_step_counter = 0
            # This frame is at a frameskip-aligned index.
            self._ctx_visual.append(
                _np_to_model_tensor(frame, self._device, self.img_size))
            self._ctx_proprio.append(
                torch.from_numpy(proprio.astype(np.float32)).view(1, 1, -1)
                .to(self._device))
            # Trim to window size.
            if len(self._ctx_visual) > self.num_hist + 1:  # +1 for the newest
                self._ctx_visual = self._ctx_visual[-self.num_hist:]
                self._ctx_proprio = self._ctx_proprio[-self.num_hist:]

    def ground_and_predict(
        self,
        gt_frame: np.ndarray,
        gt_proprio: np.ndarray,
        actions_since_gt: list[np.ndarray],
    ) -> np.ndarray:
        """Ground the WM on a GT frame and predict ahead to the current time.

        Used when a delayed GT frame arrives in the ZLI.

        Parameters
        ----------
        gt_frame, gt_proprio:
            The newly-arrived delayed GT observation.
        actions_since_gt:
            All raw actions applied since *gt_timestep+1* to the current
            timestep (newest last).

        Returns
        -------
        np.ndarray
            Predicted RGB frame for the current timestep.
        """
        # Build context from the GT frame + preceding entries from the
        # internal context window.  The context window stores frameskip-
        # aligned frames.  The GT frame at its original timestep IS
        # frameskip-aligned (because _sub_step_counter==0 at those points).
        #
        # For grounding we need ``num_hist`` context entries.  We use the
        # internal window (which contains the most recent frameskip-aligned
        # frames including this GT frame) plus the GT frame at the newest
        # position.

        ctx_vis = list(self._ctx_visual)
        ctx_prop = list(self._ctx_proprio)

        # Ensure the GT frame is in the context window.
        gt_tensor = _np_to_model_tensor(gt_frame, self._device, self.img_size)
        gt_prop_tensor = torch.from_numpy(
            gt_proprio.astype(np.float32)).view(1, 1, -1).to(self._device)

        if len(ctx_vis) < self.num_hist:
            # Pad by repeating the oldest entry.
            pad = self.num_hist - len(ctx_vis)
            ctx_vis = [ctx_vis[0]] * pad + ctx_vis
            ctx_prop = [ctx_prop[0]] * pad + ctx_prop

        # Ground: reset with the context (the last entry is the newest GT).
        self._ctx_visual = ctx_vis[-self.num_hist:]
        self._ctx_proprio = ctx_prop[-self.num_hist:]
        self._model_initialised = True

        # Build observations for rollout.
        obs_0 = {
            "visual": torch.cat(self._ctx_visual, dim=1),
            "proprio": torch.cat(self._ctx_proprio, dim=1),
        }

        # Convert buffered actions to frameskip-concatenated actions.
        n_raw = len(actions_since_gt)
        # Pad actions to a multiple of frameskip.
        pad_n = (self.frameskip - (n_raw % self.frameskip)) % self.frameskip
        padded = list(actions_since_gt)
        if pad_n > 0:
            # repeat last action for padding
            last_a = np.asarray(actions_since_gt[-1], dtype=np.float32)
            padded.extend([last_a] * pad_n)
        n_wm_steps = len(padded) // self.frameskip

        # Build concatenated actions: (n_wm_steps, action_dim)
        wm_actions = []
        for k in range(n_wm_steps):
            chunk = padded[k * self.frameskip : (k + 1) * self.frameskip]
            wm_actions.append(np.concatenate(chunk, axis=0))
        wm_actions = np.stack(wm_actions, axis=0)  # (n_wm_steps, action_dim)

        # Context actions: num_hist zero-action entries (the model encodes
        # context without real per-frame actions — training data had real
        # actions, but we approximate with zeros for the reset frames).
        ctx_actions = np.zeros((self.num_hist, self._action_dim), dtype=np.float32)

        # Full action sequence: context actions + rollout actions.
        all_actions = np.concatenate([ctx_actions, wm_actions], axis=0)
        act_tensor = torch.from_numpy(all_actions).unsqueeze(0).to(self._device)

        # Run rollout.
        with torch.no_grad():
            z_obs, _ = self._model.rollout(obs_0, act_tensor)
            decoded, _ = self._model.decode_obs(z_obs)
            rollout_vis = decoded["visual"][0]  # (T_rollout, 3, H, W)

        # rollout output = context encoding + predicted frames.
        # We want the last prediction (corresponding to the final action step).
        pred_vis = rollout_vis[self.num_hist:]  # predicted frames only

        # The prediction at index n_wm_steps - 1 corresponds to the last
        # WM step.  That's our best guess for the current raw timestep.
        if n_wm_steps > 0 and n_wm_steps <= pred_vis.shape[0]:
            self._current_prediction = _tensor_to_np(pred_vis[n_wm_steps - 1])
        elif pred_vis.shape[0] > 0:
            self._current_prediction = _tensor_to_np(pred_vis[-1])

        # Update context for future steps.
        self._ctx_visual = ctx_vis[-self.num_hist:]
        self._ctx_proprio = ctx_prop[-self.num_hist:]

        return self._current_prediction

    # -- internals -------------------------------------------------------------

    def _advance_model(self) -> None:
        """Run one autoregressive model step using the buffered actions."""
        if len(self._ctx_visual) < self.num_hist:
            return  # not enough context yet

        # Build context observations.
        obs_0 = {
            "visual": torch.cat(self._ctx_visual[-self.num_hist:], dim=1),
            "proprio": torch.cat(self._ctx_proprio[-self.num_hist:], dim=1),
        }

        # One concatenated action (frameskip raw actions).
        concat_action = np.concatenate(self._action_buf[-self.frameskip:], axis=0)
        wm_action = np.concatenate([
            np.zeros((self.num_hist, self._action_dim), dtype=np.float32),
            concat_action.reshape(1, -1),
        ], axis=0)
        act_tensor = torch.from_numpy(wm_action).unsqueeze(0).to(self._device)

        with torch.no_grad():
            z_obs, _ = self._model.rollout(obs_0, act_tensor)
            decoded, _ = self._model.decode_obs(z_obs)
            rollout_vis = decoded["visual"][0]

        # The last predicted frame.
        pred_vis = rollout_vis[self.num_hist:]
        if pred_vis.shape[0] > 0:
            self._current_prediction = _tensor_to_np(pred_vis[-1])

        # Update context: add the new prediction as the latest "frame".
        # Use the predicted visual as the new context entry.
        if pred_vis.shape[0] > 0:
            new_vis = pred_vis[-1:]  # (1, 3, H, W) in [-1, 1]
            # For proprio, we don't have a real measurement; keep the last one.
            new_prop = self._ctx_proprio[-1].clone()
            self._ctx_visual.append(new_vis)
            self._ctx_proprio.append(new_prop)

        # Trim context window.
        if len(self._ctx_visual) > self.num_hist + 5:
            self._ctx_visual = self._ctx_visual[-self.num_hist:]
            self._ctx_proprio = self._ctx_proprio[-self.num_hist:]
