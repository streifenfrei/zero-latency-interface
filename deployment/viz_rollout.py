#!/usr/bin/env python
"""Autoregressive rollout videos for DINO-WM on PushT (online env).

Loads a trained DINO-WM checkpoint, drives the ManiSkill PushT environment with
the trained PPO policy to collect ground-truth trajectories, runs autoregressive
rollouts through the world model, decodes predictions to pixels, computes PCA
on DINOv2 feature maps, and writes side-by-side MP4 videos:

    GT-RGB | GT-PCA | Pred-PCA

With ``--pixel-decoder``: an additional row with GT-RGB | Pred-RGB is appended.

The model was trained with a frameskip factor — predictions are at frameskip
intervals; the video shows every raw frame on the GT side and holds predictions
between updates.

Requires a GPU with a working Vulkan driver (SAPIEN / physx_cuda).

Examples::

    # Quick test (1 episode, latest checkpoint):
    python deployment/viz_rollout.py \
        --ckpt outputs/2026-07-06/14-25-39/checkpoints --max-episodes 1

    # PCA + pixel decoder, specific checkpoint:
    python deployment/viz_rollout.py \
        --ckpt outputs/2026-07-06/14-25-39/checkpoints \
        --max-episodes 5 --pixel-decoder

Output: one .mp4 per episode under ``--out`` (default
``logs/viz_pusht_rollout/``).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import imageio.v2 as imageio
from einops import rearrange
from omegaconf import OmegaConf

# ── repo root ────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from dino_wm.models.visual_world_model import VWorldModel
from dino_wm.models.dino import DinoV2Encoder
from dino_wm.utils import slice_trajdict_with_t

# ── constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 224
VQVAE_DECODER_SCALE = 16  # VWorldModel downsamples by this before DINOv2


# ═══════════════════════════════════════════════════════════════════════════════
# PCA visualisation of feature maps
# (adapted from zero_latency_interface/model/visualize.py)
# ═══════════════════════════════════════════════════════════════════════════════

def feats_to_pca(
    tgt_feats: torch.Tensor,
    pred_feats: torch.Tensor,
    h: int,
    w: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute PCA of ground-truth features, project both GT and predictions."""

    def to_grid(f: torch.Tensor) -> torch.Tensor:
        # (T, P, D) → (T, D, h, w)
        return f.transpose(1, 2).reshape(-1, f.shape[2], h, w)

    return _pca(to_grid(tgt_feats), to_grid(pred_feats))


def _pca(
    tgt_grid: torch.Tensor,
    pred_grid: torch.Tensor,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Fit PCA on all GT feature maps, project both GT and pred to 3-channel RGB."""
    tgt_grid = torch.nan_to_num(tgt_grid).float()
    pred_grid = torch.nan_to_num(pred_grid).float()
    num_channels = tgt_grid.shape[1]
    X = tgt_grid.permute(0, 2, 3, 1).reshape(-1, num_channels).cpu()
    if X.std() < 1e-8:
        a = np.zeros((tgt_grid.shape[2], tgt_grid.shape[3], 3), dtype=np.float32)
        return [a] * len(tgt_grid), [a] * len(pred_grid)

    with torch.autocast("cuda", enabled=False):
        mu = X.mean(0, keepdim=True)
        Xc = X - mu
        Cx = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
        Cx.diagonal().add_(1e-6)
        Wp = torch.linalg.eigh(Cx)[1][:, -3:]  # top 3 eigenvectors
        Wp[:, Wp.sum(0) < 0] *= -1  # sign consistency
        Y_all = Xc @ Wp
        m, M = Y_all.amin(0, keepdim=True), Y_all.amax(0, keepdim=True)
        d = (M - m).clamp(min=1e-6)

    def proj(f: torch.Tensor) -> np.ndarray:
        Y = ((f.cpu().flatten(1).T - mu) @ Wp).view(tgt_grid.shape[2], tgt_grid.shape[3], 3)
        return ((Y - m) / d).clamp(0, 1).numpy()

    return (
        [proj(tgt) for tgt in tgt_grid],
        [proj(pred) for pred in pred_grid],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert a C×H×W tensor in [-1, 1] to H×W×3 uint8 numpy array."""
    x = t.detach().cpu().float()
    x = x * 0.5 + 0.5  # [-1,1] → [0,1]
    x = x.clamp(0, 1).permute(1, 2, 0).mul(255).byte().numpy()
    return x


def _pca_to_uint8(pca: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """PCA float [0,1] H×W×3 → uint8, resized to target dimensions."""
    import cv2
    pca = (pca.clip(0, 1) * 255).astype(np.uint8)
    if pca.shape[0] != target_h or pca.shape[1] != target_w:
        pca = cv2.resize(pca, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return pca


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def viz_pusht_rollout(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── locate checkpoint ──────────────────────────────────────────────────
    ckpt_path = os.path.join(args.ckpt, f"model{args.ckpt_suffix}.pth")
    print(f"loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ── load hydra config ───────────────────────────────────────────────────
    hydra_yaml = os.path.join(
        os.path.dirname(args.ckpt.rstrip("/")), "hydra.yaml"
    )
    cfg = OmegaConf.load(hydra_yaml)
    num_hist = cfg.num_hist
    num_pred = cfg.num_pred
    frameskip = cfg.frameskip
    concat_dim = cfg.concat_dim
    print(
        f"config: num_hist={num_hist}  num_pred={num_pred}  "
        f"frameskip={frameskip}  concat_dim={concat_dim}"
    )

    # ── build model ─────────────────────────────────────────────────────────
    encoder = DinoV2Encoder(
        name=cfg.encoder.name, feature_key=cfg.encoder.feature_key
    )
    for param in encoder.parameters():
        param.requires_grad = False

    model = VWorldModel(
        image_size=cfg.img_size,
        num_hist=num_hist,
        num_pred=num_pred,
        encoder=encoder,
        proprio_encoder=ckpt["proprio_encoder"],
        action_encoder=ckpt["action_encoder"],
        decoder=ckpt.get("decoder"),
        predictor=ckpt["predictor"],
        proprio_dim=ckpt["proprio_encoder"].emb_dim,
        action_dim=ckpt["action_encoder"].emb_dim,
        concat_dim=concat_dim,
        num_action_repeat=cfg.num_action_repeat,
        num_proprio_repeat=cfg.num_proprio_repeat,
        train_encoder=False,
        train_predictor=False,
        train_decoder=False,
    ).to(device)
    model.eval()
    print(f"model ready — emb_dim={model.emb_dim}  epoch={ckpt.get('epoch', '?')}")
    has_decoder = ckpt.get("decoder") is not None and args.pixel_decoder

    # ── set up ManiSkill env + policy ───────────────────────────────────────
    zl_root = os.path.abspath(args.zero_latency_root)
    if zl_root not in sys.path:
        sys.path.insert(0, zl_root)

    import gymnasium as gym
    import mani_skill  # noqa: F401
    import pusht_explore  # noqa: F401
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from rl_train_pusht import Agent, PushTXYWrapper

    policy_ckpt = torch.load(
        args.policy_checkpoint, map_location=device, weights_only=False
    )
    targs = policy_ckpt["args"]

    control_mode = targs.get("control_mode", "pd_ee_delta_pose")
    action_scale = targs.get("action_scale", 0.1)
    full_state = targs.get("full_state", False)
    obs_mode = "state" if full_state else "none"

    env = gym.make(
        "PushT-XYExplore-v1",
        num_envs=1,  # single env for visualisation
        obs_mode=obs_mode,
        control_mode=control_mode,
        reward_mode="none",
        render_mode="rgb_array",
        robot_uids="panda_stick",
        ee_init_radius=targs.get("ee_init_radius", 0.0),
        ee_init_radius_min=targs.get("ee_init_radius_min", 0.12),
        sim_backend=args.sim_backend,
        max_episode_steps=args.max_episode_steps,
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

    agent = Agent(
        policy_ckpt["obs_dim"], policy_ckpt["act_dim"], targs["hidden_size"]
    ).to(device)
    agent.load_state_dict(policy_ckpt["agent"])
    agent.eval()

    proprio_dim = 2 if targs.get("no_ee_vel", False) else 4
    print(
        f"env ready — control_mode={control_mode}  action_scale={action_scale:.4f}  "
        f"proprio_dim={proprio_dim}  obs_dim={policy_ckpt['obs_dim']}"
    )

    # ── per-episode loop ────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)

    for ep in range(args.max_episodes):
        # Collect a full trajectory with the deterministic policy (no noise).
        obs, _ = venv.reset(seed=args.seed + ep)

        frames_rgb: list[torch.Tensor] = []  # each (H, W, 3) uint8 numpy
        frames_model: list[torch.Tensor] = []  # each (1, 3, 224, 224) [-1,1]
        states: list[torch.Tensor] = []  # each (1, S)
        raw_actions: list[torch.Tensor] = []  # each (1, 2) effective action

        for step in range(args.num_rollout_steps):
            # Render current scene — may be numpy (old ManiSkill) or tensor (≥3.x).
            rgb = venv.render()  # (1, H, W, 3) or (H, W, 3)
            rgb = torch.as_tensor(rgb, device="cpu")
            if rgb.ndim == 4:
                rgb = rgb[0]  # strip batch dim → (H, W, 3)
            frames_rgb.append(rgb)

            # Convert to model input format.
            rgb_t = rgb.float() / 255.0  # HWC [0,1]
            rgb_t = rearrange(rgb_t, "h w c -> 1 c h w")
            rgb_t = torch.nn.functional.interpolate(
                rgb_t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
            )
            rgb_t = (rgb_t - 0.5) / 0.5  # → [-1,1]
            frames_model.append(rgb_t.to(device))

            # Save state (compact policy observation).
            state_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if state_t.ndim == 1:
                state_t = state_t.unsqueeze(0)
            states.append(state_t)

            # Policy action (deterministic, no noise).
            obs_t = torch.nan_to_num(
                torch.as_tensor(obs, dtype=torch.float32, device=device)
            )
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)
            cmd = agent.actor_mean(obs_t).clamp(-1.0, 1.0)
            eff = env.effective_action(cmd)
            raw_actions.append(eff)

            obs, _, _, _, _ = venv.step(cmd)

        print(
            f"  episode {ep}: collected {len(frames_rgb)} frames"
        )

        # ── prepare model inputs (frameskip subsampling) ─────────────────
        T_raw = len(frames_rgb)
        # Trim to largest multiple of frameskip.
        n_sub = T_raw // frameskip
        raw_len = n_sub * frameskip

        sub_idx = list(range(0, raw_len, frameskip))

        img_sub = torch.cat([frames_model[i] for i in sub_idx], dim=0)  # (n_sub, 3, 224, 224)
        state_sub = torch.cat([states[i] for i in sub_idx], dim=0)  # (n_sub, S)

        # Concatenate actions across frameskip groups.
        act_raw = torch.cat(raw_actions[:raw_len], dim=0)  # (raw_len, 2)
        act_concat = rearrange(
            act_raw, "(n f) d -> n (f d)", f=frameskip
        )  # (n_sub, 2*fk)

        # Context: first num_hist subsampled frames.
        proprio = state_sub[:, :proprio_dim]
        obs_0 = {
            "visual": img_sub[:num_hist].unsqueeze(0),  # (1, num_hist, 3, 224, 224)
            "proprio": proprio[:num_hist].unsqueeze(0),  # (1, num_hist, P)
        }
        actions = act_concat.unsqueeze(0)  # (1, n_sub, 2*fk)

        # ── encode all GT frames for PCA reference ────────────────────────
        obs_all = {
            "visual": img_sub.unsqueeze(0),  # (1, n_sub, 3, 224, 224)
            "proprio": proprio.unsqueeze(0),  # (1, n_sub, P)
        }

        with torch.no_grad():
            # encode_obs returns {"visual": (1, T, P, D), "proprio": ...}
            z_gt_all = model.encode_obs(obs_all)
            z_gt_visual = z_gt_all["visual"]  # (1, n_sub, num_patches, emb_dim)

            # ── autoregressive rollout ─────────────────────────────────────
            # rollout returns (z_obses_dict, raw_z_tensor)
            z_obs_rollout, _ = model.rollout(obs_0, actions)
            z_pred_visual = z_obs_rollout["visual"]  # (1, n_rollout, P, D)

        # rollout includes context encoding + predictions.
        # First num_hist frames are the encoded context; the rest are predictions.
        n_rollout_frames = z_pred_visual.shape[1]

        # ── PCA visualisation ──────────────────────────────────────────────
        # Feature grid: VWorldModel resizes images to (img_size // decoder_scale) * patch_size
        # before encoding. With 224² and decoder_scale=16: 14×14 grid.
        h_feat = w_feat = cfg.img_size // VQVAE_DECODER_SCALE  # 14
        gt_pca, pred_pca = feats_to_pca(
            z_gt_visual[0], z_pred_visual[0], h_feat, w_feat
        )

        # ── pixel decoding (optional) ──────────────────────────────────────
        pred_frames_rgb = None
        if has_decoder:
            with torch.no_grad():
                rollout_decoded, _ = model.decode_obs(z_obs_rollout)
            pred_frames_rgb = rollout_decoded["visual"][0]  # (n_rollout, 3, 224, 224)

        # ── build video ────────────────────────────────────────────────────
        target_h = IMG_SIZE
        target_w = IMG_SIZE
        gap = 2  # white separator width

        out_mp4 = os.path.join(args.out, f"pusht_rollout_{ep:02d}.mp4")
        fps_video = args.fps
        writer = imageio.get_writer(out_mp4, fps=fps_video, macro_block_size=None)

        for t in range(raw_len):
            sub = t // frameskip  # which subsampled step this raw frame belongs to

            # GT RGB panel — raw render, resized to target.
            gt_rgb = frames_rgb[t].cpu().numpy()
            import cv2
            if gt_rgb.shape[0] != target_h or gt_rgb.shape[1] != target_w:
                gt_rgb = cv2.resize(
                    gt_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )

            # GT PCA panel — from subsampled step.
            gt_pca_img = _pca_to_uint8(gt_pca[sub], target_h, target_w)

            # Pred PCA panel — hold prediction at subsampled step.
            pk = sub  # in rollout, prediction at index `sub`
            if pk < len(pred_pca):
                pred_pca_img = _pca_to_uint8(pred_pca[pk], target_h, target_w)
            else:
                pred_pca_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            # Assemble row 1: GT-RGB | GT-PCA | Pred-PCA
            sep = np.ones((target_h, gap, 3), dtype=np.uint8) * 255
            row1 = np.concatenate([gt_rgb, sep, gt_pca_img, sep, pred_pca_img], axis=1)

            rows = [row1]

            # Optional row 2: GT-RGB | Pred-RGB
            if pred_frames_rgb is not None:
                pred_rgb = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                if pk < len(pred_frames_rgb):
                    pred_rgb = _tensor_to_uint8(pred_frames_rgb[pk])
                    if pred_rgb.shape[0] != target_h:
                        pred_rgb = cv2.resize(
                            pred_rgb, (target_w, target_h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                row2 = np.concatenate([gt_rgb, sep, pred_rgb], axis=1)
                # Pad row2 to match row1 width.
                if row2.shape[1] < row1.shape[1]:
                    pad_w = row1.shape[1] - row2.shape[1]
                    row2 = np.concatenate(
                        [row2, np.ones((target_h, pad_w, 3), dtype=np.uint8) * 255],
                        axis=1,
                    )
                rows.append(row2)

            frame = np.concatenate(rows, axis=0)
            writer.append_data(frame)

        writer.close()
        print(f"    wrote {out_mp4}  ({raw_len} frames)")

    # ── cleanup ─────────────────────────────────────────────────────────────
    venv.close()
    print(f"DONE → {args.out}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="DINO-WM autoregressive rollout videos (PushT online)"
    )
    ap.add_argument(
        "--ckpt", required=True, help="path to the checkpoints/ directory"
    )
    ap.add_argument(
        "--ckpt-suffix", default="_latest", help="checkpoint suffix (default _latest)"
    )
    ap.add_argument(
        "--zero-latency-root",
        default=os.path.join(REPO, "..", "zero_latency_interface"),
        help="path to zero_latency_interface repo root",
    )
    ap.add_argument(
        "--policy-checkpoint",
        default=os.path.join(
            REPO, "..", "zero_latency_interface", "checkpoints", "policy.pt"
        ),
        help="path to trained PPO policy checkpoint",
    )
    ap.add_argument(
        "--max-episodes", type=int, default=5, help="number of episodes to render"
    )
    ap.add_argument(
        "--num-rollout-steps",
        type=int,
        default=100,
        help="env steps per collected episode",
    )
    ap.add_argument(
        "--max-episode-steps", type=int, default=100, help="per-episode horizon"
    )
    ap.add_argument(
        "--sim-backend", default="physx_cuda", help="ManiSkill simulation backend"
    )
    ap.add_argument("--fps", type=int, default=20, help="output video FPS")
    ap.add_argument(
        "--pixel-decoder",
        action="store_true",
        help="include pixel-decoded prediction row",
    )
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument(
        "--out",
        default=os.path.join(REPO, "logs", "viz_pusht_rollout"),
        help="output directory for MP4 files",
    )
    args = ap.parse_args()
    viz_pusht_rollout(args)


if __name__ == "__main__":
    main()
