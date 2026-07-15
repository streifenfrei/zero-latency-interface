#!/usr/bin/env python
"""Profile DINO-WM PushT model latency, throughput, and memory on a GPU.

Loads a trained PushT checkpoint; runs timed forward/encode/predict/decode/rollout
passes with warmup; reports parameter counts, per-op latency (mean ± std),
rollout throughput (predicted frames/sec), and peak GPU memory.

Usage:
  python deployment/profile.py \
    --ckpt outputs/outputs/2026-07-06/18-46-23/checkpoints \
    --n_warmup 10 --n_timed 50
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf

# Ensure repo root is on sys.path.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from dino_wm.models.visual_world_model import VWorldModel
from dino_wm.models.dino import DinoV2Encoder


# ── helpers ──────────────────────────────────────────────────────────────────
def _params(module) -> int:
    """Total parameter count for a module (or 0 if None)."""
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def _trainable_params(module) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _fmt(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def _fmt_time(ms: float) -> str:
    if ms >= 1000:
        return f"{ms/1000:.2f} s"
    return f"{ms:.2f} ms"


def _measure(func, n_warmup=10, n_timed=50) -> dict:
    """Run *func* n_warmup + n_timed times; return {mean_ms, std_ms, times_ms}."""
    for _ in range(n_warmup):
        func()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(n_timed):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    arr = np.array(times)
    return {"mean_ms": float(arr.mean()), "std_ms": float(arr.std()),
            "min_ms": float(arr.min()), "max_ms": float(arr.max()),
            "times_ms": arr}


# ── main ─────────────────────────────────────────────────────────────────────
def profile(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"device: {device}  ({gpu_name})")
    print(f"warmup={args.n_warmup}  timed={args.n_timed}")

    # --- load model -----------------------------------------------------------
    ckpt_path = os.path.join(args.ckpt, f"model{args.ckpt_suffix}.pth")
    print(f"\nloading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    hydra_yaml = os.path.join(os.path.dirname(args.ckpt.rstrip("/")), "hydra.yaml")
    cfg = OmegaConf.load(hydra_yaml)

    encoder = DinoV2Encoder(
        name=cfg.encoder.name, feature_key=cfg.encoder.feature_key)
    for p in encoder.parameters():
        p.requires_grad = False

    model = VWorldModel(
        image_size=cfg.img_size,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
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
    ).to(device)
    model.eval()

    num_hist = cfg.num_hist
    frameskip = cfg.frameskip
    action_dim = ckpt["action_encoder"].in_chans   # 2 * frameskip = 10
    proprio_dim = ckpt["proprio_encoder"].in_chans  # 4

    print(f"config: num_hist={num_hist}  frameskip={frameskip}  "
          f"action_dim={action_dim}  proprio_dim={proprio_dim}")

    # --- parameter counts -----------------------------------------------------
    print(f"\n{'='*60}")
    print(f"{'Component':<30} {'Params':>12} {'Trainable':>12}")
    print(f"{'='*60}")
    components = [
        ("encoder (DINOv2 ViT-S/14)", model.encoder),
        ("proprio_encoder", model.proprio_encoder),
        ("action_encoder", model.action_encoder),
        ("predictor (ViT, 6L/16H)", model.predictor),
        ("decoder (VQ-VAE)", model.decoder),
    ]
    total, total_trainable = 0, 0
    for name, mod in components:
        p = _params(mod)
        t = _trainable_params(mod)
        total += p
        total_trainable += t
        print(f"{name:<30} {_fmt(p):>12} {_fmt(t):>12}")
    print(f"{'─'*60}")
    print(f"{'TOTAL':<30} {_fmt(total):>12} {_fmt(total_trainable):>12}")

    # --- build synthetic inputs matching PushT shapes --------------------------
    B = args.batch_size
    H = num_hist

    with torch.no_grad():
        # Context inputs
        visual_ctx = torch.randn(B, H, 3, cfg.img_size, cfg.img_size, device=device)
        proprio_ctx = torch.randn(B, H, proprio_dim, device=device)
        act_ctx = torch.randn(B, H, action_dim, device=device)

        obs_0 = {"visual": visual_ctx, "proprio": proprio_ctx}

        # Full forward inputs: num_hist + num_pred frames
        visual_full = torch.randn(B, H + cfg.num_pred, 3, cfg.img_size, cfg.img_size,
                                  device=device)
        proprio_full = torch.randn(B, H + cfg.num_pred, proprio_dim, device=device)
        act_full = torch.randn(B, H + cfg.num_pred, action_dim, device=device)
        obs_full = {"visual": visual_full, "proprio": proprio_full}

        # Rollout: actions for a longer horizon (e.g. 20 steps)
        n_rollout_steps = args.rollout_horizon
        act_rollout = torch.randn(B, H + n_rollout_steps, action_dim, device=device)

    # --- warmup (one forward pass to init CUDA graphs / cache) -----------------
    print(f"\nwarming up ...")
    with torch.no_grad():
        _ = model(obs_full, act_full)
        _ = model.rollout(obs_0, act_rollout)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # --- benchmark: encode ----------------------------------------------------
    print(f"\n{'='*60}")
    print(f"LATENCY  (batch={B}, num_hist={H})")
    print(f"{'='*60}")

    def _time_encode():
        with torch.no_grad():
            model.encode(obs_0, act_ctx)

    r = _measure(_time_encode, args.n_warmup, args.n_timed)
    print(f"  encode                {_fmt_time(r['mean_ms']):>10s}  ±{r['std_ms']:.2f} ms  "
          f"[{r['min_ms']:.2f}–{r['max_ms']:.2f}]")

    # --- benchmark: predict ---------------------------------------------------
    # Pre-compute encoded context
    with torch.no_grad():
        z_ctx = model.encode(obs_0, act_ctx)

    def _time_predict():
        with torch.no_grad():
            model.predict(z_ctx)

    r = _measure(_time_predict, args.n_warmup, args.n_timed)
    print(f"  predict               {_fmt_time(r['mean_ms']):>10s}  ±{r['std_ms']:.2f} ms  "
          f"[{r['min_ms']:.2f}–{r['max_ms']:.2f}]")

    # --- benchmark: decode ----------------------------------------------------
    with torch.no_grad():
        z_obs_full = model.encode_obs(obs_full)

    def _time_decode():
        with torch.no_grad():
            model.decode_obs(z_obs_full)

    r = _measure(_time_decode, args.n_warmup, args.n_timed)
    print(f"  decode                {_fmt_time(r['mean_ms']):>10s}  ±{r['std_ms']:.2f} ms  "
          f"[{r['min_ms']:.2f}–{r['max_ms']:.2f}]")

    # --- benchmark: full forward ----------------------------------------------
    def _time_forward():
        with torch.no_grad():
            model(obs_full, act_full)

    r = _measure(_time_forward, args.n_warmup, args.n_timed)
    fps = 1000 / r['mean_ms'] * B
    print(f"  forward (full)        {_fmt_time(r['mean_ms']):>10s}  ±{r['std_ms']:.2f} ms  "
          f"[{r['min_ms']:.2f}–{r['max_ms']:.2f}]  → {fps:.1f} samp/s")

    # --- benchmark: rollout step ----------------------------------------------
    def _time_rollout():
        with torch.no_grad():
            model.rollout(obs_0, act_rollout)

    r = _measure(_time_rollout, args.n_warmup, args.n_timed)
    # rollout produces H + n_rollout_steps + 1 frames of latents
    n_pred_frames = n_rollout_steps + 1  # predicted frames
    fps_rollout = n_pred_frames / (r['mean_ms'] / 1000) * B
    print(f"  rollout ({n_rollout_steps} steps)  {_fmt_time(r['mean_ms']):>10s}  ±{r['std_ms']:.2f} ms  "
          f"[{r['min_ms']:.2f}–{r['max_ms']:.2f}]")
    print(f"    → {n_pred_frames} pred frames in {r['mean_ms']:.1f} ms  "
          f"= {fps_rollout:.1f} pred fps")

    # --- benchmark: rollout + decode (end-to-end) -----------------------------
    def _time_rollout_decode():
        with torch.no_grad():
            z_obs, _ = model.rollout(obs_0, act_rollout)
            model.decode_obs(z_obs)

    r = _measure(_time_rollout_decode, args.n_warmup, args.n_timed)
    fps_e2e = n_pred_frames / (r['mean_ms'] / 1000) * B
    print(f"  rollout + decode      {_fmt_time(r['mean_ms']):>10s}  ±{r['std_ms']:.2f} ms  "
          f"[{r['min_ms']:.2f}–{r['max_ms']:.2f}]")
    print(f"    → {n_pred_frames} pred frames (latents+pixels) = {fps_e2e:.1f} fps")

    # --- benchmark: per-predict-step breakdown --------------------------------
    print(f"\n  ── per-step breakdown (1 predict + action replace + cat) ──")

    with torch.no_grad():
        z_cur = model.encode(obs_0, act_ctx)    # (1, H, N, D)
        single_act = act_rollout[:, H:H+1, :]   # (1, 1, action_dim)

    def _time_one_step():
        with torch.no_grad():
            z_pred = model.predict(z_cur[:, -H:])
            z_new = z_pred[:, -1:, ...]
            z_new = model.replace_actions_from_z(z_new, single_act)
            # torch.cat is essentially free; included anyway
            _ = torch.cat([z_cur, z_new], dim=1)

    r = _measure(_time_one_step, args.n_warmup, args.n_timed)
    fps_step = 1000 / r['mean_ms'] * B
    print(f"  1 rollout step        {_fmt_time(r['mean_ms']):>10s}  ±{r['std_ms']:.2f} ms  "
          f"→ {fps_step:.1f} steps/s")

    # --- GPU memory -----------------------------------------------------------
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        print(f"\n{'='*60}")
        print(f"GPU MEMORY")
        print(f"{'='*60}")
        print(f"  peak allocated : {peak:.0f} MiB")
        print(f"  peak reserved  : {reserved:.0f} MiB")

    # --- summary table --------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  GPU                 : {gpu_name}")
    print(f"  Total params        : {_fmt(total)} ({_fmt(total_trainable)} trainable)")
    print(f"  DINO encoder        : {_fmt(_params(model.encoder))} (frozen)")
    print(f"  ViT predictor       : {_fmt(_params(model.predictor))}")
    print(f"  VQ-VAE decoder      : {_fmt(_params(model.decoder))}")
    print(f"  Rollout fps (B={B}) : {fps_rollout:.1f} pred-frames/s (latent only)")
    print(f"  Rollout+decode fps  : {fps_e2e:.1f} pred-frames/s (end-to-end)")

    print(f"\nDONE")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Profile DINO-WM PushT model latency & throughput")
    ap.add_argument("--ckpt", required=True,
                    help="path to checkpoints/ directory")
    ap.add_argument("--ckpt_suffix", default="_latest")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--n_warmup", type=int, default=10)
    ap.add_argument("--n_timed", type=int, default=50)
    ap.add_argument("--rollout_horizon", type=int, default=20,
                    help="number of autoregressive steps for rollout benchmark")
    args = ap.parse_args()
    profile(args)


if __name__ == "__main__":
    main()
