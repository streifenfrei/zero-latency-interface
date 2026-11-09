#!/usr/bin/env python
"""Composite grid of the ZLI's DISPLAYED frames for the 3 study delays.

One fixed GT trajectory (deterministic PPO policy rollout, seeded) is run
four times: once to record the live GT frames (top row) and once per delay
(10/15/20 raw ticks = 500/750/1000 ms) recording what the ZLI displays.

The delay rows use the deterministic grounding schedule: the display runs
OPEN-LOOP for L steps (L = 11/17/23 for the three delays, frameskip 1),
then receives a fresh context grounded on the GT frame from L steps ago
and rolled forward to the present with the L known actions — then open
loop again.  The frames produced from a NEW context (one per L ticks) are
marked with a red border.  The loop runs under the study's strict
real-time tick cap (``--tick-hz``, 20 Hz); ticks whose work exceeds the
budget fall behind, exactly like the study.

Columns are sparse frames of the trajectory (every ``--stride``-th tick),
so a column shows all views at the same simulated instant.

Headless: no window is opened.  Needs the same launcher environment as the
other SAPIEN scripts (run via deployment/run_viz_zli.sh).

Usage:
    deployment/run_viz_zli.sh \\
        --ckpt outputs/outputs/2026-09-01/17-55-14/checkpoints \\
        --out logs/zli_display_grid.png
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import cv2
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from deployment.interface import _put_text
from deployment.wm_adapter import DinoWMPushtAdapter


# ── env + policy (mirror of viz_rollout.py's construction) ────────────────────

def _build_env_policy(args, device):
    """(env, venv, agent, proprio_dim) — raw wrapper path for policy actions.

    The policy emits normalized [-1, 1] commands; ManiSkillPushTEnv.step
    expects pixel space, so this mirrors viz_rollout.py's gym.make +
    PushTXYWrapper + ManiSkillVectorEnv construction, with two deltas:
    auto_reset=False and ignore_terminations=True so exactly ``--ticks``
    ticks run on one uninterrupted trajectory.
    """
    zl_root = os.path.abspath(args.zero_latency_root)
    if zl_root not in sys.path:
        sys.path.insert(0, zl_root)

    import gymnasium as gym
    import mani_skill  # noqa: F401
    import pusht_explore  # noqa: F401
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from rl_train_pusht import Agent, PushTXYWrapper

    policy_ckpt = torch.load(args.policy_checkpoint, map_location=device,
                             weights_only=False)
    targs = policy_ckpt["args"]
    control_mode = targs.get("control_mode", "pd_ee_delta_pose")
    action_scale = targs.get("action_scale", 0.1)
    full_state = targs.get("full_state", False)
    obs_mode = "state" if full_state else "none"

    env = gym.make(
        "PushT-XYExplore-v1",
        num_envs=1,
        obs_mode=obs_mode,
        control_mode=control_mode,
        reward_mode="none",
        render_mode="rgb_array",
        robot_uids="panda_stick",
        ee_init_radius=targs.get("ee_init_radius", 0.0),
        ee_init_radius_min=targs.get("ee_init_radius_min", 0.12),
        sim_backend=args.sim_backend,
        max_episode_steps=100_000,
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
        env, auto_reset=False, ignore_terminations=True, record_metrics=False
    )

    agent = Agent(policy_ckpt["obs_dim"], policy_ckpt["act_dim"],
                  targs["hidden_size"]).to(device)
    agent.load_state_dict(policy_ckpt["agent"])
    agent.eval()

    proprio_dim = (2 if targs.get("no_ee_vel", False) else 4) \
        + (4 if args.with_block_pose else 0)
    print(f"env ready — control={control_mode}  "
          f"action_scale={action_scale:.4f}  proprio_dim={proprio_dim}  "
          f"obs_dim={policy_ckpt['obs_dim']}")
    return env, venv, agent, proprio_dim


def _render_hwc(venv) -> np.ndarray:
    """venv.render() → (H, W, 3) uint8 numpy."""
    rgb = venv.render()
    rgb = torch.as_tensor(rgb, device="cpu")
    if rgb.ndim == 4:
        rgb = rgb[0]  # strip batch
    return rgb.clamp(0, 255).byte().numpy()


def _paint_tee(task, rgb=(230, 15, 25)) -> None:
    """Paint the pushed T-block a vivid red (render-material override).

    The block is built with base_color (194, 19, 22) already — this makes
    it an unambiguous bright red in the renders.  Must be applied before
    the first reset so the scene materials are captured with the new color.
    """
    import sapien
    color = np.array([*rgb, 255], dtype=np.float32) / 255.0
    for obj in task.tee._objs:
        rb = obj.find_component_by_type(sapien.render.RenderBodyComponent)
        for shape in rb.render_shapes:
            shape.material.set_base_color(color)
    print(f"painted T-block base_color=({rgb[0]}, {rgb[1]}, {rgb[2]})")


def _obs_tensor(obs, device) -> torch.Tensor:
    obs_t = torch.nan_to_num(torch.as_tensor(obs, dtype=torch.float32,
                                             device=device))
    if obs_t.ndim == 1:
        obs_t = obs_t.unsqueeze(0)
    return obs_t


# ── trajectory rows ───────────────────────────────────────────────────────────

def _priming(wm, venv, proprio_dim: int, seed: int):
    """Context priming: (num_hist-1)*frameskip zero-action steps.

    Returns (obs, hist_vis, hist_prop, actions, offset).  ``actions`` is
    indexed by ABSOLUTE raw timestep from 0 (the study's invariant — the
    async grounding worker looks actions up by timestep value).
    """
    fs, H = wm.frameskip, wm.num_hist
    obs, _ = venv.reset(seed=seed)
    hist_vis = [_render_hwc(venv)]
    hist_prop = [obs.detach().cpu().numpy()[0, :proprio_dim]
                 .astype(np.float32)]
    actions: list = []
    for _ in range((H - 1) * fs):
        obs, _, _, _, _ = venv.step(
            torch.zeros((1, 2), device=venv.device))
        # Zero action → effective_action is zero (scaled_delta(0) = 0).
        actions.append(np.zeros(2, dtype=np.float32))
        hist_vis.append(_render_hwc(venv))
        hist_prop.append(obs.detach().cpu().numpy()[0, :proprio_dim]
                         .astype(np.float32))
    return obs, hist_vis, hist_prop, actions, len(actions)


def _collect_gt_row(wm, venv, agent, proprio_dim: int, args) -> list:
    """Live GT frames of the fixed policy trajectory (post-step renders)."""
    obs, *_ = _priming(wm, venv, proprio_dim, args.seed)
    frames = []
    t_start = time.perf_counter()
    deadline = t_start + 1.0 / args.tick_hz
    for t in range(args.ticks):
        cmd = agent.actor_mean(_obs_tensor(obs, venv.device)).clamp(-1.0, 1.0)
        obs, _, _, _, _ = venv.step(cmd)
        frames.append(_render_hwc(venv))
        if t % 20 == 19:
            print(f"  [GT] t={t}", flush=True)
        # Strict real-time cap (study pacing): tick wall time = 1/tick_hz.
        rem = deadline - time.perf_counter()
        if rem > 0:
            time.sleep(rem)
        deadline += 1.0 / args.tick_hz
    print(f"  [GT] achieved {args.ticks / (time.perf_counter() - t_start):.1f} Hz",
          flush=True)
    return frames


def _collect_delay_row(wm, env, venv, agent, proprio_dim: int, args,
                       roll_len: int) -> tuple[list, list]:
    """(displayed, grounded_ticks) frames for one delay.

    Deterministic grounding schedule: the display runs open-loop for
    ``roll_len`` ticks, then re-grounds on the GT context from ``roll_len``
    ticks ago and rolls forward to the present with the known actions —
    the frame produced from each new context is recorded in
    ``grounded_ticks`` (one per ``roll_len`` ticks).
    """
    fs = wm.frameskip
    H = wm.num_hist
    obs, hist_vis, hist_prop, actions, offset = \
        _priming(wm, venv, proprio_dim, args.seed)
    # Initial context (priming frames) — the first roll_len ticks are pure
    # open-loop from here, then the context is refreshed every roll_len.
    wm.reset(hist_vis[::fs][-H:], hist_prop[::fs][-H:])

    frames = []
    grounded_ticks = []
    t_start = time.perf_counter()
    deadline = t_start + 1.0 / args.tick_hz
    for t in range(args.ticks):
        t_abs = offset + t
        cmd = agent.actor_mean(_obs_tensor(obs, venv.device)) \
            .clamp(-1.0, 1.0)
        eff = np.asarray(env.effective_action(cmd)
                         .squeeze(0).detach().cpu(), dtype=np.float32)
        obs, _, _, _, _ = venv.step(cmd)
        live = _render_hwc(venv)
        proprio = obs.detach().cpu().numpy()[0, :proprio_dim] \
            .astype(np.float32)
        actions.append(eff)
        hist_vis.append(live)
        hist_prop.append(proprio)

        if t % roll_len == 0 and t_abs - roll_len - (H - 1) * fs >= 0:
            # New context: ground on the GT frames ending roll_len steps
            # ago, then roll forward to the present with the known actions.
            ctx_end = t_abs - roll_len
            # Context frames END at the GT render from roll_len steps ago
            # (hist_vis[j] = render after raw step j-1), subsampled at the
            # training stride — frameskip apart, num_hist of them.
            wm.reset(hist_vis[ctx_end - (H - 1) * fs + 1:ctx_end + 2:fs],
                     hist_prop[ctx_end - (H - 1) * fs + 1:ctx_end + 2:fs])
            display = None
            for k in range(ctx_end + 1, t_abs + 1):
                display = wm.predict(actions[k])
            grounded_ticks.append(t)
        else:
            display = wm.predict(eff)
        frames.append(display)

        if t % 20 == 19:
            print(f"  [L{roll_len}] t={t} grounded="
                  f"{grounded_ticks[-1] if grounded_ticks else -1}",
                  flush=True)
        # Strict real-time tick cap (the study's pacing): ticks whose work
        # exceeds the budget (the grounding rolls are expensive) fall behind.
        rem = deadline - time.perf_counter()
        if rem > 0:
            time.sleep(rem)
        deadline += 1.0 / args.tick_hz

    print(f"  [L{roll_len}] achieved "
          f"{args.ticks / (time.perf_counter() - t_start):.1f} Hz; "
          f"groundings at t={grounded_ticks}", flush=True)
    return frames, grounded_ticks[::max(args.mark_stride, 1)]


# ── grid assembly (viz_rollout panel pattern: white separators) ───────────────

def _assemble_grid(rows, tick_idxs, args) -> None:
    """rows: [(label, sublabel, frames, marks)] → labelled grid PNG.

    ``marks`` (optional) lists trajectory ticks whose panels get a green
    border — the frames where the rollout was re-grounded.
    """
    panel_h, gap = args.panel_h, 2
    MARK_COLOR = (0, 0, 255)  # red border: re-grounding
    row_imgs = []
    for _label, _sub, frames, marks in rows:
        # Publishes usually fall between sampled columns — mark the first
        # sampled column at-or-after each publish tick.
        mark_set = set()
        for p in (marks or []):
            for i in tick_idxs:
                if i >= p:
                    mark_set.add(i)
                    break
        panels = []
        for i in tick_idxs:
            p = cv2.resize(frames[i], (panel_h, panel_h),
                           interpolation=cv2.INTER_LINEAR)
            if i in mark_set:
                cv2.rectangle(p, (1, 1), (panel_h - 2, panel_h - 2),
                              MARK_COLOR, 3)
            panels.append(p)
        vsep = np.ones((panel_h, gap, 3), dtype=np.uint8) * 255
        parts = []
        for p in panels:
            parts += [p, vsep]
        row_imgs.append(np.concatenate(parts[:-1], axis=1))

    hsep = np.ones((gap, row_imgs[0].shape[1], 3), dtype=np.uint8) * 255
    parts = []
    for r in row_imgs:
        parts += [r, hsep]
    block = np.concatenate(parts[:-1], axis=0)

    margin_w = 110
    margin = np.ones((block.shape[0], margin_w, 3), dtype=np.uint8) * 255
    for i, (label, sub, _frames, _marks) in enumerate(rows):
        y = i * (panel_h + gap) + panel_h // 2 - 8
        _put_text(margin, label, (8, y), color=(0, 0, 0), scale=0.5)
        if sub:
            _put_text(margin, sub, (8, y + 18), color=(90, 90, 90),
                      scale=0.45)
    canvas = np.concatenate([margin, block], axis=1)

    header_h = 24
    header = np.ones((header_h, canvas.shape[1], 3), dtype=np.uint8) * 255
    for ci, i in enumerate(tick_idxs):
        x = margin_w + ci * (panel_h + gap) + panel_h // 2 - 26
        _put_text(header, f"t={i / args.tick_hz:.1f}s", (x, 16),
                  color=(0, 0, 0), scale=0.45)
    canvas = np.concatenate([header, canvas], axis=0)

    # Frame sources emit RGB (the env render and the WM decoder both do);
    # cv2 expects BGR — without this swap every saved image has R and B
    # exchanged (the red T-block rendered blue).
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, canvas)
    print(f"wrote {out} ({canvas.shape[1]}x{canvas.shape[0]})")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Grid image of ZLI-displayed frames for 3 delays on a "
                    "fixed policy trajectory")
    ap.add_argument("--ckpt", default=os.path.join(
        REPO, "outputs", "outputs", "2026-09-01", "17-55-14", "checkpoints"),
        help="checkpoints/ dir of the WM (default: 224/fs1/hist3 "
             "frameskip-1 checkpoint)")
    ap.add_argument("--ckpt-suffix", default="_latest")
    ap.add_argument("--policy-checkpoint", default=os.path.join(
        REPO, "..", "zero_latency_interface", "checkpoints", "policy.pt"))
    ap.add_argument("--zero-latency-root", default=os.path.join(
        REPO, "..", "zero_latency_interface"))
    ap.add_argument("--delays", nargs="+", type=int, default=[10, 15, 20],
                    help="delays in raw ticks (ms = ticks * 1000 / tick-hz)")
    ap.add_argument("--roll-lengths", nargs="+", type=int,
                    default=[11, 17, 23],
                    help="grounding rollout length per delay (open-loop "
                         "steps between context updates; must match the "
                         "number of --delays; default: 11 17 23)")
    ap.add_argument("--ticks", type=int, default=40,
                    help="rollout length in ticks (default 40 = 2 s at "
                         "20 Hz)")
    ap.add_argument("--stride", type=int, default=4,
                    help="column stride in ticks (default 4 = 5 frames/s "
                         "at 20 Hz)")
    ap.add_argument("--seed", type=int, default=None,
                    help="trajectory seed (default: random per run — pass "
                         "a value to reproduce a previous figure; the seed "
                         "is shared across all rows of one run)")
    ap.add_argument("--out", default=os.path.join(REPO, "logs",
                                                  "zli_display_grid.png"))
    ap.add_argument("--panel-h", type=int, default=168)
    ap.add_argument("--mark-stride", type=int, default=1,
                    help="mark every Nth re-grounding publish (default 1 = "
                         "every publish; on slow GPUs groundings complete "
                         "nearly every tick, so raise this to thin the "
                         "markers)")
    ap.add_argument("--tick-hz", type=float, default=20.0,
                    help="strict real-time tick rate (the study's pacing — "
                         "each tick's wall time is capped at 1/tick-hz, so "
                         "the main-thread prediction contends with the "
                         "grounding worker)")
    ap.add_argument("--sim-backend", default="physx_cuda")
    ap.add_argument("--with-block-pose", action="store_true")
    ap.add_argument("--red-block", action="store_true",
                    help="paint the pushed T-block a vivid red (the GT rows "
                         "reflect it; the WM rows keep the model's learned "
                         "appearance)")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()
    if len(args.roll_lengths) != len(args.delays):
        ap.error(f"--roll-lengths ({len(args.roll_lengths)} values) must "
                 f"match --delays ({len(args.delays)} values)")
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
    print(f"seed = {args.seed} (re-run with --seed {args.seed} to "
          f"reproduce this figure)", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, venv, agent, proprio_dim = _build_env_policy(args, device)
    if args.red_block:
        _paint_tee(env.base)
    wm = DinoWMPushtAdapter(ckpt_dir=args.ckpt, ckpt_suffix=args.ckpt_suffix,
                            use_amp=args.amp)

    # Columns: the regular --stride sampling PLUS every grounding tick, so
    # the bordered (new-context) frame is actually shown next to a drifted
    # open-loop frame — otherwise the snap lands between sampled columns.
    tick_idxs = sorted(set(range(0, args.ticks, args.stride))
                       | {g for L in args.roll_lengths
                          for g in range(L, args.ticks, L)})
    print(f"GT row: {args.ticks} ticks, seed={args.seed}", flush=True)
    gt_frames = _collect_gt_row(wm, venv, agent, proprio_dim, args)
    rows = [("GT", "", gt_frames, None)]
    for d, L in zip(args.delays, args.roll_lengths):
        ms = d * 1000 / args.tick_hz
        print(f"delay row: {d} ticks ({ms:.0f} ms), rollout L={L}",
              flush=True)
        displayed, grounded = _collect_delay_row(
            wm, env, venv, agent, proprio_dim, args, L)
        rows.append((f"{ms:.0f} ms", "", displayed, grounded))

    venv.close()
    _assemble_grid(rows, tick_idxs, args)


if __name__ == "__main__":
    main()
