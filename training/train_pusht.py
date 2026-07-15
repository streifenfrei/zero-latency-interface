#!/usr/bin/env python
"""Training entry point for DINO-WM world model on PushT.

Supports both offline and online training modes:

    # Offline training (uses pre-recorded PushT dataset):
    python training/train_pusht.py --offline --epochs 100

    # Online training (generates data live with PPO policy + noise ladder):
    python training/train_pusht.py --online --epochs 100 --num-envs 64

This script is a thin orchestration layer over dino_wm's train.py. It ensures
the repo root is on sys.path so that custom modules (training.pusht_online,
deployment.*) are importable, and delegates to dino_wm's Hydra-based Trainer.

Requirements:
    - Offline: DATASET_DIR env var pointing to PushT dataset
    - Online:  ZERO_LATENCY_ROOT env var + trained PPO policy checkpoint
"""

import argparse
import os
import subprocess
import sys

# Ensure repo root is on sys.path.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DINO_WM = os.path.join(REPO, "dino_wm")

if REPO not in sys.path:
    sys.path.insert(0, REPO)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DINO-WM world model on PushT"
    )

    # Mode (mutually exclusive).
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true",
                      help="Use pre-recorded offline PushT dataset")
    mode.add_argument("--online", action="store_true",
                      help="Use online data generation (ManiSkill + PPO policy)")

    # Training hyperparameters.
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Global batch size (default: 32)")
    parser.add_argument("--frameskip", type=int, default=5,
                        help="Env steps per stored frame (default: 5)")
    parser.add_argument("--num-hist", type=int, default=3,
                        help="Context frames (default: 3)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")

    # Online-specific.
    parser.add_argument("--num-envs", type=int, default=64,
                        help="Parallel envs / noise-ladder rungs (online only, default: 64)")
    parser.add_argument("--samples-per-epoch", type=int, default=50000,
                        help="Samples per epoch (online only, default: 50000)")
    parser.add_argument("--zero-latency-root", type=str, default=None,
                        help="Path to zero_latency_interface repo (online only; "
                             "defaults to $ZERO_LATENCY_ROOT or ../zero_latency_interface)")

    # Paths.
    parser.add_argument("--ckpt-base-path", type=str,
                        default=os.path.join(REPO, "outputs"),
                        help="Base path for saving checkpoints (default: outputs/)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a specific checkpoint path")

    # Misc.
    parser.add_argument("--extra-hydra", type=str, nargs="*", default=[],
                        help="Additional Hydra overrides (e.g. 'training.lr=0.001')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the command without executing")

    args = parser.parse_args()

    # Determine environment config.
    if args.online:
        env_name = "pusht_online"
        # Resolve zero_latency_root.
        zl_root = args.zero_latency_root or os.environ.get("ZERO_LATENCY_ROOT")
        if zl_root is None:
            zl_root = os.path.join(REPO, "..", "zero_latency_interface")
        zl_root = os.path.abspath(zl_root)
        os.environ["ZERO_LATENCY_ROOT"] = zl_root
        if not os.path.isdir(zl_root):
            print(f"WARNING: zero_latency_interface not found at {zl_root}")
    else:
        env_name = "pusht"
        # DATASET_DIR is required for offline.
        if "DATASET_DIR" not in os.environ:
            print("WARNING: DATASET_DIR not set. Set it to the PushT dataset path.")

    # Build Hydra overrides.
    overrides = [
        f"env={env_name}",
        f"training.seed={args.seed}",
        f"training.batch_size={args.batch_size}",
        f"training.epochs={args.epochs}",
        f"frameskip={args.frameskip}",
        f"num_hist={args.num_hist}",
        f"ckpt_base_path={os.path.abspath(args.ckpt_base_path)}",
    ]

    if args.online:
        overrides += [
            f"env.dataset.num_envs={args.num_envs}",
            f"env.dataset.samples_per_epoch={args.samples_per_epoch}",
        ]

    if args.resume:
        overrides.append(f"resume={args.resume}")

    if args.extra_hydra:
        overrides.extend(args.extra_hydra)

    # Build command.
    train_script = os.path.join(DINO_WM, "train.py")
    cmd = [
        sys.executable, train_script,
        "--config-name", "train.yaml",
    ] + overrides

    if args.dry_run:
        print("Would run:")
        print("  " + " \\\n   ".join(cmd))
        return

    print(f"[train_pusht] mode={env_name}  epochs={args.epochs}  "
          f"batch={args.batch_size}  fs={args.frameskip}  hist={args.num_hist}")
    print(f"[train_pusht] running: {' '.join(cmd)}")

    # Run dino_wm's train.py as a subprocess.
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    subprocess.run(cmd, env=env, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
