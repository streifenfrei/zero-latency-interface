# Zero-Latency Interface (ZLI)

Training orchestration and deployment interface for the
[DINO-WM](https://github.com/gaoyuezhou/dino_wm) world model on the PushT task.

## Overview

This repo provides two main packages around the `dino_wm` core library (included
as a git submodule):

- **`training/`** — PushT world-model training orchestration (offline + online data generation)
- **`deployment/`** — Zero-latency interactive interface with simulated network delay compensation

```
zli/
├── dino_wm/          # Git submodule → gaoyuezhou/dino_wm (core library)
├── training/         # PushT training scripts, configs, SLURM jobs
└── deployment/       # Zero-latency interface, profiling, visualisation
```

## Setup

```bash
# Clone with submodule
git clone --recurse-submodules <this-repo>
cd zli

# Set up Python environment
source /mnt/cluster/environments/lelismart/uv/env
uv sync                          # base deps
uv sync --group sim              # + ManiSkill for online training
uv sync --group deploy           # + pymunk/pygame for interactive deployment
```

Set your WandB API key:
```bash
source ~/.wandb_key
```

## Training

```bash
# Offline (pre-recorded dataset)
export DATASET_DIR=/path/to/pusht_dataset
python training/train_pusht.py --offline --epochs 100

# Online (ManiSkill + PPO policy, requires GPU with Vulkan)
export ZERO_LATENCY_ROOT=/path/to/zero_latency_interface
python training/train_pusht.py --online --epochs 100 --num-envs 64

# SLURM
sbatch training/slurm/pusht_train.sbatch
```

## Deployment: Zero-Latency Interface

Requires a trained model checkpoint and a display with X11.

```bash
# ZLI mode (simulated network delay, WM compensates):
python deployment/run_zli.py --ckpt outputs/.../checkpoints --delay_steps 10

# WM-only mode (open-loop autoregressive prediction):
python deployment/run_wm_only.py --ckpt outputs/.../checkpoints

# SLURM (launches on GPU node with X11 forwarding):
sbatch deployment/slurm/pusht_zli.sbatch
```

### Controls

| Key | Action |
|-----|--------|
| WASD | Move PushT agent |
| Q/E/Z/C | Diagonal movement |
| ESC | Quit |

### Modes

- **`zli`** — Simulated network delay on GT frames; WM compensates by predicting ahead
- **`wm_only`** — WM runs open-loop from initial context; env runs alongside for comparison
- **`passthrough`** — Direct env display (no WM, no delay) — baseline

## Profiling

```bash
python deployment/profile.py --ckpt outputs/.../checkpoints --batch_size 1 --rollout_horizon 20
```

## Rollout Visualisation

```bash
python deployment/viz_rollout.py --ckpt outputs/.../checkpoints --max-episodes 5
```

## Submodule

The `dino_wm/` directory is a git submodule pointing to the upstream paper
repository at <https://github.com/gaoyuezhou/dino_wm>. To update:

```bash
git submodule update --remote dino_wm
```
