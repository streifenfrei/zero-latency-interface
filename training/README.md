# Training DINO-WM on PushT

## Quick Start

### Offline Training (pre-recorded dataset)

```bash
export DATASET_DIR=/path/to/pusht_dataset

python training/train_pusht.py --offline --epochs 100
```

Dataset path is read from `$DATASET_DIR` (config resolves `${oc.env:DATASET_DIR}`).

### Online Training (ManiSkill + PPO policy)

```bash
export ZERO_LATENCY_ROOT=/path/to/zero_latency_interface

python training/train_pusht.py --online --epochs 100 --num-envs 64
```

Requires:
- A trained PPO policy at `$ZERO_LATENCY_ROOT/checkpoints/policy.pt`
- GPU with Vulkan driver (SAPIEN / physx_cuda backend)
- `mani-skill` + `gymnasium` installed (`uv sync --group sim`)

### SLURM

```bash
sbatch training/slurm/pusht_train.sbatch

# With overrides:
EPOCHS=200 BATCH_SIZE=64 FRAMESKIP=3 sbatch training/slurm/pusht_train.sbatch
```

## Overridable Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 32 | Global batch size |
| `--frameskip` | 5 | Env steps per stored frame |
| `--num-hist` | 3 | Context frames |
| `--seed` | 0 | Random seed |
| `--num-envs` | 64 | Parallel envs (online only) |
| `--samples-per-epoch` | 50000 | Samples/epoch (online only) |
| `--ckpt-base-path` | `outputs/` | Checkpoint save path |
| `--resume` | auto | Specific checkpoint to resume from |

## Output

Checkpoints are saved to `{ckpt_base_path}/outputs/{date}/{time}/checkpoints/`:
- `model_latest.pth` — latest checkpoint (always overwritten)
- `model_{epoch}.pth` — epoch checkpoint

Training auto-resumes from `model_latest.pth` if it exists.
