#!/usr/bin/env python
"""Study-results figures: metric vs delay condition, no_wm vs wm lines.

Reads the per-run results CSV produced by ``deployment/run_study.py`` and
renders one PNG figure per metric.  Each figure plots two lines — ``no_wm``
and ``wm`` — against the delay condition (``delay_n``), after reducing the
variance in two stages:

  1. Intra-participant: each participant's repeated trials within a
     (participant, delay, setting) cell are averaged first (the schedule's
     ABBA design repeats each condition within a block).
  2. Inter-participant: across participants, per (delay, setting), the mean
     is plotted with SEM error bars (std(ddof=1)/sqrt(n)).

Defaults (see flags):
  * only ``phase == "main"`` rows (practice runs are familiarisation);
  * only ``attempt == 1`` rows (retries are excluded — ``--all-attempts``
    includes them);
  * completion time is averaged over SUCCESSFUL attempts only (timed-out
    rows are censored at the timeout value and would bias the mean);
  * ``ssim_mean``/``lpips_mean`` exist only for the wm condition and render
    as a single line.

Usage:
    python deployment/plot_study_results.py \\
        --results results/study_results.csv --out logs/study_figures
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np

# Schema source (import-light — deliberately NOT run_study, which would
# pull the torch/WM stack into a CSV-to-PNG job).
from deployment.study_schema import FIELDNAMES, METRIC_FIELDS  # noqa: E402

# ── figure configuration ─────────────────────────────────────────────────────
SETTINGS = ("no_wm", "wm")
COLORS = {"no_wm": "#2a78d6", "wm": "#eb6834"}   # palette slots 1/2 (validated)
METRIC_LABELS = {
    "completion_time_s": "Completion time (s)",
    "success_rate": "Success rate",
    "ee_path_len_m": "EE path length (m)",
    "block_path_len_m": "T-block path length (m)",
    "pushes": "Push onsets",
    "reversals": "Direction reversals",
    "idle_fraction": "Idle fraction",
    "duty_cycle": "Input duty cycle",
    "sparc": "SPARC (input smoothness)",
    "tick_rate_mean_hz": "Mean tick rate (Hz)",
    "tick_rate_min_hz": "Min tick rate (Hz)",
    "ssim_mean": "SSIM (wm only)",
    "lpips_mean": "LPIPS (wm only)",
    "completion_ticks": "Completion time (ticks)",
}
DEFAULT_METRICS = tuple(METRIC_LABELS)
SUCCESS_ONLY_METRICS = {"completion_time_s", "completion_ticks"}


# ── loading / aggregation ────────────────────────────────────────────────────

def load_rows(path: Path, all_attempts: bool) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("phase", "main") != "main":
                continue
            if not all_attempts and int(r.get("attempt", 1)) != 1:
                continue
            rows.append(r)
    if not rows:
        raise SystemExit(f"[PLOT] no usable rows in {path}")
    return rows


def _float(v) -> float | None:
    try:
        x = float(v)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def cell_stats(rows: list[dict], metric: str) -> dict[int, dict[str, tuple]]:
    """Two-stage reduction over (delay, setting) cells.

    Returns ``{delay: {setting: (mean, sem, n_participants)}}``.
    """
    # Stage 1: intra-participant mean per (participant, delay, setting).
    per_participant: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if metric == "success_rate":
            v = 1.0 if str(r.get("success", "")).strip().lower() == "true" \
                else 0.0
        else:
            v = _float(r.get(metric))
            if v is None:
                continue
        per_participant[
            (r["participant_id"], int(r["delay_n"]), r["setting"])
        ].append(v)

    participant_cells: dict[tuple, float] = {}
    for (pid, delay, setting), vals in per_participant.items():
        participant_cells[(pid, delay, setting)] = float(np.mean(vals))

    # Stage 2: across-participant mean + SEM per (delay, setting).
    out: dict[int, dict[str, tuple]] = {}
    by_cell: dict[tuple, list[float]] = defaultdict(list)
    for (pid, delay, setting), v in participant_cells.items():
        by_cell[(delay, setting)].append(v)
    for (delay, setting), vals in by_cell.items():
        vals = np.asarray(vals, dtype=np.float64)
        mean = float(vals.mean())
        sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 \
            else 0.0
        out.setdefault(int(delay), {})[setting] = (mean, sem, len(vals))
    return out


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_metric(rows: list[dict], metric: str, out_path: Path,
                show_title: bool) -> None:
    success_only = metric in SUCCESS_ONLY_METRICS
    sel = [r for r in rows] if not success_only else [
        r for r in rows
        if str(r.get("success", "")).strip().lower() == "true"
    ]
    if not sel:
        print(f"[PLOT] skip {metric}: no rows")
        return
    stats = cell_stats(sel, metric)
    delays = sorted(stats)
    if not delays:
        print(f"[PLOT] skip {metric}: no data")
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    for setting in SETTINGS:
        xs, means, sems = [], [], []
        for d in delays:
            cell = stats[d].get(setting)
            if cell is None:
                continue
            mean, sem, n = cell
            xs.append(d)
            means.append(mean)
            sems.append(sem)
        if not xs:
            continue
        xs = np.asarray(xs, dtype=float)
        means = np.asarray(means)
        sems = np.asarray(sems)
        ax.errorbar(
            xs, means, yerr=sems, label=setting,
            color=COLORS[setting], linewidth=2, marker="o", markersize=5,
            capsize=3, elinewidth=1.2, zorder=3,
        )
        # Direct label at the line end (identity never color-alone).
        ax.annotate(
            setting, xy=(xs[-1], means[-1]), xytext=(6, 0),
            textcoords="offset points", va="center",
            fontsize=9, color="#0b0b0b", fontweight="bold",
        )

    ax.set_xlabel("Delay (raw timesteps)")
    ax.set_ylabel(METRIC_LABELS[metric])
    if metric == "success_rate":
        ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(delays)
    ax.grid(True, axis="y", color="#c9c8c4", linewidth=0.6, alpha=0.6)
    ax.grid(False, axis="x")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#8a8984")
    ax.spines["bottom"].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.xaxis.label.set_color("#0b0b0b")
    ax.yaxis.label.set_color("#0b0b0b")

    ns = [cell[2] for d in delays for cell in stats[d].values()]
    if show_title:
        ax.set_title(f"{METRIC_LABELS[metric]}  (n={min(ns)}–{max(ns)} "
                     f"participants per cell)", fontsize=10,
                     color="#0b0b0b", loc="left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"[PLOT] wrote {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Study results figures: metric vs delay, no_wm vs wm")
    ap.add_argument("--results", default=os.path.join(REPO, "results",
                                                      "study_results.csv"),
                    help="results CSV from run_study.py")
    ap.add_argument("--out", default=os.path.join(REPO, "logs",
                                                  "study_figures"),
                    help="output directory for PNG figures")
    ap.add_argument("--metrics", default=None,
                    help=f"comma-separated metric names "
                         f"(default: all: {','.join(DEFAULT_METRICS)})")
    ap.add_argument("--all-attempts", action="store_true",
                    help="include retry attempts (default: attempt 1 only)")
    ap.add_argument("--no-title", action="store_true",
                    help="omit figure titles")
    args = ap.parse_args()

    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        unknown = set(metrics) - set(METRIC_LABELS)
        if unknown:
            ap.error(f"unknown metrics: {sorted(unknown)}")
    else:
        metrics = list(DEFAULT_METRICS)

    rows = load_rows(Path(args.results), args.all_attempts)
    print(f"[PLOT] {len(rows)} usable rows from {args.results}")
    out_dir = Path(args.out)
    for metric in metrics:
        plot_metric(rows, metric, out_dir / f"{metric}.png",
                    not args.no_title)
    print(f"[PLOT] done → {out_dir}")


if __name__ == "__main__":
    main()
