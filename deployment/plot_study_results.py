#!/usr/bin/env python
"""Study-results figures: metric vs delay condition, no_wm vs wm lines.

Reads the per-run results CSV produced by ``deployment/run_study.py`` and
renders one PNG figure per metric.  Each figure plots two lines — ``no_wm``
and ``wm`` (rendered as ``baseline`` / ``compensated``) — against the delay
condition (``delay_n``), after reducing the
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
  * ``timeout_rate`` is derived from the ``timed_out`` flag and is NOT
    success-filtered (timeouts are the metric);
  * ``ssim_mean``/``lpips_mean`` exist only for the wm condition and render
    as a single line;
  * the TLX items (mental, performance, frustration, plus the two custom
    perceived-responsiveness and perceived-authenticity scales) render
    like any other metric on a fixed 0–100 y-range; practice rows carry
    no ratings, so the practice reference line is absent there;
  * every condition figure shows the delay axis in milliseconds
    (10/15/20 ticks → 500/750/1000 ms at the 20 Hz tick rate);
  * ``completion_time_sim_s`` is derived from ``completion_ticks`` at the
    study tick rate (20 Hz); it is figure-only (no LMM power block — a
    rescaling of ``completion_ticks``);
  * practice runs (undelayed GT familiarisation) render as a dashed
    horizontal reference line with an SEM band spanning every condition
    figure (baseline colour — practice is an undelayed view), and get
    their own figure ``practice_curve.png`` (trial index vs simulated
    completion time, per-participant lines plus mean ± SEM).

Required-sample-size analysis (stdout, ``--no-power`` to skip):
  For the two success-only time metrics (``completion_time_s`` and
  ``completion_ticks``) the script additionally fits the planned
  within-subject model — once per delay and once pooled across delays
  with delay dummies (delay-15/20) as fixed effects —

      log(metric) ~ condition + trial_index + (1 + condition | participant)

  with statsmodels' MixedLM (REML; condition = wm dummy, trial_index =
  session_trial, centred).  From the fitted effect, the SD of the random
  condition slope and the trial-level residual SD it derives the SD of
  the within-participant condition difference,

      sd_diff = sqrt(sigma_slope^2 + 2 * sigma_resid^2 / m)
      d_z     = effect / sd_diff

  where ``m`` is the mean number of (successful) trials per (participant,
  setting) cell among the rows used, and inverts the exact noncentral-t
  power of the two-tailed paired t-test at alpha = 0.05 to report the
  number of participants needed per delay at 80/90/95% power.  A naive
  paired-t estimate from per-participant log-deltas is printed alongside
  as a cross-check.  statsmodels is imported lazily: if it is absent the
  figures are still produced and only the power block is skipped, with
  an install hint.

Usage:
    python deployment/plot_study_results.py \\
        --results results/study_results.csv --out logs/study_figures
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import warnings
from collections import Counter, defaultdict
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
# Display names for the two conditions (keys stay the CSV setting values).
SETTING_LABELS = {"no_wm": "baseline", "wm": "compensated"}
# Baseline blue darkened — the subtle tag inside the practice band.
PRACTICE_LABEL_COLOR = "#1b4e8b"
METRIC_LABELS = {
    "completion_time_s": "Completion time (s)",
    "success_rate": "Success rate",
    "timeout_rate": "Timeout rate",
    "ee_path_len_m": "EE path length (m)",
    "block_path_len_m": "T-block path length (m)",
    "pushes": "Push onsets",
    "reversals": "Direction reversals",
    "idle_fraction": "Idle fraction",
    "duty_cycle": "Input duty cycle",
    "sparc": "SPARC (input smoothness)",
    "tick_rate_mean_hz": "Mean tick rate (Hz)",
    "tick_rate_min_hz": "Min tick rate (Hz)",
    "ssim_mean": "SSIM",
    "lpips_mean": "LPIPS",
    "completion_ticks": "Completion time (ticks)",
    "completion_time_sim_s": "Completion time (s)",
    "tlx_mental": "TLX: mental demand (0–100)",
    "tlx_performance": "TLX: performance (0–100)",
    "tlx_frustration": "TLX: frustration (0–100)",
    "tlx_responsiveness": "Perceived responsiveness (0–100)",
    "tlx_authenticity": "Perceived authenticity (0–100)",
}
DEFAULT_METRICS = tuple(METRIC_LABELS)
# 0–100 rating scales — fixed y-range so the endpoint meaning is visible.
RATING_METRICS = {"tlx_mental", "tlx_performance", "tlx_frustration",
                  "tlx_responsiveness", "tlx_authenticity"}
SUCCESS_ONLY_METRICS = {"completion_time_s", "completion_ticks",
                        "completion_time_sim_s"}
# Derived metrics render as figures only: the LMM power analysis on a
# rescaled column (log(x/a) = log(x) - log(a) shifts the intercept only)
# would reproduce the source metric's output, so it is skipped.
POWER_SKIP_METRICS = {"completion_time_sim_s"}

# Study tick rate (run_study's --fps / make_result_tables.py's --tick-hz):
# one raw timestep = 1000 / STUDY_FPS ms of simulated time.  All condition
# figures show the delay axis in milliseconds (10/15/20 ticks →
# 500/750/1000 ms).
STUDY_FPS = 20.0
MS_PER_TICK = 1000.0 / STUDY_FPS
X_LABEL = "Delay (ms)"


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


def load_practice_rows(path: Path) -> list[dict]:
    """Practice runs (undelayed GT familiarisation), unfiltered.

    Unlike main rows, attempts are NOT filtered: the final results CSV
    stores exactly one row per scheduled practice trial (for a handful of
    trials the surviving row is a retry, ``attempt`` 2/3), so every
    familiarisation trial is represented exactly once.
    """
    return [r for r in csv.DictReader(open(path, newline=""))
            if r.get("phase", "main") == "practice"]


def _float(v) -> float | None:
    try:
        x = float(v)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def derive_sim_metrics(rows: list[dict]) -> None:
    """Add simulated-time columns derived from recorded ticks, in place.

    ``completion_time_sim_s`` = completion_ticks / STUDY_FPS (success-only
    filtering still applies downstream).  Rows without ticks get ``None``
    so the metric aggregators skip them.
    """
    for r in rows:
        ticks = _float(r.get("completion_ticks"))
        r["completion_time_sim_s"] = ticks / STUDY_FPS if ticks is not None \
            else None


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
        elif metric == "timeout_rate":
            v = 1.0 if str(r.get("timed_out", "")).strip().lower() == "true" \
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


def practice_stats(rows: list[dict], metric: str) -> tuple | None:
    """Two-stage mean ± SEM over practice runs, pooled across trials.

    Stage 1: per-participant mean over their practice runs; stage 2:
    across-participant mean ± SEM.  Returns ``(mean, sem, n)`` or ``None``
    if the metric has no practice values.
    """
    if not rows:
        return None
    per_participant: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if metric == "success_rate":
            v = 1.0 if str(r.get("success", "")).strip().lower() == "true" \
                else 0.0
        elif metric == "timeout_rate":
            v = 1.0 if str(r.get("timed_out", "")).strip().lower() == "true" \
                else 0.0
        else:
            v = _float(r.get(metric))
            if v is None:
                continue
        per_participant[r["participant_id"]].append(v)
    vals = [float(np.mean(v)) for v in per_participant.values() if v]
    if not vals:
        return None
    vals = np.asarray(vals, dtype=np.float64)
    sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 \
        else 0.0
    return float(vals.mean()), sem, len(vals)


# ── power / required-sample-size (mixed model) ───────────────────────────────
POWER_TARGETS = (0.80, 0.90, 0.95)
POWER_ALPHA = 0.05
POWER_MAX_N = 2000

_SM_MODULE = None      # lazily imported statsmodels.regression.mixed_linear_model
_SM_WARNED = False     # one-time missing-install hint


def _t_crit(df, alpha=POWER_ALPHA):
    from scipy import stats
    return stats.t.ppf(1 - alpha / 2, df)


def _paired_t_power(n, d_z, alpha=POWER_ALPHA):
    """Exact two-tailed paired-t power via the noncentral t distribution."""
    from scipy import stats
    if n < 2:
        return 0.0
    df = n - 1
    t_c = _t_crit(df, alpha)
    ncp = abs(d_z) * math.sqrt(n)
    return (1.0 - stats.nct.cdf(t_c, df, ncp)
            + stats.nct.cdf(-t_c, df, ncp))


def solve_n(d_z, target=0.80, alpha=POWER_ALPHA, n_max=POWER_MAX_N):
    """Smallest participant count whose exact paired-t power reaches target."""
    if not math.isfinite(d_z) or d_z == 0.0:
        return None
    with warnings.catch_warnings():
        # scipy's nct cdf emits spurious divide-by-zero RuntimeWarnings
        # while the scan explores small powers — harmless here.
        warnings.simplefilter("ignore", RuntimeWarning)
        for n in range(2, n_max + 1):
            if _paired_t_power(n, d_z, alpha) >= target:
                return n
    return None


def required_n(effect, sigma_slope, sigma_resid, m, targets=POWER_TARGETS):
    """Effective group size from mixed-model variance components.

    ``sd_diff`` is the SD of the per-participant condition difference
    (the varcomp planning calculation)::

        sd_diff = sqrt(sigma_slope^2 + 2 * sigma_resid^2 / m)
        d_z     = effect / sd_diff

    and ``n`` solves the exact noncentral-t paired test at each target
    power.  Returns ``(sd_diff, d_z, {power: n})``.
    """
    if (not math.isfinite(effect) or not math.isfinite(sigma_slope)
            or not math.isfinite(sigma_resid) or m <= 0):
        return math.nan, math.nan, {t: None for t in targets}
    sd_diff = math.sqrt(sigma_slope ** 2 + 2 * sigma_resid ** 2 / m)
    if not math.isfinite(sd_diff) or sd_diff <= 0.0:
        return sd_diff, math.nan, {t: None for t in targets}
    d_z = effect / sd_diff
    return sd_diff, d_z, {t: solve_n(d_z, t) for t in targets}


def _ensure_statsmodels():
    """One-time lazy import of statsmodels' MixedLM module (or None)."""
    global _SM_MODULE, _SM_WARNED
    if _SM_MODULE is None and not _SM_WARNED:
        try:
            from statsmodels.regression import mixed_linear_model as _sm
        except ImportError:
            _sm = None
        if _sm is None:
            _SM_WARNED = True
            print("[POWER] statsmodels not installed — skipping required-n "
                  "analysis.")
            print("[POWER]        install:  source "
                  "/mnt/cluster/environments/lelismart/uv/env && "
                  "uv add statsmodels")
        else:
            _SM_MODULE = _sm
    return _SM_MODULE


def _power_records(rows: list[dict], metric: str, delay: int | None = None):
    """Success-only, positive-valued rows as
    ``(pid, delay_n, setting, log y, session_trial)``."""
    recs = []
    for r in rows:
        if str(r.get("success", "")).strip().lower() != "true":
            continue
        if delay is not None and int(r["delay_n"]) != delay:
            continue
        v = _float(r.get(metric))
        if v is None or v <= 0.0:
            continue
        recs.append((str(r["participant_id"]), int(r["delay_n"]),
                     r["setting"], math.log(v), float(int(r["session_trial"]))))
    return recs


def _power_design(recs, delay_dummies=()):
    """Model matrices for ``log(y) ~ wm + trial (+ delay dummies)``.

    Returns ``(endog, X, Z, groups)`` with ``X = [1, wm, t_centred,
    *dummies]`` and ``Z = X[:, :2]`` (random intercept + wm slope).
    """
    wm = np.array([1.0 if rec[2] == "wm" else 0.0 for rec in recs])
    trial = np.array([rec[4] for rec in recs])
    endog = np.array([rec[3] for rec in recs])
    trial = trial - trial.mean()   # centring: no effect on the wm slope
    cols = [np.ones_like(wm), wm, trial]
    for d in delay_dummies:
        cols.append(np.array([1.0 if rec[1] == d else 0.0 for rec in recs]))
    X = np.column_stack(cols)
    return endog, X, X[:, :2], [rec[0] for rec in recs]


def _cell_m(recs):
    """Mean trials per (participant, setting) cell among the records."""
    counts = Counter((rec[0], rec[2]) for rec in recs)
    return float(np.mean(list(counts.values()))) if counts else 0.0


def _fit_lmm(endog, X, Z, groups):
    """REML fit of ``endog ~ X`` with random effects ``Z`` per group.

    Returns ``(info, error)``: info is a dict of scalars (effect, se,
    sigma_slope, sigma_resid, flags) or None on failure; error explains.
    """
    sm = _ensure_statsmodels()
    if sm is None:
        return None, "statsmodels missing"
    md = sm.MixedLM(endog=endog, exog=X, groups=groups, exog_re=Z)
    flags = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = md.fit(reml=True, method=["lbfgs", "powell"])
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
            return None, f"{type(exc).__name__}: {exc}"
        for w in caught:
            text = str(w.message)
            if any(k in text for k in ("boundary", "converge", "singular",
                                       "not positive definite", "Retrying")):
                flags.append(text)
    if not getattr(res, "converged", True):
        flags.append("fit did not converge")

    effect = float(res.fe_params[1])
    se = float(res.bse_fe[1])
    if not math.isfinite(se):
        se = math.nan
        flags.append("SE unavailable (Hessian not positive definite)")
    sigma_slope = float(np.sqrt(max(float(res.cov_re[1, 1]), 0.0)))
    sigma_resid = float(np.sqrt(max(float(res.scale), 0.0)))
    return {"effect": effect, "se": se, "sigma_slope": sigma_slope,
            "sigma_resid": sigma_resid, "flags": flags}, None


def print_power_analysis(rows: list[dict], metric: str) -> None:
    """Required sample size for ``metric`` from the planned LMM.

    Prints per-cell descriptives, one LMM fit per delay plus a pooled
    (delay-adjusted) fit, a naive paired-t cross-check, and caveats.
    """
    try:
        _print_power_analysis(rows, metric)
    except Exception as exc:  # the power block must never kill the plots
        print(f"[POWER] {metric}: analysis failed ({type(exc).__name__}: "
              f"{exc}) — figures unaffected")


def _print_power_analysis(rows: list[dict], metric: str) -> None:
    if _ensure_statsmodels() is None:
        return
    sel = [r for r in rows
           if str(r.get("success", "")).strip().lower() == "true"]
    if not sel:
        print(f"[POWER] skip {metric}: no successful rows")
        return
    stats_cells = cell_stats(sel, metric)
    delays = sorted(stats_cells)

    print()
    print(f"[POWER] ── {metric} ({METRIC_LABELS[metric]}) ──"
          f"{'─' * (78 - 38 - len(metric) - len(METRIC_LABELS[metric]))}")
    print("[POWER] per-cell descriptives (raw units; mean ± SD across "
          "participants of each")
    print("[POWER] participant's mean over their successful trials)")
    print(f"[POWER] {'delay':>7} {'setting':>11} {'n_part':>6} "
          f"{'mean':>12} {'sd':>10}")
    for d in delays:
        for setting in SETTINGS:
            cell = stats_cells[d].get(setting)
            if cell is None:
                continue
            mean, sem, n = cell
            sd = sem * math.sqrt(n) if n > 1 else math.nan
            sd_s = f"{sd:10.3f}" if math.isfinite(sd) else f"{'n/a':>10}"
            print(f"[POWER] {d:>7} {SETTING_LABELS[setting]:>11} {n:>6} "
                  f"{mean:12.3f} {sd_s}")

    print("[POWER]")
    print("[POWER] LMM  log(y) ~ condition + trial_index + "
          "(1 + condition | participant)")
    print("[POWER]      MixedLM REML (lbfgs→powell); condition = wm dummy, "
          "trial_index = session_trial (centred)")
    print("[POWER]      m = mean successful trials per (participant, "
          "setting) cell")
    print(f"[POWER] {'delay':>7} {'effect':>9} {'SE':>9} {'Δ%':>8} "
          f"{'sd_slope':>9} {'sd_resid':>9} {'m':>7} {'n_obs':>6} "
          f"{'d_z':>7} {'n80':>5} {'n90':>5} {'n95':>5}")
    fit_rows = []
    for d in delays:
        recs = _power_records(sel, metric, delay=d)
        if not recs:
            continue
        y, X, Z, groups = _power_design(recs)
        info, err = _fit_lmm(y, X, Z, groups)
        fit_rows.append((str(d), info, err, len(recs), _cell_m(recs)))
    if len(delays) > 1:
        recs_all = _power_records(sel, metric)
        y, X, Z, groups = _power_design(recs_all, delay_dummies=delays[1:])
        info, err = _fit_lmm(y, X, Z, groups)
        fit_rows.append(("pooled", info, err, len(recs_all),
                         _cell_m(recs_all)))

    for label, info, err, n_obs, m in fit_rows:
        if info is None:
            print(f"[POWER] {label:>7}  fit failed: {err}")
            continue
        sd_diff, d_z, ns = required_n(info["effect"], info["sigma_slope"],
                                      info["sigma_resid"], m)
        pct = (math.exp(info["effect"]) - 1.0) * 100.0
        nstr = " ".join(f"{ns[t]:>5}" if ns[t] is not None else f"{'-':>5}"
                        for t in POWER_TARGETS)
        print(f"[POWER] {label:>7} {info['effect']:+9.4f} "
              f"{info['se']:9.4f} {pct:+7.1f}% {info['sigma_slope']:9.4f} "
              f"{info['sigma_resid']:9.4f} {m:7.3f} {n_obs:>6} "
              f"{d_z:+7.2f} {nstr}")
        for flag in info["flags"]:
            print(f"[POWER]      fit flag ({label}): {flag}")

    print("[POWER] naive cross-check (paired t on per-participant mean "
          "log-deltas; no model)")
    print(f"[POWER] {'delay':>7} {'n_pairs':>7} {'mean_delta':>11} "
          f"{'sd_delta':>9} {'d_z':>7} {'n80':>5}")
    for d in delays:
        recs = _power_records(sel, metric, delay=d)
        by_cell = defaultdict(list)
        for pid, _d, setting, logy, _t in recs:
            by_cell[(pid, setting)].append(logy)
        deltas = []
        for (pid, setting), vals in by_cell.items():
            if setting == "no_wm" and (pid, "wm") in by_cell:
                deltas.append(np.mean(by_cell[(pid, "wm")]) - np.mean(vals))
        if not deltas:
            continue
        deltas = np.asarray(deltas)
        sd_delta = deltas.std(ddof=1) if len(deltas) > 1 else 0.0
        d_z = deltas.mean() / sd_delta if sd_delta > 0 else math.nan
        n80 = solve_n(d_z, 0.80) if math.isfinite(d_z) else None
        n80_s = f"{n80:>5}" if n80 is not None else f"{'-':>5}"
        print(f"[POWER] {d:>7} {len(deltas):>7} {deltas.mean():+11.4f} "
              f"{sd_delta:9.4f} {d_z:+7.2f} {n80_s}")

    print("[POWER] caveats")
    n_part = len({r["participant_id"] for r in sel})
    if n_part < 10:
        print(f"[POWER]   * pilot n = {n_part} (< 10): variance components — "
              "especially sd_slope — are barely")
        print("[POWER]     identified; the naive cross-check brackets the "
              "plausible range. Treat these")
        print("[POWER]     n as planning estimates, not precise targets.")
    print("[POWER]   * add ~20% for dropout / unusable sessions before "
          "recruiting.")
    print("[POWER]   * n is per delay; every participant runs all delays "
          "(within-subject), so the")
    print("[POWER]     cohort is max(n over delays) + dropout, not the sum.")
    print("[POWER]   * times are success-only: timed-out trials are "
          "excluded, not imputed — the")
    print("[POWER]     effect is conditional on success.")


# ── plotting ─────────────────────────────────────────────────────────────────

def _style_ax(ax) -> None:
    """Shared frame styling (no top/right spines, muted ticks/grid)."""
    ax.grid(True, axis="y", color="#c9c8c4", linewidth=0.6, alpha=0.6)
    ax.grid(False, axis="x")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#8a8984")
    ax.spines["bottom"].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.xaxis.label.set_color("#0b0b0b")
    ax.yaxis.label.set_color("#0b0b0b")


def _place_line_labels(ax, anchors: list[tuple]) -> None:
    """Direct condition labels under/over the lines, at the same x.

    ``anchors``: ``(name, x, mean, sem)`` in draw order (baseline first,
    compensated second).  Each label clears its own SEM bar: normally
    BELOW it; when the two labels would overlap (lines nearly coincident,
    e.g. timeout_rate / success_rate) the compensated label flips ABOVE
    its bar so the two labels bracket the lines instead of stacking.
    Call AFTER tight_layout and a draw so the axes height — hence points
    per data unit — is final.
    """
    if not anchors:
        return
    fig = ax.figure
    ylo, yhi = ax.get_ylim()
    span = max(yhi - ylo, 1e-12)
    # get_window_extent returns display PIXELS — convert to points so the
    # text-height estimate below shares its units.
    ax_h_pt = ax.get_window_extent().height * 72.0 / fig.dpi
    pt_per_unit = ax_h_pt / span
    text_h = 12.0 / pt_per_unit    # fontsize-9 line + inter-label gap
    pad = 0.04 * span              # breathing room beyond the bar

    # Desired positions: below-bar top / above-bar bottom per anchor.
    pos = [(name, x, mean - sem - pad, mean + sem + pad)
           for name, x, mean, sem in anchors]
    flip = False
    if len(pos) == 2:
        b1, b2 = pos[0][2], pos[1][2]
        # The two below positions overlap (intervals [b - text_h, b]).
        if b2 > b1 - text_h and b2 - text_h < b1:
            flip = True

    label_bottom, label_top = math.inf, -math.inf
    for i, (name, x, below, above) in enumerate(pos):
        above_line = flip and i == len(pos) - 1
        y = above if above_line else below
        va = "bottom" if above_line else "top"
        ax.annotate(
            name, xy=(x, y), va=va, ha="right",
            fontsize=9, color="#0b0b0b", fontweight="bold",
        )
        label_bottom = min(label_bottom, y - text_h)
        label_top = max(label_top, y + text_h)
    # Expand the limits so bracketing labels are never clipped.
    if label_bottom < ylo:
        ax.set_ylim(bottom=label_bottom)
    if label_top > yhi:
        ax.set_ylim(top=label_top)


def plot_metric(rows: list[dict], metric: str, out_path: Path,
                practice_rows: list[dict] | None = None) -> None:
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
    x_scale = lambda d: d * MS_PER_TICK

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")

    label_anchors: list[tuple] = []
    for setting in SETTINGS:
        xs, means, sems = [], [], []
        for d in delays:
            cell = stats[d].get(setting)
            if cell is None:
                continue
            mean, sem, n = cell
            xs.append(x_scale(d))
            means.append(mean)
            sems.append(sem)
        if not xs:
            continue
        xs = np.asarray(xs, dtype=float)
        means = np.asarray(means)
        sems = np.asarray(sems)
        ax.errorbar(
            xs, means, yerr=sems, label=SETTING_LABELS[setting],
            color=COLORS[setting], linewidth=2, marker="o", markersize=5,
            capsize=3, elinewidth=1.2, zorder=3,
        )
        # Collect the label anchor (name, x, mean, sem) — placement is
        # deferred to _place_line_labels after layout, so both labels can
        # share the same x without covering the error bars or each other.
        label_anchors.append(
            (SETTING_LABELS[setting], float(xs[-1]), float(means[-1]),
             float(sems[-1])))
    # Single-line metrics (ssim/lpips — wm only) need no condition label.
    if len(label_anchors) < 2:
        label_anchors = []

    # Practice runs: dashed horizontal reference line at the practice mean
    # with an SEM band, spanning the figure (undelayed familiarisation,
    # baseline colour) — NOT a delay-0 condition.  Success-only metrics
    # use successful practice runs.  Y-limits are expanded (never shrunk)
    # so the reference stays visible when practice is faster than every
    # condition.
    if practice_rows is not None:
        psel = [r for r in practice_rows
                if not success_only
                or str(r.get("success", "")).strip().lower() == "true"]
        pstat = practice_stats(psel, metric)
        if pstat is not None:
            pmean, psem, _pn = pstat
            ylo, yhi = ax.get_ylim()
            pad = 0.04 * (yhi - ylo)
            ax.set_ylim(min(ylo, pmean - psem - pad),
                        max(yhi, pmean + psem + pad))
            ax.axhspan(pmean - psem, pmean + psem, color=COLORS["no_wm"],
                       alpha=0.12, zorder=1)
            ax.axhline(pmean, color=COLORS["no_wm"], linestyle="--",
                       linewidth=1.2, zorder=2)
            ax.annotate(
                "practice", xy=(0.0, pmean - psem),
                xycoords=("axes fraction", "data"),
                xytext=(6, 2), textcoords="offset points", va="bottom",
                fontsize=9, color=PRACTICE_LABEL_COLOR,
            )

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(METRIC_LABELS[metric])
    if metric in ("success_rate", "timeout_rate"):
        ax.set_ylim(-0.02, 1.05)
    elif metric in RATING_METRICS:
        ax.set_ylim(0, 100)
    xticks = [x_scale(d) for d in delays]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:g}" for t in xticks])
    _style_ax(ax)

    fig.tight_layout()
    fig.canvas.draw()
    _place_line_labels(ax, label_anchors)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="#ffffff")
    plt.close(fig)
    print(f"[PLOT] wrote {out_path}")


def plot_practice_curve(practice_rows: list[dict], out_path: Path) -> None:
    """Practice learning curve: trial index vs simulated completion time.

    Faint per-participant lines plus the across-participant mean ± SEM
    (``completion_ticks / STUDY_FPS``).  ALL runs are included, success or
    timeout — early failures are the interesting part of familiarisation.
    """
    if not practice_rows:
        print("[PLOT] skip practice curve: no practice rows")
        return

    # Per-participant completion time by practice trial index.
    per: dict[str, dict[int, float]] = defaultdict(dict)
    for r in practice_rows:
        ticks = _float(r.get("completion_ticks"))
        if ticks is None:
            continue
        per[r["participant_id"]][int(r["session_trial"])] = ticks / STUDY_FPS
    if not per:
        print("[PLOT] skip practice curve: no completion_ticks values")
        return

    trials = sorted({t for d in per.values() for t in d})

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")

    for d in per.values():
        xs = sorted(d)
        ys = [d[t] for t in xs]
        ax.plot(xs, ys, color=COLORS["no_wm"], linewidth=1, alpha=0.22,
                zorder=1)

    means, sems = [], []
    for t in trials:
        vals = [d[t] for d in per.values() if t in d]
        vals = np.asarray(vals, dtype=np.float64)
        means.append(float(vals.mean()))
        sems.append(float(vals.std(ddof=1) / np.sqrt(len(vals)))
                    if len(vals) > 1 else 0.0)
    ax.errorbar(trials, means, yerr=sems, color=COLORS["no_wm"], linewidth=2,
                marker="o", markersize=5, capsize=3, elinewidth=1.2,
                zorder=3)

    ax.set_xlabel("Practice trial")
    ax.set_ylabel("Completion time (simulated s)")
    ax.set_xticks(trials)
    _style_ax(ax)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="#ffffff")
    plt.close(fig)
    print(f"[PLOT] wrote {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Study results figures: metric vs delay, no_wm vs wm")
    ap.add_argument("--results", default=os.path.join(REPO, "results", "final",
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
    ap.add_argument("--no-power", action="store_true",
                    help="skip the required-sample-size (LMM) analysis")
    args = ap.parse_args()

    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        unknown = set(metrics) - set(METRIC_LABELS)
        if unknown:
            ap.error(f"unknown metrics: {sorted(unknown)}")
    else:
        metrics = list(DEFAULT_METRICS)

    rows = load_rows(Path(args.results), args.all_attempts)
    derive_sim_metrics(rows)
    practice_rows = load_practice_rows(Path(args.results))
    derive_sim_metrics(practice_rows)
    print(f"[PLOT] {len(rows)} usable main rows "
          f"(+{len(practice_rows)} practice rows) from {args.results}")
    out_dir = Path(args.out)
    for metric in metrics:
        plot_metric(rows, metric, out_dir / f"{metric}.png",
                    practice_rows=practice_rows)
        if (not args.no_power and metric in SUCCESS_ONLY_METRICS
                and metric not in POWER_SKIP_METRICS):
            print_power_analysis(rows, metric)
    plot_practice_curve(practice_rows, out_dir / "practice_curve.png")
    print(f"[PLOT] done → {out_dir}")


if __name__ == "__main__":
    main()
