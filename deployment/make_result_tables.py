#!/usr/bin/env python
"""Copyable LaTeX result tables for the user study.

Reads the per-run results CSV (produced by ``run_study.py``) and prints
three LaTeX float blocks (``\\begin{table}...\\end{table}``) to stdout —
stdout carries NOTHING but LaTeX; all diagnostics go to stderr:

  Table 1 — timing budget: component latencies (τp predictor, τq decoder,
            encoder) against the per-frame tick budget τf = 50 ms, with the
            look-ahead gain G(D) = D·τp/(τf − τp) and k+G = D + G per delay
            (frameskip = 1, k = D).  G appears on the τp row only; the
            other rows show their latency, backing "the decoder is the
            expensive component".
  Table 2 — completion statistics per delay × condition: n, mean/median
            completion time in SIMULATED seconds (completion_ticks /
            tick rate; success-only), timeout rate, plus the test block:
            MixedLM main effects + condition×delay interaction on log
            time, stratified Cox PH (timeouts as right-censored), and
            per-delay log-rank tests.
  Table 3 — rating statistics: per TLX item the wm−no_wm difference with
            95 % paired-t CI (pooled across delays), plus the frustration
            condition×delay interaction model.

Usage:
    python deployment/make_result_tables.py \\
        --results results/final/study_results.csv [--output tables.tex]
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

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# ── shared helpers ────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    """Diagnostic to stderr — stdout is reserved for LaTeX."""
    print(f"[TABLES] {msg}", file=sys.stderr, flush=True)


def load_rows(path: Path, attempts: str = "last",
              phase: str = "main") -> list[dict]:
    """Rows of one phase, one row per (participant, session_trial) slot.

    ``attempts`` resolves retried trial slots: "last" keeps the final
    logged attempt (post-save retries supersede earlier attempts; in-run
    fault-discards never log a row, so a lone attempt-2 row is kept as
    the slot's only outcome), "first" keeps the earliest, "all" keeps
    every attempt.  Mirror of plot_study_results.load_rows otherwise.
    """
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("phase", "main") != phase:
                continue
            rows.append(r)
    if not rows:
        raise SystemExit(f"[TABLES] no usable rows in {path}")
    if attempts == "all":
        return rows
    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r["participant_id"], int(r["session_trial"]))
        a = int(r.get("attempt", 1))
        cur = best.get(key)
        if cur is None or (a > int(cur.get("attempt", 1))
                           if attempts == "last" else
                           a < int(cur.get("attempt", 1))):
            best[key] = r
    return list(best.values())


def _float(v) -> float | None:
    """Mirror of plot_study_results._float."""
    try:
        x = float(v)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _fmt_ms(x: float) -> str:
    return f"{x:.2f}"


def _fmt_sec(x: float) -> str:
    return f"{x:.1f}"


def _fmt_g(x: float) -> str:
    """2 decimals < 10, 1 decimal >= 10 (2–3 significant digits)."""
    return f"{x:.2f}" if x < 10.0 else f"{x:.1f}"


def _fmt_p(p: float) -> str:
    if not math.isfinite(p):
        return "---"
    if p < 0.001:
        return "$<.001$"
    return f"{p:.3f}".lstrip("0") or ".000"


def _fmt_est(x: float) -> str:
    return f"{x:.3f}"


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{_fmt_est(lo)}, {_fmt_est(hi)}]"


def _fmt_chi(stat: float, df: int) -> str:
    return f"$\\chi^2({df}) = {stat:.2f}$"


def _cell_label(delay: int, setting: str) -> str:
    """LaTeX-safe condition label."""
    return setting.replace("_", "\\_")


# ── stats plumbing (pattern from plot_study_results._fit_lmm) ─────────────────

_SM_MODULE = None
_SM_WARNED = False
_FIT_LADDERS = (["lbfgs", "powell"], ["powell"], ["bfgs"], ["nm"])


def _ensure_statsmodels():
    """One-time lazy import of statsmodels (or None + install hint)."""
    global _SM_MODULE, _SM_WARNED
    if _SM_MODULE is None and not _SM_WARNED:
        try:
            from statsmodels.regression import mixed_linear_model as _sm
        except ImportError:
            _sm = None
        if _sm is None:
            _SM_WARNED = True
            _log("statsmodels not installed — skipping inferential blocks. "
                 "Install with: uv pip install --python .venv/bin/python "
                 "statsmodels")
        else:
            _SM_MODULE = _sm
    return _SM_MODULE


def _fit_mixed(endog, X, Z, groups, reml=True):
    """MixedLM fit with a manual method ladder.

    ``MixedLM.fit(method=["lbfgs","powell"])`` raises (not falls back) on
    some models (e.g. the frustration fit: LinAlgError on lbfgs), so each
    rung is tried with try/except.  Returns ``(res_or_None, flags, error)``.
    """
    sm = _ensure_statsmodels()
    if sm is None:
        return None, [], "statsmodels missing"
    md = sm.MixedLM(endog=endog, exog=X, groups=groups, exog_re=Z)
    flags: list = []
    res, error = None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for ladder in _FIT_LADDERS:
            try:
                res = md.fit(reml=reml, method=ladder)
                error = None
                break
            except (np.linalg.LinAlgError, ValueError,
                    FloatingPointError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                flags.append(f"method {ladder} failed ({exc})")
        if res is not None:
            for w in caught:
                text = str(w.message)
                if any(k in text for k in ("boundary", "converge", "singular",
                                           "not positive definite",
                                           "Retrying")):
                    flags.append(text)
    if res is None:
        return None, flags, error or "all methods failed"
    if not getattr(res, "converged", True):
        flags.append("fit did not converge")
    return res, flags, None


def _fe_stats(res, k_fe: int):
    """Fixed-effect estimates/SE/p/CI computed MANUALLY.

    For matrix-based MixedLM, ``res.pvalues`` is misaligned and
    ``res.conf_int()`` includes random-effect rows — so p and the 95 % CI
    are derived from ``fe_params``/``bse_fe`` via the normal approximation.
    """
    from scipy import stats as st
    b = np.asarray(res.fe_params[:k_fe], dtype=float)
    se = np.asarray(res.bse_fe[:k_fe], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = b / se
        p = 2.0 * st.norm.sf(np.abs(z))
    ci_lo = b - st.norm.ppf(0.975) * se
    ci_hi = b + st.norm.ppf(0.975) * se
    return b, se, p, np.column_stack([ci_lo, ci_hi])


def _wald(est, cov, R):
    """Manual Wald test (res.wald_test is broken via patsy for MixedLM)."""
    from scipy import stats as st
    est = np.asarray(est, dtype=float)
    cov = np.asarray(cov, dtype=float)
    R = np.asarray(R, dtype=float)
    Rb = R @ est
    RV = R @ cov @ R.T
    try:
        stat = float(Rb @ np.linalg.inv(RV) @ Rb)
    except np.linalg.LinAlgError:
        stat = math.nan
    return stat, int(R.shape[0]), float(st.chi2.sf(stat, R.shape[0]))


# ── Table 1: timing budget ────────────────────────────────────────────────────

def gain_frames(delay: int, tau_e_ms: float, tau_p_ms: float,
                tau_f_ms: float) -> float | None:
    """G(k) = (τe + k·τp)/(τf − τp) — the corrected horizon bound
    (main paper, Appendix C): encoder cost plus k predictor passes,
    amortised against the per-frame budget after one predictor pass."""
    denom = tau_f_ms - tau_p_ms
    if denom <= 0:
        return None
    return (tau_e_ms + delay * tau_p_ms) / denom


def horizon_bound(delay: int, g: float | None) -> float | None:
    """k + 2⌈G⌉ − 1 — the Appendix C bound on the required horizon."""
    if g is None:
        return None
    return delay + 2 * math.ceil(g) - 1


def table1_tex(args) -> str:
    delays = [int(d) for d in args.delays]
    spec = "l r" + " | r r" * len(delays)
    head1 = " & $\\tau$ (ms)" + "".join(
        f" & \\multicolumn{{2}}{{c}}{{$k{{=}}{d}$ ({d * 1000 / args.tick_hz:.0f} ms)}}"
        for d in delays)
    head2 = " & " + "".join(
        " & $G$ & $k{+}2\\lceil G\\rceil{-}1$" for _ in delays)
    lines = [
        "\\begin{tabular}{%s}" % spec,
        "\\hline",
        head1 + " \\\\",
        "\\cline{3-%d}" % (2 + 2 * len(delays)),
        head2 + " \\\\",
        "\\hline",
    ]
    comp_rows = [
        ("Predictor $\\tau_p$", args.pred_ms, True),
        ("Decoder $\\tau_q$", args.dec_ms, False),
        ("Encoder $\\tau_e$", args.enc_ms, False),
        ("Tick budget $\\tau_f$", args.tick_ms, False),
    ]
    for label, tau, is_pred in comp_rows:
        cells = [label, _fmt_ms(tau)]
        if is_pred:
            for d in delays:
                g = gain_frames(d, args.enc_ms, args.pred_ms, args.tick_ms)
                cells.append(_fmt_g(g))
                cells.append(_fmt_g(horizon_bound(d, g)))
        else:
            cells += ["\\multicolumn{2}{c}{---}"] * len(delays)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")

    caption = (args.caption_1 +
               " Component latencies from the main paper's Appendix C "
               "($\\tau_e$, $\\tau_p$, $\\tau_q$), against the per-frame tick "
               "budget $\\tau_f$; $G(k)=(\\tau_e+k\\tau_p)/(\\tau_f-\\tau_p)$ "
               "(the corrected horizon gain) and the resulting horizon bound "
               "$k+2\\lceil G\\rceil-1$ for a delay of $k$ frames "
               "(1 frame = 50 ms).")
    return _float_block(caption, args.label_1, "\n".join(lines))


# ── Table 2: completion statistics ────────────────────────────────────────────

def _participant_cell_seconds(rows, tick_hz: float) -> dict:
    """(pid, delay, setting) -> mean simulated seconds over SUCCESSFUL runs."""
    per_participant: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if str(r.get("success", "")).strip().lower() != "true":
            continue
        v = _float(r.get("completion_ticks"))
        if v is None:
            continue
        per_participant[(r["participant_id"], int(r["delay_n"]),
                         r["setting"])].append(v / tick_hz)
    return {k: float(np.mean(v)) for k, v in per_participant.items()}


def completion_descriptives(rows, tick_hz: float, delays,
                            practice_rows=None) -> dict:
    """Per (delay, setting): run counts, timeout %, mean/median/SEM seconds.

    With ``practice_rows`` (undelayed GT familiarization runs), a
    descriptive-only reference cell keyed ``(0, "GT")`` is added.
    """
    n_runs = Counter()
    n_success = Counter()
    n_timeout = Counter()
    for r in rows:
        cell = (int(r["delay_n"]), r["setting"])
        if r["setting"] not in ("no_wm", "wm"):
            continue
        n_runs[cell] += 1
        if str(r.get("success", "")).strip().lower() == "true":
            n_success[cell] += 1
        if str(r.get("timed_out", "")).strip().lower() == "true":
            n_timeout[cell] += 1
    cells = _participant_cell_seconds(rows, tick_hz)
    by_cell: dict[tuple, list[float]] = defaultdict(list)
    for (pid, d, s), v in cells.items():
        by_cell[(d, s)].append(v)
    out = {}
    for d in delays:
        for s in ("no_wm", "wm"):
            vals = np.asarray(by_cell.get((d, s), []), dtype=float)
            out[(d, s)] = {
                "n_runs": n_runs.get((d, s), 0),
                "n_success": n_success.get((d, s), 0),
                "n_part": int(vals.size),
                "timeout_pct": 100.0 * n_timeout.get((d, s), 0)
                / max(n_runs.get((d, s), 1), 1),
                "mean_s": float(vals.mean()) if vals.size else math.nan,
                "median_s": float(np.median(vals)) if vals.size else math.nan,
                "sem_s": float(vals.std(ddof=1) / np.sqrt(vals.size))
                if vals.size > 1 else 0.0,
            }
    if practice_rows:
        # All practice runs are undelayed GT — one cell per participant
        # (their mean over the 10 familiarization runs), never delay-split.
        pp: dict[str, list[float]] = defaultdict(list)
        n_runs_p = n_timeout_p = 0
        for r in practice_rows:
            n_runs_p += 1
            if str(r.get("timed_out", "")).strip().lower() == "true":
                n_timeout_p += 1
            if str(r.get("success", "")).strip().lower() != "true":
                continue
            v = _float(r.get("completion_ticks"))
            if v is None:
                continue
            pp[r["participant_id"]].append(v / tick_hz)
        vals = np.asarray([float(np.mean(v)) for v in pp.values()],
                          dtype=float)
        out[(0, "GT")] = {
            "n_runs": n_runs_p,
            "n_success": sum(len(v) for v in pp.values()),
            "n_part": int(vals.size),
            "timeout_pct": 100.0 * n_timeout_p / max(n_runs_p, 1),
            "mean_s": float(vals.mean()) if vals.size else math.nan,
            "median_s": float(np.median(vals)) if vals.size else math.nan,
            "sem_s": float(vals.std(ddof=1) / np.sqrt(vals.size))
            if vals.size > 1 else 0.0,
        }
    return out


def table2_descriptive_tex(rows, args, practice_rows=None) -> str:
    desc = completion_descriptives(rows, args.tick_hz,
                                   [int(d) for d in args.delays],
                                   practice_rows)
    lines = [
        "\\begin{tabular}{l l r r r}",
        "\\hline",
        "Delay & Condition & \\multicolumn{2}{c}{Completion time (s)} & "
        "Timeout (\\%) \\\\",
        " & & Mean & Median & \\\\",
        "\\hline",
    ]
    if (0, "GT") in desc:
        c = desc[(0, "GT")]
        lines.append(" & ".join(
            ["0", "GT (practice)",
             _fmt_sec(c["mean_s"]) if math.isfinite(c["mean_s"]) else "---",
             _fmt_sec(c["median_s"]) if math.isfinite(c["median_s"])
             else "---",
             f"{c['timeout_pct']:.1f}"]) + " \\\\")
    for d in [int(d) for d in args.delays]:
        pair = (desc[(d, "no_wm")], desc[(d, "wm")])
        # Bold the better (lower) value of the two conditions per column.
        best = {}
        for m in ("mean_s", "median_s", "timeout_pct"):
            vals = [pair[0][m], pair[1][m]]
            finite = [v for v in vals if math.isfinite(v)]
            best[m] = min(finite) if finite else None
        for si, s in enumerate(("no_wm", "wm")):
            c = pair[si]
            cells = [str(d) if si == 0 else "", _cell_label(d, s)]
            for m, fmt in (("mean_s", _fmt_sec), ("median_s", _fmt_sec),
                           ("timeout_pct", lambda v: f"{v:.1f}")):
                v = c[m]
                if not math.isfinite(v):
                    cells.append("---")
                    continue
                txt = fmt(v)
                if best[m] is not None and v <= best[m] + 1e-9:
                    txt = f"\\textbf{{{txt}}}"
                cells.append(txt)
            lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def cox_completion(rows, delays, tick_hz: float):
    """Delay-stratified Cox PH with participant-clustered robust SEs.

    Matches the main paper's specification: delay as stratum (the delay
    main effects are absorbed, so only condition and condition×delay are
    estimable) with the covariance clustered on participant.  Times are
    simulated seconds (ticks / tick rate; HRs are scale-invariant).
    """
    from statsmodels.duration.hazard_regression import PHReg
    recs = [(str(r["participant_id"]), int(r["delay_n"]), r["setting"],
             float(r["completion_ticks"]) / tick_hz,
             1.0 if str(r.get("success", "")).strip().lower() == "true"
             else 0.0)
            for r in rows if r["setting"] in ("no_wm", "wm")
            and _float(r.get("completion_ticks")) is not None]
    endog = np.array([t for _, _, _, t, _ in recs], dtype=float)
    status = np.array([s for _, _, _, _, s in recs], dtype=float)
    wm = np.array([1.0 if s == "wm" else 0.0 for _, _, s, _, _ in recs])
    # Delay dummies are collinear with the delay strata — the main-effects
    # fit is condition only; the full fit adds the condition×delay terms.
    X_main = np.column_stack([wm])
    cols_full = [wm]
    for d in delays[1:]:
        cols_full.append(wm * np.array([1.0 if dd == d else 0.0
                                        for _, dd, _, _, _ in recs]))
    X_full = np.column_stack(cols_full)
    strata = np.asarray([int(d) for _, d, _, _, _ in recs])
    cluster = np.asarray([pid for pid, _, _, _, _ in recs])
    out = {"n": len(recs)}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res_main = PHReg(endog, X_main, status=status, strata=strata,
                             ties="breslow").fit(groups=cluster)
            res_full = PHReg(endog, X_full, status=status, strata=strata,
                             ties="breslow").fit(groups=cluster)
        for w in caught:
            _log(f"Cox: {w.message}")
        R = np.zeros((len(delays[1:]), X_full.shape[1]))
        for j in range(len(delays[1:])):
            R[j, 1 + j] = 1.0
        stat, df, p_int = _wald(res_full.params, res_full.cov_params(), R)
        out.update({"hr": np.exp(res_main.params),
                    "ci": np.exp(res_main.conf_int()),
                    "p": res_main.pvalues,
                    "int_stat": stat, "int_df": df, "int_p": p_int})
    except Exception as exc:
        _log(f"Cox failed: {type(exc).__name__}: {exc}")
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def table4_tex(rows, args) -> str:
    """Standalone Cox tests float (no Test column — Cox only)."""
    delays = [int(d) for d in args.delays]
    cox = cox_completion(rows, delays, args.tick_hz)
    lines = [
        "\\begin{tabular}{l r r r}",
        "\\hline",
        "Effect & Estimate & 95\\% CI & $p$ \\\\",
        "\\hline",
    ]
    if "error" in cox:
        lines.append(
            f"\\multicolumn{{4}}{{l}}{{Cox model failed: {cox['error']}}} "
            "\\\\")
    else:
        rows_out = [("Condition (wm)",
                     f"HR = {_fmt_est(cox['hr'][0])}",
                     _fmt_ci(*cox["ci"][0]), _fmt_p(cox["p"][0])),
                    ("Cond $\\times$ Delay",
                     _fmt_chi(cox["int_stat"], cox["int_df"]),
                     "---", _fmt_p(cox["int_p"]))]
        for e, est, ci, p in rows_out:
            lines.append(" & ".join([e, est, ci, p]) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    caption = (args.caption_4 +
               " Cox proportional-hazards model of time-to-completion in "
               "simulated seconds (ticks / "
               f"{args.tick_hz:g}\\,Hz), the main paper's specification: "
               "delay as stratum with participant-clustered robust standard "
               "errors (delay main effects are absorbed by the strata, so "
               "only the condition HR and its interaction with delay are "
               "estimable). Timed-out runs enter as right-censored "
               "observations; HR greater than 1 indicates faster "
               "completion.")
    return _float_block(caption, args.label_4, "\n".join(lines))


def table2_tex(rows, args, practice_rows=None) -> str:
    body = table2_descriptive_tex(rows, args, practice_rows)
    caption = (args.caption_2 +
               " Completion time is simulated (ticks / 20 Hz) over "
               "successful trials. Bold marks the better (lower) value "
               "of the two conditions per column. Each trial slot "
               "contributes its final logged attempt (post-save retries "
               "supersede earlier attempts; in-run fault-discards never "
               "log).")
    if _row_jitters(rows):
        caption += (" Pooled across both jitter amplitudes (±2 and ±10 "
                    "frames) — this pilot contains no constant-delay "
                    "condition.")
    if practice_rows:
        caption += (" Delay 0 are the undelayed GT practice "
                    "(familiarization) runs, shown for reference only.")
    return _float_block(caption, args.label_2, body)


# ── Table 3: rating statistics ────────────────────────────────────────────────

def _row_jitters(rows) -> list:
    """Sorted nominal jitter amplitudes present in the data ([] if none)."""
    from deployment.study_schema import jitter_level
    return sorted({j for r in rows
                   if (j := jitter_level(r)) is not None})


def jitter_descriptives(rows, tick_hz: float, delays, jitters) -> dict:
    """Per (delay, setting, jitter): run counts, timeout %, mean/median
    simulated seconds (success-only participant cell means), and the
    realized delay mean/std (ms)."""
    from deployment.study_schema import jitter_level
    n_runs, n_timeout = Counter(), Counter()
    pp: dict[tuple, list[float]] = defaultdict(list)
    real: dict[tuple, list[tuple]] = defaultdict(list)
    for r in rows:
        j = jitter_level(r)
        if j is None:
            continue
        cell = (int(r["delay_n"]), r["setting"], j)
        n_runs[cell] += 1
        if str(r.get("timed_out", "")).strip().lower() == "true":
            n_timeout[cell] += 1
        dm, ds = _float(r.get("delay_mean_ms")), _float(r.get("delay_std_ms"))
        if dm is not None and ds is not None:
            real[cell].append((dm, ds))
        if str(r.get("success", "")).strip().lower() == "true":
            v = _float(r.get("completion_ticks"))
            if v is not None:
                pp[(r["participant_id"],) + cell].append(v / tick_hz)
    cells = {k: float(np.mean(v)) for k, v in pp.items()}
    by_cell: dict[tuple, list[float]] = defaultdict(list)
    for (pid, d, s, j), v in cells.items():
        by_cell[(d, s, j)].append(v)
    out = {}
    for d in delays:
        for s in ("no_wm", "wm"):
            for j in jitters:
                vals = np.asarray(by_cell.get((d, s, j), []), dtype=float)
                rv = np.asarray(real.get((d, s, j), []), dtype=float)
                out[(d, s, j)] = {
                    "n_runs": n_runs.get((d, s, j), 0),
                    "n_part": int(vals.size),
                    "timeout_pct": 100.0 * n_timeout.get((d, s, j), 0)
                    / max(n_runs.get((d, s, j), 1), 1),
                    "mean_s": float(vals.mean()) if vals.size else math.nan,
                    "median_s": float(np.median(vals))
                    if vals.size else math.nan,
                    "delay_mean_ms": float(rv[:, 0].mean())
                    if rv.size else math.nan,
                    "delay_std_ms": float(rv[:, 1].mean())
                    if rv.size else math.nan,
                }
    return out


def jitter_lmm(rows, tick_hz, delays, jitters):
    """MixedLM on log simulated seconds: condition + jitter + delay
    dummies + condition×jitter, per-participant cell means."""
    from deployment.study_schema import jitter_level
    jhi = max(jitters)
    pp: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        j = jitter_level(r)
        if j is None:
            continue
        if str(r.get("success", "")).strip().lower() != "true":
            continue
        v = _float(r.get("completion_ticks"))
        if v is None:
            continue
        pp[(r["participant_id"], int(r["delay_n"]), r["setting"],
            j)].append(v / tick_hz)
    recs = sorted((k, np.mean(v)) for k, v in pp.items())
    if not recs:
        return None, "no successful jitter rows"
    endog = np.log(np.array([v for _, v in recs], dtype=float))
    wm = np.array([1.0 if s == "wm" else 0.0
                   for (_, d, s, j), _ in recs])
    jh = np.array([1.0 if j == jhi else 0.0
                   for (_, d, s, j), _ in recs])
    cols = [np.ones_like(wm), wm, jh]
    for d in delays[1:]:
        cols.append(np.array([1.0 if dd == d else 0.0
                              for (_, dd, _, _), _ in recs]))
    cols.append(wm * jh)  # condition×jitter interaction
    X = np.column_stack(cols)
    groups = [pid for (pid, _, _, _), _ in recs]
    res, flags, error = _fit_mixed(endog, X, X[:, :2], groups, reml=True)
    for fl in flags:
        _log(f"jitter LMM: {fl}")
    if res is None:
        return None, error
    b, se, p, ci = _fe_stats(res, X.shape[1])
    R = np.zeros((1, X.shape[1]))
    R[0, -1] = 1.0
    stat, df, p_int = _wald(res.fe_params[:X.shape[1]],
                            res.cov_params()[:X.shape[1], :X.shape[1]], R)
    return {"n": len(recs), "b": b, "se": se, "p": p, "ci": ci,
            "int_stat": stat, "int_df": df, "int_p": p_int,
            "flags": flags}, None


def table5_tex(rows, args) -> str:
    """Jitter ablation: completion + realized delay stats per cell, and
    the jitter-effect tests."""
    delays = [int(d) for d in args.delays]
    jitters = _row_jitters(rows)
    desc = jitter_descriptives(rows, args.tick_hz, delays, jitters)
    lines = [
        "\\begin{tabular}{l l r r r r r r r}",
        "\\hline",
        "Delay & Condition & Jitter & $n$ & Mean & Median & Timeout (\\%) "
        "& Real. mean & Real. std \\\\",
        " & & (fr.) & & (s) & (s) & & (ms) & (ms) \\\\",
        "\\hline",
    ]
    for d in delays:
        for si, s in enumerate(("no_wm", "wm")):
            for ji, j in enumerate(jitters):
                c = desc[(d, s, j)]
                cells = [
                    str(d) if si == 0 and ji == 0 else "",
                    _cell_label(d, s) if ji == 0 else "",
                    f"$\\pm${j}",
                    str(c["n_runs"]),
                    _fmt_sec(c["mean_s"]) if math.isfinite(c["mean_s"])
                    else "---",
                    _fmt_sec(c["median_s"]) if math.isfinite(c["median_s"])
                    else "---",
                    f"{c['timeout_pct']:.1f}",
                    _fmt_sec(c["delay_mean_ms"]) if math.isfinite(
                        c["delay_mean_ms"]) else "---",
                    _fmt_sec(c["delay_std_ms"]) if math.isfinite(
                        c["delay_std_ms"]) else "---",
                ]
                lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")

    if not args.no_stats:
        jm, jerr = jitter_lmm(rows, args.tick_hz, delays, jitters)
        lines += ["", "\\vspace{0.6em}", "",
                  "\\begin{tabular}{l l r r r}",
                  "\\hline",
                  "Test & Effect & Estimate & 95\\% CI & $p$ \\\\",
                  "\\hline"]
        if jm is None:
            lines.append(f"\\multicolumn{{5}}{{l}}{{Jitter model failed: "
                         f"{jerr}}} \\\\")
        else:
            jhi = max(jitters)
            rows_out = [
                ("Jitter LMM", f"Jitter ${jhi}$ vs ${min(jitters)}$",
                 _fmt_est(jm["b"][2]), _fmt_ci(*jm["ci"][2]),
                 _fmt_p(jm["p"][2])),
                ("Jitter LMM", "Cond $\\times$ Jitter",
                 _fmt_chi(jm["int_stat"], jm["int_df"]),
                 "---", _fmt_p(jm["int_p"])),
            ]
            for t, e, est, ci, p in rows_out:
                lines.append(" & ".join([t, e, est, ci, p]) + " \\\\")
        lines.append("\\hline")
        lines.append("\\end{tabular}")

    caption = (args.caption_5 +
               " Jitter ablation: per delay, condition and jitter amplitude "
               "(frames), completion time in simulated seconds over "
               "successful trials and the realized per-run delay mean/std. "
               "The model row tests the log-time jitter effect with "
               "condition and delay controlled. Each trial slot contributes "
               "its final logged attempt (post-save retries supersede "
               "earlier attempts; in-run fault-discards never log).")
    return _float_block(caption, args.label_5, "\n".join(lines))


TLX_ITEMS = [
    ("Mental demand", "tlx_mental"),
    ("Performance", "tlx_performance"),
    ("Frustration", "tlx_frustration"),
    ("Responsiveness", "tlx_responsiveness"),
    ("Authenticity", "tlx_authenticity"),
]


def tlx_cells(rows) -> dict:
    """(pid, delay, setting) -> {item: value}; scans ALL attempts.

    Attempt-2 rows fill cells missing on attempt 1 (verified: the union
    gives exactly one answer per participant × delay × setting cell).
    """
    cells: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        for item, col in TLX_ITEMS:
            v = _float(r.get(col))
            if v is None:
                continue
            cells[(r["participant_id"], int(r["delay_n"]),
                   r["setting"])][item] = v
    return cells


def tlx_paired_deltas(rows) -> dict:
    """Per TLX item: wm−no_wm paired delta, 95 % t CI, t, p, n."""
    from scipy import stats as st
    cells = tlx_cells(rows)
    per: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for (pid, _d, s), dct in cells.items():
        for item, v in dct.items():
            per[pid][s][item].append(v)
    out = {}
    for item, _col in TLX_ITEMS:
        deltas = []
        for pid, by_setting in per.items():
            wm_vals = by_setting.get("wm", {}).get(item, [])
            no_vals = by_setting.get("no_wm", {}).get(item, [])
            if wm_vals and no_vals:
                deltas.append(np.mean(wm_vals) - np.mean(no_vals))
        n = len(deltas)
        if n == 0:
            out[item] = {"n": 0}
            continue
        arr = np.asarray(deltas, dtype=float)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1)) if n > 1 else 0.0
        se = sd / math.sqrt(n)
        t = mean / se if se > 0 else math.nan
        p = float(st.t.sf(abs(t), n - 1) * 2) if math.isfinite(t) \
            else math.nan
        ci = st.t.interval(0.95, n - 1, loc=mean, scale=se) \
            if n > 1 else (mean, mean)
        out[item] = {"mean": mean, "ci": ci, "t": t, "p": p, "n": n}
    return out


def frustration_lmm(rows, delays):
    """frustration ~ condition*delay + (1|participant) over TLX cells."""
    cells = tlx_cells(rows)
    recs = sorted((k, v["Frustration"]) for k, v in cells.items()
                  if "Frustration" in v)
    if not recs:
        return None, "no frustration answers"
    endog = np.array([v for _, v in recs], dtype=float)
    wm = np.array([1.0 if s == "wm" else 0.0
                   for (_, d, s), _ in recs])
    cols = [np.ones_like(wm), wm]
    for d in delays[1:]:
        cols.append(np.array([1.0 if dd == d else 0.0
                              for (_, dd, _), _ in recs]))
    for d in delays[1:]:
        cols.append(wm * np.array([1.0 if dd == d else 0.0
                                   for (_, dd, _), _ in recs]))
    X = np.column_stack(cols)
    groups = [pid for (pid, _, _), _ in recs]
    res, flags, error = _fit_mixed(endog, X, X[:, :1], groups, reml=True)
    for fl in flags:
        _log(f"frustration LMM: {fl}")
    if res is None:
        return None, error
    b, se, p, ci = _fe_stats(res, X.shape[1])
    k_int = 2 + len(delays[1:])
    R = np.zeros((len(delays[1:]), X.shape[1]))
    for j in range(len(delays[1:])):
        R[j, k_int + j] = 1.0
    stat, df, p_int = _wald(res.fe_params[:X.shape[1]],
                            res.cov_params()[:X.shape[1], :X.shape[1]], R)
    return {"n": len(recs), "b": b, "se": se, "p": p, "ci": ci,
            "int_stat": stat, "int_df": df, "int_p": p_int,
            "flags": flags}, None


def table3_descriptive_tex(rows, args) -> str:
    deltas = tlx_paired_deltas(rows)
    lines = [
        "\\begin{tabular}{l r r r r r}",
        "\\hline",
        "TLX item & $\\Delta$ (wm $-$ no\\_wm) & 95\\% CI & $t$ & $p$ & $n$ \\\\",
        "\\hline",
    ]
    for item, _col in TLX_ITEMS:
        d = deltas.get(item, {"n": 0})
        if d["n"] == 0:
            continue
        lines.append(" & ".join([
            item,
            f"{d['mean']:.2f}",
            f"[{d['ci'][0]:.2f}, {d['ci'][1]:.2f}]",
            f"{d['t']:.2f}",
            _fmt_p(d["p"]),
            str(d["n"]),
        ]) + " \\\\")
    lines.append("\\hline")
    lines.append(
        "\\multicolumn{6}{l}{\\footnotesize Performance scale reversed: "
        "higher = better (all other items higher = worse); negative "
        "$\\Delta$ under wm therefore means \\emph{worse} perceived "
        "performance.} \\\\")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def table3_stats_tex(rows, args) -> str:
    delays = [int(d) for d in args.delays]
    fr, error = frustration_lmm(rows, delays)
    lines = [
        "\\begin{tabular}{l l r r r}",
        "\\hline",
        "Test & Effect & Estimate & 95\\% CI & $p$ \\\\",
        "\\hline",
    ]
    if fr is None:
        lines.append(
            f"\\multicolumn{{5}}{{l}}{{Frustration model failed: {error}}} "
            "\\\\")
    else:
        rows_out = [("Frustration LMM", "Condition (wm)",
                     _fmt_est(fr["b"][1]), _fmt_ci(*fr["ci"][1]),
                     _fmt_p(fr["p"][1]))]
        for j, d in enumerate(delays[1:]):
            rows_out.append(("Frustration LMM", f"Delay {d} vs {delays[0]}",
                             _fmt_est(fr["b"][2 + j]),
                             _fmt_ci(*fr["ci"][2 + j]),
                             _fmt_p(fr["p"][2 + j])))
        rows_out.append(("Frustration LMM", "Cond $\\times$ Delay",
                         _fmt_chi(fr["int_stat"], fr["int_df"]),
                         "---", _fmt_p(fr["int_p"])))
        for t, e, est, ci, p in rows_out:
            lines.append(" & ".join([t, e, est, ci, p]) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def table3_tex(rows, args) -> str:
    body = table3_descriptive_tex(rows, args)
    if not args.no_stats:
        body += "\n\n\\vspace{0.6em}\n\n" + table3_stats_tex(rows, args)
    caption = (args.caption_3 +
               " Mean paired differences (wm $-$ no\\_wm) pooled across "
               "delays, 95\\% confidence intervals from the paired $t$ "
               "distribution.")
    return _float_block(caption, args.label_3, body)


# ── assembly ──────────────────────────────────────────────────────────────────

def _float_block(caption: str, label: str, body: str) -> str:
    return (f"% generated by: make_result_tables.py {' '.join(sys.argv[1:])}\n"
            f"\\begin{{table}}[t]\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            f"\\begin{{center}}\n"
            f"{body}\n"
            f"\\end{{center}}\n"
            f"\\end{{table}}\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Copyable LaTeX result tables for the user study")
    ap.add_argument("--results", default=os.path.join(
        REPO, "results", "final", "study_results.csv"),
        help="results CSV (default: results/final/study_results.csv)")
    ap.add_argument("--output", default=None,
                    help="write the LaTeX fragment to this file instead of "
                         "stdout")
    ap.add_argument("--tables", nargs="+", default=["1", "2", "3", "4", "5"],
                    help="which tables to emit (default: 1 2 3 4 5; "
                         "4 = Cox tests, 5 = jitter ablation — auto-skipped "
                         "when the data has no jitter)")
    ap.add_argument("--pred-ms", type=float, default=6.0,
                    help="predictor latency τp in ms (Appendix C default "
                         "6.0)")
    ap.add_argument("--dec-ms", type=float, default=13.0,
                    help="decoder latency τq in ms (Appendix C default "
                         "13.0)")
    ap.add_argument("--enc-ms", type=float, default=11.0,
                    help="encoder latency τe in ms (Appendix C default "
                         "11.0)")
    ap.add_argument("--tick-ms", type=float, default=50.0,
                    help="per-frame tick budget τf in ms (default 50)")
    ap.add_argument("--delays", nargs="+", type=int, default=[10, 15, 20],
                    help="delay values in frames (default: 10 15 20)")
    ap.add_argument("--tick-hz", type=float, default=20.0,
                    help="tick rate for simulated seconds (default 20)")
    ap.add_argument("--caption-1", default="Timing budget. ",
                    help="Table 1 caption (LaTeX, prepended)")
    ap.add_argument("--caption-2", default="Completion statistics. ",
                    help="Table 2 caption (LaTeX, prepended)")
    ap.add_argument("--caption-3", default="TLX ratings. ",
                    help="Table 3 caption (LaTeX, prepended)")
    ap.add_argument("--caption-4", default="Inferential tests. ",
                    help="Table 4 caption (LaTeX, prepended)")
    ap.add_argument("--caption-5", default="Jitter ablation. ",
                    help="Table 5 caption (LaTeX, prepended)")
    ap.add_argument("--label-1", default="tab:timing")
    ap.add_argument("--label-2", default="tab:completion")
    ap.add_argument("--label-3", default="tab:tlx")
    ap.add_argument("--label-4", default="tab:tests")
    ap.add_argument("--label-5", default="tab:jitter")
    ap.add_argument("--attempts", choices=["first", "last", "all"],
                    default="last",
                    help="retry rule per (participant, trial) slot: 'last' "
                         "(default) keeps the final logged attempt — "
                         "post-save retries supersede earlier attempts, and "
                         "in-run fault-discards never log; 'first' keeps the "
                         "earliest; 'all' keeps every attempt")
    ap.add_argument("--include-practice", action="store_true",
                    help="add the undelayed GT practice runs as a delay-0 "
                         "reference row in the completion table "
                         "(descriptive only; excluded from the tests)")
    ap.add_argument("--no-stats", action="store_true",
                    help="descriptives only; skip all inferential blocks "
                         "(works without statsmodels)")
    args = ap.parse_args()
    args.delays = [int(d) for d in args.delays]

    rows = load_rows(Path(args.results), args.attempts)
    tlx_rows = load_rows(Path(args.results), attempts="all")
    practice_rows = load_rows(Path(args.results), args.attempts,
                              phase="practice") if args.include_practice \
        else None
    _log(f"{len(rows)} main rows "
         f"(attempts={args.attempts}), "
         f"{len(tlx_cells(tlx_rows))} TLX cells"
         + (f", {len(practice_rows)} practice rows included"
            if practice_rows else ""))

    parts = []
    for t in args.tables:
        try:
            if t == "1":
                parts.append(table1_tex(args))
            elif t == "2":
                parts.append(table2_tex(rows, args, practice_rows))
            elif t == "3":
                # NOTE: the TLX items were not properly administered in the
                # pilot — for the rebuttal report run with --tables "1 2 4 5".
                parts.append(table3_tex(tlx_rows, args))
            elif t == "4":
                if args.no_stats:
                    _log("table 4 (Cox tests) skipped — --no-stats")
                    continue
                parts.append(table4_tex(rows, args))
            elif t == "5":
                if not _row_jitters(rows):
                    _log("table 5 (jitter ablation) skipped — no jitter "
                         "data in this CSV")
                    continue
                if args.no_stats:
                    _log("table 5 (jitter ablation) — stats block skipped "
                         "by --no-stats")
                parts.append(table5_tex(rows, args))
            else:
                _log(f"unknown table id {t!r} (expected 1/2/3/4/5) — "
                     f"skipped")
        except Exception as exc:
            _log(f"table {t} failed: {type(exc).__name__}: {exc}")
    tex = "\n\n".join(parts) + "\n"
    if args.output:
        with open(args.output, "w") as f:
            f.write(tex)
        _log(f"wrote {args.output}")
    else:
        sys.stdout.write(tex)


if __name__ == "__main__":
    main()
