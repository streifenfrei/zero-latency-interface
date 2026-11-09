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


def load_rows(path: Path, all_attempts: bool, phase: str = "main") -> list[dict]:
    """Rows of one phase (mirror of plot_study_results.load_rows)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("phase", "main") != phase:
                continue
            if not all_attempts and int(r.get("attempt", 1)) != 1:
                continue
            rows.append(r)
    if not rows:
        raise SystemExit(f"[TABLES] no usable rows in {path}")
    return rows


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

def gain_frames(delay: int, tau_ms: float, tau_f_ms: float) -> float | None:
    """G(D) = D·τ/(τf − τ)."""
    if tau_f_ms <= tau_ms:
        return None
    return delay * tau_ms / (tau_f_ms - tau_ms)


def table1_tex(args) -> str:
    delays = [int(d) for d in args.delays]
    spec = "l r" + " | r r" * len(delays)
    head1 = " & $\\tau$ (ms)" + "".join(
        f" & \\multicolumn{{2}}{{c}}{{$k{{=}}{d}$ ({d * 1000 / args.tick_hz:.0f} ms)}}"
        for d in delays)
    head2 = " & " + "".join(" & $G$ & $k{+}G$" for _ in delays)
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
        ("Encoder", args.enc_ms, False),
        ("Tick budget $\\tau_f$", args.tick_ms, False),
    ]
    for label, tau, is_pred in comp_rows:
        cells = [label, _fmt_ms(tau)]
        if is_pred:
            for d in delays:
                g = gain_frames(d, args.pred_ms, args.tick_ms)
                cells.append(_fmt_g(g))
                cells.append(_fmt_g(d + g))
        else:
            cells += ["\\multicolumn{2}{c}{---}"] * len(delays)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")

    caption = (args.caption_1 +
               " Component latencies (batch 1, RTX 2080 Ti, frameskip 1 — "
               "one model step per frame) against the per-frame tick budget "
               "$\\tau_f$; $G(k)=k\\tau_p/(\\tau_f-\\tau_p)$ look-ahead steps "
               "and $k{+}G$ total predictor passes for a delay of $k$ frames "
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
    """Stratified Cox PH on simulated seconds (ticks / tick rate).

    Hazard ratios are invariant to time scaling, so the fit equals one on
    raw ticks — the /tick_hz conversion keeps the units consistent with
    the descriptive completion table.
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
    cols_main = [wm]
    for d in delays[1:]:
        cols_main.append(np.array([1.0 if dd == d else 0.0
                                   for _, dd, _, _, _ in recs]))
    cols_full = cols_main + [
        wm * np.array([1.0 if dd == d else 0.0
                       for _, dd, _, _, _ in recs])
        for d in delays[1:]]
    # NO intercept: aliased under strata.  HR rows come from the
    # MAIN-effects fit; the interaction Wald from the full fit.
    X_main = np.column_stack(cols_main)
    X_full = np.column_stack(cols_full)
    strata = np.asarray([pid for pid, _, _, _, _ in recs])
    out = {"n": len(recs)}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res_main = PHReg(endog, X_main, status=status, strata=strata,
                             ties="breslow").fit()
            res_full = PHReg(endog, X_full, status=status, strata=strata,
                             ties="breslow").fit()
        for w in caught:
            _log(f"Cox: {w.message}")
        k_int = len(cols_main)
        R = np.zeros((len(delays[1:]), X_full.shape[1]))
        for j in range(len(delays[1:])):
            R[j, k_int + j] = 1.0
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
                     _fmt_ci(*cox["ci"][0]), _fmt_p(cox["p"][0]))]
        for j, d in enumerate(delays[1:]):
            rows_out.append((f"Delay {d} vs {delays[0]}",
                             f"HR = {_fmt_est(cox['hr'][1 + j])}",
                             _fmt_ci(*cox["ci"][1 + j]),
                             _fmt_p(cox["p"][1 + j])))
        rows_out.append(("Cond $\\times$ Delay",
                         _fmt_chi(cox["int_stat"], cox["int_df"]),
                         "---", _fmt_p(cox["int_p"])))
        for e, est, ci, p in rows_out:
            lines.append(" & ".join([e, est, ci, p]) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    caption = (args.caption_4 +
               " Cox proportional-hazards model of time-to-completion in "
               "simulated seconds (ticks / "
               f"{args.tick_hz:g}\\,Hz) with condition (no-WM vs WM), "
               "delay, and their interaction as covariates; timed-out runs "
               "enter as right-censored observations at the timeout limit. "
               "Participants are included as strata, so each condition "
               "contrast is a within-participant comparison. Hazard ratios "
               "(HR) greater than 1 indicate faster completion; the "
               "interaction row is a joint Wald test whether the WM "
               "advantage varies across delays.")
    return _float_block(caption, args.label_4, "\n".join(lines))


def table2_tex(rows, args, practice_rows=None) -> str:
    body = table2_descriptive_tex(rows, args, practice_rows)
    caption = (args.caption_2 +
               " Completion time is simulated (ticks / 20 Hz) over "
               "successful trials. Bold marks the better (lower) value "
               "of the two conditions per column.")
    if practice_rows:
        caption += (" Delay 0 are the undelayed GT practice "
                    "(familiarization) runs, shown for reference only.")
    return _float_block(caption, args.label_2, body)


# ── Table 3: rating statistics ────────────────────────────────────────────────

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
    ap.add_argument("--tables", nargs="+", default=["1", "2", "3", "4"],
                    help="which tables to emit (default: 1 2 3 4; "
                         "4 = Cox tests)")
    ap.add_argument("--pred-ms", type=float, default=6.45,
                    help="predictor latency τp in ms (fs=1/224 on RTX 2080 "
                         "Ti; default 6.45)")
    ap.add_argument("--dec-ms", type=float, default=13.43,
                    help="decoder latency τq in ms (default 13.43)")
    ap.add_argument("--enc-ms", type=float, default=5.25,
                    help="encoder latency in ms (default 5.25)")
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
    ap.add_argument("--label-1", default="tab:timing")
    ap.add_argument("--label-2", default="tab:completion")
    ap.add_argument("--label-3", default="tab:tlx")
    ap.add_argument("--label-4", default="tab:tests")
    ap.add_argument("--all-attempts", action="store_true",
                    help="include retry attempts in the statistics")
    ap.add_argument("--include-practice", action="store_true",
                    help="add the undelayed GT practice runs as a delay-0 "
                         "reference row in the completion table "
                         "(descriptive only; excluded from the tests)")
    ap.add_argument("--no-stats", action="store_true",
                    help="descriptives only; skip all inferential blocks "
                         "(works without statsmodels)")
    args = ap.parse_args()
    args.delays = [int(d) for d in args.delays]

    rows = load_rows(Path(args.results), args.all_attempts)
    tlx_rows = load_rows(Path(args.results), all_attempts=True)
    practice_rows = load_rows(Path(args.results), args.all_attempts,
                              phase="practice") if args.include_practice \
        else None
    _log(f"{len(rows)} main rows "
         f"(attempts={'all' if args.all_attempts else 'first'}), "
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
                parts.append(table3_tex(tlx_rows, args))
            elif t == "4":
                if args.no_stats:
                    _log("table 4 (Cox tests) skipped — --no-stats")
                    continue
                parts.append(table4_tex(rows, args))
            else:
                _log(f"unknown table id {t!r} (expected 1/2/3/4) — skipped")
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
