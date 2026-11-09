#!/usr/bin/env python3
"""
Reproduce every statistic reported in the paper from the study CSV.

Output is organised by paper section so each number can be checked against the
text.  Numbers marked [PAPER] are the ones quoted in the main text or tables.

Usage
-----
  python analyse_study.py study_results.csv
  python analyse_study.py study_results.csv --latex      # also emit LaTeX rows
  python analyse_study.py study_results.csv --wall       # wall-clock time base

Conventions
-----------
* Primary time base is SIMULATED task time: completion_ticks / 20 Hz.
* Completion time is log-transformed before averaging over participants,
  because trial-level times are right-skewed and the effect is multiplicative.
* Paired comparisons are computed on participant-level means, n = 22.
* Censored (capped) trials are excluded from descriptive comparisons and
  right-censored at their own censoring time in the Cox model.

Requires: pandas, numpy, scipy, lifelines
"""

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

TICK_HZ = 20.0
DELAY_MS = {10: 500, 15: 750, 20: 1000}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def hdr(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def sub(title):
    print()
    print(f"--- {title} " + "-" * max(0, 68 - len(title)))


def paired(pv, log=False):
    """Paired comparison on a participant x setting table. Returns dict."""
    pv = pv.dropna()
    a, b = pv["no_wm"], pv["wm"]
    if log:
        a, b = np.log(a), np.log(b)
    d = b - a
    n = len(d)
    t, p = stats.ttest_rel(b, a)
    se = d.std(ddof=1) / np.sqrt(n)
    ci = stats.t.ppf(0.975, n - 1) * se
    out = dict(n=n, mean_a=pv["no_wm"].mean(), mean_b=pv["wm"].mean(),
               diff=d.mean(), lo=d.mean() - ci, hi=d.mean() + ci,
               t=t, p=p, dz=abs(d.mean() / d.std(ddof=1)),
               n_better=int((d < 0).sum()), n_higher=int((d > 0).sum()))
    if log:
        out["pct_reduction"] = (1 - np.exp(d.mean())) * 100
    return out


def line(label, r, pct=False):
    if pct:
        print(f"  {label:<26} {r['pct_reduction']:+6.1f}%  "
              f"t({r['n']-1})={r['t']:+6.2f}  p={r['p']:.5f}  "
              f"dz={r['dz']:.2f}  faster={r['n_better']}/{r['n']}")
    else:
        print(f"  {label:<26} {r['mean_a']:8.3f} {r['mean_b']:8.3f}  "
              f"{r['diff']:+8.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
              f"t({r['n']-1})={r['t']:+6.2f}  p={r['p']:.4f}")


def pv_of(d, col, index="participant_id"):
    return d.pivot_table(index=index, columns="setting", values=col,
                         aggfunc="mean")


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #

def sec_sample(df, m):
    hdr("1. SAMPLE  (App: Participants)")
    first = df.groupby("participant_id").first()
    print(f"  participants                [PAPER] {df.participant_id.nunique()}")
    print(f"  measured trials             [PAPER] {len(m)}")
    print(f"  practice trials                     {(df.phase=='practice').sum()}")
    for c in ["age_band", "gender", "handedness",
              "gaming_hours_week", "gamepad_hours_week"]:
        counts = first[c].value_counts().to_dict()
        print(f"  {c:<26} {counts}")
    t = pd.to_datetime(df.timestamp_iso)
    dur = df.assign(t=t).groupby("participant_id").t.apply(
        lambda s: (s.max() - s.min()).total_seconds() / 60)
    print(f"  session minutes             [PAPER] median {dur.median():.1f} "
          f"(range {dur.min():.1f}-{dur.max():.1f})")

    sub("design realisation")
    cells = m.pivot_table(index="delay_n", columns="setting",
                          values="T", aggfunc="count")
    print(cells.to_string())
    rows = first.schedule_row.value_counts().sort_index().to_dict()
    print(f"  schedule rows used          [PAPER] {rows}")
    att = m.attempt.value_counts().sort_index().to_dict()
    print(f"  attempts                    [PAPER] {att}  "
          f"(first-attempt {att.get(1,0)}, retried {len(m)-att.get(1,0)})")
    print(f"  recompiles>0                        {(m.recompiles_during_run>0).sum()}")
    print(f"  frame pairs evaluated       [PAPER] {int(m.n_pairs.sum()):,}"
          f"  dropped {int(m.n_pairs_dropped.sum())}")


def sec_timebase(m):
    hdr("2. TIME BASE AND CENSORING  (App: Time base and censoring)")
    sub("realised wall-clock update rate (Hz)")
    tr = m.pivot_table(index="delay_n", columns="setting",
                       values="tick_rate_mean_hz", aggfunc="mean")
    tr.index = [DELAY_MS[i] for i in tr.index]
    print("  [PAPER]")
    print(tr.round(2).to_string())

    sub("censoring times among capped trials (control steps)")
    cap = m[m.timed_out]
    g = cap.groupby("setting").completion_ticks.agg(["count", "mean",
                                                     "min", "max"])
    print("  [PAPER]")
    print(g.round(1).to_string())
    print(f"  overall cap rate            [PAPER] "
          f"{100*m.timed_out.mean():.1f}%")


def sec_completion(m, base):
    hdr(f"3. COMPLETION TIME  (Sec. Results)   [time base: {base}]")
    comp = m[~m.timed_out]

    sub("primary: log-transformed, participant means")
    r = paired(pv_of(comp, "T"), log=True)
    line("completed only    [PAPER]", r, pct=True)
    r_all = paired(pv_of(m, "T"), log=True)
    line("incl. capped      [PAPER]", r_all, pct=True)

    sub("sensitivity to time base / transform")
    for lab, col, lg in [("simulated, log", "T_sim", True),
                         ("simulated, raw", "T_sim", False),
                         ("wall clock, log", "T_wall", True),
                         ("wall clock, raw", "T_wall", False)]:
        pv = pv_of(comp, col).dropna()
        if lg:
            d = np.log(pv.wm) - np.log(pv.no_wm)
            red = (1 - np.exp(d.mean())) * 100
            t = stats.ttest_rel(np.log(pv.wm), np.log(pv.no_wm))
        else:
            red = (1 - pv.wm.mean() / pv.no_wm.mean()) * 100
            t = stats.ttest_rel(pv.wm, pv.no_wm)
        print(f"  {lab:<20} {red:5.1f}%   t={t.statistic:+.2f}")

    sub("per delay, completed only, log")
    print("  [PAPER]")
    for k, ms in DELAY_MS.items():
        s = comp[comp.delay_n == k]
        pv = pv_of(s, "T").dropna()
        d = np.log(pv.wm) - np.log(pv.no_wm)
        t, p = stats.ttest_rel(np.log(pv.wm), np.log(pv.no_wm))
        print(f"    {ms:>5} ms   {(1-np.exp(d.mean()))*100:5.1f}%   "
              f"t({len(pv)-1})={t:+6.2f}  p={p:.4f}  n={len(pv)}")

    sub("median completion time by delay (for descriptive delay effect)")
    med = comp.pivot_table(index="delay_n", columns="setting",
                           values="T", aggfunc="median")
    med.index = [DELAY_MS[i] for i in med.index]
    print("  [PAPER]")
    print(med.round(1).to_string())

    sub("timeout rates")
    pv = pv_of(m, "timed_out")
    w = stats.wilcoxon(pv.wm, pv.no_wm)
    print(f"  [PAPER] baseline {pv.no_wm.mean()*100:.1f}%  "
          f"compensated {pv.wm.mean()*100:.1f}%  "
          f"Wilcoxon p={w.pvalue:.3f}")
    rate = m.pivot_table(index="delay_n", columns="setting",
                         values="timed_out", aggfunc="mean")
    rate.index = [DELAY_MS[i] for i in rate.index]
    print(rate.round(3).to_string())


def sec_cox(m):
    hdr("4. COX MODEL  (Sec. Results, Table: tests)")
    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test
    except ImportError:
        print("  lifelines not installed -- skipping "
              "(pip install lifelines)")
        return

    d = m.copy()
    d["E"] = (~d.timed_out).astype(int)
    d["wm"] = (d.setting == "wm").astype(int)
    d["d750"] = (d.delay_n == 15).astype(int)
    d["d1000"] = (d.delay_n == 20).astype(int)
    d["wm_d750"] = d.wm * d.d750
    d["wm_d1000"] = d.wm * d.d1000
    d["delay"] = d.delay_n
    d["pid"] = d.participant_id
    print(f"  events {d.E.sum()} / {len(d)}   censored {(1-d.E).sum()}")

    sub("PH check on UNSTRATIFIED model (why we stratify)")
    cols = ["T", "E", "wm", "d750", "d1000"]
    un = CoxPHFitter().fit(d[cols], "T", "E")
    print(un.summary[["exp(coef)", "exp(coef) lower 95%",
                      "exp(coef) upper 95%", "p"]].round(4).to_string())
    pht = proportional_hazard_test(un, d[cols], time_transform="rank")
    print("  scaled Schoenfeld residuals:   [PAPER]")
    print(pht.summary.round(4).to_string())

    sub("PRIMARY: stratified by delay, clustered by participant")
    st = CoxPHFitter().fit(d[["T", "E", "wm", "delay", "pid"]], "T", "E",
                           strata=["delay"], cluster_col="pid")
    s = st.summary
    print(f"  [PAPER] compensated vs baseline   "
          f"HR = {s.loc['wm','exp(coef)']:.3f}  "
          f"[{s.loc['wm','exp(coef) lower 95%']:.3f}, "
          f"{s.loc['wm','exp(coef) upper 95%']:.3f}]   "
          f"p = {s.loc['wm','p']:.4f}")

    st_u = CoxPHFitter().fit(d[["T", "E", "wm", "delay"]], "T", "E",
                             strata=["delay"])
    pht2 = proportional_hazard_test(st_u, d[["T", "E", "wm", "delay"]],
                                    time_transform="rank")
    print("  PH check on stratified model:   [PAPER]")
    print(pht2.summary.round(4).to_string())

    sub("condition x delay interaction (LR test, stratified)")
    full = CoxPHFitter().fit(
        d[["T", "E", "wm", "delay", "wm_d750", "wm_d1000"]], "T", "E",
        strata=["delay"])
    lr = 2 * (full.log_likelihood_ - st_u.log_likelihood_)
    print(f"  [PAPER] chi2(2) = {lr:.3f}   "
          f"p = {stats.chi2.sf(lr, 2):.4f}")


def sec_control(m):
    hdr("5. CONTROL BEHAVIOUR  (Sec. Results, Table: secondary)")
    comp = m[~m.timed_out].copy()

    sub("primary behavioural measures")
    for col, lab in [("idle_fraction", "Idle fraction"),
                     ("rev_rate", "Reversal rate (1/s)")]:
        r = paired(pv_of(comp, col))
        line(lab + "  [PAPER]", r)
        print(f"  {'':<26} lower/higher under wm: "
              f"{r['n_better']}/{r['n']} lower, "
              f"{r['n_higher']}/{r['n']} higher")

    sub("secondary metrics table")
    print(f"  {'metric':<22}{'base':>9}{'comp':>9}{'diff':>10}"
          f"{'95% CI':>22}{'p':>9}")
    for col, lab in [("idle_fraction", "Idle fraction"),
                     ("rev_rate", "Reversal rate (1/s)"),
                     ("pushes", "Pushes"),
                     ("ee_path_len_m", "EE path length (m)"),
                     ("block_path_len_m", "Block path length (m)"),
                     ("sparc", "SPARC")]:
        if col not in comp.columns:
            continue
        r = paired(pv_of(comp, col))
        print(f"  {lab:<22}{r['mean_a']:9.3f}{r['mean_b']:9.3f}"
              f"{r['diff']:+10.3f}"
              f"  [{r['lo']:+7.3f}, {r['hi']:+7.3f}]{r['p']:9.4f}")

    sub("trend across delay within each condition (Friedman)")
    print("  [PAPER]")
    for col in ["idle_fraction", "rev_rate"]:
        for setting in ["no_wm", "wm"]:
            s = comp[comp.setting == setting]
            w = s.pivot_table(index="participant_id", columns="delay_n",
                              values=col, aggfunc="mean").dropna()
            fr = stats.friedmanchisquare(*[w[c] for c in w.columns])
            vals = " -> ".join(f"{w[c].mean():.2f}" for c in w.columns)
            print(f"    {col:<15} {setting:<6} {vals}   "
                  f"chi2={fr.statistic:.2f}  p={fr.pvalue:.4f}  n={len(w)}")


def sec_fidelity(m):
    hdr("6. PREDICTION FIDELITY  (Sec. Results)")
    wm = m[m.setting == "wm"].copy()
    comp = wm[~wm.timed_out]

    sub("fidelity by delay")
    f = wm.groupby("delay_n")[["ssim_mean", "lpips_mean"]].mean()
    f.index = [DELAY_MS[i] for i in f.index]
    print("  [PAPER]")
    print(f.round(4).to_string())
    rho = stats.spearmanr(wm.ssim_mean, wm.lpips_mean)[0]
    print(f"  SSIM~LPIPS redundancy       [PAPER] rho = {rho:+.3f}")

    sub("within-delay: LPIPS vs outcome (Spearman)")
    print("  [PAPER]")
    for label, subset, col in [("completion time", comp, "T"),
                               ("reversals", comp, "reversals")]:
        vals = []
        for k in sorted(DELAY_MS):
            s = subset[subset.delay_n == k]
            r, p = stats.spearmanr(s.lpips_mean, s[col])
            vals.append(f"{DELAY_MS[k]}ms rho={r:+.2f} p={p:.4f}")
        print(f"    LPIPS vs {label:<18} " + " | ".join(vals))
    vals = []
    for k in sorted(DELAY_MS):
        s = wm[wm.delay_n == k]
        r, p = stats.spearmanr(s.lpips_mean, s.timed_out.astype(int))
        vals.append(f"{DELAY_MS[k]}ms rho={r:+.2f} p={p:.4f}")
    print("    LPIPS vs timeout           " + " | ".join(vals))

    sub("LPIPS vs rated authenticity")
    rated = m[m.tlx_authenticity.notna() & (m.setting == "wm")]
    r, p = stats.spearmanr(rated.lpips_mean, rated.tlx_authenticity)
    print(f"  [PAPER] condition level      rho = {r:+.3f}  p = {p:.4f}  "
          f"n = {len(rated)}")
    g = rated.copy()
    for c in ["lpips_mean", "tlx_authenticity"]:
        g[c + "_c"] = g[c] - g.groupby("participant_id")[c].transform("mean")
    r2, p2 = stats.spearmanr(g.lpips_mean_c, g.tlx_authenticity_c)
    print(f"  [PAPER] within participants  rho = {r2:+.3f}  p = {p2:.4f}")


def sec_tlx(m):
    hdr("7. SUBJECTIVE RATINGS  (Sec. Results, App: tlx)")
    items = ["tlx_mental", "tlx_performance", "tlx_frustration",
             "tlx_responsiveness", "tlx_authenticity"]
    rated = m[m.tlx_mental.notna()]
    print(f"  rated trials                [PAPER] {len(rated)}  "
          f"({len(rated)//rated.participant_id.nunique()} per participant)")

    sub("pooled across delays")
    print(f"  {'item':<22}{'delta':>9}{'95% CI':>22}{'t':>8}{'p':>9}")
    for it in items:
        r = paired(pv_of(rated, it))
        print(f"  {it:<22}{r['diff']:+9.2f}"
              f"  [{r['lo']:+7.2f}, {r['hi']:+7.2f}]"
              f"{r['t']:+8.2f}{r['p']:9.4f}   [PAPER]")

    sub("by delay (baseline / compensated)")
    for it in items:
        row = []
        for k in sorted(DELAY_MS):
            s = rated[rated.delay_n == k]
            pv = pv_of(s, it)
            row.append(f"{DELAY_MS[k]}ms {pv.no_wm.mean():.1f}/"
                       f"{pv.wm.mean():.1f}")
        print(f"  {it:<22} " + "  ".join(row) + "   [PAPER]")

    sub("responsiveness advantage per delay")
    for k in sorted(DELAY_MS):
        s = rated[rated.delay_n == k]
        r = paired(pv_of(s, "tlx_responsiveness"))
        print(f"    {DELAY_MS[k]:>5} ms   delta = {r['diff']:+.1f}   [PAPER]")


def sec_power(m):
    hdr("8. SAMPLE SIZE AND SENSITIVITY  (App: stats)")
    import math

    def pw(dz, n, a=0.05):
        dfree = n - 1
        ncp = dz * math.sqrt(n)
        c = stats.t.ppf(1 - a / 2, dfree)
        return (1 - stats.nct.cdf(c, dfree, ncp)
                + stats.nct.cdf(-c, dfree, ncp))

    n = m.participant_id.nunique()
    lo, hi = 0.01, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pw(mid, n) < 0.80 else (lo, mid)
    print(f"  minimum detectable dz at n={n}, 80% power, alpha=.05 "
          f"two-tailed:  [PAPER] {hi:.3f}")

    comp = m[~m.timed_out]
    r = paired(pv_of(comp, "T"), log=True)
    print(f"  observed dz, completion time            [PAPER] {r['dz']:.3f}")
    ri = paired(pv_of(comp, "idle_fraction"))
    print(f"  observed dz, idle fraction              [PAPER] {ri['dz']:.3f}")

    sub("pilot vs out-of-pilot (internal pilot check)")
    ids = sorted(m.participant_id.unique())
    pilot, rest = ids[:6], ids[6:]
    for lab, group in [("pilot P001-P006", pilot),
                       ("post-pilot", rest), ("full sample", ids)]:
        s = comp[comp.participant_id.isin(group)]
        rr = paired(pv_of(s, "T"), log=True)
        print(f"    {lab:<18} n={rr['n']:>2}  dz={rr['dz']:.3f}  "
              f"reduction={rr['pct_reduction']:.1f}%   [PAPER]")


def sec_perparticipant(m, latex):
    hdr("9. PER-PARTICIPANT DATA  (App: perparticipant)")
    comp = m[~m.timed_out]
    t = pv_of(comp, "T").round(1)
    i = pv_of(comp, "idle_fraction").round(2)
    r = pv_of(comp, "rev_rate").round(2)
    row = m.groupby("participant_id").schedule_row.first()
    slower = []
    for p in t.index:
        if t.loc[p, "wm"] > t.loc[p, "no_wm"]:
            slower.append(p)
        s = (f"{p} & {row[p]} & {t.loc[p,'no_wm']:.1f} / {t.loc[p,'wm']:.1f}"
             f" & {i.loc[p,'no_wm']:.2f} / {i.loc[p,'wm']:.2f}"
             f" & {r.loc[p,'no_wm']:.2f} / {r.loc[p,'wm']:.2f} \\\\")
        print("  " + (s if latex else s.replace(" \\\\", "")))
    print(f"\n  slower under compensation   [PAPER] {slower}")


# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--wall", action="store_true",
                    help="use wall-clock time instead of simulated task time")
    ap.add_argument("--latex", action="store_true",
                    help="emit per-participant rows as LaTeX")
    a = ap.parse_args(argv)

    df = pd.read_csv(a.csv)
    m = df[df.phase == "main"].copy()

    m["T_sim"] = m.completion_ticks / TICK_HZ
    m["T_wall"] = m.completion_time_s
    m["T"] = m.T_wall if a.wall else m.T_sim
    m["rev_rate"] = m.reversals / m["T"]
    base = "WALL CLOCK" if a.wall else "SIMULATED (ticks/20 Hz)"

    print(f"file: {a.csv}")
    print(f"time base: {base}")
    if a.wall:
        print("WARNING: the paper reports simulated task time; "
              "these numbers will not match.")

    sec_sample(df, m)
    sec_timebase(m)
    sec_completion(m, base)
    sec_cox(m)
    sec_control(m)
    sec_fidelity(m)
    sec_tlx(m)
    sec_power(m)
    sec_perparticipant(m, a.latex)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())