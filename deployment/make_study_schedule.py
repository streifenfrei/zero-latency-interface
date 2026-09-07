#!/usr/bin/env python3
"""
Generate a counterbalanced trial schedule for a teleoperation latency user study.

Design
------
Within-subjects, blocked by delay condition:

  * Delay conditions are presented in blocks. Block order across participants
    follows a Williams design (a Latin square balanced for first-order
    carryover), so every delay follows every other delay equally often.
  * Inside each block the WM / no-WM setting alternates in an ABBA pattern, so
    the critical comparison is adjacent in time at matched delay and is immune
    to within-block drift, fatigue and warm-up.
  * Participant i is assigned schedule row (i mod cycle_length). Recruitment can
    therefore be sequential and open-ended; balance is exact at every multiple
    of cycle_length and off by at most one participant per row in between.

Write the CSV once, before recruiting, and have the experiment software look up
participant_id -> rows from the file rather than generating order at runtime.

Example
-------
  python make_schedule.py --delays 0 200 400 --trials 4 --participants 24 \
      --practice 10 --spares 3 -o schedule.csv

Spare trials carry no delay/setting: a replacement inherits the condition of
whatever trial was discarded.  They exist only to pre-commit the initial-pose
seed, so a failed trial cannot be re-rolled into an easier configuration.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict


# --------------------------------------------------------------------------- #
# Design construction
# --------------------------------------------------------------------------- #

def williams_square(n):
    """Latin square balanced for first-order carryover.

    Returns a list of rows, each a permutation of range(n).
    n even -> n rows.  n odd -> 2n rows (square plus its mirror).
    """
    if n < 1:
        raise ValueError("need at least one condition")
    if n == 1:
        return [[0]]

    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            if j % 2 == 0:
                v = (i + j // 2) % n
            else:
                v = (i + n - 1 - (j - 1) // 2) % n
            row.append(v)
        rows.append(row)

    if n % 2 == 1:
        rows = rows + [list(reversed(r)) for r in rows]
    return rows


def abba_pattern(trials_per_setting, start):
    """ABBA-style alternation of two settings.

    Produces 2 * trials_per_setting entries, balanced in count and in mean
    serial position, so a linear within-block trend cannot favour either
    setting.  `start` is 0 or 1 and fixes which setting comes first.
    """
    a, b = start, 1 - start
    seq = []
    for i in range(trials_per_setting):
        seq.extend([a, b] if i % 2 == 0 else [b, a])
    return seq


# --------------------------------------------------------------------------- #
# Schedule generation
# --------------------------------------------------------------------------- #

def build_schedule(delays, trials, participants, cycle_length=None,
                   practice=0, spares=0, settings=("no_wm", "wm"),
                   practice_delay=None, first_pid=1, seed_base=1000):
    """Return (rows, cycle_length, n_orders) where rows is a list of dicts."""
    orders = williams_square(len(delays))
    n_orders = len(orders)
    natural_cycle = n_orders * 2          # x2 for which setting starts a block

    if cycle_length is None:
        cycle_length = natural_cycle

    if practice_delay is None:
        practice_delay = min(delays)

    rows = []
    for p in range(participants):
        pid = f"P{first_pid + p:03d}"
        r = p % cycle_length
        delay_order = orders[r % n_orders]
        start_setting = (r // n_orders) % 2

        trial_no = 0

        # ---- practice: undelayed, no world model, not analysed -------------
        for t in range(practice):
            trial_no += 1
            rows.append(dict(
                participant_id=pid, schedule_row=r, phase="practice",
                block_index=0, delay_n=practice_delay,
                setting="none", trial_in_block=t + 1,
                session_trial=trial_no, is_spare=0,
                seed=seed_base + 10000 * p + trial_no,
            ))

        # ---- measured blocks -----------------------------------------------
        for b, d_idx in enumerate(delay_order):
            # flip the starting setting each block so it is not confounded
            # with block position within a participant
            block_start = (start_setting + b) % 2
            pattern = abba_pattern(trials, block_start)
            for t, s in enumerate(pattern):
                trial_no += 1
                rows.append(dict(
                    participant_id=pid, schedule_row=r, phase="main",
                    block_index=b + 1, delay_n=delays[d_idx],
                    setting=settings[s], trial_in_block=t + 1,
                    session_trial=trial_no, is_spare=0,
                    seed=seed_base + 10000 * p + trial_no,
                ))

        # ---- spares: a condition-agnostic pool of pre-committed seeds ------
        # The delay/setting of a replacement is forced by whichever trial was
        # discarded, so it is not pre-assigned here.  What is pre-committed is
        # the seed, which removes any freedom to re-roll the initial pose after
        # a failure.  Consume the pool in order and record which spare replaced
        # which discarded trial.
        for k in range(spares):
            trial_no += 1
            rows.append(dict(
                participant_id=pid, schedule_row=r, phase="spare",
                block_index="", delay_n="", setting="",
                trial_in_block=k + 1,
                session_trial=trial_no, is_spare=1,
                seed=seed_base + 10000 * p + trial_no,
            ))

    return rows, cycle_length, n_orders


# --------------------------------------------------------------------------- #
# Balance verification
# --------------------------------------------------------------------------- #

def verify(rows, delays, settings, cycle_length, n_orders, participants):
    main = [r for r in rows if r["phase"] == "main"]
    out = []
    ok = True

    out.append(f"participants          : {participants}")
    out.append(f"schedule cycle length : {cycle_length}"
               f"  (natural = {n_orders * 2})")
    complete = participants // cycle_length
    rem = participants % cycle_length
    out.append(f"complete cycles       : {complete}"
               + (f"  (+{rem} partial -- rows 0..{rem - 1} over-represented)"
                  if rem else "  (exactly balanced)"))
    if rem:
        ok = False
    out.append(f"measured trials/person: {len(main) // participants}")
    out.append("")

    # delay x block position
    out.append("delay x block position (want equal columns):")
    pos = defaultdict(Counter)
    for r in main:
        pos[r["delay_n"]][r["block_index"]] += 1
    blocks = sorted({r["block_index"] for r in main})
    out.append("  delay_n | " + " ".join(f"blk{b:>2}" for b in blocks))
    for d in delays:
        out.append(f"  {d:>8} | "
                   + " ".join(f"{pos[d][b] // 1:>5}" for b in blocks))
    out.append("")

    # setting x within-block position
    out.append("setting x within-block position (want equal rows):")
    sp = defaultdict(Counter)
    for r in main:
        sp[r["setting"]][r["trial_in_block"]] += 1
    tpos = sorted({r["trial_in_block"] for r in main})
    out.append("  setting | " + " ".join(f"t{t:>3}" for t in tpos))
    for s in settings:
        out.append(f"  {s:>7} | " + " ".join(f"{sp[s][t]:>4}" for t in tpos))
    out.append("")

    # mean serial position of each setting (linear-drift check)
    msp = defaultdict(list)
    for r in main:
        msp[r["setting"]].append(r["trial_in_block"])
    out.append("mean within-block position (want identical):")
    for s in settings:
        out.append(f"  {s:>7} : {sum(msp[s]) / len(msp[s]):.3f}")
    out.append("")

    # first-order carryover between delay blocks
    out.append("delay->delay carryover pairs (want equal counts):")
    pairs = Counter()
    by_p = defaultdict(list)
    for r in main:
        by_p[r["participant_id"]].append(r)
    for p, trs in by_p.items():
        seq = []
        for r in trs:
            if not seq or seq[-1] != r["delay_n"]:
                seq.append(r["delay_n"])
        for a, b in zip(seq, seq[1:]):
            pairs[(a, b)] += 1
    if pairs:
        counts = sorted(pairs.values())
        for (a, b), c in sorted(pairs.items()):
            out.append(f"  {a:>6} -> {b:>6} : {c}")
        spread = counts[-1] - counts[0]
        out.append(f"  max-min = {spread}"
                   + ("  (balanced)" if spread == 0 else "  (imbalanced)"))
        if spread:
            ok = False
    out.append("")

    # per-cell trial counts
    cell = Counter((r["delay_n"], r["setting"]) for r in main)
    counts = sorted(cell.values())
    out.append(f"trials per (delay, setting) cell: "
               f"min {counts[0]}, max {counts[-1]}"
               + ("  (equal)" if counts[0] == counts[-1] else "  (UNEQUAL)"))
    if counts[0] != counts[-1]:
        ok = False

    return "\n".join(out), ok


# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate a counterbalanced user-study schedule.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--delays", type=int, nargs="+", default=[10, 15, 20, 25],
                    help="delay conditions")
    ap.add_argument("--trials", type=int, default=4,
                    help="measured trials per setting per delay block "
                         "(block length = 2 x this)")
    ap.add_argument("--participants", type=int, default=48,
                    help="number of participants to generate rows for")
    ap.add_argument("--rows", type=int, default=None, dest="cycle_length",
                    help="override the schedule cycle length k "
                         "(default: the natural balanced value)")
    ap.add_argument("--practice", type=int, default=10,
                    help="practice trials at the start (not analysed)")
    ap.add_argument("--spares", type=int, default=3,
                    help="size of the per-participant pool of pre-committed "
                         "replacement seeds; condition-agnostic, consumed in "
                         "order when a trial is discarded")
    ap.add_argument("--practice-delay", type=int, default=None, metavar="MS",
                    help="delay used during practice (default: lowest delay)")
    ap.add_argument("--settings", nargs=2, default=["no_wm", "wm"],
                    metavar=("A", "B"), help="the two setting labels")
    ap.add_argument("--first-pid", type=int, default=1,
                    help="numeric part of the first participant id")
    ap.add_argument("--seed-base", type=int, default=1000,
                    help="base for reproducible per-trial init-pose seeds")
    ap.add_argument("-o", "--out", default="schedule.csv",
                    help="output CSV path")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the verification report")
    a = ap.parse_args(argv)

    if len(set(a.delays)) != len(a.delays):
        ap.error("--delays must be distinct")
    if a.trials < 1:
        ap.error("--trials must be >= 1")

    rows, cycle_length, n_orders = build_schedule(
        delays=a.delays, trials=a.trials, participants=a.participants,
        cycle_length=a.cycle_length, practice=a.practice, spares=a.spares,
        settings=tuple(a.settings), practice_delay=a.practice_delay,
        first_pid=a.first_pid, seed_base=a.seed_base)

    fields = ["participant_id", "schedule_row", "phase", "block_index",
              "delay_n", "setting", "trial_in_block", "session_trial",
              "is_spare", "seed"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    if not a.quiet:
        report, ok = verify(rows, a.delays, tuple(a.settings),
                            cycle_length, n_orders, a.participants)
        print(report, file=sys.stderr)
        print(file=sys.stderr)
        if a.cycle_length is not None and a.cycle_length != n_orders * 2:
            print(f"WARNING: cycle length {a.cycle_length} != natural "
                  f"{n_orders * 2}; carryover balance is not guaranteed.",
                  file=sys.stderr)
        if not ok:
            print("NOTE: schedule is not perfectly balanced -- see above. "
                  "Stop recruitment at a multiple of the cycle length, or "
                  "include serial position as a covariate.", file=sys.stderr)

    print(f"wrote {len(rows)} rows to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())