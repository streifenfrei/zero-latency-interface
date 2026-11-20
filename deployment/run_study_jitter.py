#!/usr/bin/env python
"""Second user study: per-tick uniform jitter instead of constant delay.

Identical to ``run_study.py`` in every respect (UI, prompts, TLX, intake,
metrics, retry, resume — all imported) except the delay model:

    d_t = max(1, delay_n + U(-J, +J))   per tick, J = --jitter (frames)

The per-tick draws are seeded from the schedule's ``seed`` column, so every
attempt of a given row sees the identical jitter pattern.  The realized
per-run delay statistics are recorded in two extra results columns:
``delay_mean_ms`` and ``delay_std_ms``.

Defaults differ from ``run_study.py`` only in ``--results-dir results_jitter/``
(separate data) and the new ``--jitter`` flag.

Usage:
    deployment/run_study_jitter.sh --ckpt outputs/.../checkpoints \\
        --policy-checkpoint checkpoints/policy.pt \\
        --gamepad --schedule deployment/schedule.csv --fps 20 [--jitter 5]
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# CRITICAL: pin SDL_VIDEODRIVER=x11 BEFORE importing the deployment package.
# `import deployment.run_study` runs deployment/__init__.py first, which
# imports deployment.gamepad — and gamepad.py setdefaults SDL_VIDEODRIVER to
# "dummy" at its own import time.  Without this pin, dummy wins and
# run_study's own x11 pin becomes a no-op → set_mode "succeeds" invisibly,
# flip() no-ops, and the UI loop spins with no window.
os.environ.setdefault("SDL_VIDEODRIVER", "x11")

# Must be imported after the pin: all UI helpers are reached through the rs
# namespace so the imported functions close over run_study's module globals
# (_win, _show_fps, _fps_hist, ...).
import deployment.run_study as rs  # noqa: E402
from deployment.delay import JitterDelay  # noqa: E402


def _add_jitter_flags(parser) -> None:
    """argparse hook invoked at the end of run_study.parse_args."""
    parser.add_argument(
        "--jitter", type=int, default=5,
        help="fallback jitter amplitude J in frames when the schedule has "
             "no jitter_n column: per tick, d_t = delay_n + U(-J, +J), "
             "floored at 1; the RNG is seeded from the schedule's seed "
             "column, so each trial's jitter pattern is exactly "
             "reproducible")
    parser.set_defaults(results_dir="results_jitter")
    parser.set_defaults(schedule=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "schedule_jitter.csv"))


def parse_args():
    args = rs.parse_args(extra=_add_jitter_flags)
    if args.jitter < 0:
        raise SystemExit("--jitter must be >= 0")
    return args


def main() -> None:
    args = parse_args()

    gamepad = None
    if args.gamepad:
        gamepad = rs.GamepadInput(
            speed=args.gamepad_speed,
            deadzone=args.gamepad_deadzone,
            fixed_magnitude=not args.gamepad_no_fixed_magnitude,
            precision_scale=args.gamepad_precision_scale,
            boost_scale=args.gamepad_boost_scale,
            invert_x=args.gamepad_invert_x,
            invert_y=args.gamepad_invert_y,
            swap_xy=args.gamepad_swap_xy,
            debug=args.gamepad_debug,
        )
        gamepad.setup()

    # Open the study window BEFORE the world model (loading screen while the
    # adapter compiles).  NOTE: assign the MODULE global — wait_for_button /
    # poll_input read rs._win, not a local.
    W, H = args.display_size, int(args.display_size * 0.6)
    rs._win = rs._StudyWindow(rs.WINDOW_NAME, W, H)
    if args.compile:
        rs._win.show(rs.loading_canvas(
            W, H, "Compiling world model...",
            "torch.compile inference graphs — first launch can take a minute"))

    wm = rs.DinoWMPushtAdapter(ckpt_dir=args.ckpt,
                               ckpt_suffix=args.ckpt_suffix,
                               use_amp=args.amp, torch_compile=args.compile,
                               compile_mode=args.compile_mode)
    env_kwargs = dict(
        env_id=args.env_id,
        policy_checkpoint=args.policy_checkpoint,
        control_mode=args.control_mode,
        action_scale=args.action_scale,
        step_size=args.step_size,
        max_input=args.gamepad_speed if args.gamepad else 60.0,
        fixed_magnitude=args.fixed_magnitude,
        level_ee=args.level_ee,
        push_height=args.push_height,
        sim_backend=args.sim_backend,
        max_episode_steps=args.max_episode_steps,
        with_velocity=not args.no_velocity,
        with_block_pose=args.with_block_pose,
        goal_tolerance=args.goal_tolerance,
        workspace_bounds=tuple(args.workspace_bounds)
        if args.workspace_bounds else None,
        render_size=wm.img_size,
    )
    env = rs.ManiSkillPushTEnv(**env_kwargs)
    env.seed(0)  # env RNG stays fixed; the jitter RNG is the per-row seed

    results_dir = rs.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    # Reduce the practice block to a single familiarization run per
    # participant (the constant study keeps its full practice block).
    schedule = {
        pid: ([r for r in rows if r["phase"] == "practice"][:1]
              + [r for r in rows if r["phase"] != "practice"])
        for pid, rows in rs.load_schedule(rs.Path(args.schedule)).items()}
    results_path = results_dir / "study_results.csv"
    state_path = results_dir / "session_state.json"
    rs.shelve_incompatible_results(results_path)
    completed, attempts, max_trial_id = rs.load_results(results_path)
    print(f"[STUDY-JITTER] schedule: {len(schedule)} participants, "
          f"{sum(len(v) for v in schedule.values())} non-spare runs, "
          f"jitter=+-{args.jitter} frames")

    def mk_button(id_: str, x1: int, x2: int, label: str,
                  key=None, centered: bool = False) -> rs.Button:
        yc = H // 2 if centered else int(H * 0.62)
        return rs.Button(id_, (x1, yc - 28, x2, yc + 28), label, key)

    try:
        while True:
            if rs._win.quit_requested:  # window close button
                break
            # ── home: New Trial, or resume/restart an interrupted one ─────
            stale = rs.load_session_state(state_path)
            if stale is not None:
                r_buttons = [
                    mk_button("resume", W // 2 - 270, W // 2 - 10,
                              f"[R] Resume (continue at run "
                              f"{stale.get('session_trial', '?')})",
                              key=ord("r"), centered=True),
                    mk_button("restart", W // 2 + 10, W // 2 + 270,
                              "[S] Restart from run 1",
                              key=ord("s"), centered=True),
                ]
                bid, esc = rs.wait_for_button(
                    gamepad, W, H, r_buttons,
                    "Unfinished trial detected",
                    "An earlier trial was interrupted.",
                    ["Resume continues from that run; Restart begins the",
                     "trial's sequence from the first run."])
                if esc:
                    continue
                pid = stale["participant_id"]
                trial_id = int(stale.get("trial_id") or max_trial_id)
                restart = (bid == "restart")
                if restart:
                    rs.clear_session_state(state_path)
                    rs.delete_trial_rows(results_path, trial_id)
                    completed = {k for k in completed if k[0] != pid}
                    attempts = {k: v for k, v in attempts.items()
                                if k[0] != pid}
                    intake_answers = None
                    restart = False
                else:
                    intake_answers = rs.load_intake(results_path, trial_id)
                if intake_answers is None:
                    intake = rs.intake_questionnaire(gamepad, W, H)
                    intake_answers = intake if intake else {
                        k: "" for k in rs.INTAKE_ANSWER_FIELDS}
            else:
                home_buttons = [
                    mk_button("new", W // 2 - 130, W // 2 + 130,
                              "New Trial", key=ord("n"), centered=True),
                ]
                bid, esc = rs.wait_for_button(gamepad, W, H, home_buttons,
                                              "", "", [])
                if bid is None:
                    if esc:  # ESC on the home screen quits the app
                        rs._win.quit_requested = True
                    continue
                pid = None
                for cand in schedule:
                    if any((cand, int(r["session_trial"])) not in completed
                           for r in schedule[cand]):
                        pid = cand
                        break
                if pid is None:
                    done_buttons = [mk_button("home", W // 2 - 130,
                                              W // 2 + 130,
                                              "[H] All done", key=ord("h"))]
                    rs.wait_for_button(gamepad, W, H, done_buttons,
                                       "All participants complete",
                                       "No unfinished runs remain.")
                    continue
                trial_id = max_trial_id + 1
                max_trial_id = trial_id
                restart = False
                print(f"[STUDY-JITTER] new trial {trial_id} -> {pid}")

                intake = rs.intake_questionnaire(gamepad, W, H)
                intake_answers = intake if intake else {
                    k: "" for k in rs.INTAKE_ANSWER_FIELDS}
            if pid not in schedule:
                print(f"[STUDY-JITTER] participant {pid} has no schedule rows")
                continue

            # ── run the participant's rows in order ──────────────────────────
            rows = schedule[pid]
            practice_rows = [r for r in rows if r["phase"] == "practice"]
            main_rows = [r for r in rows if r["phase"] != "practice"]

            def run_label(row: dict) -> str:
                if row["phase"] == "practice":
                    idx = practice_rows.index(row) + 1
                    return f"Practice - {idx}/{len(practice_rows)}"
                idx = main_rows.index(row) + 1
                return f"Trial - {idx}/{len(main_rows)}"

            i = 0
            last_saved = False
            first_run = True  # New Trial starts the first run immediately
            while i < len(rows):
                row = rows[i]
                key_r = (pid, int(row["session_trial"]))
                if key_r in completed and not restart:
                    i += 1
                    continue
                attempt = attempts.get(key_r, 0) + 1
                attempts[key_r] = attempt

                if last_saved:
                    choice = rs.inter_run_prompt(
                        gamepad, W, H, run_label(row),
                        run_label(rows[i + 1]) if i + 1 < len(rows) else "",
                        completed_run=True,
                        do_tlx=bool(row.get("do_tlx")))
                    if choice is None:
                        break
                    if choice == "retry":
                        i -= 1  # re-run the just-saved row
                        row = rows[i]
                        key_r = (pid, int(row["session_trial"]))
                        attempt = attempts.get(key_r, 0) + 1
                        attempts[key_r] = attempt
                    last_saved = False
                elif not first_run or row.get("do_tlx"):
                    choice = rs.inter_run_prompt(
                        gamepad, W, H, run_label(row), "",
                        completed_run=False,
                        do_tlx=bool(row.get("do_tlx")))
                    if choice is None:
                        break
                else:
                    choice = "advance"  # start the first run immediately
                first_run = False

                rs.save_session_state(state_path, pid,
                                      int(row["session_trial"]), attempt,
                                      trial_id)
                # The one behavioral delta: a seeded JitterDelay per row
                # (fresh instance each attempt → identical jitter pattern
                # for retries of the same row).  The amplitude comes from
                # the schedule's jitter_n column (the jitter-ablated design)
                # with --jitter as the fallback for schedules without it.
                jitter = int(row.get("jitter_n") or args.jitter)
                delay = (JitterDelay(row["delay_n"], jitter,
                                     int(row["seed"]))
                         if row["phase"] != "practice" else None)
                result, metrics_dict = rs.run_row(
                    env, wm, gamepad, W, H, row,
                    args.task_timeout_minutes, args.fps, run_label(row),
                    delay=delay)
                if result == "retry":  # supervisor R: restart this run
                    rs.clear_session_state(state_path)
                    last_saved = False
                    continue
                if result is None:  # aborted — no save
                    if rs._win.quit_requested:
                        break
                    rs.clear_session_state(state_path)
                    last_saved = False
                    continue

                rs.clear_session_state(state_path)
                # Record the REALIZED jitter for this run (before the
                # out_row build so the standard float formatting applies).
                if delay is not None:
                    real = np.asarray(delay.realized, dtype=np.float64)
                    if real.size:
                        metrics_dict["delay_mean_ms"] = \
                            float(real.mean()) * 1000.0 / args.fps
                        metrics_dict["delay_std_ms"] = \
                            float(real.std()) * 1000.0 / args.fps
                tlx_answers = None
                if row.get("do_tlx"):
                    tlx_answers = rs.tlx_questionnaire(gamepad, W, H)
                out_row = {
                    "participant_id": pid,
                    "schedule_row": row["schedule_row"],
                    "block_index": row["block_index"],
                    "delay_n": row["delay_n"],
                    "jitter_n": row.get("jitter_n", ""),
                    "phase": row["phase"],
                    "setting": row["setting"],
                    "trial_in_block": row["trial_in_block"],
                    "session_trial": row["session_trial"],
                    "attempt": attempt,
                    "trial_id": trial_id,
                    "completion_time_s": f"{result['completion_time_s']:.3f}",
                    "success": result["success"],
                    "timed_out": result["timed_out"],
                    "goal_tolerance": args.goal_tolerance,
                    "timestamp_iso": rs.time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                out_row.update({k: f"{v:.6f}" if isinstance(v, float)
                                else v for k, v in metrics_dict.items()})
                out_row.update(intake_answers)
                if tlx_answers:
                    out_row.update(tlx_answers)
                rs.append_result_row(results_path, out_row)
                completed.add(key_r)
                last_saved = True
                print(f"[STUDY-JITTER] trial {trial_id} saved {pid} "
                      f"run {row['session_trial']} attempt {attempt} "
                      f"t={result['completion_time_s']:.2f}s "
                      f"success={result['success']}")

            if i >= len(rows):
                d_buttons = [mk_button("home", W // 2 - 140, W // 2 + 140,
                                       "[H] Back to Home", key=ord("h"))]
                rs.wait_for_button(gamepad, W, H, d_buttons,
                                   "Trial complete",
                                   f"Trial {trial_id} finished all "
                                   f"{len(rows)} scheduled runs.",
                                   primary_id="home", gamepad_continue=True)
    finally:
        if gamepad is not None and gamepad.connected:
            gamepad.close()
        rs._win.close()


if __name__ == "__main__":
    main()
