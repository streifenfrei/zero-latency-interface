"""Headless check: does zli ground-and-predict show the present or the past?

Runs the real ManiSkill env + DINO-WM adapter without any GUI and compares,
at every tick, the WM prediction against (a) the live GT frame and (b) the
delayed GT frame.  If the WM works, MSE(pred, live) << MSE(pred, delayed).

Two drivers:
  main()        — synchronous ground_and_predict every tick (regression
                  baseline for the adapter's grounding math).
  async_main()  — the interface's async scheme: predict() every tick, launch
                  a background ground_and_predict_async worker per the launch
                  rules (idle + newer frame, no cancellation), chase the
                  present, publish.  Thread-safety smoke test included.

Usage (from repo root, with the X11/xcb preload):
    LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libX11.so.6:/usr/lib/x86_64-linux-gnu/libxcb.so.1" \
        .venv/bin/python deployment/test_zli_headless.py [--sync | --async | --all]
"""

import argparse
import os
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np

from deployment.interface import zli_tick
from deployment.mani_skill_env import ManiSkillPushTEnv
from deployment.wm_adapter import DinoWMPushtAdapter

CKPT = os.path.join(REPO, "outputs/2026-07-06/18-46-23/checkpoints")
POLICY = "/mnt/cluster/workspaces/lelismart/negative_latency/" \
         "zero_latency_interface/checkpoints/policy.pt"
AX = 10.0  # pixel-space constant action (x only), 1/6 deflection
def mse(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(((a - b) ** 2).mean())


def make_env_wm(ckpt=CKPT, render_size=None):
    """Mirror PushTInterface.run()/_collect_context() exactly.

    In particular, _action_history must be indexed by ABSOLUTE raw timestep
    (matching self._timestep) from t=0 — the same invariant _collect_context
    now maintains by recording an action for every context-collection step,
    not just the post-grounding ticks.  Returns
    ``(env, wm, hist_vis, hist_prop, actions, offset)`` where ``offset`` is
    the absolute timestep of the first post-grounding tick — ``gt_ts``/
    ``get_timestep()`` in the caller must use ``offset + t``, never a
    from-scratch 0-based ``t``, or this test stops exercising the real
    indexing path.
    """
    # WM first so the env renders at the WM's training resolution (mirrors
    # the interface's auto-derivation), e.g. 448 for the high-res checkpoint.
    wm = DinoWMPushtAdapter(ckpt_dir=ckpt)
    env_kw = dict(policy_checkpoint=POLICY, fixed_magnitude=False)
    if render_size is None:
        render_size = wm.img_size
    env_kw["render_size"] = render_size
    env = ManiSkillPushTEnv(**env_kw)
    env.seed(0)
    fs = wm.frameskip
    obs, _ = env.reset()
    frames = [obs["visual"]]
    props = [obs["proprio"]]
    actions: list = []
    for _ in range((wm.num_hist - 1) * fs):
        obs, _, _, info = env.step(np.array([0, 0], dtype=np.float32))
        actions.append(np.asarray(info["effective_action"], dtype=np.float32))
        frames.append(obs["visual"])
        props.append(obs["proprio"])
    wm.reset(frames[::fs][-wm.num_hist:], props[::fs][-wm.num_hist:])
    offset = len(actions)
    return env, wm, list(frames), list(props), actions, offset


def main(delay=10, ticks=150, ckpt=CKPT):
    """Sync baseline: ground_and_predict every tick (the pre-async scheme)."""
    env, wm, hist_vis, hist_prop, actions, offset = make_env_wm(ckpt)
    fs = wm.frameskip
    mse_pred_live, mse_pred_del, mse_del_live = [], [], []

    for t in range(ticks):
        t_abs = offset + t
        obs, _, _, info = env.step(np.array([AX, 0], dtype=np.float32))
        live = obs["visual"]
        actions.append(np.asarray(info["effective_action"], dtype=np.float32))
        hist_vis.append(live)
        hist_prop.append(obs["proprio"])

        if t < delay:
            continue
        d_vis = hist_vis[-(delay + 1)]
        d_prop = hist_prop[-(delay + 1)]
        gt_ts = t_abs - delay
        acts_full = actions[-((wm.num_hist - 1) * fs + delay):]

        wm.update_ground_truth(d_vis, d_prop, gt_ts)
        pred = wm.ground_and_predict(d_vis, d_prop, gt_ts, acts_full)

        mse_pred_live.append(mse(pred, live))
        mse_pred_del.append(mse(pred, d_vis))
        mse_del_live.append(mse(d_vis, live))

    print(f"[SYNC] delay={delay} ticks={len(mse_pred_live)}")
    print(f"  MSE(pred, live)    = {np.mean(mse_pred_live):8.2f}")
    print(f"  MSE(pred, delayed) = {np.mean(mse_pred_del):8.2f}")
    print(f"  MSE(delayed, live) = {np.mean(mse_del_live):8.2f}")
    env.close()
    return np.mean(mse_pred_live), np.mean(mse_del_live)


def run_async(delay, ticks, seed_action=AX, random_actions=False,
              pace=0.03, seed=0, ckpt=CKPT):
    """Mirror the interface's async zli rules exactly.

    Calls the SAME ``deployment.interface.zli_tick`` helper the real
    interface (and run_study.py) use — this is a regression test for that
    shared code path, not just the adapter's own methods.

    ``actions`` and ``gt_ts``/``get_timestep()`` all use the SAME absolute
    raw-timestep numbering _collect_context/PushTInterface use (offset by
    the grounding-phase step count) — this is what makes the test able to
    catch an offset/indexing bug between the two, not just a from-scratch
    0-based scheme that trivially stays consistent with itself.
    """
    env, wm, hist_vis, hist_prop, actions, offset = make_env_wm(ckpt)
    rng = np.random.default_rng(seed)
    ground_thread = None
    last_grounded = -1
    worker_errors: list = []
    publishes = 0
    last_pub_ts = -1
    mse_pred_live, mse_pred_del, mse_del_live = [], [], []
    mse_pub_live, mse_launch_live = [], []
    pred_at_launch = None

    # zli_tick launches its worker thread directly (no wrapper) — capture
    # any exception via the threading excepthook instead, scoped to this
    # call so it doesn't leak into unrelated code.
    prev_hook = threading.excepthook
    threading.excepthook = lambda args: worker_errors.append(args.exc_value)

    for t in range(ticks):
        t_abs = offset + t
        if random_actions:
            a = rng.uniform(-60, 60, size=2).astype(np.float32)
        else:
            a = np.array([seed_action, 0], dtype=np.float32)
        obs, _, _, info = env.step(a)
        live = obs["visual"]
        eff_action = np.asarray(info["effective_action"], dtype=np.float32)
        actions.append(eff_action)
        hist_vis.append(live)
        hist_prop.append(obs["proprio"])

        received = None
        if t >= delay:
            d_vis = hist_vis[-(delay + 1)]
            d_prop = hist_prop[-(delay + 1)]
            gt_ts = t_abs - delay
            received = (d_vis, d_prop, gt_ts)

        prev_last_grounded = last_grounded
        pred, _, ground_thread, last_grounded = zli_tick(
            wm, received, actions, lambda: t_abs, eff_action, live,
            ground_thread, last_grounded,
        )
        if last_grounded != prev_last_grounded:
            # zli_tick just launched a new grounding this tick — pred is
            # the chain's prediction at launch time, before that worker
            # could possibly have published anything yet.
            pred_at_launch = pred

        if t >= delay:
            # Detect publishes (worker sets wm.last_grounding_ts).
            pub_ts = wm.last_grounding_ts
            if pub_ts != last_pub_ts:
                last_pub_ts = pub_ts
                publishes += 1
                if pred_at_launch is not None:
                    mse_pub_live.append(mse(pred, live))
                    mse_launch_live.append(mse(pred_at_launch, live))

            mse_pred_live.append(mse(pred, live))
            mse_pred_del.append(mse(pred, hist_vis[-(delay + 1)]))
            mse_del_live.append(mse(hist_vis[-(delay + 1)], live))

            if t % 20 == 19:
                # phase diagnostics: is the prediction still tracking?
                print(f"  t={t}: ee_x={obs['proprio'][0]:+.3f}  "
                      f"MSE(pred,live)={mse(pred, live):7.1f}  "
                      f"MSE(pred,del)={mse(pred, hist_vis[-(delay + 1)]):7.1f}  "
                      f"lastpub={wm.last_grounding_ts}", flush=True)

        time.sleep(pace)  # mimic the 20 Hz loop so the worker overlaps ticks

    threading.excepthook = prev_hook
    if ground_thread is not None:
        ground_thread.join()
    env.close()

    pl, dl = np.mean(mse_pred_live), np.mean(mse_del_live)
    print(f"[ASYNC] delay={delay} ticks={ticks} random={random_actions} "
          f"publishes={publishes} worker_errors={len(worker_errors)}")
    print(f"  MSE(pred, live)    = {pl:8.2f}")
    print(f"  MSE(pred, delayed) = {np.mean(mse_pred_del):8.2f}")
    print(f"  MSE(delayed, live) = {dl:8.2f}")
    print(f"  last_grounding_ts  = {wm.last_grounding_ts}")
    if mse_pub_live:
        print(f"  publish: MSE(pub, live)={np.mean(mse_pub_live):.2f}  "
              f"MSE(chain@launch, live)={np.mean(mse_launch_live):.2f}")
    for exc in worker_errors:
        print(f"  WORKER ERROR: {exc!r}")
    return pl, dl, publishes, worker_errors, wm.last_grounding_ts, \
        np.mean(mse_pub_live), np.mean(mse_launch_live)


def async_main(ckpt=CKPT, delays=(10, 20, 50), ticks=150, seed_action=AX):
    """The async scheme at several delays, plus a random-action stress pass."""
    results = {}
    for delay in delays:
        pl, dl, pubs, errs, last_ts, mpl, mll = \
            run_async(delay, ticks, seed_action=seed_action, ckpt=ckpt)
        results[delay] = (pl, dl, pubs, errs, last_ts, mpl, mll)
    # Thread-safety smoke test: random actions, shorter pass.
    pl, dl, pubs, errs, last_ts, mpl, mll = \
        run_async(10, 300, random_actions=True, pace=0.02, ckpt=ckpt)
    results["random"] = (pl, dl, pubs, errs, last_ts, mpl, mll)

    ok = True
    for key, (pl, dl, pubs, errs, last_ts, mpl, mll) in results.items():
        if errs:
            print(f"[FAIL] {key}: worker errors: {errs}")
            ok = False
        if pl >= dl:
            print(f"[FAIL] {key}: pred not closer to live than delayed "
                  f"({pl:.2f} >= {dl:.2f})")
            ok = False
        if pubs == 0:
            print(f"[FAIL] {key}: no grounding published")
            ok = False
        # predict() holds the same frame between 5-tick model updates, so the
        # published vs launch-chain comparison needs tolerance (equality is
        # fine; published must not be CLEARLY worse).
        if mpl is not None and mll is not None and mpl > mll * 1.5 + 20:
            print(f"[FAIL] {key}: published chain clearly worse than launch "
                  f"chain ({mpl:.2f} > {mll:.2f})")
            ok = False
    print("[ASYNC] ALL PASS" if ok else "[ASYNC] FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sync", "async", "all"],
                        default="async")
    parser.add_argument("--ckpt", default=CKPT,
                        help="checkpoints/ directory of the WM to test")
    parser.add_argument("--delays", default="10,20,50",
                        help="comma-separated delays for the async suite")
    parser.add_argument("--ticks", type=int, default=150)
    parser.add_argument("--action", type=float, default=AX,
                        help="constant x action in pixel space (default 10)")
    args = parser.parse_args()
    delays = tuple(int(d) for d in args.delays.split(","))
    if args.mode in ("sync", "all"):
        main(delay=10, ckpt=args.ckpt)
    if args.mode in ("async", "all"):
        sys.exit(async_main(ckpt=args.ckpt, delays=delays, ticks=args.ticks,
                            seed_action=args.action))
