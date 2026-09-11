"""Per-run metrics for the user-study UI.

Everything here is designed around one hard constraint: measuring must NOT
degrade simulation speed.  The per-tick work in :meth:`RunMetrics.record` is
fixed-size numpy arithmetic (~microseconds vs a 20-60 ms tick); the expensive
parts — SPARC (post-run FFT) and the per-frame SSIM/LPIPS comparison — run
off the critical path (SPARC at finalize time, SSIM/LPIPS in a background
collector thread that a bounded, drop-newest queue feeds).

State vector layout (from PushTXYWrapper._build_obs, the sibling repo):
    state[0:2] = ee xy (world metres)
    state[2:4] = ee velocity
    state[4:6] = T-block xy (world metres)
    state[6:8] = block cos/sin
    state[8:12] = goal xy + cos/sin
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np


# ── SPARC (spectral arc length; Balasubramanian et al., JNER 2015) ────────────

def sparc(v, fs: float, fc: float = 10.0, vbar: float = 0.05) -> float:
    """Spectral arc length of a movement profile ``v`` sampled at ``fs`` Hz.

    Steps per the published discrete algorithm (Balasubramanian et al.,
    JNER 2015, as documented by PyEyesWeb): centre the profile, normalise
    amplitude to unit max, take the rFFT with zero-padding, normalise the
    magnitude spectrum by its maximum, apply the adaptive cutoff (last bin
    <= ``fc`` Hz whose normalised magnitude is >= ``vbar``), then return
    the negative arc length of the spectrum with a frequency-normalised
    step.  More negative = smoother.  Returns NaN for degenerate (flat/too
    short) profiles.

    Known sensitivity (verified with synthetic profiles): light additive
    noise can INCREASE |SPARC| under this discrete form even though the
    sign convention says smoother = more negative.  The metric is still
    deterministic and identical across study conditions, so between-
    condition comparisons remain valid; interpret absolute smoothness
    rankings with care.  Canonical check: a minimum-jerk bell profile at
    20 Hz yields ≈ -4.6.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.size < 4:
        return float("nan")
    v = v - v.mean()
    amp = float(np.abs(v).max())
    if amp < 1e-9:
        return float("nan")
    v = v / amp
    nfft = 2 ** int(np.ceil(np.log2(v.size)) + 1)
    V = np.abs(np.fft.rfft(v, nfft))
    V = V / V.max()
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    above = np.flatnonzero((freqs <= fc) & (V >= vbar))
    kc = int(above[-1]) if above.size else 0
    if kc < 1:
        # Only the DC-ish bin cleared the noise floor (e.g. a step-like
        # profile whose spectrum falls off immediately).  Give the arc one
        # step instead of NaN so non-flat-but-spiky profiles still get a
        # defined (near-zero = unsmooth) value; truly flat profiles are
        # already filtered by the amplitude check above.
        kc = 1
    df = float(freqs[1] - freqs[0])
    arc = np.sqrt(np.diff(V[:kc + 1]) ** 2 + (df / fc) ** 2)
    return -float(np.sum(arc))


def _segments(v, idle_thresh: float, min_len: int = 10,
              max_gap: int = 3) -> list[tuple[int, int]]:
    """(start, end) inclusive index pairs of active segments in ``v``.

    Active = above ``idle_thresh``; gaps of <= ``max_gap`` ticks are bridged;
    segments shorter than ``min_len`` ticks are dropped.  A mostly-idle
    series must be segmented before SPARC, or zero-padding distortion from
    the idle stretches dominates the spectrum.
    """
    active = np.asarray(v) > idle_thresh
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return []
    segs = []
    start = int(idx[0])
    prev = start
    for i in idx[1:]:
        i = int(i)
        if i - prev > max_gap + 1:
            segs.append((start, prev))
            start = i
        prev = i
    segs.append((start, prev))
    return [(s, e) for s, e in segs if e - s + 1 >= min_len]


def sparc_mean(v, fs: float, idle_thresh: float, fc: float = 10.0,
               vbar: float = 0.05):
    """Mean SPARC over the active segments of ``v``.

    Returns ``(mean, n_segments)``.  Falls back to the nonzero-trimmed
    series when no segment survives; ``(nan, 0)`` for an empty/degenerate
    input.
    """
    segs = _segments(v, idle_thresh)
    vals = [sparc(v[s:e + 1], fs, fc, vbar) for s, e in segs]
    vals = [x for x in vals if not np.isnan(x)]
    if vals:
        return float(np.mean(vals)), len(vals)
    trimmed = np.asarray(v)[np.asarray(v) > idle_thresh]
    if trimmed.size >= 4:
        s = sparc(trimmed, fs, fc, vbar)
        return (float(s), 1) if not np.isnan(s) else (float("nan"), 0)
    return float("nan"), 0


# ── frame-pair collector (SSIM / LPIPS, off the critical path) ────────────────

def _to_lpips_tensor(img: np.ndarray):
    """(H, W, 3) uint8 RGB → (1, 3, H, W) float tensor in [-1, 1]."""
    import torch
    t = torch.from_numpy(img.astype(np.float32))
    t = t.permute(2, 0, 1).unsqueeze(0)
    return t / 255.0 * 2.0 - 1.0


class _FramePairCollector:
    """Background consumer for (wm_frame, gt_frame) pairs.

    Main thread submits downscaled uint8 pairs with non-blocking,
    drop-newest semantics; this thread computes grayscale SSIM (+ LPIPS when
    the package imports cleanly) entirely on CPU, keeping the GPU free for
    the sim and world model.
    """

    def __init__(self, queue_size: int = 16) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._pairs: list[tuple[int, float, float]] = []  # (ts, ssim, lpips)
        self._n_dropped = 0
        self._thread: Optional[threading.Thread] = None

    @property
    def n_dropped(self) -> int:
        return self._n_dropped

    def submit(self, ts: int, wm_small: np.ndarray,
               gt_small: np.ndarray) -> None:
        try:
            self._q.put_nowait((ts, wm_small, gt_small))
        except queue.Full:
            self._n_dropped += 1  # drop newest — unbiased subsample

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="metric-collector")
        self._thread.start()

    def stop(self, discard: bool = False) -> None:
        """Drain/join the collector.  ``discard`` skips pending results."""
        if self._thread is None:
            return
        if discard:
            while True:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
        self._q.put(None)  # blocking is fine: the consumer is draining
        self._thread.join()
        self._thread = None

    def results(self):
        """(ssim_mean, ssim_std, lpips_mean, n_pairs) — NaN where absent."""
        if not self._pairs:
            return float("nan"), float("nan"), float("nan"), 0
        ssims = np.array([p[1] for p in self._pairs], dtype=np.float64)
        lp = np.array([p[2] for p in self._pairs], dtype=np.float64)
        lpips_ok = np.isfinite(lp)
        lpips_mean = float(lp[lpips_ok].mean()) if lpips_ok.any() \
            else float("nan")
        return (float(ssims.mean()), float(ssims.std()),
                lpips_mean, len(self._pairs))

    def _loop(self) -> None:
        from skimage.metrics import structural_similarity
        lpips_fn = None
        try:
            import lpips  # noqa: F401
            lpips_fn = lpips.LPIPS(net="alex")
            print("[METRICS] LPIPS ready (net=alex, CPU)")
        except Exception as exc:  # noqa: BLE001 — degrade to SSIM-only
            print(f"[METRICS] LPIPS unavailable ({exc}); SSIM only.")
        while True:
            item = self._q.get()
            if item is None:
                return
            ts, a, b = item
            ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            s = float(structural_similarity(ga, gb, data_range=255))
            l = float("nan")
            if lpips_fn is not None:
                try:
                    l = float(lpips_fn(_to_lpips_tensor(a),
                                       _to_lpips_tensor(b)).item())
                except Exception:
                    l = float("nan")
            self._pairs.append((ts, s, l))


# ── per-run metrics ───────────────────────────────────────────────────────────

class RunMetrics:
    """Online accumulators + offline post-processing for one run attempt.

    Parameters
    ----------
    fps:
        Target tick rate — used for SPARC's sampling frequency and reported
        as the reference for tick-rate comparability.
    collect_pairs:
        Enable the background SSIM/LPIPS frame-pair collector (the 'wm'
        condition only).
    pair_every:
        Submit a frame pair every K-th tick (default 5 → 4 Hz at 20 fps).
    pair_resize:
        Downscale side length for submitted frames (default 128).
    contact_hi / contact_lo:
        Block-speed hysteresis thresholds for the contact-onset proxy, in
        metres per tick (defaults ≈ 0.02 / 0.005 m/s at 20 Hz, ~100x above
        physx rest noise).  Calibrate on a pilot run if needed.
    idle_thresh:
        Input magnitude below which a tick counts as idle, pixel space
        (default 9 ≈ the gamepad deadzone 0.15 x speed 60).
    rev_min_amp:
        Minimum per-axis input amplitude for a sign change to count as a
        direction reversal (default 12 px/tick — inert for the
        fixed-magnitude gamepad, which is always ±60 when nonzero).
    """

    def __init__(self, fps: int = 20, *, collect_pairs: bool = False,
                 pair_every: int = 5, queue_size: int = 16,
                 pair_resize: int = 128,
                 contact_hi: float = 1e-3, contact_lo: float = 2.5e-4,
                 idle_thresh: float = 9.0, rev_min_amp: float = 12.0) -> None:
        self.fps = fps
        self._collect_pairs = collect_pairs
        self._pair_every = pair_every
        self._pair_resize = pair_resize
        self._contact_hi = contact_hi
        self._contact_lo = contact_lo
        self._idle_thresh = idle_thresh
        self._rev_min_amp = rev_min_amp

        self.ee_path = 0.0
        self.block_path = 0.0
        self.n_contacts = 0
        self._moving = False
        self._was_moving = False
        self.n_reversals = 0
        self._last_sign: list[Optional[int]] = [None, None]
        self.n_idle = 0
        self.n_ticks = 0
        self._prev_state: Optional[np.ndarray] = None
        self.input_mag: list[float] = []
        self._dt_sum = 0.0
        self._dt_max = 0.0
        self._tick = 0
        self._t_start = None
        self._collector = _FramePairCollector(queue_size) \
            if collect_pairs else None

    # -- online ---------------------------------------------------------------

    def start(self) -> None:
        self._t_start = time.perf_counter()
        if self._collector is not None:
            self._collector.start()

    def record(self, action, state, dt: float,
               frame_pair: Optional[tuple[np.ndarray, np.ndarray]] = None
               ) -> None:
        a = np.asarray(action, dtype=np.float32)
        # Copy the state: callers may reuse/mutate their buffer in place,
        # which would alias _prev_state and zero out every delta.
        s = np.array(state, dtype=np.float32, copy=True)
        if self._prev_state is not None:
            self.ee_path += float(np.abs(s[0:2] - self._prev_state[0:2]).sum())
            d = s[4:6] - self._prev_state[4:6]
            self.block_path += float(np.abs(d).sum())
            spd = float(np.hypot(*d))
            self._moving = spd >= (self._contact_hi if not self._moving
                                   else self._contact_lo)
            if self._moving and not self._was_moving:
                self.n_contacts += 1
            self._was_moving = self._moving
        self._prev_state = s

        for i in (0, 1):
            if abs(a[i]) >= self._rev_min_amp:
                sg = 1 if a[i] > 0 else -1
                if self._last_sign[i] is not None and sg != self._last_sign[i]:
                    self.n_reversals += 1
                self._last_sign[i] = sg

        mag = float(np.hypot(*a))
        self.input_mag.append(mag)
        self.n_idle += int(mag <= self._idle_thresh)
        self.n_ticks += 1
        self._dt_sum += dt
        self._dt_max = max(self._dt_max, dt)
        self._tick += 1

        if frame_pair is not None and self._collector is not None \
                and self._tick % self._pair_every == 0:
            wm, gt = frame_pair
            small_wm = cv2.resize(wm, (self._pair_resize, self._pair_resize),
                                  interpolation=cv2.INTER_AREA)
            small_gt = cv2.resize(gt, (self._pair_resize, self._pair_resize),
                                  interpolation=cv2.INTER_AREA)
            self._collector.submit(self._tick, small_wm, small_gt)

    # -- offline --------------------------------------------------------------

    def finalize(self, discard: bool = False) -> dict:
        """Stop the collector and return the flat metrics dict."""
        if self._collector is not None:
            self._collector.stop(discard=discard)
            if discard:
                ssim_mean = ssim_std = lpips_mean = float("nan")
                n_pairs = 0
                n_pairs_dropped = self._collector.n_dropped
            else:
                ssim_mean, ssim_std, lpips_mean, n_pairs = \
                    self._collector.results()
                n_pairs_dropped = self._collector.n_dropped
        else:
            ssim_mean = ssim_std = lpips_mean = float("nan")
            n_pairs = n_pairs_dropped = 0

        sparc_mean_v, sparc_n = sparc_mean(self.input_mag, self.fps,
                                           self._idle_thresh)
        idle_frac = (self.n_idle / self.n_ticks) if self.n_ticks else float("nan")
        rate_mean = (self.n_ticks / self._dt_sum) if self._dt_sum > 0 \
            else float("nan")
        rate_min = (1.0 / self._dt_max) if self._dt_max > 0 else float("nan")

        return {
            "ee_path_len_m": float(self.ee_path),
            "block_path_len_m": float(self.block_path),
            "pushes": int(self.n_contacts),
            "reversals": int(self.n_reversals),
            "idle_fraction": float(idle_frac),
            "duty_cycle": float(1.0 - idle_frac) if np.isfinite(idle_frac)
            else float("nan"),
            "sparc": float(sparc_mean_v),
            "sparc_n_segments": int(sparc_n),
            "ssim_mean": float(ssim_mean),
            "ssim_std": float(ssim_std),
            "lpips_mean": float(lpips_mean),
            "n_pairs": int(n_pairs),
            "n_pairs_dropped": int(n_pairs_dropped),
            "tick_rate_mean_hz": float(rate_mean),
            "tick_rate_min_hz": float(rate_min),
            # Completion time in env ticks (== number of recorded steps) —
            # the pacing-independent complement of the wall-clock time.
            "completion_ticks": int(self.n_ticks),
        }
