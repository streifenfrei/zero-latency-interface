"""Column schemas shared by the study UI and its analysis tools.

Kept import-light on purpose: ``plot_study_results.py`` (and any future
analysis script) must NOT have to import the WM/torch stack just to know
the CSV column names.
"""

import math

METRIC_FIELDS = [
    "ee_path_len_m", "block_path_len_m", "pushes", "reversals",
    "idle_fraction", "duty_cycle", "sparc", "sparc_n_segments",
    "ssim_mean", "ssim_std", "lpips_mean", "n_pairs", "n_pairs_dropped",
    "tick_rate_mean_hz", "tick_rate_min_hz",
    "completion_ticks",
    "recompiles_during_run",
    "delay_mean_ms", "delay_std_ms",
]

FIELDNAMES = [
    "participant_id", "schedule_row", "block_index", "delay_n", "jitter_n",
    "phase", "setting", "trial_in_block", "session_trial", "attempt",
    "trial_id",
    "age_band", "gender", "handedness", "gaming_hours_week",
    "gamepad_hours_week",
    "completion_time_s", "success", "timed_out", "goal_tolerance",
    "timestamp_iso",
] + METRIC_FIELDS + [
    "tlx_mental", "tlx_performance", "tlx_frustration",
    "tlx_responsiveness", "tlx_authenticity",
]


def jitter_level(row: dict):
    """Nominal jitter amplitude (frames) for a results row, or None.

    Reads the ``jitter_n`` column when present (jitter-ablated study);
    otherwise derives it from the realized ``delay_std_ms`` — ±2 frames
    ≈ 71 ms, ±10 frames ≈ 316 ms at 20 Hz — so pilot data recorded before
    the column existed can still be grouped.
    """
    j = str(row.get("jitter_n", "") or "").strip()
    if j:
        return int(j)
    try:
        s = float(row["delay_std_ms"])
    except (TypeError, ValueError):
        return None
    if math.isnan(s):
        return None
    return 2 if s < 150.0 else 10

NAME_FIELDNAMES = ["participant_id", "name", "timestamp_iso"]
