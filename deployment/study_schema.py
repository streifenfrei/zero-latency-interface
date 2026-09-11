"""Column schemas shared by the study UI and its analysis tools.

Kept import-light on purpose: ``plot_study_results.py`` (and any future
analysis script) must NOT have to import the WM/torch stack just to know
the CSV column names.
"""

METRIC_FIELDS = [
    "ee_path_len_m", "block_path_len_m", "pushes", "reversals",
    "idle_fraction", "duty_cycle", "sparc", "sparc_n_segments",
    "ssim_mean", "ssim_std", "lpips_mean", "n_pairs", "n_pairs_dropped",
    "tick_rate_mean_hz", "tick_rate_min_hz",
    "completion_ticks",
]

FIELDNAMES = [
    "participant_id", "schedule_row", "block_index", "delay_n", "phase",
    "setting", "trial_in_block", "session_trial", "attempt", "trial_id",
    "age_band", "gender", "handedness", "gaming_hours_week",
    "gamepad_hours_week",
    "completion_time_s", "success", "timed_out", "goal_tolerance",
    "timestamp_iso",
] + METRIC_FIELDS + [
    "tlx_mental", "tlx_performance", "tlx_frustration",
    "tlx_responsiveness", "tlx_authenticity",
]

NAME_FIELDNAMES = ["participant_id", "name", "timestamp_iso"]
