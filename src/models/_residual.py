# src/models/_residual.py
"""
Shared machinery for the residual-learning models.

Instead of predicting metered load directly, predict its DEVIATION from a naive
same-hour-last-week baseline, and add the baseline back at inference:

    baseline[h] = mean of the past 7 days' PRELIMINARY load at clock-hour h   -> (24,)
    train target : y_residual = y_metered - baseline
    inference    : pred_metered = model.predict(X) + baseline

Features are unchanged and the estimator is unchanged — only the target moves. The
baseline is derived per-sample from that sample's OWN lookback window, so nothing is fit
on the training set and it cannot leak; and it reads only the preliminary load, which is
published in real time. That makes the residual models deployable on exactly the same
forecast path as everything else.

The baseline is recovered from the feature matrix itself rather than re-read from the
cleaned CSV. That is what lets predict() keep the same signature as the plain models — and
it means the forecast path (which runs off predict.csv, not the cleaned CSV) needs no
special plumbing at all.
"""

import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error)

BASELINE_MODES = ('hourly', 'scalar')


def print_metrics(tag, y_true, y_pred):
    """One-line score. Used to report the naive baseline ALONE next to the model's — the
    residual models are only worth their complexity if they beat the thing they lean on."""
    t, p = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    print(f"  {tag:28s} MAPE {mean_absolute_percentage_error(t, p) * 100:6.2f}%  "
          f"MAE {mean_absolute_error(t, p):8.1f}  "
          f"RMSE {np.sqrt(mean_squared_error(t, p)):8.1f}  "
          f"ME {np.mean(p - t):+8.1f}")


def baseline_from_windows(win, mode='hourly'):
    """(N, lookback) of raw-MW preliminary load -> (N, 24) baseline.

    'hourly' : baseline[h] = mean of the past 7 days at clock-hour h. The window ends at
               the midnight cutoff, so window position k sits at clock-hour k % 24 and
               win[:, h::24] is exactly the 7 values for hour h. (DST shifts this by an
               hour a couple of times a year.)
    'scalar' : one number per day (mean of the whole window), broadcast across all 24 h.
    """
    if mode not in BASELINE_MODES:
        raise ValueError(f"baseline mode must be one of {BASELINE_MODES}, got {mode!r}")
    if mode == 'scalar':
        return np.repeat(win.mean(axis=1, keepdims=True), 24, axis=1)
    return np.stack([win[:, h::24].mean(axis=1) for h in range(24)], axis=1)


def tree_baseline(X_opt, mode='hourly', lookback=168):
    """Baseline for the 2D matrix — the lookback window is sitting right there in columns."""
    cols = [f'load_estimated_h{k}' for k in range(lookback)]
    missing = [c for c in cols if c not in X_opt.columns]
    if missing:
        raise KeyError(
            f"2D matrix is missing {len(missing)} load_estimated_h* column(s) "
            f"(first: {missing[0]}). The residual model reads the preliminary-load window "
            f"straight out of the features; without it there is no baseline."
        )
    return baseline_from_windows(X_opt[cols].values, mode)


def sequence_baseline(X_3d, seq_scaler, mode='hourly'):
    """Baseline for the 3D matrix.

    Channel 0 of the window is Load_Estimated, standardized by the sequence scaler that
    build_timeseries_matrix fitted. Undo that one channel to get raw MW — the same 168 h
    window the 2D path reads, so both models see an identical baseline.
    """
    win = X_3d[:, :, 0] * seq_scaler.scale_[0] + seq_scaler.mean_[0]
    return baseline_from_windows(win, mode)
