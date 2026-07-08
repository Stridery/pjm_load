"""Macro (multi-week) context features.

Summarize the THREE weeks before the forecast cutoff into a handful of scalars,
giving the model long-horizon context beyond the 168h lookback window without
lengthening the sequence. These are added alongside the forecast-day calendar
features (as broadcast constants in the 3D matrix / appended columns in the 2D
matrix), NOT mixed into the per-timestep lookback window.

Windows (position-based, each 7 days = 168h), most recent first:
    W1 = [cutoff-168, cutoff)      last week (= the lookback period)
    W2 = [cutoff-336, cutoff-168)
    W3 = [cutoff-504, cutoff-336)

Features (7), in MACRO_FEATURE_NAMES order:
  1. macro_q05_slope_d1 : q05(W1 night) - q05(W2 night)   week-over-week slope of the
     macro_q05_slope_d2 : q05(W2 night) - q05(W3 night)   "pure trough" base load
       night = hours 01:00-06:00; q05 = 5th percentile (harder base than the mean)
  2. macro_energy_ratio : sum(load W1) / sum(load W2)      total-energy (AUC) ratio
  3. macro_err_bias_w1  : mean(actual - estimated) over W1   historical estimate-error
     macro_err_bias_w2  :   ... over W2                       bias momentum, per week
     macro_err_bias_w3  :   ... over W3
  4. macro_lag168_err   : mean |load(t-k) - load(t-168-k)|, k=1..LAST_HOURS
       recent divergence from the same clock-time one week ago

All computed from raw MW load. In the 3D (neural) matrix these are standardized
by the caller; in the 2D (tree) matrix they are kept raw like every other column.
"""

import numpy as np

MACRO_WINDOW_HOURS = 504                 # 3 weeks of history required before cutoff
_WEEK = 168
_LAG = 168
_LAST_HOURS = 6                          # "the last few hours" for the lag-168 error
_NIGHT_HOURS = np.array([1, 2, 3, 4, 5, 6])   # 01:00-06:00 trough window

MACRO_FEATURE_NAMES = [
    'macro_q05_slope_d1',
    'macro_q05_slope_d2',
    'macro_energy_ratio',
    'macro_err_bias_w1',
    'macro_err_bias_w2',
    'macro_err_bias_w3',
    'macro_lag168_err',
]


def compute_macro_features(actual_raw, est_raw, ept_hours, cutoff_pos):
    """Return a length-7 float32 vector of macro features for one sample.

    actual_raw : 1D array of metered load (MW), full contiguous hourly series.
    est_raw    : 1D array of estimated load (MW), aligned with actual_raw.
    ept_hours  : 1D array of hour-of-day (0-23), aligned with the series.
    cutoff_pos : row index of the forecast cutoff; only data strictly before it
                 is used. Caller must guarantee cutoff_pos >= MACRO_WINDOW_HOURS.
    """
    c = cutoff_pos
    w1 = slice(c - _WEEK, c)
    w2 = slice(c - 2 * _WEEK, c - _WEEK)
    w3 = slice(c - 3 * _WEEK, c - 2 * _WEEK)

    # (1) pure-trough 5th-percentile week-over-week slope
    def night_q05(sl):
        h = ept_hours[sl]
        vals = actual_raw[sl][np.isin(h, _NIGHT_HOURS)]
        return float(np.quantile(vals, 0.05)) if len(vals) else 0.0

    q1, q2, q3 = night_q05(w1), night_q05(w2), night_q05(w3)
    slope_d1 = q1 - q2
    slope_d2 = q2 - q3

    # (2) total-energy (AUC) ratio W1 / W2
    e1 = float(actual_raw[w1].sum())
    e2 = float(actual_raw[w2].sum())
    energy_ratio = e1 / e2 if e2 != 0 else 1.0

    # (3) per-week mean estimate-error bias (actual - estimated)
    bias_w1 = float(np.mean(actual_raw[w1] - est_raw[w1]))
    bias_w2 = float(np.mean(actual_raw[w2] - est_raw[w2]))
    bias_w3 = float(np.mean(actual_raw[w3] - est_raw[w3]))

    # (4) recent vs same-time-last-week absolute error over the last LAST_HOURS
    recent = actual_raw[c - _LAST_HOURS:c]
    lagged = actual_raw[c - _LAST_HOURS - _LAG:c - _LAG]
    lag168_err = float(np.mean(np.abs(recent - lagged))) if len(recent) == len(lagged) and len(recent) else 0.0

    return np.array([slope_d1, slope_d2, energy_ratio,
                     bias_w1, bias_w2, bias_w3, lag168_err], dtype='float32')
