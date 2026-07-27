"""Forecast-weather features: how far the forecast departs from the lookback window.

Every model here has so far seen the forecast day's CALENDAR and nothing else about it —
no weather at all. This module supplies the rest, from the archived forecasts in
cleaned/forecast.csv (see src/data_crawler/open_meteo_forecast.py for why those are real
forecasts and not observations relabelled).

Which 48 hours, and why 48
--------------------------
A sample cuts off on day d, reads the 168 h BEFORE the cutoff, and predicts day d+1. So
everything from the cutoff to the end of the target day is future: all of day d AND all of
day d+1. Day d is a blind spot in the current design — the lookback cannot reach it and the
calendar does not describe it — yet it is where the heat builds that day d+1's load pays
out. Hence both days.

Those 48 hours are exactly one row-block of the forecast table:

    issue_date = T - 2   (= d - 1)      lead_day 1 -> day d      lead_day 2 -> day T

and that also settles the lead-alignment question outright. The forecast is issued the day
BEFORE the cutoff day, so even its latest model run (23Z = ~18:00 EPT on d-1) predates a
midnight cutoff on d by hours. There is no run that we use but would not have had.

Levels vs departures
--------------------
The headline features are DEPARTURES, not levels. The model already sees 168 h of observed
temperature; `fc_temp = 90F` is largely redundant with that, whereas "8F hotter than the
recent norm for this hour" is not. Taking the difference also cancels the season, the site,
and the constant bias between the forecast model and the ERA5 archive the window is built
from (measured at +0.25F for lead 1, +0.87F for lead 2 — a constant, since the lead is fixed).

The reference is the window's own per-clock-hour mean, the same construction the residual
models use for load (src/models/_residual.baseline_from_windows). One deliberate difference:
that one slices by POSITION (`win[:, h::24]`) and its docstring concedes DST shifts it by an
hour twice a year. Here the window rows are grouped by their TRUE EPT hour, which is exact,
costs nothing (the builders already hold `ept_hours`), and stays correct when
`latest_info_hour` is not 0 and the window is no longer seven whole days.

Three ABSOLUTE anchors survive the delta treatment on purpose. Load responds to temperature
non-linearly: +8F on a 92F base is worth far more MW than +8F on a 62F base, and pure
departures cannot express which one it is. The neural head could perhaps recover the level
by combining features; a depth-3 tree picking columns out of ~3700 will not.

Timezone
--------
The forecast table is indexed by naive LOCAL time; the cleaned load frame is indexed by UTC
with Datetime_EPT as a column. Everything here joins on EPT date and clock hour — never on
the UTC index, which is 4-5 h off and would silently line up against the wrong day.
"""

import numpy as np
import pandas as pd

BASE_TEMP_F = 65.0                      # same degree-day base as thermal_features
PEAK_HOURS = np.arange(14, 20)          # 14:00-19:00, the late-afternoon load peak
FC_ISSUE_OFFSET_DAYS = 2                # target day T -> forecast issued on T - 2

FC_FEATURE_NAMES = [f'fc_dev_h{h}' for h in range(24)] + [
    # --- target day vs the window ---
    'fc_d1_cdd_dev',        # day-total CDD minus the window's daily mean CDD
    'fc_d1_hdd_dev',
    # --- the ramp d-1 (observed) -> d (forecast) -> d+1 (forecast) ---
    'fc_d0_max_dev',        # day d max minus the last observed day's max: the first step
    'fc_d1_max_dev',        # target day max minus day d max: still climbing, or turning over
    'fc_2d_cdd_buildup',    # both forecast days' CDD against two window-average days
    # --- absolute anchors (see module docstring) ---
    'fc_d1_temp_max',
    'fc_d1_temp_peak',      # mean over PEAK_HOURS
    'fc_heatwave_days',     # 0/1/2: forecast days at or above the train-fitted heat threshold
]

N_FC_FEATURES = len(FC_FEATURE_NAMES)
assert N_FC_FEATURES == 32, N_FC_FEATURES


def load_forecast_table(path):
    """cleaned/forecast.csv -> {issue_date: (day_d_24h, day_d1_24h)} of temperatures.

    Only issue dates flagged `is_complete` are kept — a day forecast for part of its 48 h
    cannot be turned into features, and a silently short array would poison the departures
    rather than fail. Callers treat a missing key as "this sample is not forecastable" and
    skip the sample, exactly as they already do for a day whose target cannot be squashed
    to 24 h.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[df['is_complete'] == 1]
    if df.empty:
        return {}

    # Pivot to one row per issue date, columns (lead, clock hour). The forecast table is a
    # flat 24-rows-per-day local grid, so no (issue, lead, hour) cell is ever written twice
    # and the aggregation below never actually aggregates.
    wide = (df.assign(_hour=df.index.hour,
                      _issue=pd.to_datetime(df['issue_date']).dt.date)
              .pivot_table(index='_issue', columns=['lead_day', '_hour'], values='Temp_F'))

    cols_d0 = [(1, h) for h in range(24)]      # lead 1 = the day after issue   = day d
    cols_d1 = [(2, h) for h in range(24)]      # lead 2 = two days after issue  = day d+1
    missing = [c for c in cols_d0 + cols_d1 if c not in wide.columns]
    if missing:
        raise ValueError(
            f"Forecast table is missing {len(missing)} (lead, hour) column(s), first "
            f"{missing[0]}. Every complete issue date must carry both leads over all 24 "
            f"clock hours; re-run the crawler rather than filling the gaps here."
        )

    wide = wide[wide[cols_d0 + cols_d1].notna().all(axis=1)]
    d0 = wide[cols_d0].to_numpy('float32')
    d1 = wide[cols_d1].to_numpy('float32')
    return {issue: (d0[i], d1[i]) for i, issue in enumerate(wide.index)}


def forecast_issue_date(target_day):
    """The issue date whose 48 h covers day `target_day` and the day before it."""
    return pd.Timestamp(target_day).date() - pd.Timedelta(days=FC_ISSUE_OFFSET_DAYS).to_pytimedelta()


def _hourly_reference(win_temp, win_hours):
    """Per-clock-hour mean temperature over the lookback window, grouped by true EPT hour.

    Robust to DST (a fall-back hour simply contributes one extra sample to its slot) and to
    a window that is not a whole number of days. An hour with no observations at all falls
    back to the window mean rather than producing a NaN that would spread through the
    departures and out into every model that consumes them.
    """
    ref = np.empty(24, dtype='float64')
    overall = float(np.mean(win_temp))
    for h in range(24):
        sel = win_temp[win_hours == h]
        ref[h] = float(sel.mean()) if len(sel) else overall
    return ref


def compute_forecast_features(fc_d0, fc_d1, win_temp, win_hours, heat_threshold):
    """The 32 forecast features for one sample, in FC_FEATURE_NAMES order.

    fc_d0, fc_d1   : (24,) forecast temperature (F) for day d and the target day d+1
    win_temp       : (lookback,) observed temperature over the sample's lookback window
    win_hours      : (lookback,) EPT clock hour of each window row
    heat_threshold : train-fitted heat-wave threshold from thermal_features.build_thermal_references
    """
    ref = _hourly_reference(win_temp, win_hours)
    dev = fc_d1 - ref                                    # the headline: 24 hourly departures

    # Window daily means, so a day total can be compared against a day.
    win_days = max(len(win_temp) / 24.0, 1e-9)
    win_cdd_daily = float(np.maximum(0.0, win_temp - BASE_TEMP_F).sum()) / win_days
    win_hdd_daily = float(np.maximum(0.0, BASE_TEMP_F - win_temp).sum()) / win_days

    d0_cdd = float(np.maximum(0.0, fc_d0 - BASE_TEMP_F).sum())
    d1_cdd = float(np.maximum(0.0, fc_d1 - BASE_TEMP_F).sum())
    d1_hdd = float(np.maximum(0.0, BASE_TEMP_F - fc_d1).sum())

    # Last observed day, matching compute_thermal_static's `d1 = slice(c - 24, c)` so the
    # ramp's first step is measured against exactly the day that feature calls "previous".
    prev_max = float(np.max(win_temp[-24:]))
    d0_max, d1_max = float(fc_d0.max()), float(fc_d1.max())

    heat_days = int(float(fc_d0.mean()) >= heat_threshold) + \
                int(float(fc_d1.mean()) >= heat_threshold)

    return np.array([
        *dev,
        d1_cdd - win_cdd_daily,
        d1_hdd - win_hdd_daily,
        d0_max - prev_max,
        d1_max - d0_max,
        (d0_cdd + d1_cdd) - 2.0 * win_cdd_daily,
        d1_max,
        float(fc_d1[PEAK_HOURS].mean()),
        float(heat_days),
    ], dtype='float32')
