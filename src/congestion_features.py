"""Congestion features: how far the grid departed from its own recent norm on the day the
lookback cannot see.

Built from PJM Day-Ahead marginal value (transmission-constraint shadow prices), aggregated
RTO-wide — see src/data_crawler/pjm_marginal_value.py for why Day-Ahead and why RTO-wide.
These are STATIC (per-sample) features, added beside the macro / thermal / forecast blocks,
NOT channels in the 168 h lookback window: the window ends at the cutoff, and the signal
lives on the day AFTER it (see "the blind-spot day"), so a per-hour channel could not carry
the useful part and would cost the tree models 168 columns each on top of ~3700.


The blind-spot day
------------------
A sample cuts off at 00:00 of day d and predicts day d+1, so day d is invisible: the
lookback ends before it and the calendar does not describe it. src/forecast_features fills
half that gap with forecast weather; this fills the other half.

    d-7 ---- 168 h lookback ---- d-1 | CUTOFF = d 00:00 | day d (blind) | d+1 = TARGET

Day d's Day-Ahead market cleared at ~13:30 EPT on d-1, hours before the cutoff, so it is
genuinely ex-ante. Day d+1's clears at ~13:30 on day d — AFTER the cutoff — so the target
day's own congestion is not available and is deliberately not used. Measured against the
2025-26 level shift, explanatory power decays 0.086 (target day, unavailable) -> 0.085
(day d) -> 0.050 (48 h), so day d retains essentially all of it.


Why everything here is a DEPARTURE
----------------------------------
Version 1 of this block carried both levels ("how congested was day d") and departures ("how
much more congested than lately"). A LightGBM-residual gain breakdown settled which form
works, and it was not close:

    cong_d0_dev     day d logsum - trailing mean    3.307 %   <- 69% of the whole block
    cong_d0_peak    day d peak logsum (level)       0.968 %
    cong_d0_logsum  day d logsum (level)            0.062 %
    cong_w1_logsum  trailing mean (level)           0.013 %
    cong_d0_max_hrs hours above a threshold         0.002 %

The same underlying quantity was ~50x more useful written as a departure than as a level,
and the whole block was 4.79% of gain against 2.09% for the average forecast feature — so
the information is real, it was just mostly spelled the wrong way.

The reason is the same one that makes this block help residual targets and not direct ones.
About 70% of what congestion carries is a proxy for system-wide LOAD LEVEL (measured: the
block scores R2 0.117 against a weather+calendar residual, but only 0.036 once RTO-wide load
is controlled for), and the models already see the level directly through 168 h of
Load_Estimated plus the 3-week macro block. Subtracting a trailing mean removes exactly that
redundant component and leaves the ~30% that is genuinely new. A residual model, whose target
is itself a deviation from a same-hour-last-week baseline, is framed to use that directly.

So v2 writes every quantity as a departure from its own trailing mean, over three window
lengths so the model can pick the timescale, and keeps exactly one absolute anchor — the
peak-hours level, the only level form that earned its place. That mirrors src/forecast_features,
which takes departures for the same reason but keeps three absolute anchors because load
responds to temperature non-linearly.

The trailing mean INCLUDES day d. Excluding it would only rescale each departure by
w/(w-1) — a monotone transform, invisible to trees and absorbed by the scaler for the nets —
so the simpler definition is kept, and it is the one the 3.307% measurement was taken on.


DST
---
The per-day grid is a naive 24-hour local grid. On the fall-back day the duplicated 01:00
was already merged when the hourly table was built; on the spring-forward day the missing
02:00 reindexes to zero and reads as one non-binding hour. Two days a year, inside a daily
aggregate — noted rather than special-cased.
"""

import os

import numpy as np
import pandas as pd

PEAK_HOURS = np.arange(14, 20)          # 14:00-19:00, same block src/forecast_features uses

# Trailing-mean windows, in days: short / primary / long. The primary window is the one the
# secondary quantities (peak, max, n) are measured against; logsum gets all three, since it
# is the quantity the winning feature was built from and the right timescale is not obvious.
W_SHORT, W_PRIMARY, W_LONG = 3, 7, 21

CONG_FEATURE_NAMES = [
    # --- the blind-spot day against its own recent norm ---
    'cong_d0_dev',        # day d logsum      - trailing W_PRIMARY mean   (v1's best feature)
    'cong_d0_dev_s',      # ... against a 3-day norm   — a sharper, more local comparison
    'cong_d0_dev_l',      # ... against a 21-day norm  — is this a whole congested stretch
    'cong_d0_peak_dev',   # day d peak-hours logsum - its own trailing mean
    'cong_d0_max_dev',    # day d worst single constraint - its own trailing mean
    'cong_d0_n_dev',      # day d constraint count - its own trailing mean  (breadth)
    # --- medium-term drift of the norm itself ---
    'cong_w1_slope',      # trailing W_PRIMARY mean - trailing W_LONG mean
    # --- the one absolute anchor (see the docstring: levels mostly do not work, this one did)
    'cong_d0_peak',       # day d peak-hours logsum, as a level
]

N_CONG_FEATURES = len(CONG_FEATURE_NAMES)
assert N_CONG_FEATURES == 8, N_CONG_FEATURES

# Day-Ahead results for day d post at ~13:30 EPT on day d-1. A sample whose cutoff is on
# d-1 (latest_info_hour > 9) must therefore cut off at 14:00 or later, or day d's congestion
# would not yet exist at forecast time.
DA_POST_HOUR_EPT = 14


def assert_available(latest_info_hour):
    """Raise if the configured cutoff would read a Day-Ahead result that had not posted yet.

    latest_info_hour <= 9 puts the cutoff on the target's own previous day (day d) at that
    hour, which is a full night after day d's results posted — always safe. Above 9 the
    cutoff moves back to d-1, where only 14:00 onwards clears the ~13:30 posting time.
    """
    if 9 < latest_info_hour < DA_POST_HOUR_EPT:
        raise ValueError(
            f"latest_info_hour={latest_info_hour} puts the cutoff on day d-1 at "
            f"{latest_info_hour}:00 EPT, before the Day-Ahead market for day d posts "
            f"(~13:30 EPT). The congestion features would be leaky. Use <= 9 (cutoff on day d) "
            f"or >= {DA_POST_HOUR_EPT}.")


def build_congestion_table(path):
    """cleaned/congestion.csv -> per-EPT-date DataFrame of the features, ready to look up.

    Built once per run (like src/feature_engine._load_forecast) because every sample reads a
    different day out of the same table, and re-aggregating 48k hourly rows per sample would
    dominate the build.

    Rows are only written for hours that had a binding constraint, so each day is reindexed
    to its own 24 hours and the rest filled with zero — "nothing bound" is a real state, not
    a gap. A day with NO rows at all is different: RTO-wide, every day of 2021-2026 has at
    least one binding constraint, so an entirely absent day means the fetch failed. Those days
    are marked unusable and their samples get dropped rather than silently fed zeros.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing. Run the crawler (it fetches Day-Ahead marginal value and "
            f"writes this file), or set PJM_CONGESTION=0 to build without the congestion "
            f"block — the pre-congestion feature set is still fully reproducible.")
    h = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if h.empty:
        return pd.DataFrame(columns=CONG_FEATURE_NAMES + ['_usable'])

    real_days = set(h.index.normalize())            # before the grid fill — the fetch sentinel

    grid = pd.date_range(h.index.min().normalize(),
                         h.index.max().normalize() + pd.Timedelta(hours=23), freq='h')
    h = h.reindex(grid).fillna(0.0)
    h['lmax'] = np.log1p(h['max_sp'])
    h['lsum'] = np.log1p(h['sum_sp'])

    day = h.index.normalize()
    peak = h.index.hour.isin(PEAK_HOURS)
    g, gp = h.groupby(day), h[peak].groupby(day[peak])

    # The four daily base quantities. Everything below is a difference between two of these
    # or between one and its own trailing mean; none of them ships as a level except peak.
    base = pd.DataFrame({
        'logsum': g['lsum'].mean(),        # total depth, day mean
        'peak':   gp['lsum'].mean(),       # total depth over 14:00-19:00
        'max':    g['lmax'].max(),         # worst single constraint of the day
        'n':      g['n'].sum(),            # how many constraints bound (breadth)
    }).astype('float64')

    # min_periods=1 so the first days of the record still produce a value instead of poisoning
    # every sample whose window reaches back into them.
    roll = lambda col, w: base[col].rolling(w, min_periods=1).mean()

    t = pd.DataFrame({
        'cong_d0_dev':      base['logsum'] - roll('logsum', W_PRIMARY),
        'cong_d0_dev_s':    base['logsum'] - roll('logsum', W_SHORT),
        'cong_d0_dev_l':    base['logsum'] - roll('logsum', W_LONG),
        'cong_d0_peak_dev': base['peak']   - roll('peak',   W_PRIMARY),
        'cong_d0_max_dev':  base['max']    - roll('max',    W_PRIMARY),
        'cong_d0_n_dev':    base['n']      - roll('n',      W_PRIMARY),
        'cong_w1_slope':    roll('logsum', W_PRIMARY) - roll('logsum', W_LONG),
        'cong_d0_peak':     base['peak'],
    })[CONG_FEATURE_NAMES]

    t['_usable'] = [d in real_days for d in t.index]
    t.index = t.index.date
    return t


def compute_congestion_features(day_table, target_day):
    """The features for one sample, in CONG_FEATURE_NAMES order, or None if unavailable.

    `target_day` is the day being forecast; everything is read off the day BEFORE it — the
    blind-spot day d. Returning None marks the sample unforecastable, which the builders
    handle the same way they handle a day with no complete weather forecast: drop it.
    """
    d = pd.Timestamp(target_day).date() - pd.Timedelta(days=1).to_pytimedelta()
    if d not in day_table.index:
        return None
    row = day_table.loc[d]
    if not row['_usable']:
        return None
    return row[CONG_FEATURE_NAMES].to_numpy(dtype='float32')
