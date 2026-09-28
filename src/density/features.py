"""The paper's fifteen predictors, assembled for this repo's day-ahead convention.

Li et al. Section VI lists them exactly:

    "eleven of the fifteen features are lagged observations with 24, 25, 26, 27, 28, 48, 72,
     96, 120, 144, and 168 hours lagged; the remaining four of the fifteen features are the
     hour of the day, day of the week, month of the year, and a binary feature indicating
     whether a weekend is involved."

No weather. That is the paper's choice, not an omission here — the contribution is the
combination step, and the base models are deliberately generic. Keeping the feature set
exactly as published is what makes the reported numbers comparable.


The lag shift, and why it is forced
-----------------------------------
Their shortest lag is 24 h, which only works if the forecast is issued at the END of the day
before the target: hour k of day D then reaches back to hour k of day D-1.

This repo forecasts day D+1 from a cutoff at day D 00:00, so the last observed hour is day
D-1 23:00 and the target is a further day out. Target hour k of day D+1 sits 25+k hours after
that last observation, so a lag valid for EVERY target hour must be at least 48. Lags 24-28
are simply not observable here — lag 24 never, lags 25-28 only for k < 4.

So the whole list shifts by 24, which preserves its structure rather than truncating it:

    paper     24 25 26 27 28 | 48  72  96 120 144 168
    here      48 49 50 51 52 | 72  96 120 144 168 192

The first block is still "the same hour and the four before it, on the most recent fully
observed day"; the second is still six daily lags out to a week. Eleven features either way.

One consequence worth stating rather than discovering later: this puts the baseline on the
SAME information set as the repo's 16 point models (data through D-1, predict D+1) and 24 h
behind a D-1 -> D issue time (data through D-1, predict D).
"""

import numpy as np
import pandas as pd

# Lags in hours, in the paper's order, shifted +24 (see the module docstring).
LAGS = (48, 49, 50, 51, 52, 72, 96, 120, 144, 168, 192)
LAG_NAMES = [f'load_lag{h}' for h in LAGS]
CALENDAR_NAMES = ['hour', 'dayofweek', 'month', 'is_weekend']
FEATURE_NAMES = LAG_NAMES + CALENDAR_NAMES

MIN_HISTORY = max(LAGS)


def build(df, ept, load):
    """(X, y, rows) — the 15-feature matrix over every hour that has a complete lag set.

    Every hour is its own sample; there is no per-day assembly, because all eleven lags are
    at least 48 h and are therefore observable for any target hour under this repo's cutoff.
    `rows` indexes back into the cleaned frame so the caller can slice train/validation/test
    without re-deriving timestamps.

    The paper keeps both `dayofweek` and `is_weekend` even though one implies the other,
    reporting that models carrying both did better; that is reproduced rather than tidied.
    """
    load = np.asarray(load, dtype=float)
    n = len(load)
    rows = np.arange(MIN_HISTORY, n)

    cols = [load[rows - h] for h in LAGS]
    cols.append(ept[rows].hour.values.astype(float))
    cols.append(ept[rows].dayofweek.values.astype(float))
    cols.append(ept[rows].month.values.astype(float))
    cols.append((ept[rows].dayofweek.values >= 5).astype(float))

    return np.column_stack(cols), load[rows], rows


def split_indices(ept_rows, t2_days, t3_days, test_start):
    """Boolean masks for the paper's T1/T2/T3/T4 partition, in that order.

    T1 trains the base models, T2 tunes/selects them, T3 optimises the combination weights,
    and T4 is the held-out test. The paper uses 10:1:1:1 on two years of ISO-NE data; here T4
    is pinned to the same test period as the other two baselines instead, and T2/T3 take a
    fixed number of days immediately before it. T3 only has to pin down fifteen weights —
    the paper itself uses five weeks — so spending more of the record on it would buy nothing
    and cost the base models their training data.
    """
    t3_start = pd.Timestamp(test_start) - pd.Timedelta(days=t3_days)
    t2_start = t3_start - pd.Timedelta(days=t2_days)
    return (ept_rows < t2_start,
            (ept_rows >= t2_start) & (ept_rows < t3_start),
            (ept_rows >= t3_start) & (ept_rows < test_start),
            ept_rows >= test_start)
