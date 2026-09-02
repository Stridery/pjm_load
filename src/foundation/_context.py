"""Locating each target day's context window and forecast steps in the cleaned CSV.

This is the file that has to be right, because everything it produces is compared against
numbers the 16 point models produced through a completely different path (the cached matrices).
The two must line up exactly or the comparison is meaningless.

The geometry, for latest_info_hour = 0
--------------------------------------
The forecast cutoff is the target day's PREVIOUS midnight, and the lookback window ENDS there
(feature_engine slices [cutoff_pos - lookback, cutoff_pos), excluding the cutoff row itself).
So for target day d:

    context   ... (d-2) 23:00        <- last observed hour
    step 1-24     (d-1) 00:00-23:00  <- the BLIND-SPOT DAY, forecast but discarded
    step 25-48    (d)   00:00-23:00  <- the target day, what we keep

Verified against the matrix builder: for target 2023-09-29 the 168 h window is
2023-09-21 00:00 -> 2023-09-27 23:00, and the first target hour is step 25.

Which is why the horizon is 48 and not 24. A model asked for 24 steps forecasts the blind-spot
day and scores it against the target day's actuals.

DST
---
The cleaned index is UTC and perfectly contiguous (checked: every gap is exactly 1 h), so step
k is always row cutoff_pos + k - 1 and no calendar arithmetic is needed to find it. The target
day itself may hold 23 or 25 EPT hours; steps are selected by matching the row's EPT date and
then normalised to 24 slots by the matrix builder's own _normalize_to_24h, so DST is handled
identically on both sides.
"""

import numpy as np
import pandas as pd

from src.feature_engine import _normalize_to_24h

# The same-hour-last-week baseline, as a time-invariant series definition.
#
# src/models/_residual.baseline_from_windows takes the mean of the 7 same-clock-hour values in
# the 168 h window ending at the cutoff. For a target-day row t that window's 7 values sit at
# t-48, t-72, ... t-192 -- NOT t-24..t-168, because the blind-spot day lies in between. Writing
# the lags out here makes the residual SERIES fed to a model use exactly the same definition as
# the baseline added back afterwards; a half-day mismatch between the two would inject a bias
# that looks like model error. The identity is asserted in test/test_foundation_context.py.
def baseline_lags(extra_lag_days=0):
    """The 7 same-clock-hour lags, pushed back by however stale the context series is.

    With the preliminary series the nearest usable lag is 48 h (the blind-spot day sits at 24).
    A context that is `extra_lag_days` staler -- metered load, which trails preliminary by about
    a day -- cannot see lag 48 either, so every lag shifts by 24 h per extra day. Getting this
    wrong would build the baseline out of values that do not exist yet at forecast time, which
    is leakage that no shape check would catch.
    """
    base = 48 + 24 * extra_lag_days
    return tuple(base + 24 * k for k in range(7))


def baseline_warmup(extra_lag_days=0):
    return max(baseline_lags(extra_lag_days))


BASELINE_LAGS = baseline_lags()          # 48, 72, ..., 192 -- the preliminary-series default
BASELINE_WARMUP = baseline_warmup()      # 192 h before r(t) exists


def load_frame(cleaned_path):
    """The cleaned hourly record, sorted, with the EPT date/hour arrays everything keys off."""
    df = pd.read_csv(cleaned_path, index_col=0, parse_dates=True).sort_index()
    ept = pd.to_datetime(df['Datetime_EPT'])
    return df, ept.dt.date.values, ept.dt.hour.values


def cutoff_positions(ept_dates, ept_hours, target_days, latest_info_hour=0, extra_lag_days=0):
    """Row index of each target day's forecast cutoff, or -1 where it cannot be located.

    Mirrors feature_engine's rule: for latest_info_hour <= 9 the cutoff is the PREVIOUS day at
    that hour; beyond 9 it steps back one further day. Only the <= 9 branch is reachable from
    this repo's configs, but the rule is written out rather than assumed.

    extra_lag_days pushes the cutoff further back for a context series that is not current as of
    the real cutoff. Metered load is the case that matters: it trails preliminary by ~1 day, so
    a run reading metered must pretend the cutoff was a day earlier. Stepping whole DAYS rather
    than subtracting 24 rows keeps it right across DST, where a day is not 24 rows.
    """
    lookup = {}
    for pos, (d, h) in enumerate(zip(ept_dates, ept_hours)):
        if h == latest_info_hour and d not in lookup:
            lookup[d] = pos

    back = (1 if latest_info_hour <= 9 else 2) + int(extra_lag_days)
    out = np.full(len(target_days), -1, dtype=np.int64)
    for i, day in enumerate(target_days):
        cut_day = (pd.Timestamp(day) - pd.Timedelta(days=back)).date()
        out[i] = lookup.get(cut_day, -1)
    return out


def baseline_series(series, extra_lag_days=0):
    """r(t)'s baseline: the mean of the 7 same-clock-hour values at baseline_lags().

    NaN for the first baseline_warmup() hours, which have no complete set of lags.
    """
    lags = baseline_lags(extra_lag_days)
    est = np.asarray(series, dtype=float)
    out = np.full(len(est), np.nan)
    stack = np.full((len(lags), len(est)), np.nan)
    for i, lag in enumerate(lags):
        stack[i, lag:] = est[:-lag]
    ok = ~np.isnan(stack).any(axis=0)
    out[ok] = stack[:, ok].mean(axis=0)
    return out


def residual_series(series, extra_lag_days=0):
    """The context series minus its own same-hour-last-week baseline — what 'residual' mode feeds."""
    return np.asarray(series, dtype=float) - baseline_series(series, extra_lag_days)


def day_row_span(ept_dates):
    """{EPT date: (first_row, last_row)} — every day's row range in the cleaned frame.

    Used instead of a fixed step count so the horizon requirement is what the target day
    ACTUALLY needs (48 normally, 49 across a DST change) rather than a blanket 50. Demanding
    50 rows past the cutoff silently dropped the final test day, whose day exists in full but
    which the CSV simply does not extend 50 hours beyond -- exactly the kind of quiet
    one-day-short test set that makes a comparison wrong without making it fail.
    """
    order = np.argsort(ept_dates, kind='stable')
    spans = {}
    for pos in order:
        d = ept_dates[pos]
        if d in spans:
            spans[d] = (spans[d][0], pos)
        else:
            spans[d] = (pos, pos)
    return spans


def steps_needed(cutoff_pos, target_days, spans):
    """Steps from the cutoff to the last hour of each target day (-1 if the day is absent)."""
    out = np.full(len(target_days), -1, dtype=np.int64)
    for i, day in enumerate(target_days):
        span = spans.get(day)
        if span is not None and cutoff_pos[i] >= 0:
            out[i] = span[1] - cutoff_pos[i] + 1
    return out


def usable(cutoff_pos, context_hours, horizon_steps, n_rows, spans, target_days, warmup=0):
    """Mask of target days this run can actually score.

    Requires: a located cutoff; enough history for the context plus any warmup the series
    definition needs; the target day present in the frame; and a horizon long enough to reach
    the day's final hour. Days that fail are dropped and REPORTED by the caller, never
    zero-filled.
    """
    need = steps_needed(cutoff_pos, target_days, spans)
    return ((cutoff_pos >= 0)
            & (cutoff_pos >= context_hours + warmup)
            & (need > 0)
            & (need <= horizon_steps)
            & (cutoff_pos + need <= n_rows))


def gather_contexts(series, cutoff_pos, context_hours):
    """(N, L) — the L hours ENDING at the cutoff, exclusive, one row per target day."""
    series = np.asarray(series, dtype=float)
    idx = cutoff_pos[:, None] - context_hours + np.arange(context_hours)[None, :]
    return series[idx]


def gather_steps(series, cutoff_pos, horizon):
    """(N, H) — the series' own values at the forecast steps, NaN-padded past the frame's end.

    Used to run the baseline through exactly the same step -> EPT-hour -> 24-slot path the
    model's forecast takes, so the two are aligned by construction rather than by a second
    implementation that has to be kept in agreement.
    """
    series = np.asarray(series, dtype=float)
    idx = cutoff_pos[:, None] + np.arange(horizon)[None, :]
    inside = idx < len(series)
    out = np.full(idx.shape, np.nan)
    out[inside] = series[idx[inside]]
    return out


def steps_to_day24(preds, cutoff_pos, target_days, ept_dates, ept_hours):
    """(N, H) forecast steps -> (N, 24) aligned to each target day's EPT hours.

    Step k of row i is cleaned-frame row cutoff_pos[i] + k. Steps landing on the target day's
    EPT date are kept and handed to the matrix builder's own DST normaliser, so a 23- or
    25-hour day is folded to 24 slots exactly as the training matrices were.
    """
    n, horizon = preds.shape
    out = np.full((n, 24), np.nan)
    for i in range(n):
        rows = cutoff_pos[i] + np.arange(horizon)
        inside = rows < len(ept_dates)          # the frame may end mid-horizon
        rows, step_vals = rows[inside], preds[i][inside]
        keep = ept_dates[rows] == target_days[i]
        if not keep.any():
            continue
        day24 = _normalize_to_24h(step_vals[keep], ept_hours[rows[keep]])
        if day24 is not None:
            out[i] = day24
    return out
