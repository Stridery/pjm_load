"""Covariate channels for TimesFM 3.0, assembled around the metered-load context.

The geometry, and why it is not the one the point models use
-----------------------------------------------------------
TimesFM 3.0 forecasts the METERED series directly, so the context is metered load — and
metered is published a day later than everything else. Measured on both zones, unverified
metered trails preliminary by exactly 24 h. With the repo's cutoff at (D-1) 00:00:

    preliminary / weather / price   available through (D-2) 23:00
    metered                         available through (D-3) 23:00

TimesFM's past_only covariates are right-aligned to the CONTEXT window (the forecaster slices
them `[:, -context_len:]`), so they cannot end a day later than the target series does. Rather
than let that silently truncate them, the whole window steps back one day and keeps the point
models' START date: days D-8..D-3, six days, 144 hours instead of seven days and 168.

    context / past_only   (D-8) 00:00 .. (D-3) 23:00     144 h
    horizon               (D-2) 00:00 .. (D)   23:00      72 steps
                          steps  1-24 = D-2, 25-48 = D-1, 49-72 = the target day
    past_future           the 144 h above PLUS those 72   216 h

Which channels can be past_future
---------------------------------
A past_future covariate must be known for all 216 hours AT FORECAST TIME — the last 48 of
which are D-1 and D, days that have not happened. Their values can only come from a forecast,
never from an observation; reading the observation would be reading the future.

That leaves exactly two families, and the weather one is narrower than it looks. Measured on
the archive (dom, 1967 complete issue dates):

    Temp_F            missing 0.00%     -> usable on every issue date
    the other nine    missing 51.30%    -> 1009 of 1967 issue dates have none of them

So requiring all ten weather variables would halve the sample count, from ~1900 usable days to
~950, to buy nine extra channels. Temperature alone is kept, which is also what the point
models do: every one of the 31 fc_* features is derived from these same 24 forecast
temperatures, and the ridge probe that attributed 80.8% of the daily residual to "forecast
weather" was measuring temperature and nothing else.

  past_future   Temp_F (observed to D-2, forecast for D-1 and D) + 8 calendar channels
  past_only     preliminary load, the three RT price channels, and the other ten weather
                variables as OBSERVATIONS — all confined to the context window, because none
                of them exists for D-1 or D at forecast time.

The target day therefore reaches the model as 24 forecast temperatures plus its calendar,
which is precisely the information the point models get for it. The comparison is between two
ways of USING that information, not between different information.
"""

import numpy as np
import pandas as pd

from src.config import WEATHER_COLS
from src.forecast_features import forecast_issue_date
from src.lmp_features import signed_log

CONTEXT_HOURS = 144      # 6 days: the point models' 7-day window minus metered's day of lag
HORIZON_STEPS = 73       # (D-3) 23:00 -> (D) 23:00, plus one for a fall-back day's 25th hour

# The only weather variable the forecast archive covers completely, hence the only one that
# can span the horizon.
FUTURE_WEATHER = 'Temp_F'

# past_only: observations that stop at the context end. The price channels are switched by
# PJM_LMP, exactly like the point models' price block, so the run tag already distinguishes
# the two arms and an A/B is two runs.
# The load channel follows PJM_TARGET. For load and rt_price that is preliminary load; for
# load_met it is metered load, which is also the CONTEXT series and is therefore removed again
# by the `exclude` argument -- leaving that class with exactly one load series, the one being
# forecast, exactly as the point models now see it.
def _past_only_load():
    from src.config import SERIES_COL
    return [SERIES_COL]
PAST_ONLY_PRICE = ['LMP_RT_Total', 'LMP_RT_Energy', 'LMP_RT_Cong']
# Prices are signed-log'd for the same reason the lookback channels are (median 36, max 4392,
# 22% of congestion hours negative); load is left in MW, which the model normalises itself.
PAST_ONLY_LOG = set(PAST_ONLY_PRICE)


def past_only_columns(include_prices, exclude=()):
    """Observed channels for the context window: load, optionally prices, then the weather
    variables the forecast archive cannot carry into the horizon.

    `exclude` drops whichever column is already the CONTEXT series. With a price target the
    context is LMP_RT_Total, and leaving it here too would hand the model the same series twice
    -- harmless arithmetically, but it misreports the channel count and wastes a covariate slot.
    """
    obs_weather = [c for c in WEATHER_COLS if c != FUTURE_WEATHER]
    cols = _past_only_load() + (PAST_ONLY_PRICE if include_prices else []) + obs_weather
    return [c for c in cols if c not in set(exclude)]

CALENDAR_CHANNELS = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                     'month_sin', 'month_cos', 'is_weekend', 'is_holiday']


def _calendar(ept_index):
    """Deterministic calendar channels for any stretch of EPT hours, past or future."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    e = pd.DatetimeIndex(ept_index)
    hol = USFederalHolidayCalendar().holidays(start=e.min().normalize(),
                                              end=e.max().normalize())
    return np.stack([
        np.sin(2 * np.pi * e.hour / 24), np.cos(2 * np.pi * e.hour / 24),
        np.sin(2 * np.pi * e.dayofweek / 7), np.cos(2 * np.pi * e.dayofweek / 7),
        np.sin(2 * np.pi * e.month / 12), np.cos(2 * np.pi * e.month / 12),
        (e.dayofweek >= 5).astype(float), e.normalize().isin(hol).astype(float),
    ])


def load_forecast_hourly(path):
    """cleaned/forecast.csv -> {issue_date: (48,) array of forecast temperature}.

    lead 1's 24 hours followed by lead 2's — i.e. day D-1 then day D for an issue date of D-2.
    Only temperature: see the module docstring for why the other nine variables cannot be used
    without halving the sample count.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[df['is_complete'] == 1]
    if df.empty:
        return {}
    wide = (df.assign(_hour=df.index.hour, _issue=pd.to_datetime(df['issue_date']).dt.date)
              .pivot_table(index='_issue', columns=['lead_day', '_hour'], values=FUTURE_WEATHER))
    cols = [(lead, h) for lead in (1, 2) for h in range(24)]
    if any(c not in wide.columns for c in cols):
        return {}
    block = wide[cols].to_numpy(dtype='float32')
    return {issue: block[i] for i, issue in enumerate(wide.index) if np.isfinite(block[i]).all()}


def build(frame, ept_dates, ept_hours, cutoff_pos, target_days, fc_hourly,
          include_prices=True, context_hours=CONTEXT_HOURS, horizon=HORIZON_STEPS,
          exclude=()):
    """(past_only, past_future) for a batch of samples, or (None, None) rows where unusable.

    Returns two lists aligned with `target_days`; an entry is None when that day's 48 h weather
    forecast is missing, which makes the sample unforecastable exactly as it does for the point
    models — the caller drops it rather than filling anything in.
    """
    temp = frame[FUTURE_WEATHER].to_numpy(dtype='float32')               # (T,)
    po_cols = past_only_columns(include_prices, exclude)
    missing = [c for c in po_cols if c not in frame.columns]
    if missing:
        raise KeyError(f"The cleaned frame has no {missing}. Re-run the crawler with PJM_LMP=1, "
                       f"or run this adapter with prices off.")
    po_raw = np.stack([
        signed_log(frame[c].to_numpy(dtype='float32')) if c in PAST_ONLY_LOG
        else frame[c].to_numpy(dtype='float32')
        for c in po_cols
    ])                                                                   # (n_past_only, T)
    ept = pd.DatetimeIndex(pd.to_datetime(frame['Datetime_EPT']))

    past_only, past_future = [], []
    for cut, day in zip(cutoff_pos, target_days):
        lo, hi = cut - context_hours, cut                                # context window
        fut = np.arange(cut, cut + horizon)                              # the 72 forecast rows
        fc = fc_hourly.get(forecast_issue_date(day))
        if fc is None or fut[-1] >= temp.shape[0]:
            past_only.append(None)
            past_future.append(None)
            continue

        # Weather over the horizon, resolved PER ROW rather than by slicing at fixed offsets.
        # Day D-2 is observed by the cutoff; D-1 and D can only come from the archived
        # forecast's lead 1 and lead 2. Looking each row up by its own EPT date and hour is
        # what makes this DST-safe: a 23- or 25-hour day shifts the offsets, and a fixed
        # `fut[:24]` splice would then read an observation from D-1 — a leak that no shape
        # check would catch. A day whose 72 steps do not land exactly on those three dates is
        # dropped instead of being patched.
        d1 = day - pd.Timedelta(days=1).to_pytimedelta()
        d2 = day - pd.Timedelta(days=2).to_pytimedelta()
        fut_t = np.empty(horizon, dtype='float32')
        bad = False
        for k, r in enumerate(fut):
            dr, hr = ept_dates[r], ept_hours[r]
            if dr == d2:
                fut_t[k] = temp[r]                 # observed
            elif dr == d1:
                fut_t[k] = fc[hr]                  # forecast, lead 1
            elif dr == day:
                fut_t[k] = fc[24 + hr]             # forecast, lead 2
            else:
                bad = True
                break
        if bad:
            past_only.append(None)
            past_future.append(None)
            continue
        t_full = np.concatenate([temp[lo:hi], fut_t])[None, :]           # (1, L+horizon)
        cal = _calendar(ept[np.r_[np.arange(lo, hi), fut]])                   # (8, 216)

        past_future.append(np.concatenate([t_full, cal]))
        past_only.append(po_raw[:, lo:hi])
    return past_only, past_future


def channel_names(include_prices=True, exclude=()):
    return {'past_only': past_only_columns(include_prices, exclude),
            'past_future': [FUTURE_WEATHER] + CALENDAR_CHANNELS}
