"""
Daily serving run for one zone: fetch → forecast a rolling window of days → freeze-merge into
the committed store.

Forecasts a band of days [last_complete - (window-1) .. last_complete + horizon]:
  - the trailing `horizon` days reach past the data — the genuine day-ahead forecast, no metered
    yet;
  - the earlier days already happened, so re-forecasting them reproduces the day-ahead forecast
    that WAS made for them (forecast weather is archived; the store freezes the first value
    anyway), and their metered fills in as it verifies.
The store then keeps only the most recent `keep_days`.
"""

import numpy as np
import pandas as pd

from src.config import CRAWLER_CONFIG
from src.serving import table
from src.serving.fetch import fetch_recent
from src.serving.predict import forecast_day, available_models

HORIZON_DAYS = 2      # forecast reaches this many days past the last complete day of data
WINDOW_DAYS  = 8      # how many recent days to (re)forecast each run — self-seeds the store


def _complete_days(df):
    """EPT dates that have a row reaching hour 23 (a full day; DST-safe — see the DST fix)."""
    ept = pd.to_datetime(df['Datetime_EPT'])
    last_hour = ept.groupby(ept.dt.date).max().dt.hour
    return sorted(last_hour.index[last_hour == 23])


def _extend_future(df, future_days, tz):
    """Append calendar-only rows for days past the data so build_*_sample can read their
    calendar. Only the calendar columns are filled; load/weather stay NaN (the lookback ends at
    the day before, and the forecast weather comes from forecast.csv)."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    add = []
    for d in future_days:
        local = pd.date_range(pd.Timestamp(d), periods=24, freq='h', tz=tz)   # DST-aware hours
        utc = local.tz_convert('UTC')
        e = local.tz_localize(None)
        hol = USFederalHolidayCalendar().holidays(start=e.min(), end=e.max())
        add.append(pd.DataFrame({
            'Datetime_EPT': e.astype('datetime64[ns]'),
            'month_sin': np.sin(2 * np.pi * e.month / 12), 'month_cos': np.cos(2 * np.pi * e.month / 12),
            'dayofweek': e.dayofweek, 'is_weekend': (e.dayofweek >= 5).astype(int),
            'is_holiday': e.normalize().isin(hol).astype(int),
        }, index=utc.tz_convert('UTC')))
    out = pd.concat([df] + add)
    return out[~out.index.duplicated(keep='first')].sort_index()


def _day_hours(df, target_day):
    """(datetime_local_str, datetime_utc, real_hour) for each hour the target day actually has."""
    ept = pd.to_datetime(df['Datetime_EPT'])
    sub = df[ept.dt.date.values == target_day].sort_index()
    e = pd.to_datetime(sub['Datetime_EPT'])
    return list(zip(e.dt.strftime('%Y-%m-%d %H:%M:%S'), sub.index, e.dt.hour.values))


def run_zone(zone, keep_days=10):
    frame, metered = fetch_recent(zone)   # metered = best-available (incl. unverified), for scoring
    models = available_models()

    complete = _complete_days(frame)
    if not complete:
        raise SystemExit(f"{zone}: no complete day in the fetched window.")
    last = complete[-1]
    future = [ (pd.Timestamp(last) + pd.Timedelta(days=k)).date() for k in range(1, HORIZON_DAYS + 1) ]
    frame = _extend_future(frame, future, CRAWLER_CONFIG['timezone'])

    targets = [d for d in complete[-WINDOW_DAYS:]] + future

    rows = []
    for d in targets:
        try:
            preds = forecast_day(frame, d, models=models)
        except Exception as e:
            print(f"  {zone} {d}: skipped ({e})")
            continue
        for local_s, utc, h in _day_hours(frame, d):
            r = {'datetime_utc': utc, 'datetime': local_s,
                 'metered': (float(metered[utc]) if utc in metered.index
                             and pd.notna(metered[utc]) else np.nan)}
            for m, p in preds.items():
                r[f'{m}_pred'] = round(float(p[h]), 1)
            rows.append(r)

    fresh = pd.DataFrame(rows).set_index('datetime_utc').sort_index()
    return table.update(zone, fresh, keep_days=keep_days)


def main(keep_days=10):
    # One zone per process: config (MATRIX_DIR / MODEL_ROOT / FORECAST_PATH) is fixed from
    # PJM_DATASET at import, so the workflow invokes this once per zone with the env set.
    import src.config as cfg
    print(f"\n=== serving run: {cfg.DATASET} ===")
    run_zone(cfg.DATASET, keep_days=keep_days)


if __name__ == '__main__':
    main()
