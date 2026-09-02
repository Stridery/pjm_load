"""
Fetch the recent window serving needs — reusing the training crawler's building blocks, scoped
to the last ~1-2 years and KEEPING the unlabelled tail.

Two differences from the training crawler (run_pipeline):
  - it returns the FULL cleaned frame (every row, including the most recent days that have
    Load_Estimated + weather but no metered yet). Training drops those; serving's lookback
    reads Load_Estimated, so it needs them.
  - it only fetches the current + previous year. Serving needs ~40 days of history, not six
    years, and every feature is now computed from that window alone — no train-fitted constant
    has to travel with the weights.

The forecast weather is the SAME source and format as training (open_meteo_forecast's
historical-forecast archive, keyed by issue_date/lead_day), so the forecast-day features are
byte-identical to what the models were trained on — no serving-specific forecast path.
"""

import glob
import os
import time
from datetime import date

import pandas as pd

from src.config import (CRAWLER_CONFIG, LOAD_ESTIMATED_DIVISOR, FORECAST_PATH,
                        CONGESTION_CONFIG)
from src.data_crawler import open_meteo as om
from src.data_crawler import open_meteo_forecast as omf
from src.data_crawler.aligner import merge_and_align
from src.data_crawler.pjm_load import fetch_load
from src.data_crawler.pipeline import load_metered, load_preliminary, _fetch_or_load_weather
from src.data_processor import clean_and_engineer, clean_forecast


# The daily job serves TWO congestion arms per zone, and MATRIX_TAG is baked from
# PJM_CONGESTION at config import — so each arm is its own process, and each would otherwise
# re-pull two years of PJM load, both Open-Meteo archives and the DA marginal value to produce
# byte-identical inputs. That doubles the runtime and the load placed on APIs we do not own.
#
# So artifacts younger than this are reused instead of re-fetched. Freshness is checked PER
# ARTIFACT, not behind one flag, so the arms may run in either order: a nocong run never touches
# congestion.csv, and a cong run that follows it still refreshes that one file.
#
# 180 min is far longer than a daily job takes end to end and far shorter than a day, so
# consecutive arms share one fetch while tomorrow's run always fetches afresh.
REUSE_MAX_AGE_MIN = 180


def _fresh(path, max_age_min=None):
    """True if `path` exists and was written within the reuse window.

    The window is read at CALL time, not bound as a default argument. As a default it was
    captured when this module was defined, so setting fetch.REUSE_MAX_AGE_MIN = 0 to force a
    full refresh silently did nothing — the reuse path ran anyway and the "full" run being
    timed was a cached one.
    """
    window = REUSE_MAX_AGE_MIN if max_age_min is None else max_age_min
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < window * 60


def _best_metered(raw_dir):
    """Best-available metered load as a UTC-indexed MW series — INCLUDING the recent unverified
    tail. pipeline.load_metered deliberately drops that tail (the training label must be verified
    metered), but the real-time test scores against the freshest actual there is: unverified
    (~2-3 d lag), upgraded to verified as it finalises. That upgrade happens naturally because
    the store re-reads this each day."""
    files = sorted(glob.glob(os.path.join(raw_dir, 'metered', 'hrl_load_metered_*.csv')))
    if not files:
        return pd.Series(dtype=float)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['datetime_beginning_utc'] = pd.to_datetime(df['datetime_beginning_utc'], utc=True)
    df = (df.sort_values('datetime_beginning_utc')
            .drop_duplicates('datetime_beginning_utc', keep='first')
            .set_index('datetime_beginning_utc'))
    return df['mw'].astype(float)


def fetch_recent(zone, days=45, data_root='data', reuse=True):
    """Return (frame, metered): the cleaned recent frame (UTC index, incl. the no-metered tail)
    for feature building, and the best-available metered series (incl. unverified) for scoring.
    Also refreshes the zone's forecast.csv. `days` is how much tail to hand back (>= 40 to cover
    the 21-day macro window + heat-streak margin before the last forecastable day).

    reuse=True loads artifacts written within REUSE_MAX_AGE_MIN instead of re-fetching them, so
    the second congestion arm of the same zone costs no network at all. Pass reuse=False to force
    a full refresh."""
    tz = CRAWLER_CONFIG['timezone']
    location = CRAWLER_CONFIG['location_name']
    years = [date.today().year - 1, date.today().year]   # previous+current: always >= 40 days

    raw_dir      = os.path.join(data_root, zone, 'raw')
    weather_dir  = os.path.join(raw_dir, 'weather')
    forecast_dir = os.path.join(raw_dir, 'forecast')
    os.makedirs(weather_dir, exist_ok=True)
    os.makedirs(forecast_dir, exist_ok=True)

    cleaned_dir = os.path.join(data_root, zone, 'cleaned')
    os.makedirs(cleaned_dir, exist_ok=True)
    cleaned_path = os.path.join(cleaned_dir, '_serving_cleaned.csv')
    fc_path      = os.path.join(cleaned_dir, 'forecast.csv')
    cong_path    = os.path.join(cleaned_dir, 'congestion.csv')

    if reuse and _fresh(cleaned_path) and _fresh(fc_path):
        # Written minutes ago by this zone's other arm. Read back rather than re-derived: the
        # frame round-trips exactly (tz-aware UTC index, Datetime_EPT, and the no-metered tail
        # rows the lookback needs are all preserved in the CSV).
        age = (time.time() - os.path.getmtime(cleaned_path)) / 60
        print(f"  reusing serving data fetched {age:.0f} min ago ({cleaned_path})")
        cleaned = pd.read_csv(cleaned_path, index_col=0, parse_dates=True)
    else:
        cleaned = _fetch_and_clean(zone, years, tz, location, raw_dir, weather_dir,
                                   forecast_dir, cleaned_dir, data_root)

    # 5. Day-Ahead marginal value → congestion.csv, where _load_congestion reads it. Checked on
    # its own timestamp: only the cong arm writes it, so reusing the frame above says nothing
    # about whether THIS file is current.
    if CONGESTION_CONFIG['enabled'] and not (reuse and _fresh(cong_path)):
        from src.data_crawler.pjm_marginal_value import fetch_marginal_value, clean_marginal_value
        fetch_marginal_value(years, raw_dir)
        clean_marginal_value(raw_dir, cong_path)

    cutoff = cleaned.index.max() - pd.Timedelta(days=days)
    return cleaned[cleaned.index > cutoff].copy(), _best_metered(raw_dir)


def _fetch_and_clean(zone, years, tz, location, raw_dir, weather_dir, forecast_dir,
                     cleaned_dir, data_root):
    """The actual network pass: PJM load + Open-Meteo observed + forecast archive -> cleaned frame."""
    # 1. PJM load (metered + preliminary) — force-fetch these two years.
    fetch_load(zone, years, years, raw_dir)
    metered     = load_metered(os.path.join(raw_dir, 'metered'))
    preliminary = load_preliminary(os.path.join(raw_dir, 'preliminary'))
    divisor = LOAD_ESTIMATED_DIVISOR.get(zone, 1.0)
    if divisor != 1.0 and 'Load_Estimated' in preliminary.columns:
        preliminary['Load_Estimated'] = preliminary['Load_Estimated'] / divisor

    # 2. Observed weather + 3. forecast archive (same fetchers as training).
    lat, lon = om.geocode(location)
    weather = pd.concat([_fetch_or_load_weather(lat, lon, y, weather_dir, tz, skip_existing=False)
                         for y in years])
    weather = weather[~weather.index.duplicated(keep='first')].sort_index()
    for y in years:
        omf.fetch_or_load_forecast_year(lat, lon, y, forecast_dir, timezone=tz, skip_existing=False)

    # 4. Align + clean into the full frame (keeps the no-metered tail).
    merged = merge_and_align(metered=metered, preliminary=preliminary, weather=weather, timezone=tz)
    tmp = os.path.join(cleaned_dir, '_serving_merged.csv')
    merged.rename(columns={'Load_Metered': 'Load'}).to_csv(tmp)
    cleaned = clean_and_engineer(tmp, os.path.join(cleaned_dir, '_serving_cleaned.csv'))
    os.remove(tmp)

    # forecast.csv where _load_forecast (the feature layer) reads it.
    clean_forecast(forecast_dir, os.path.join(cleaned_dir, 'forecast.csv'))
    assert FORECAST_PATH.endswith(os.path.join(zone, 'cleaned', 'forecast.csv')) or zone not in FORECAST_PATH
    return cleaned
