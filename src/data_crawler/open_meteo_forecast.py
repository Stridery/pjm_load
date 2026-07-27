"""Archived weather FORECASTS from Open-Meteo — what was forecast, not what happened.

Why this exists
---------------
Every model in this repo currently sees the forecast day's CALENDAR and nothing
else about it — no weather at all (see src/prediction_engine.FORECAST_HORIZON_DAYS).
This module fetches the other half: for each issue date D, the 48 h forecast issued
that day, covering D+1 00:00 – D+2 23:00.

It must be a FORECAST, never an observation. Training on observed weather would
teach the model that tomorrow's temperature is known exactly; at deployment it gets
a forecast with real error, and the backtest would have been fiction. So the values
here are pulled from `{var}_previous_dayN`, which returns what the model run N days
before a given hour predicted for that hour:

    issue date D -> valid times on D+1  are  previous_day1
                 -> valid times on D+2  are  previous_day2

Measured against the ERA5 actuals already on disk (Richmond, July 2025), the error
grows with lead time exactly as a real forecast must:

    best-match      RMSE 2.46 F      previous_day2   RMSE 4.12 F
    previous_day1   RMSE 2.61 F      previous_day5   RMSE 5.21 F

Three things worth knowing before reading the code
--------------------------------------------------
1. ONE endpoint covers past AND future. The historical-forecast archive serves
   previous_dayN for valid times days ahead of now (a run from today does forecast
   tomorrow), so there is no separate "live" mode — one call per year, and the
   current year's call reaches past today into the days we actually forecast.
   The live endpoint (api.open-meteo.com) does NOT support previous_dayN.

2. TEN variables, not eleven. `soil_temperature_0_to_7cm` comes back empty from
   every forecast endpoint — the archive, previous runs, and the live forecast API
   alike. Only the ERA5 archive has it. Its OBSERVED values still reach the model
   through the lookback window; there is simply no forecast of it to be had.

3. Temperature sets the start date; the other nine are sparser and that is fine.
   Measured coverage at Richmond (non-null hours per year, previous_day1):

                        2021   2022   2023   2024   2025
       temperature_2m   6796   8760   8708   8344   8760     from 2021-03-23
       the other nine      0      0      0   8344   8760     from 2024-01-19

   Load is overwhelmingly temperature-driven, so the crawl runs from temperature's
   start and the other nine are simply null until the archive picks them up. They
   ride along in the same request at no extra cost and become usable on their own
   as it grows. Starting from 2024 instead, to have all ten dense, would halve the
   forecastable history for variables that barely move load.

   Coverage is not gapless even where it exists (2023 is 52 h short, 2024 starts on
   the 19th), and whole days go missing. The consumer must therefore check each issue
   date for a complete 24 h rather than assume presence.

Years before the archive begins return an empty frame with a warning rather than
raising: the load history reaches back to 2020 and the caller may legitimately ask
for all of it.

Output layout (long format, one file per year of VALID time)
------------------------------------------------------------
    data/{zone}/raw/forecast/forecast_{year}.csv

        time         naive local time of the forecast hour  (the valid time)
        issue_date   the day the forecast was issued        (= valid date - lead_day)
        lead_day     1 or 2
        <10 weather columns, same names and units as raw/weather/>

Timestamps follow the same convention as raw/weather/: naive local time on a flat
24-rows-per-day grid, DST included. The aligner and the matrix builders already
speak that convention, so nothing downstream needs a special case.
"""

import logging
import os
from datetime import date, timedelta

import pandas as pd
import requests

from ._retry import with_retry
from .open_meteo import COLUMN_RENAME

logger = logging.getLogger(__name__)

_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# The ten variables that exist as a forecast. Deliberately derived from the archive
# crawler's rename map minus soil temperature, so adding a variable there and here
# cannot silently drift apart.
_UNAVAILABLE = {"soil_temperature_0_to_7cm"}
FORECAST_VARS = [v for v in COLUMN_RENAME if v not in _UNAVAILABLE]
FORECAST_COLS = [COLUMN_RENAME[v] for v in FORECAST_VARS]

# D+1 and D+2 — the two days a forecast issued on D can cover, and exactly the
# horizon src/prediction_engine.py already reaches with calendar features alone.
LEAD_DAYS = (1, 2)

# First day the previous-run archive holds a temperature forecast, measured by
# scanning full years and taking the first non-null hour. Temperature is the variable
# that matters for load, so it — not the point where all ten turn dense (2024-01-19) —
# is what the crawl starts from.
ARCHIVE_START = date(2021, 3, 23)


@with_retry(max_attempts=5, backoff_base=3.0)
def _fetch_chunk(lat, lon, start_date, end_date, timezone):
    """One request: all 10 variables x both leads over a date range."""
    hourly = [f"{v}_previous_day{d}" for v in FORECAST_VARS for d in LEAD_DAYS]
    resp = requests.get(
        _FORECAST_URL,
        params={
            "latitude":           lat,
            "longitude":          lon,
            "start_date":         start_date,
            "end_date":           end_date,
            "hourly":             ",".join(hourly),
            "timezone":           timezone,
            "wind_speed_unit":    "mph",
            "temperature_unit":   "fahrenheit",
            "precipitation_unit": "inch",
        },
        timeout=300,
    )
    resp.raise_for_status()
    hourly_block = resp.json().get("hourly", {})
    if not hourly_block or "time" not in hourly_block:
        raise ValueError("Empty or malformed hourly block in Open-Meteo response")
    df = pd.DataFrame(hourly_block)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def _reshape(wide):
    """(valid time x var_previous_dayN) -> long rows keyed by (issue_date, lead_day).

    The API indexes by VALID time and tags each column with how far back the run was.
    What the feature layer wants is the opposite: given an issue date, the 48 hours
    that were forecast from it. Since previous_dayN at valid date V came from the run
    on V - N days, the issue date is just that subtraction.
    """
    frames = []
    for lead in LEAD_DAYS:
        cols = {f"{v}_previous_day{lead}": COLUMN_RENAME[v] for v in FORECAST_VARS}
        present = [c for c in cols if c in wide.columns]
        if not present:
            continue
        part = wide[present].rename(columns=cols).apply(pd.to_numeric, errors="coerce")
        if part.notna().sum().sum() == 0:
            continue                      # lead entirely outside the archive
        part = part.copy()
        part["lead_day"] = lead
        part["issue_date"] = (part.index.normalize() - pd.Timedelta(days=lead)).date
        frames.append(part)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames)
    out.index.name = "time"
    # Sort by what the file is FOR — reading forward through issue dates, then through
    # the 48 h each one covers — rather than by valid time, which interleaves the two leads.
    out = out.reset_index().sort_values(["issue_date", "time"]).set_index("time")
    return out[["issue_date", "lead_day"] + FORECAST_COLS]


def fetch_forecast_year(lat, lon, year, timezone="America/New_York", horizon_days=2):
    """All archived forecasts whose VALID time falls in `year`.

    For the current year the window is extended `horizon_days` past today, which is
    the point of the exercise: those rows are the forecast for the days that have not
    happened yet. The archive serves them because a run from today really does predict
    tomorrow — no separate live fetch needed.

    Returns an empty DataFrame (with a warning) for years before the archive begins.
    """
    if year < ARCHIVE_START.year:
        logger.warning(
            "  forecast %d: no archive — Open-Meteo's previous-run archive starts %s, "
            "so this year has no forecast data at all (the load history does reach back "
            "here; those days simply cannot be used by a forecast-weather model).",
            year, ARCHIVE_START,
        )
        return pd.DataFrame()

    start = max(date(year, 1, 1), ARCHIVE_START)
    end = min(date(year, 12, 31), date.today() + timedelta(days=horizon_days))
    if end < start:
        logger.warning("  forecast %d: window is empty (%s > %s) — skipped", year, start, end)
        return pd.DataFrame()

    logger.info("Fetching Open-Meteo forecast archive  year=%d  %s → %s  lat=%.4f lon=%.4f",
                year, start, end, lat, lon)
    wide = _fetch_chunk(lat, lon, str(start), str(end), timezone)
    long = _reshape(wide)
    if long.empty:
        logger.warning("  forecast %d: archive returned no usable values", year)
        return long

    # Report on the thing that actually determines usability: not row count, but how
    # many issue dates carry a COMPLETE 48 h temperature forecast. The archive drops
    # whole days, so a year can look full and still be unusable in places.
    per_issue = long.groupby("issue_date")["Temp_F"].apply(lambda s: int(s.notna().sum()))
    complete = int((per_issue == 24 * len(LEAD_DAYS)).sum())
    empty_cols = [c for c in FORECAST_COLS if long[c].isna().all()]

    logger.info("  → %d rows over %d issue date(s), %s → %s",
                len(long), len(per_issue), long["issue_date"].min(), long["issue_date"].max())
    logger.info("     %d issue date(s) with a complete %d h temperature forecast%s",
                complete, 24 * len(LEAD_DAYS),
                "" if complete == len(per_issue) else f"  ({len(per_issue) - complete} partial)")
    if empty_cols:
        logger.info("     no archive yet for: %s", ", ".join(empty_cols))
    return long


def fetch_or_load_forecast_year(lat, lon, year, forecast_dir, timezone="America/New_York",
                                skip_existing=True, horizon_days=2):
    """Cached per-year fetch, mirroring _fetch_or_load_weather in pipeline.py.

    The current year is ALWAYS re-fetched even when caching is on: its file grows every
    day, and the newest issue dates — the ones that cover the days we actually want to
    forecast — are precisely the ones a stale cache would be missing.
    """
    path = os.path.join(forecast_dir, f"forecast_{year}.csv")
    is_current = year >= date.today().year

    if skip_existing and not is_current and os.path.exists(path):
        logger.info("  forecast %d: loading from cache (%s)", year, path)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # issue_date round-trips through the CSV as text, while a freshly fetched year
        # carries real dates. Restore it here so the two paths are interchangeable — a run
        # that mixes cached and fetched years would otherwise compare str against date.
        df["issue_date"] = pd.to_datetime(df["issue_date"]).dt.date
        return df

    df = fetch_forecast_year(lat, lon, year, timezone=timezone, horizon_days=horizon_days)
    if df.empty:
        return df

    os.makedirs(forecast_dir, exist_ok=True)
    df.to_csv(path)
    logger.info("  forecast %d: saved → %s  (%d rows)%s",
                year, path, len(df), "  [refreshed: current year]" if is_current else "")
    return df
