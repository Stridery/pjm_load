"""
PJM Dataminer 2 REST API client.

Fetches three load time-series for a given load area and year:
  - Metered Load   (verified hourly actuals → used as label y)
  - Estimated Load (preliminary unverified hourly data → historical feature)
  - Load Forecast  (official 7-day-ahead forecasts → future reference feature)

All returned DataFrames are indexed by UTC-aware Timestamps;
timezone conversion to Eastern is handled by aligner.py.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

from ._retry import with_retry

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.pjm.com/api/v1"
_PAGE_SIZE = 50_000


def _make_session(api_key: str) -> requests.Session:
    s = requests.Session()
    if api_key:
        s.headers["Ocp-Apim-Subscription-Key"] = api_key
    s.headers["Accept"] = "application/json"
    return s


def _monthly_windows(year: int) -> list[tuple[str, str]]:
    """Return a list of (start, end) UTC strings for each month of the year."""
    windows = []
    for month in range(1, 13):
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1) - timedelta(hours=1)
        else:
            end = datetime(year, month + 1, 1) - timedelta(hours=1)
        windows.append((
            start.strftime("%Y-%m-%d %H:%M:%S"),
            end.strftime("%Y-%m-%d %H:%M:%S"),
        ))
    return windows


@with_retry(max_attempts=5, backoff_base=3.0, exceptions=(requests.RequestException, ValueError))
def _fetch_page(
    session: requests.Session,
    endpoint: str,
    params: list[tuple],
) -> list[dict]:
    resp = session.get(f"{_BASE_URL}/{endpoint}", params=params, timeout=90)
    if resp.status_code == 401:
        raise PermissionError(
            "PJM API returned 401 Unauthorized. "
            "Check that PJM_API_KEY is set correctly in CRAWLER_CONFIG."
        )
    resp.raise_for_status()
    body = resp.json()
    # DataMiner 2 wraps rows under "items" key
    return body.get("items", body if isinstance(body, list) else [])


def _paginate(
    session: requests.Session,
    endpoint: str,
    base_params: list[tuple],
) -> pd.DataFrame:
    """Exhaust all pages for an endpoint using startRow / rowCount pagination."""
    all_rows: list[dict] = []
    start_row = 1
    while True:
        params = base_params + [("startRow", start_row), ("rowCount", _PAGE_SIZE)]
        page = _fetch_page(session, endpoint, params)
        if not page:
            break
        all_rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        start_row += _PAGE_SIZE
    return pd.DataFrame(all_rows)


def _fetch_hourly_load(
    session: requests.Session,
    load_area: str,
    year: int,
    is_verified: str,          # "TRUE" | "FALSE"
    col_out: str,
) -> pd.DataFrame:
    """
    Pull hrl_load_metered month-by-month and return a single-column DataFrame.

    PJM date-range filter: pass the same field name twice (start, end).
    """
    frames: list[pd.DataFrame] = []
    for start, end in _monthly_windows(year):
        logger.info(
            "  hrl_load_metered [%s] is_verified=%s  %s",
            load_area, is_verified, start[:7],
        )
        params = [
            ("fields",                   "datetime_beginning_utc,load_area,mw,is_verified"),
            ("datetime_beginning_utc",   start),
            ("datetime_beginning_utc",   end),
            ("load_area",                load_area),
            ("is_verified",              is_verified),
        ]
        df = _paginate(session, "hrl_load_metered", params)
        if not df.empty:
            frames.append(df)

    if not frames:
        logger.warning("No data from hrl_load_metered [%s] %d is_verified=%s", load_area, year, is_verified)
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["datetime_beginning_utc"] = pd.to_datetime(
        result["datetime_beginning_utc"], utc=True
    )
    result = (
        result
        .sort_values("datetime_beginning_utc")
        .drop_duplicates(subset="datetime_beginning_utc")
        .set_index("datetime_beginning_utc")
    )
    return result[["mw"]].rename(columns={"mw": col_out})


def fetch_metered_load(api_key: str, load_area: str, year: int) -> pd.DataFrame:
    """Verified final hourly metered load → label y."""
    logger.info("=== Fetching Metered Load [%s] %d ===", load_area, year)
    session = _make_session(api_key)
    return _fetch_hourly_load(session, load_area, year, "TRUE", "Load_Metered")


def fetch_estimated_load(api_key: str, load_area: str, year: int) -> pd.DataFrame:
    """Preliminary (unverified) hourly estimated load → historical feature."""
    logger.info("=== Fetching Estimated Load [%s] %d ===", load_area, year)
    session = _make_session(api_key)
    df = _fetch_hourly_load(session, load_area, year, "FALSE", "Load_Estimated")
    if df.empty:
        logger.warning(
            "Estimated load returned empty for %s %d — "
            "preliminary data may not be retained in the archive.",
            load_area, year,
        )
    return df


def fetch_load_forecast(api_key: str, forecast_area: str, year: int) -> pd.DataFrame:
    """
    Official 7-day-ahead load forecasts.

    For each target hour we keep the most-recent forecast published at least
    12 hours in advance (approximates the day-ahead forecast horizon).
    """
    logger.info("=== Fetching Load Forecast [%s] %d ===", forecast_area, year)
    session = _make_session(api_key)

    frames: list[pd.DataFrame] = []
    for start, end in _monthly_windows(year):
        logger.info("  load_frcstd_7_day [%s] %s", forecast_area, start[:7])
        params = [
            ("fields",                           "evaluated_at_utc,forecast_area,"
                                                 "forecast_hourbeginning_utc,forecast_load_mw"),
            ("forecast_hourbeginning_utc",       start),
            ("forecast_hourbeginning_utc",       end),
            ("forecast_area",                    forecast_area),
        ]
        df = _paginate(session, "load_frcstd_7_day", params)
        if not df.empty:
            frames.append(df)

    if not frames:
        logger.warning("No forecast data for %s %d", forecast_area, year)
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["evaluated_at_utc"] = pd.to_datetime(df["evaluated_at_utc"], utc=True)
    df["forecast_hourbeginning_utc"] = pd.to_datetime(
        df["forecast_hourbeginning_utc"], utc=True
    )

    # Keep only forecasts with at least 12 h lead time (day-ahead proxy)
    df["lead_h"] = (
        (df["forecast_hourbeginning_utc"] - df["evaluated_at_utc"])
        .dt.total_seconds() / 3600
    )
    df = df[df["lead_h"] >= 12].copy()

    # For each target hour, pick the freshest qualifying forecast
    df = df.sort_values("evaluated_at_utc", ascending=False)
    df = df.drop_duplicates(subset="forecast_hourbeginning_utc", keep="first")
    df = df.set_index("forecast_hourbeginning_utc").sort_index()

    return df[["forecast_load_mw"]].rename(columns={"forecast_load_mw": "Load_Forecast"})
