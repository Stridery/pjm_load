"""Open-Meteo data fetcher: geocoding + hourly historical weather archive."""

import logging

import pandas as pd
import requests

from ._retry import with_retry

logger = logging.getLogger(__name__)

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# All hourly variables requested from the archive endpoint
_HOURLY_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "dew_point_2m",
    "relative_humidity_2m",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "precipitation",
    "cloud_cover",
    "soil_temperature_0_to_7cm",
]

# Open-Meteo variable names → our canonical column names
COLUMN_RENAME: dict[str, str] = {
    "temperature_2m":           "Temp_F",
    "apparent_temperature":     "ApparentTemp_F",
    "dew_point_2m":             "Dewpoint_F",
    "relative_humidity_2m":     "RelativeHumidity_pct",
    "shortwave_radiation":      "SolarRadiation_Wm2",
    "wind_speed_10m":           "WindSpeed_mph",
    "wind_direction_10m":       "WindDirection_deg",
    "wind_gusts_10m":           "WindGusts_mph",
    "precipitation":            "Precip_in",
    "cloud_cover":              "CloudCover_pct",
    "soil_temperature_0_to_7cm": "SoilTemp0_7cm_F",
}


@with_retry(max_attempts=5, backoff_base=2.0)
def geocode(location_name: str) -> tuple[float, float]:
    """Return (latitude, longitude) for a human-readable place name."""
    resp = requests.get(
        _GEOCODING_URL,
        params={"name": location_name, "count": 1, "language": "en", "format": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        raise ValueError(f"Geocoding returned no results for '{location_name}'")
    lat, lon = results[0]["latitude"], results[0]["longitude"]
    logger.info("Geocoded '%s' → (%.4f, %.4f)", location_name, lat, lon)
    return lat, lon


@with_retry(max_attempts=5, backoff_base=3.0)
def _fetch_archive_chunk(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    timezone: str,
) -> pd.DataFrame:
    resp = requests.get(
        _ARCHIVE_URL,
        params={
            "latitude":           lat,
            "longitude":          lon,
            "start_date":         start_date,
            "end_date":           end_date,
            "hourly":             ",".join(_HOURLY_VARS),
            "timezone":           timezone,
            "wind_speed_unit":    "mph",
            "temperature_unit":   "fahrenheit",
            "precipitation_unit": "inch",
        },
        timeout=120,
    )
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError("Empty or malformed hourly block in Open-Meteo response")
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").rename(columns=COLUMN_RENAME)
    return df


def fetch_weather_year(
    lat: float,
    lon: float,
    year: int,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """
    Fetch all hourly weather variables for a full calendar year.

    For the current year, end_date is capped at yesterday — matching how far the
    preliminary load series reaches, so the two line up and the recent hours can
    be used for prediction.

    Note on the source: the archive endpoint serves ERA5 only up to ~today-5, then
    silently falls back to ECMWF IFS for the remaining days (same column names,
    ~1F different values). Fetching past today-5 therefore mixes sources at the
    tail. Accepted deliberately: stopping 5 days short would leave NaN weather
    inside the lookback window and make recent-day prediction impossible.

    Timestamps in the returned DataFrame are naive local time in `timezone`.
    """
    from datetime import date, timedelta

    start_date = f"{year}-01-01"
    latest = date(year, 12, 31)
    cutoff = date.today() - timedelta(days=1)
    end_date = str(min(latest, cutoff))

    logger.info(
        "Fetching Open-Meteo archive  year=%d  lat=%.4f  lon=%.4f  end=%s",
        year, lat, lon, end_date,
    )
    df = _fetch_archive_chunk(lat, lon, start_date, end_date, timezone)
    logger.info("  → %d hourly rows fetched for %d", len(df), year)
    return df
