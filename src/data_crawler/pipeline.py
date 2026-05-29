"""
Orchestration pipeline: manual PJM CSVs + Open-Meteo weather → joined CSV.

Expected raw file layout
------------------------
    data/{zone}/raw/metered/hrl_load_metered_{year}.csv     ← manually downloaded
    data/{zone}/raw/preliminary/hrl_load_prelim_{year}.csv  ← manually downloaded
    data/{zone}/raw/weather_{year}.csv                      ← auto-fetched

Output
------
    data/{zone}/joined/merged_pjm_load_weather.csv

Usage
-----
    from src.data_crawler import run_pipeline
    run_pipeline(zone="dom2")                         # auto-detects year range from files
    run_pipeline(zone="dom2", start_year=2022, end_year=2023)
"""

import glob
import logging
import os
import re
import sys

import pandas as pd

from . import open_meteo as om
from .aligner import merge_and_align

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_utc(series: pd.Series) -> pd.Series:
    """Parse a datetime series and ensure it is UTC-aware."""
    s = pd.to_datetime(series)
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    else:
        s = s.dt.tz_convert("UTC")
    return s


def _sorted_csvs(directory: str, pattern: str) -> list[str]:
    """Return sorted list of files matching glob pattern."""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in: {directory}"
        )
    return files


def _years_from_files(files: list[str]) -> list[int]:
    """Extract 4-digit years from filenames."""
    years = []
    for f in files:
        m = re.search(r"(\d{4})\.csv$", os.path.basename(f))
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def load_metered(metered_dir: str) -> pd.DataFrame:
    """
    Concatenate all hrl_load_metered_*.csv files and return a clean DataFrame.

    PJM export columns used:
        datetime_beginning_utc, datetime_beginning_ept, mw

    Returns UTC-aware DataFrame with columns: Datetime_EPT, Load_Metered.
    """
    files = _sorted_csvs(metered_dir, "hrl_load_metered_*.csv")
    logger.info("Found %d metered load file(s): %s – %s",
                len(files), os.path.basename(files[0]), os.path.basename(files[-1]))

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    df["datetime_beginning_utc"] = _parse_utc(df["datetime_beginning_utc"])
    df["datetime_beginning_ept"] = pd.to_datetime(df["datetime_beginning_ept"])
    df = df.set_index("datetime_beginning_utc")
    df.index.name = "Datetime_UTC"
    df = df.rename(columns={
        "datetime_beginning_ept": "Datetime_EPT",
        "mw":                     "Load_Metered",
    })
    df = df[["Datetime_EPT", "Load_Metered"]]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    logger.info("Metered load: %d rows  (%s → %s)",
                len(df), df.index[0], df.index[-1])
    return df


def load_preliminary(preliminary_dir: str) -> pd.DataFrame:
    """
    Concatenate all hrl_load_prelim_*.csv files and return a clean DataFrame.

    PJM export columns used:
        datetime_beginning_utc, prelim_load_avg_hourly

    Returns UTC-aware single-column DataFrame: Load_Estimated.
    """
    files = _sorted_csvs(preliminary_dir, "hrl_load_prelim_*.csv")
    logger.info("Found %d preliminary load file(s): %s – %s",
                len(files), os.path.basename(files[0]), os.path.basename(files[-1]))

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    df["datetime_beginning_utc"] = _parse_utc(df["datetime_beginning_utc"])
    df = df.set_index("datetime_beginning_utc")
    df.index.name = "Datetime_UTC"
    df = df.rename(columns={"prelim_load_avg_hourly": "Load_Estimated"})
    df = df[["Load_Estimated"]]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    logger.info("Preliminary load: %d rows  (%s → %s)",
                len(df), df.index[0], df.index[-1])
    return df


# ---------------------------------------------------------------------------
# Weather fetching with per-year caching
# ---------------------------------------------------------------------------

def _fetch_or_load_weather(
    lat: float,
    lon: float,
    year: int,
    weather_dir: str,
    timezone: str,
    skip_existing: bool,
) -> pd.DataFrame:
    path = os.path.join(weather_dir, f"weather_{year}.csv")
    if skip_existing and os.path.exists(path):
        logger.info("  weather %d: loading from cache (%s)", year, path)
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = om.fetch_weather_year(lat, lon, year, timezone=timezone)
    df.to_csv(path)
    logger.info("  weather %d: fetched and saved → %s  (%d rows)", year, path, len(df))
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    start_year: int | None = None,
    end_year: int | None = None,
    *,
    zone: str | None = None,
    location_name: str | None = None,
    timezone: str | None = None,
    data_root: str = "data",
    skip_existing: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: concat manual PJM CSVs + crawl Open-Meteo weather → joined CSV.

    Parameters not supplied here are read from ``config.CRAWLER_CONFIG``.
    Year range is auto-detected from the metered load filenames if not specified.

    Steps
    -----
    1. Concat all hrl_load_metered_*.csv  → metered load DataFrame.
    2. Concat all hrl_load_prelim_*.csv   → preliminary load DataFrame.
    3. Geocode location; fetch weather per year (cached in raw/).
    4. Align all series to metered load UTC index and left-join.
    5. Save joined CSV to data/{zone}/joined/merged_pjm_load_weather.csv.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.config import CRAWLER_CONFIG  # type: ignore

    cfg           = CRAWLER_CONFIG
    zone          = (zone          or cfg["pjm_zone"]).lower()
    location_name = location_name  or cfg["location_name"]
    timezone      = timezone       or cfg["timezone"]

    raw_dir         = os.path.join(data_root, zone, "raw")
    metered_dir     = os.path.join(raw_dir, "metered")
    preliminary_dir = os.path.join(raw_dir, "preliminary")
    weather_dir     = os.path.join(raw_dir, "weather")
    joined_dir      = os.path.join(data_root, zone, "joined")
    os.makedirs(weather_dir, exist_ok=True)
    os.makedirs(joined_dir,  exist_ok=True)
    joined_path = os.path.join(joined_dir, "merged_pjm_load_weather.csv")

    # Step 1 & 2 – Load and concat PJM CSVs
    logger.info("=== Step 1: Loading metered load from %s ===", metered_dir)
    metered = load_metered(metered_dir)

    logger.info("=== Step 2: Loading preliminary load from %s ===", preliminary_dir)
    preliminary = load_preliminary(preliminary_dir)

    # Auto-detect year range from metered files (override if caller specified)
    metered_files = _sorted_csvs(metered_dir, "hrl_load_metered_*.csv")
    detected_years = _years_from_files(metered_files)
    start_year = start_year or detected_years[0]
    end_year   = end_year   or detected_years[-1]
    logger.info("Year range: %d – %d", start_year, end_year)

    # Step 3 – Geocode + fetch weather year-by-year
    logger.info("=== Step 3: Geocoding '%s' ===", location_name)
    lat, lon = om.geocode(location_name)

    weather_frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        weather_frames.append(
            _fetch_or_load_weather(lat, lon, year, weather_dir, timezone, skip_existing)
        )

    all_weather = pd.concat(weather_frames)
    all_weather = all_weather[~all_weather.index.duplicated(keep="first")].sort_index()

    # Step 4 – Align to metered UTC index and join
    logger.info("=== Step 4: Aligning and joining ===")
    merged = merge_and_align(
        metered=metered,
        preliminary=preliminary,
        weather=all_weather,
        timezone=timezone,
    )

    # Step 5 – Save joined CSV
    merged.to_csv(joined_path)
    logger.info(
        "=== Joined CSV saved → %s  (%d rows × %d cols) ===",
        joined_path, *merged.shape,
    )

    # Step 6 – Clean and engineer features
    logger.info("=== Step 6: Cleaning and feature engineering ===")
    from src.data_processor import clean_and_engineer  # type: ignore

    cleaned_dir  = os.path.join(data_root, zone, "cleaned")
    os.makedirs(cleaned_dir, exist_ok=True)
    cleaned_path = os.path.join(cleaned_dir, "cleaned_pjm_load_weather.csv")

    # clean_and_engineer expects a 'Load' column; rename from 'Load_Metered'
    clean_input_path = os.path.join(cleaned_dir, "_merged_for_cleaning.csv")
    merged.rename(columns={"Load_Metered": "Load"}).to_csv(clean_input_path)

    cleaned = clean_and_engineer(clean_input_path, cleaned_path)
    os.remove(clean_input_path)

    logger.info(
        "=== Cleaned CSV saved → %s  (%d rows × %d cols) ===",
        cleaned_path, *cleaned.shape,
    )
    return cleaned
