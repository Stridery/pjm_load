"""
Orchestration pipeline: manual PJM CSVs + Open-Meteo weather → joined CSV.

Expected raw file layout
------------------------
    data/{zone}/raw/metered/hrl_load_metered_{year}.csv     ← manually downloaded
    data/{zone}/raw/preliminary/hrl_load_prelim_{year}.csv  ← manually downloaded
    data/{zone}/raw/weather/weather_{year}.csv              ← auto-fetched (observed)
    data/{zone}/raw/forecast/forecast_{year}.csv            ← auto-fetched (forecast)

Output
------
    data/{zone}/joined/merged_pjm_load_weather.csv     load + observed weather
    data/{zone}/cleaned/cleaned_pjm_load_weather.csv   labelled rows  → training
    data/{zone}/cleaned/predict.csv                    all rows, no Load → forecasting

The forecast files are a side output on their own index (issue date x lead), not part
of the joined/cleaned frames — see Step 3b for why.

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

import numpy as np
import pandas as pd

from src.config import LOAD_ESTIMATED_DIVISOR
from . import open_meteo as om
from . import open_meteo_forecast as omf
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
    files = _sorted_csvs(preliminary_dir, "hrl_load_prelim*.csv")
    logger.info("Found %d preliminary load file(s): %s – %s",
                len(files), os.path.basename(files[0]), os.path.basename(files[-1]))

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    df["datetime_beginning_utc"] = _parse_utc(df["datetime_beginning_utc"])
    df = df.set_index("datetime_beginning_utc")
    df.index.name = "Datetime_UTC"
    df = df.rename(columns={"prelim_load_avg_hourly": "Load_Estimated"})
    df = df[["Load_Estimated"]]
    # Replace outliers beyond median ± 10×IQR with linear interpolation
    q1, q3 = df["Load_Estimated"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 10 * iqr
    df.loc[df["Load_Estimated"] > upper, "Load_Estimated"] = np.nan
    df["Load_Estimated"] = df["Load_Estimated"].interpolate(method="linear").ffill().bfill()
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
    forecast_dir    = os.path.join(raw_dir, "forecast")
    joined_dir      = os.path.join(data_root, zone, "joined")
    os.makedirs(weather_dir,  exist_ok=True)
    os.makedirs(forecast_dir, exist_ok=True)
    os.makedirs(joined_dir,   exist_ok=True)
    joined_path = os.path.join(joined_dir, "merged_pjm_load_weather.csv")

    # Step 1 & 2 – Load and concat PJM CSVs
    logger.info("=== Step 1: Loading metered load from %s ===", metered_dir)
    metered = load_metered(metered_dir)

    logger.info("=== Step 2: Loading preliminary load from %s ===", preliminary_dir)
    preliminary = load_preliminary(preliminary_dir)

    # Scale a regional-aggregate preliminary series down to this zone's own magnitude. BGE has
    # no zone-level preliminary — PJM only publishes the MIDATL aggregate (~8.8x BGE) — so we
    # divide it here, at the source, before it is joined and written. One divide keeps every
    # later stage byte-for-byte identical to a zone whose preliminary really is its own load.
    divisor = LOAD_ESTIMATED_DIVISOR.get(zone, 1.0)
    if divisor != 1.0 and "Load_Estimated" in preliminary.columns:
        preliminary["Load_Estimated"] = preliminary["Load_Estimated"] / divisor
        logger.info("Scaled Load_Estimated for zone '%s' by 1/%.2f (regional aggregate -> zone scale)",
                    zone, divisor)

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

    # Step 3b – Archived weather FORECASTS (what was predicted, not what happened).
    #
    # Deliberately a SIDE OUTPUT: it lands in raw/forecast/, beside raw/weather/, and is
    # never joined into `merged`. Two reasons, and both matter.
    #
    #   Shape.  This is not another hourly column on the same index. It is, per issue
    #           date, the 48 h that were forecast from it — the same valid hour appears
    #           under two different issue dates at two different lead times. Flattening
    #           that into the hourly frame would have to pick one lead and throw the
    #           other away.
    #   Blast radius. The archive only reaches back to ~2021-04, while the load history
    #           starts 2020-01. Joining it in would put NaNs in the joined frame for the
    #           first 15 months, which the prediction-view guard below (rightly) refuses
    #           to write. Keeping it separate means the cleaned CSVs stay byte-identical
    #           and every existing model keeps training on exactly what it trained on
    #           before — the forecast features get spliced in at matrix-build time, for
    #           the runs that ask for them.
    logger.info("=== Step 3b: Fetching archived weather forecasts ===")
    fc_frames = [
        omf.fetch_or_load_forecast_year(lat, lon, year, forecast_dir,
                                        timezone=timezone, skip_existing=skip_existing)
        for year in range(start_year, end_year + 1)
    ]
    fc_rows = sum(len(f) for f in fc_frames)
    if fc_rows:
        issue_days = sorted({d for f in fc_frames if not f.empty
                             for d in f["issue_date"].unique()})
        logger.info(
            "Forecast archive: %d rows over %d issue date(s), %s → %s  →  %s",
            fc_rows, len(issue_days), issue_days[0], issue_days[-1], forecast_dir,
        )
    else:
        logger.warning(
            "Forecast archive: nothing fetched for %d-%d. Every year requested predates "
            "%s, so no forecast-weather model can be trained on this range.",
            start_year, end_year, omf.ARCHIVE_START,
        )

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
    from src.data_processor import clean_and_engineer, clean_forecast  # type: ignore

    cleaned_dir  = os.path.join(data_root, zone, "cleaned")
    os.makedirs(cleaned_dir, exist_ok=True)
    cleaned_path = os.path.join(cleaned_dir, "cleaned_pjm_load_weather.csv")

    # clean_and_engineer expects a 'Load' column; rename from 'Load_Metered'
    clean_input_path = os.path.join(cleaned_dir, "_merged_for_cleaning.csv")
    merged.rename(columns={"Load_Metered": "Load"}).to_csv(clean_input_path)

    cleaned = clean_and_engineer(clean_input_path, cleaned_path)
    os.remove(clean_input_path)

    # Step 6b – Stitch the per-year forecast shards into one file, on their own index.
    # Separate from the load/weather frame for the reasons in Step 3b; consolidated here
    # so the feature layer reads one path and never has to know the shards exist.
    clean_forecast(forecast_dir, os.path.join(cleaned_dir, "forecast.csv"))

    # Step 7 – Split into the training view and the forecast view.
    #
    # Metered (verified) lags ~7 days behind preliminary, so the most recent hours have
    # Load_Estimated + weather but no Load. Every model INPUT comes from Load_Estimated
    # and weather; only the label needs metered. So those hours are not junk — they are
    # the days we forecast.
    #
    # train  : labelled rows only. Identical in shape to what training always consumed,
    #          so nothing downstream changes: no NaN labels reach the loss, none leak
    #          into the tail split (sklearn's MAPE does not skip NaN — a single NaN
    #          label in the test set turns the reported MAPE into `nan`), and
    #          split_idx = int(len(df) * (1 - test_frac)) keeps meaning what it meant.
    # predict: EVERY row, with Load dropped. Not just the unlabelled tail — forecasting
    #          one day reads 504 h (21 d) of history for the macro features, so the
    #          recent days are worthless without the labelled history in front of them.
    #          Dropping the column makes it impossible to train or score against a NaN.
    predict_path = os.path.join(cleaned_dir, "predict.csv")

    train   = cleaned[cleaned["has_label"] == 1].drop(columns=["has_label"])
    predict = cleaned.drop(columns=["Load", "is_valid"], errors="ignore")

    # Dropping the unlabelled rows is only safe because they form a single block at the
    # END. The lookback is sliced BY POSITION (data_array[cutoff-168 : cutoff]), so a
    # hole punched in the middle would make "168 rows" silently span more than 168 hours
    # and corrupt every window that crosses it — without raising. Refuse to write a
    # training file that would do that.
    step = train.index.to_series().diff().dropna()
    holes = step[step != pd.Timedelta("1h")]
    if len(holes):
        raise ValueError(
            f"Training view is not contiguous: {len(holes)} gap(s) in the labelled "
            f"hours, first at {holes.index[0]}. Metered must have no interior holes — "
            f"backfill them (PJM's unverified rows are fine) and re-run. Writing this "
            f"file would silently corrupt every lookback window that crosses a gap."
        )

    na_cols = predict.columns[predict.isna().any()].tolist()
    if na_cols:
        raise ValueError(
            f"Prediction view has NaNs in {na_cols}. A NaN anywhere inside a 168 h "
            f"lookback window silently turns that day's whole forecast into NaN, so "
            f"this must be fixed upstream rather than filled here."
        )

    train.to_csv(cleaned_path)
    logger.info(
        "=== Training CSV saved → %s  (%d rows × %d cols, %s → %s) ===",
        cleaned_path, *train.shape, train.index[0], train.index[-1],
    )

    predict.to_csv(predict_path)
    n_target = int((predict["has_label"] == 0).sum())
    logger.info(
        "=== Prediction CSV saved → %s  (%d rows × %d cols; %d h with no metered yet "
        "= %.0f forecastable days) ===",
        predict_path, *predict.shape, n_target, n_target / 24,
    )
    return train
