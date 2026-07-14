"""
UTC-based alignment: joins metered load (base), preliminary load, and weather.

Metered load provides the canonical UTC index.
Preliminary load (UTC-aware) is left-joined directly.
Weather (naive local time) is converted to UTC before joining.
DST fall-back duplicates are resolved by keeping the first occurrence.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _weather_to_utc(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Convert naive local-time weather index to UTC-aware."""
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    else:
        # ambiguous="NaT": marks the ambiguous DST fall-back hour as NaT
        # (at most 1 row per year) rather than failing — those rows are dropped below.
        idx = idx.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
    df = df.copy()
    df.index = idx
    df = df[df.index.notna()]   # drop the NaT DST-ambiguous rows
    df.index.name = "Datetime_UTC"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def merge_and_align(
    metered: pd.DataFrame,      # UTC-aware index; cols: Datetime_EPT, Load_Metered
    preliminary: pd.DataFrame,  # UTC-aware index; col:  Load_Estimated (may be empty)
    weather: pd.DataFrame,      # naive local-time index; weather feature cols
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """
    Join all series onto metered load's UTC index.

    Preliminary load and weather are left-joined so no metered rows are dropped.
    Index is UTC-aware; Datetime_EPT is the first non-index column.

    Column order in output
    ----------------------
    Datetime_EPT | Load_Metered | Load_Estimated | <weather cols>
    """
    met = metered.copy()
    if met.index.tz is None:
        met.index = met.index.tz_localize("UTC")

    # Time — not metered — is the canonical index. Metered (verified) lags ~7 days
    # behind preliminary, so an outer join keeps the recent hours that have
    # Load_Estimated but no Load_Metered yet. Those rows are exactly what the model
    # forecasts on: every model input comes from Load_Estimated + weather, and only
    # the training label needs metered. The old metered-as-base left-join dropped
    # them outright, which is what made the dataset unusable for live prediction.
    if not preliminary.empty:
        pre = preliminary.copy()
        if pre.index.tz is None:
            pre.index = pre.index.tz_localize("UTC")
        df = met.join(pre, how="outer")
        n_unlabeled = int(df["Load_Metered"].isna().sum())
        logger.info(
            "Aligning on the union of metered+preliminary hours: %d rows "
            "(%d metered, %d with Load_Estimated but no metered yet)",
            len(df), len(met), n_unlabeled,
        )
    else:
        logger.warning("Preliminary load is empty — Load_Estimated column will be absent.")
        df = met

    # Datetime_EPT comes from the metered export, so the preliminary-only rows have
    # none. Derive it from the UTC index for exactly those rows; existing values are
    # left untouched so nothing about the labelled history changes.
    missing_ept = df["Datetime_EPT"].isna()
    if missing_ept.any():
        derived = df.index[missing_ept].tz_convert(timezone).tz_localize(None)
        df.loc[missing_ept, "Datetime_EPT"] = derived
        logger.info("  Derived Datetime_EPT for %d preliminary-only rows", int(missing_ept.sum()))

    # Left-join weather (convert to UTC first)
    weather_utc = _weather_to_utc(weather, timezone)
    df = df.join(weather_utc, how="left")
    logger.info(
        "  Joined weather (%d cols): %d / %d rows have weather data",
        len(weather_utc.columns),
        df[weather_utc.columns[0]].notna().sum() if len(weather_utc.columns) else 0,
        len(df),
    )

    df = df.sort_index()

    # Fill NaN weather values caused by DST fall-back ambiguous hours being
    # dropped during tz_localize. Forward-fill propagates the adjacent value
    # (equivalent to using the nearest hour's reading).
    weather_cols = [c for c in df.columns if c not in ["Datetime_EPT", "Load_Metered", "Load_Estimated"]]
    if df[weather_cols].isna().any().any():
        df[weather_cols] = df[weather_cols].ffill().bfill()

    # Sanity checks
    dup_count = df.index.duplicated().sum()
    if dup_count:
        logger.warning("Duplicate UTC timestamps after merge: %d — dropping.", dup_count)
        df = df[~df.index.duplicated(keep="first")]

    gap_count = int(
        (df.index.to_series().diff().dt.total_seconds().dropna() > 3600 * 1.5).sum()
    )
    if gap_count:
        logger.warning("Found %d gap(s) > 1.5 h in the merged index.", gap_count)

    logger.info("Merge complete: %d rows × %d cols", *df.shape)
    return df
