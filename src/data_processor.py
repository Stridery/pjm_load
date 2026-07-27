# src/data_processor.py
import glob
import pandas as pd
import numpy as np
import os
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

def merge_raw_data(load_path, weather_path, output_path):
    print("=== Merging Raw Data ===")
    df_load = pd.read_csv(load_path)
    df_load['date'] = pd.to_datetime(df_load['date'])
    df_load = df_load.set_index('date')
    df_load.rename(columns={'load': 'Load'}, inplace=True)

    df_weather = pd.read_csv(weather_path)
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather = df_weather.set_index('time')

    df = df_load.join(df_weather, how='inner')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"- Merged data saved to {output_path}. Shape: {df.shape}")
    return df

FC_LEAD_DAYS = (1, 2)                 # the two days a forecast issued on D covers
FC_HOURS_PER_ISSUE = 24 * len(FC_LEAD_DAYS)


def clean_forecast(raw_forecast_dir, output_path):
    """Stitch the per-year raw forecast shards into one table the feature layer can use.

    The raw files are split by the year of the VALID time, which cuts issue dates in half
    at every year boundary: a forecast issued 2021-12-30 has its lead-1 day in
    forecast_2021.csv and its lead-2 day in forecast_2022.csv. Nothing downstream should
    have to know that, so it is reassembled here — the same role Step 4's join plays for
    the load and weather series.

    Cleaning, in the same spirit as clean_and_engineer: FLAG, don't drop. Whole days go
    missing from the archive (2023-12-27 → 2024-01-18 is a three-week hole), and a day
    forecast for only part of its 48 h cannot be turned into features. Rather than delete
    those rows and leave the caller wondering why a date vanished, every row carries
    `is_complete` for its issue date, and the feature builder filters on it — exactly how
    `is_valid` works for the load.

    The index is renamed to Datetime_EPT because that is what the matrix builders key on
    (`ept_dt = pd.to_datetime(df['Datetime_EPT'])`); the forecast table joins to the rest
    of the pipeline by local date and hour, and the column name should say so.

    Returns the stitched DataFrame (empty if there are no shards).
    """
    print("=== Consolidating Weather Forecasts ===")
    files = sorted(glob.glob(os.path.join(raw_forecast_dir, "forecast_*.csv")))
    if not files:
        print(f"- No forecast shards in {raw_forecast_dir} — skipped.")
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(f, index_col=0, parse_dates=True) for f in files])
    df.index.name = "Datetime_EPT"
    df["issue_date"] = pd.to_datetime(df["issue_date"]).dt.date

    # A given (valid hour, lead) is forecast exactly once. A duplicate would mean two
    # shards disagree about the same number, so say so rather than silently keeping one.
    dup = df.reset_index().duplicated(subset=["Datetime_EPT", "lead_day"]).sum()
    if dup:
        print(f"- WARNING: {dup} duplicate (valid hour, lead) row(s) across shards — keeping first.")
        df = df[~df.reset_index().duplicated(subset=["Datetime_EPT", "lead_day"]).values]

    # Sorted by what the file is FOR: walk forward through issue dates, and within each,
    # through the 48 h it covers. Sorting by valid time instead interleaves the two leads.
    df = df.reset_index().sort_values(["issue_date", "Datetime_EPT"]).set_index("Datetime_EPT")

    # Completeness is judged on TEMPERATURE alone. The other nine columns are empty before
    # 2024-01-19 (the archive simply does not go back further for them), so requiring them
    # would mark three quarters of the history incomplete over variables that barely move load.
    hours = df.groupby("issue_date")["Temp_F"].transform(lambda s: s.notna().sum())
    df["is_complete"] = (hours == FC_HOURS_PER_ISSUE).astype(int)

    front = ["issue_date", "lead_day", "is_complete"]
    df = df[front + [c for c in df.columns if c not in front]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)

    per_issue = df.groupby("issue_date")["is_complete"].max()
    n_ok = int(per_issue.sum())
    print(f"- Forecast data saved to {output_path}. "
          f"{len(df)} rows, {len(per_issue)} issue dates "
          f"({n_ok} complete / {len(per_issue) - n_ok} partial), "
          f"{per_issue.index.min()} → {per_issue.index.max()}")
    return df


def clean_and_engineer(input_path, output_path):
    print("=== Cleaning and Feature Engineering ===")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    if 'POP_pct' in df.columns:
        df = df.drop(columns=['POP_pct'])

    # The most recent hours have Load_Estimated but no verified Load yet (metered
    # lags ~7 days). Those rows are the prediction set — the model's inputs all come
    # from Load_Estimated + weather, only the label needs metered. Record it before
    # anything touches the Load column.
    df['has_label'] = df['Load'].notna().astype(int)

    # Handle impossible loads. limit_area='inside' is load-bearing: a plain
    # interpolate() pads TRAILING NaNs with the last valid value, which would
    # fabricate labels for the entire unlabelled tail — silently, and only when a
    # Load<=0 row happens to exist to trigger this branch.
    invalid_mask = df['Load'] <= 0
    if invalid_mask.any():
        df.loc[invalid_mask, 'Load'] = np.nan
        df['Load'] = df['Load'].interpolate(method='linear', limit_area='inside')

    # Temporal features — derived from EPT (local Eastern time), not UTC index
    ept = pd.to_datetime(df['Datetime_EPT'])
    df['hour'] = ept.dt.hour
    df['month'] = ept.dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek'] = ept.dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Holidays
    cal = calendar()
    holidays = cal.holidays(start=ept.min(), end=ept.max())
    df['is_holiday'] = ept.dt.normalize().isin(holidays).astype(int)

    # Outlier detection (3-sigma)
    df['group_mean'] = df.groupby(['month', 'hour'])['Load'].transform('mean')
    df['group_std'] = df.groupby(['month', 'hour'])['Load'].transform('std')
    df['z_score'] = (df['Load'] - df['group_mean']) / (df['group_std'] + 1e-6)
    # is_valid is the TRAINING mask: a row must be both labelled and non-outlier.
    # Unlabelled rows already fall out via NaN z-score comparing False, but say so
    # explicitly — the mask must never let a NaN label reach the loss.
    df['is_valid'] = ((df['z_score'].abs() <= 3.0) & (df['has_label'] == 1)).astype(int)
    
    df = df.drop(columns=['group_mean', 'group_std', 'z_score'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"- Cleaned data saved to {output_path}. Outliers tagged.")
    return df