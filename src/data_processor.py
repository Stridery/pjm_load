# src/data_processor.py
import pandas as pd
import numpy as np
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
    df.to_csv(output_path)
    print(f"- Merged data saved to {output_path}. Shape: {df.shape}")
    return df

def clean_and_engineer(input_path, output_path):
    print("=== Cleaning and Feature Engineering ===")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    if 'POP_pct' in df.columns:
        df = df.drop(columns=['POP_pct'])

    # Handle impossible loads
    invalid_mask = df['Load'] <= 0
    if invalid_mask.any():
        df.loc[invalid_mask, 'Load'] = np.nan
        df['Load'] = df['Load'].interpolate(method='linear')

    # Temporal features
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek'] = df.index.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Holidays
    cal = calendar()
    holidays = cal.holidays(start=df.index.min(), end=df.index.max())
    df['is_holiday'] = df.index.normalize().isin(holidays).astype(int)

    # Outlier detection (3-sigma)
    df['group_mean'] = df.groupby(['month', 'hour'])['Load'].transform('mean')
    df['group_std'] = df.groupby(['month', 'hour'])['Load'].transform('std')
    df['z_score'] = (df['Load'] - df['group_mean']) / (df['group_std'] + 1e-6)
    df['is_valid'] = (df['z_score'].abs() <= 3.0).astype(int)
    
    df = df.drop(columns=['group_mean', 'group_std', 'z_score'])
    df.to_csv(output_path)
    print(f"- Cleaned data saved to {output_path}. Outliers tagged.")
    return df