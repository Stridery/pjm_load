# src/feature_engine.py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib
from src.config import TREE_FEATURE_CONFIG, TRANSFORMER_FEATURE_CONFIG


def build_or_load_matrix(cleaned_path, matrix_dir, lookback_hours=None, latest_info_hour=None):
    """生成 2D CSV 矩阵，特征是打平的 (Flattened)"""
    if lookback_hours is None:
        lookback_hours = TREE_FEATURE_CONFIG['lookback_hours']
    if latest_info_hour is None:
        latest_info_hour = TREE_FEATURE_CONFIG['latest_info_hour']

    x_path = os.path.join(matrix_dir, f'X_opt_lb{lookback_hours}_h{latest_info_hour}.csv')
    y_path = os.path.join(matrix_dir, f'y_opt_lb{lookback_hours}_h{latest_info_hour}.csv')

    if os.path.exists(x_path) and os.path.exists(y_path):
        print("=== Loading Pre-built 2D Matrix ===")
        return pd.read_csv(x_path, index_col=0, parse_dates=True), \
               pd.read_csv(y_path, index_col=0, parse_dates=True)

    print("=== Constructing 2D Matrix from Cleaned Data ===")
    df_final = pd.read_csv(cleaned_path, index_col=0, parse_dates=True)
    df_final = df_final.sort_index()
    X_list, y_list = [], []
    unique_days = pd.Series(df_final.index.date).unique()

    weather_cols = ['Temp_F', 'Dewpoint_F', 'HeatIndex_F', 'SolarRadiation_Wm2', 'Windchill_F',
                    'WindSpeed_mph', 'WindDirection_deg', 'CloudCover_pct', 'Precip_in', 'RelativeHumidity_pct']
    feature_cols = ['Load'] + weather_cols
    data_array = df_final[feature_cols].values

    for i in tqdm(range(len(unique_days) - 1), desc="Building 2D Samples"):
        today, tomorrow = unique_days[i], unique_days[i+1]

        # 确定cutoff时间点：<=9取当天该小时，>9取前一天该小时
        if latest_info_hour <= 9:
            cutoff_date, cutoff_hour = today, latest_info_hour
        else:
            if i == 0:
                continue
            cutoff_date, cutoff_hour = unique_days[i - 1], latest_info_hour

        cutoff_rows = np.where((df_final.index.date == cutoff_date) & (df_final.index.hour == cutoff_hour))[0]
        if len(cutoff_rows) == 0 or cutoff_rows[0] < lookback_hours:
            continue
        cutoff_pos = cutoff_rows[0]

        y_tomorrow = df_final.loc[df_final.index.date == tomorrow, 'Load'].values
        if len(y_tomorrow) != 24: continue

        tmrw_valid = 1 if df_final.loc[df_final.index.date == tomorrow, 'is_valid'].all() else 0

        past_window = data_array[cutoff_pos - lookback_hours : cutoff_pos]  # (lookback_hours, n_features)

        f = {'timestamp': tomorrow}
        for k in range(lookback_hours):
            f[f'load_h{k}'] = past_window[k, 0]
        for j, col in enumerate(weather_cols):
            for k in range(lookback_hours):
                f[f'{col}_h{k}'] = past_window[k, j + 1]

        today_meta = df_final.loc[df_final.index.date == today].iloc[0]
        tmrw_meta = df_final.loc[df_final.index.date == tomorrow].iloc[0]
        f.update({
            'today_month': today_meta['month'],
            'today_hour_sin': today_meta['hour_sin'], 'today_hour_cos': today_meta['hour_cos'],
            'today_month_sin': today_meta['month_sin'], 'today_month_cos': today_meta['month_cos'],
            'today_dayofweek': today_meta['dayofweek'],
            'tmrw_is_weekend': tmrw_meta['is_weekend'], 'tmrw_is_holiday': tmrw_meta['is_holiday'],
            'is_target_valid': tmrw_valid
        })

        y_dict = {'timestamp': tomorrow}
        for h in range(24): y_dict[f'h{h}'] = y_tomorrow[h]
        X_list.append(f); y_list.append(y_dict)

    X_opt = pd.DataFrame(X_list).set_index('timestamp')
    y_opt = pd.DataFrame(y_list).set_index('timestamp')
    os.makedirs(matrix_dir, exist_ok=True)
    X_opt.to_csv(x_path); y_opt.to_csv(y_path)
    return X_opt, y_opt


def build_timeseries_matrix(cleaned_path, matrix_dir, lookback_hours=None, latest_info_hour=None):
    if lookback_hours is None:
        lookback_hours = TRANSFORMER_FEATURE_CONFIG['lookback_hours']
    if latest_info_hour is None:
        latest_info_hour = TRANSFORMER_FEATURE_CONFIG['latest_info_hour']

    x_path = os.path.join(matrix_dir, f'X_3d_lb{lookback_hours}_h{latest_info_hour}.npy')
    y_path = os.path.join(matrix_dir, f'y_3d_lb{lookback_hours}_h{latest_info_hour}.npy')
    mask_path = os.path.join(matrix_dir, f'mask_3d_lb{lookback_hours}_h{latest_info_hour}.npy')
    timestamp_path = os.path.join(matrix_dir, f'timestamps_3d_lb{lookback_hours}_h{latest_info_hour}.npy')
    scaler_path = os.path.join(matrix_dir, f'scaler_ts_lb{lookback_hours}_h{latest_info_hour}.pkl')
    os.makedirs(matrix_dir, exist_ok=True)

    if all(os.path.exists(p) for p in [x_path, y_path, mask_path, timestamp_path]):
        print(f"=== Loading 3D Matrix (lb={lookback_hours}, h={latest_info_hour}) ===")
        return np.load(x_path), np.load(y_path), np.load(mask_path), np.load(timestamp_path, allow_pickle=True)

    print(f"=== Constructing 3D Matrix (lb={lookback_hours}, h={latest_info_hour}) ===")
    df = pd.read_csv(cleaned_path, index_col=0, parse_dates=True)
    df = df.sort_index()

    feature_cols = [
        'Load',
        'Temp_F', 'Dewpoint_F', 'HeatIndex_F',
        'SolarRadiation_Wm2', 'Windchill_F',
        'WindSpeed_mph', 'WindDirection_deg',
        'CloudCover_pct', 'Precip_in',
        'RelativeHumidity_pct',
        'hour_sin', 'hour_cos',
        'month_sin', 'month_cos',
        'dayofweek',
        'is_weekend'
    ]
    split_idx = int(len(df) * (1 - TRANSFORMER_FEATURE_CONFIG['test_frac']))
    scaler = StandardScaler()
    scaler.fit(df.iloc[:split_idx][feature_cols])
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    joblib.dump(scaler, scaler_path)

    X_list, y_list, valid_mask_list, timestamps_list = [], [], [], []
    data_array = df_scaled[feature_cols].values
    load_array = df_scaled['Load'].values
    is_valid_array = df['is_valid'].values
    timestamps = df.index
    unique_days = pd.Series(timestamps.date).unique()

    for i in tqdm(range(len(unique_days) - 1), desc="Building 3D Samples"):
        today, tomorrow = unique_days[i], unique_days[i + 1]

        # cutoff：<=9取当天该小时，>9取前一天该小时
        if latest_info_hour <= 9:
            cutoff_date, cutoff_hour = today, latest_info_hour
        else:
            if i == 0:
                continue
            cutoff_date, cutoff_hour = unique_days[i - 1], latest_info_hour

        cutoff_rows = np.where((timestamps.date == cutoff_date) & (timestamps.hour == cutoff_hour))[0]
        if len(cutoff_rows) == 0 or cutoff_rows[0] < lookback_hours:
            continue
        cutoff_pos = cutoff_rows[0]

        # y永远是tomorrow（unique_days[i+1]）的完整24小时，从0点开始
        tmrw_midnight_rows = np.where((timestamps.date == tomorrow) & (timestamps.hour == 0))[0]
        if len(tmrw_midnight_rows) == 0 or tmrw_midnight_rows[0] + 24 > len(df):
            continue
        target_start = tmrw_midnight_rows[0]

        X_window = data_array[cutoff_pos - lookback_hours : cutoff_pos]
        y_window = load_array[target_start : target_start + 24]
        is_seq_valid = is_valid_array[target_start : target_start + 24].all()

        X_list.append(X_window)
        y_list.append(y_window)
        valid_mask_list.append(is_seq_valid)
        timestamps_list.append(timestamps[target_start])

    X_3d = np.array(X_list, dtype='float32')
    y_3d = np.array(y_list, dtype='float32')
    mask_3d = np.array(valid_mask_list, dtype=bool)
    timestamps_3d = np.array(timestamps_list)

    np.save(x_path, X_3d)
    np.save(y_path, y_3d)
    np.save(mask_path, mask_3d)
    np.save(timestamp_path, timestamps_3d)

    print(f"Done. Generated {len(X_3d)} daily samples. Valid samples: {mask_3d.sum()}")
    return X_3d, y_3d, mask_3d, timestamps_3d

def _split_indices(n, strategy, test_frac, random_state=42):
    n_test = int(n * test_frac)
    if strategy == 'head':
        test_pos = np.arange(n_test)
        train_pos = np.arange(n_test, n)
    elif strategy == 'tail':
        test_pos = np.arange(n - n_test, n)
        train_pos = np.arange(n - n_test)
    elif strategy == 'random':
        test_pos = np.random.default_rng(random_state).choice(n, size=n_test, replace=False)
        train_pos = np.setdiff1d(np.arange(n), test_pos)
    else:
        raise ValueError(f"Unknown split_strategy '{strategy}'. Choose 'head', 'tail', or 'random'.")
    return train_pos, test_pos


def get_train_test_split(X_opt, y_opt, strategy=None, test_frac=None, random_state=None):
    if strategy is None:
        strategy = TREE_FEATURE_CONFIG['split_strategy']
    if test_frac is None:
        test_frac = TREE_FEATURE_CONFIG['test_frac']
    if random_state is None:
        random_state = TREE_FEATURE_CONFIG['random_state']

    train_pos, test_pos = _split_indices(len(X_opt), strategy, test_frac, random_state)
    train_idx = X_opt.index[train_pos]
    test_idx = X_opt.index[test_pos]

    X_test_raw = X_opt.loc[test_idx]
    y_test = y_opt.loc[test_idx]
    X_train_raw = X_opt.loc[train_idx]
    y_train_raw = y_opt.loc[train_idx]

    train_mask = X_train_raw['is_target_valid'] == 1
    X_train = X_train_raw[train_mask].drop(columns=['is_target_valid'])
    y_train = y_train_raw[train_mask]

    X_test = X_test_raw.drop(columns=['is_target_valid'])
    return X_train, y_train, X_test, y_test