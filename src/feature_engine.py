# src/feature_engine.py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib
from src.config import TREE_FEATURE_CONFIG, TRANSFORMER_FEATURE_CONFIG, WEATHER_COLS, EMBARGO_DAYS
from src.macro_features import compute_macro_features, MACRO_FEATURE_NAMES, MACRO_WINDOW_HOURS


def _normalize_to_24h(load_vals, ept_hour_vals):
    """
    Normalize a DST transition day's load to exactly 24 EPT-hour slots.

    Fall Back  (25 rows): the repeated 1:00 AM is averaged into one value.
    Spring Fwd (23 rows): the missing 2:00 AM is linearly interpolated.
    Returns a length-24 float array, or None for unexpected day lengths.
    """
    n = len(load_vals)
    if n == 24:
        return np.array(load_vals, dtype=float)

    if n == 25:  # Fall Back: average duplicate EPT hour
        sums   = np.zeros(24)
        counts = np.zeros(24, dtype=int)
        for val, h in zip(load_vals, ept_hour_vals):
            sums[h]   += val
            counts[h] += 1
        return sums / np.maximum(counts, 1)

    if n == 23:  # Spring Forward: interpolate missing EPT hour (2:00 AM)
        hour_to_val = {int(h): float(v) for h, v in zip(ept_hour_vals, load_vals)}
        vals = np.array([hour_to_val.get(h, np.nan) for h in range(24)])
        nans = np.isnan(vals)
        if nans.any():
            x = np.arange(24)
            vals[nans] = np.interp(x[nans], x[~nans], vals[~nans])
        return vals

    return None  # unexpected length — caller skips this day


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

    ept_dt    = pd.to_datetime(df_final['Datetime_EPT'])
    ept_dates = ept_dt.dt.date.values
    ept_hours = ept_dt.dt.hour.values

    weather_cols = WEATHER_COLS
    feature_cols = ['Load_Estimated'] + weather_cols
    data_array   = df_final[feature_cols].values
    load_raw     = df_final['Load'].values
    est_raw      = df_final['Load_Estimated'].values   # for macro features
    valid_raw    = df_final['is_valid'].values

    min_history = max(lookback_hours, MACRO_WINDOW_HOURS)   # lookback + 3-week macro
    unique_days = np.unique(ept_dates)
    X_list, y_list = [], []

    for i in tqdm(range(len(unique_days) - 1), desc="Building 2D Samples"):
        today, tomorrow = unique_days[i], unique_days[i + 1]

        if latest_info_hour <= 9:
            cutoff_date, cutoff_hour = today, latest_info_hour
        else:
            if i == 0:
                continue
            cutoff_date, cutoff_hour = unique_days[i - 1], latest_info_hour

        cutoff_rows = np.where((ept_dates == cutoff_date) & (ept_hours == cutoff_hour))[0]
        if len(cutoff_rows) == 0 or cutoff_rows[0] < min_history:
            continue
        cutoff_pos = cutoff_rows[0]

        tmrw_pos  = np.where(ept_dates == tomorrow)[0]
        y_tomorrow = _normalize_to_24h(load_raw[tmrw_pos], ept_hours[tmrw_pos])
        if y_tomorrow is None:
            continue

        tmrw_valid = 1 if valid_raw[tmrw_pos].all() else 0
        past_window = data_array[cutoff_pos - lookback_hours : cutoff_pos]

        f = {'timestamp': tomorrow}
        for j, col in enumerate(feature_cols):
            for k in range(lookback_hours):
                f[f'{col.lower()}_h{k}'] = past_window[k, j]

        # Forecast-day (tomorrow) calendar features appended after the flattened
        # window. Hour features dropped; calendar taken from the forecast day
        # (mirrors the 3D matrix). is_target_valid kept for train-time filtering.
        tmrw_meta = df_final.iloc[tmrw_pos[0]]
        f.update({
            'tmrw_month_sin':  tmrw_meta['month_sin'],
            'tmrw_month_cos':  tmrw_meta['month_cos'],
            'tmrw_dow_sin':    np.sin(2 * np.pi * tmrw_meta['dayofweek'] / 7),
            'tmrw_dow_cos':    np.cos(2 * np.pi * tmrw_meta['dayofweek'] / 7),
            'tmrw_is_weekend': tmrw_meta['is_weekend'],
            'tmrw_is_holiday': tmrw_meta['is_holiday'],
        })

        # 3-week macro features (raw MW, unscaled — trees are scale-invariant).
        macro_raw = compute_macro_features(load_raw, est_raw, ept_hours, cutoff_pos)
        for nm, val in zip(MACRO_FEATURE_NAMES, macro_raw):
            f[nm] = float(val)
        f['is_target_valid'] = tmrw_valid

        y_dict = {'timestamp': tomorrow}
        for h in range(24):
            y_dict[f'h{h}'] = y_tomorrow[h]
        X_list.append(f)
        y_list.append(y_dict)

    X_opt = pd.DataFrame(X_list).set_index('timestamp')
    y_opt = pd.DataFrame(y_list).set_index('timestamp')
    os.makedirs(matrix_dir, exist_ok=True)
    X_opt.to_csv(x_path)
    y_opt.to_csv(y_path)
    return X_opt, y_opt


def build_timeseries_matrix(cleaned_path, matrix_dir, lookback_hours=None, latest_info_hour=None):
    if lookback_hours is None:
        lookback_hours = TRANSFORMER_FEATURE_CONFIG['lookback_hours']
    if latest_info_hour is None:
        latest_info_hour = TRANSFORMER_FEATURE_CONFIG['latest_info_hour']

    x_path         = os.path.join(matrix_dir, f'X_3d_lb{lookback_hours}_h{latest_info_hour}.npy')
    y_path         = os.path.join(matrix_dir, f'y_3d_lb{lookback_hours}_h{latest_info_hour}.npy')
    mask_path      = os.path.join(matrix_dir, f'mask_3d_lb{lookback_hours}_h{latest_info_hour}.npy')
    timestamp_path = os.path.join(matrix_dir, f'timestamps_3d_lb{lookback_hours}_h{latest_info_hour}.npy')
    scaler_path       = os.path.join(matrix_dir, f'scaler_ts_lb{lookback_hours}_h{latest_info_hour}.pkl')
    y_scaler_path     = os.path.join(matrix_dir, f'y_scaler_lb{lookback_hours}_h{latest_info_hour}.pkl')
    macro_scaler_path = os.path.join(matrix_dir, f'macro_scaler_lb{lookback_hours}_h{latest_info_hour}.pkl')
    os.makedirs(matrix_dir, exist_ok=True)

    if all(os.path.exists(p) for p in [x_path, y_path, mask_path, timestamp_path]):
        print(f"=== Loading 3D Matrix (lb={lookback_hours}, h={latest_info_hour}) ===")
        return np.load(x_path), np.load(y_path), np.load(mask_path), np.load(timestamp_path, allow_pickle=True)

    print(f"=== Constructing 3D Matrix (lb={lookback_hours}, h={latest_info_hour}) ===")
    df = pd.read_csv(cleaned_path, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Lookback (per-timestep) features: only continuous physical signals.
    # Calendar/cyclic features are NOT put in the window — hour-of-day is implicit
    # in the 24-dim output (and the expert routing), and month/day-of-week belong
    # to the FORECAST day, so they are appended below as broadcast constants.
    feature_cols = ['Load_Estimated'] + WEATHER_COLS
    split_idx = int(len(df) * (1 - TRANSFORMER_FEATURE_CONFIG['test_frac']))

    scaler = StandardScaler()
    scaler.fit(df.iloc[:split_idx][feature_cols])
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    joblib.dump(scaler, scaler_path)

    y_scaler = StandardScaler()
    y_scaler.fit(df.iloc[:split_idx][['Load']])
    load_scaled = y_scaler.transform(df[['Load']])[:, 0]
    joblib.dump(y_scaler, y_scaler_path)

    ept_dt    = pd.to_datetime(df['Datetime_EPT'])
    ept_dates = ept_dt.dt.date.values
    ept_hours = ept_dt.dt.hour.values

    data_array    = df_scaled[feature_cols].values
    load_array    = load_scaled
    actual_raw    = df['Load'].values              # raw MW, for macro features
    est_raw       = df['Load_Estimated'].values    # raw MW, for macro features
    is_valid_array = df['is_valid'].values
    timestamps    = df.index
    unique_days   = np.unique(ept_dates)

    # Enough history for both the lookback window and the 3-week macro features.
    min_history = max(lookback_hours, MACRO_WINDOW_HOURS)

    X_list, macro_list, y_list, valid_mask_list, timestamps_list = [], [], [], [], []

    for i in tqdm(range(len(unique_days) - 1), desc="Building 3D Samples"):
        today, tomorrow = unique_days[i], unique_days[i + 1]

        if latest_info_hour <= 9:
            cutoff_date, cutoff_hour = today, latest_info_hour
        else:
            if i == 0:
                continue
            cutoff_date, cutoff_hour = unique_days[i - 1], latest_info_hour

        cutoff_rows = np.where((ept_dates == cutoff_date) & (ept_hours == cutoff_hour))[0]
        if len(cutoff_rows) == 0 or cutoff_rows[0] < min_history:
            continue
        cutoff_pos = cutoff_rows[0]

        tmrw_pos = np.where(ept_dates == tomorrow)[0]
        y_window = _normalize_to_24h(load_array[tmrw_pos], ept_hours[tmrw_pos])
        if y_window is None:
            continue

        is_seq_valid = bool(is_valid_array[tmrw_pos].all())
        X_window     = data_array[cutoff_pos - lookback_hours : cutoff_pos]
        macro_raw    = compute_macro_features(actual_raw, est_raw, ept_hours, cutoff_pos)

        # Forecast-day (tomorrow) calendar features, broadcast across the window.
        # These condition the prediction on the day being forecast, not the lookback.
        tmrw_row = df.iloc[tmrw_pos[0]]
        tmrw_dow = tmrw_row['dayofweek']
        tmrw_meta = np.array([
            tmrw_row['month_sin'],                     # forecast-day month (season phase)
            tmrw_row['month_cos'],
            np.sin(2 * np.pi * tmrw_dow / 7),          # forecast-day day-of-week
            np.cos(2 * np.pi * tmrw_dow / 7),
            float(tmrw_row['is_weekend']),
            float(tmrw_row['is_holiday']),
        ], dtype='float32')
        tmrw_broadcast = np.tile(tmrw_meta, (lookback_hours, 1))
        X_window = np.concatenate([X_window, tmrw_broadcast], axis=1)

        X_list.append(X_window)
        macro_list.append(macro_raw)
        y_list.append(y_window)
        valid_mask_list.append(is_seq_valid)
        timestamps_list.append(tomorrow)  # EPT date, consistent with tree-model index

    X_3d = np.array(X_list, dtype='float32')                    # (N, 168, seq+calendar)
    y_3d = np.array(y_list, dtype='float32')
    mask_3d = np.array(valid_mask_list, dtype=bool)
    timestamps_3d = np.array(timestamps_list)

    # Macro features: standardize (fit on the first 1-test_frac samples, like the
    # sequence scaler), then broadcast across the window and append after calendar.
    macro_arr = np.array(macro_list, dtype='float32')          # (N, 7), raw
    m_split = int(len(macro_arr) * (1 - TRANSFORMER_FEATURE_CONFIG['test_frac']))
    macro_scaler = StandardScaler().fit(macro_arr[:m_split])
    macro_scaled = macro_scaler.transform(macro_arr).astype('float32')
    joblib.dump(macro_scaler, macro_scaler_path)
    macro_bc = np.repeat(macro_scaled[:, None, :], lookback_hours, axis=1)   # (N, 168, 7)
    X_3d = np.concatenate([X_3d, macro_bc], axis=2)

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


def apply_embargo(train_idx, val_idx, embargo):
    """Drop the last `embargo` samples of train and of val (chronological/tail
    split), leaving a gap before val and before test respectively so a sample's
    3-week feature window never straddles a split boundary. No-op if embargo<=0.
    """
    if not embargo or embargo <= 0:
        return train_idx, val_idx
    train_idx = np.sort(np.asarray(train_idx))
    val_idx   = np.sort(np.asarray(val_idx))
    train_idx = train_idx[:-embargo] if len(train_idx) > embargo else train_idx[:0]
    val_idx   = val_idx[:-embargo]   if len(val_idx)   > embargo else val_idx[:0]
    return train_idx, val_idx


def get_train_test_split(X_opt, y_opt, strategy=None, test_frac=None, random_state=None):
    if strategy is None:
        strategy = TREE_FEATURE_CONFIG['split_strategy']
    if test_frac is None:
        test_frac = TREE_FEATURE_CONFIG['test_frac']
    if random_state is None:
        random_state = TREE_FEATURE_CONFIG['random_state']

    train_pos, test_pos = _split_indices(len(X_opt), strategy, test_frac, random_state)
    # Embargo: drop the last EMBARGO_DAYS train samples so the 3-week feature
    # window of no train sample overlaps the test period (test kept intact).
    train_pos, _ = apply_embargo(train_pos, test_pos, EMBARGO_DAYS)
    train_idx = X_opt.index[train_pos]
    test_idx  = X_opt.index[test_pos]

    X_test_raw  = X_opt.loc[test_idx]
    y_test      = y_opt.loc[test_idx]
    X_train_raw = X_opt.loc[train_idx]
    y_train_raw = y_opt.loc[train_idx]

    train_mask = X_train_raw['is_target_valid'] == 1
    X_train = X_train_raw[train_mask].drop(columns=['is_target_valid'])
    y_train = y_train_raw[train_mask]

    X_test = X_test_raw.drop(columns=['is_target_valid'])
    return X_train, y_train, X_test, y_test
