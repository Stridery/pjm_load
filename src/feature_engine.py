# src/feature_engine.py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib
from src.config import TREE_FEATURE_CONFIG, TRANSFORMER_FEATURE_CONFIG

# Plain windowed features, used as-is (monotonic w.r.t. load -> no split needed).
WINDOWED_FEATURES = ['Load_Estimated']

# Piecewise "degree-day" style splits. Each source column is split at a balance
# point into a below-threshold and an above-threshold non-negative feature, so a
# U-shaped relation to load becomes two monotonic features. `square=True` then
# squares both, capturing the super-linear cooling/heating response to load.
# (source_column, balance_point, below_name, above_name, square)
FEATURE_SPLITS = [
    ('ApparentTemp_F',     65.0,  'HDD',      'CDD',      True),  # heating / cooling degree, squared
    ('SolarRadiation_Wm2', 170.0, 'SOLAR_LO', 'SOLAR_HI', True),
]


def _split_below_above(values, balance, square=False):
    """Split values at `balance` into (below, above), both non-negative.
    If `square`, square both sides."""
    below = np.maximum(0.0, balance - values)
    above = np.maximum(0.0, values - balance)
    if square:
        below, above = below ** 2, above ** 2
    return below, above


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

    # Surgery: windowed features = plain features + piecewise weather splits.
    feature_cols = list(WINDOWED_FEATURES)
    arrays       = [df_final[c].values for c in WINDOWED_FEATURES]
    for src, bal, lo_name, hi_name, square in FEATURE_SPLITS:
        below, above = _split_below_above(df_final[src].values, bal, square)
        feature_cols += [lo_name, hi_name]
        arrays       += [below, above]
    data_array = np.column_stack(arrays)
    load_raw     = df_final['Load'].values
    valid_raw    = df_final['is_valid'].values

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
        if len(cutoff_rows) == 0 or cutoff_rows[0] < lookback_hours:
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

        today_meta = df_final[ept_dates == today].iloc[0]
        # Surgery: keep only the cyclical month encoding. hour_sin/cos are dropped
        # (constant — every 2D sample is stamped at hour 0, so they carry no signal).
        # is_target_valid is not a model feature — it is a control column used by
        # get_train_test_split for masking and dropped before fitting.
        f.update({
            'today_month_sin':  today_meta['month_sin'],
            'today_month_cos':  today_meta['month_cos'],
            'is_target_valid':  tmrw_valid,
        })

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
    scaler_path    = os.path.join(matrix_dir, f'scaler_ts_lb{lookback_hours}_h{latest_info_hour}.pkl')
    y_scaler_path  = os.path.join(matrix_dir, f'y_scaler_lb{lookback_hours}_h{latest_info_hour}.pkl')
    os.makedirs(matrix_dir, exist_ok=True)

    if all(os.path.exists(p) for p in [x_path, y_path, mask_path, timestamp_path]):
        print(f"=== Loading 3D Matrix (lb={lookback_hours}, h={latest_info_hour}) ===")
        return np.load(x_path), np.load(y_path), np.load(mask_path), np.load(timestamp_path, allow_pickle=True)

    print(f"=== Constructing 3D Matrix (lb={lookback_hours}, h={latest_info_hour}) ===")
    df = pd.read_csv(cleaned_path, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Surgery: plain features + piecewise weather splits + cyclical month.
    # hour_sin/cos are dropped: the window always ends at the fixed cutoff hour, so
    # each timestep maps to the same hour-of-day across samples -> purely positional
    # (cross-sample std ~0.03), redundant with the sequence model's own ordering.
    split_cols = []
    for src, bal, lo_name, hi_name, square in FEATURE_SPLITS:
        df[lo_name], df[hi_name] = _split_below_above(df[src], bal, square)
        split_cols += [lo_name, hi_name]
    feature_cols = list(WINDOWED_FEATURES) + split_cols + ['month_sin', 'month_cos']
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
    is_valid_array = df['is_valid'].values
    timestamps    = df.index
    unique_days   = np.unique(ept_dates)

    X_list, y_list, valid_mask_list, timestamps_list = [], [], [], []

    for i in tqdm(range(len(unique_days) - 1), desc="Building 3D Samples"):
        today, tomorrow = unique_days[i], unique_days[i + 1]

        if latest_info_hour <= 9:
            cutoff_date, cutoff_hour = today, latest_info_hour
        else:
            if i == 0:
                continue
            cutoff_date, cutoff_hour = unique_days[i - 1], latest_info_hour

        cutoff_rows = np.where((ept_dates == cutoff_date) & (ept_hours == cutoff_hour))[0]
        if len(cutoff_rows) == 0 or cutoff_rows[0] < lookback_hours:
            continue
        cutoff_pos = cutoff_rows[0]

        tmrw_pos = np.where(ept_dates == tomorrow)[0]
        y_window = _normalize_to_24h(load_array[tmrw_pos], ept_hours[tmrw_pos])
        if y_window is None:
            continue

        is_seq_valid = bool(is_valid_array[tmrw_pos].all())
        # Surgery: no tomorrow-calendar broadcast — only the kept feature_cols.
        X_window     = data_array[cutoff_pos - lookback_hours : cutoff_pos]

        X_list.append(X_window)
        y_list.append(y_window)
        valid_mask_list.append(is_seq_valid)
        timestamps_list.append(tomorrow)  # EPT date, consistent with tree-model index

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
