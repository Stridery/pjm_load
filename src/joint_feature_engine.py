# src/joint_feature_engine.py
"""
Joint feature engine: merges cleaned CSVs from multiple zones into a single
wide dataset and builds a 3D matrix where X contains all zones' features and
y contains each zone's 24-hour load target concatenated in zone order.

Zone list is config-driven; adding a third zone only requires updating
JOINT_ZONES in config.py — no code changes needed here.
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .feature_engine import _normalize_to_24h

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Build joint cleaned CSV
# ---------------------------------------------------------------------------

def build_joint_cleaned(zones: list, data_root: str = 'data') -> str:
    """
    Load each zone's cleaned CSV, prefix every column with the zone name,
    inner-join on UTC index, and save to data/joint_{zones}/cleaned/.

    The resulting file has columns like dom_Load, dom_Temp_F, bge_Load, etc.
    Temporal columns (hour_sin, ...) are also prefixed — the matrix builder
    uses the first zone's temporal columns as shared features.

    Returns the path to the saved CSV.
    """
    joint_dataset = 'joint_' + '_'.join(zones)
    out_dir = os.path.join(data_root, joint_dataset, 'cleaned')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'joint_cleaned.csv')

    dfs = []
    for zone in zones:
        path = os.path.join(data_root, zone, 'cleaned', 'cleaned_pjm_load_weather.csv')
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.columns = [f'{zone}_{col}' for col in df.columns]
        dfs.append(df)
        logger.info('Loaded %s cleaned: %d rows', zone, len(df))

    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.join(df, how='inner')

    combined.to_csv(out_path)
    logger.info('Joint cleaned saved → %s  (%d rows × %d cols)', out_path, *combined.shape)
    return out_path


# ---------------------------------------------------------------------------
# Step 2: Build joint 3D matrix
# ---------------------------------------------------------------------------

def build_joint_timeseries_matrix(
    zones: list,
    weather_cols: dict,            # {zone: [col_name, ...]}
    joint_cleaned_path: str,
    matrix_dir: str,
    lookback_hours: int = 168,
    latest_info_hour: int = 0,
    test_frac: float = 0.1,
) -> tuple:
    """
    Build a joint 3D matrix from the merged cleaned CSV.

    X shape : (N, lookback_hours, n_features)
      n_features = Σ zone_features_per_zone + 6 shared temporal + 4 tomorrow meta
      zone features = Load_Estimated + weather cols (per zone)

    y shape : (N, 24 × len(zones))
      [zone0_h0..h23 | zone1_h0..h23 | ...]  all in scaled space

    Returns
    -------
    X_3d, y_3d, mask_3d, timestamps, y_scalers
      y_scalers: {zone: StandardScaler fitted on that zone's Load}
    """
    lb, h = lookback_hours, latest_info_hour
    x_path    = os.path.join(matrix_dir, f'X_joint_lb{lb}_h{h}.npy')
    y_path    = os.path.join(matrix_dir, f'y_joint_lb{lb}_h{h}.npy')
    mask_path = os.path.join(matrix_dir, f'mask_joint_lb{lb}_h{h}.npy')
    ts_path   = os.path.join(matrix_dir, f'ts_joint_lb{lb}_h{h}.npy')
    xs_path   = os.path.join(matrix_dir, f'x_scaler_joint_lb{lb}_h{h}.pkl')
    os.makedirs(matrix_dir, exist_ok=True)

    ys_paths = {
        z: os.path.join(matrix_dir, f'y_scaler_joint_{z}_lb{lb}_h{h}.pkl')
        for z in zones
    }

    if all(os.path.exists(p) for p in [x_path, y_path, mask_path, ts_path]):
        logger.info('=== Loading Joint 3D Matrix (lb=%d, h=%d) ===', lb, h)
        y_scalers = {z: joblib.load(ys_paths[z]) for z in zones}
        return (
            np.load(x_path),
            np.load(y_path),
            np.load(mask_path),
            np.load(ts_path, allow_pickle=True),
            y_scalers,
        )

    logger.info('=== Constructing Joint 3D Matrix (lb=%d, h=%d) ===', lb, h)
    df = pd.read_csv(joint_cleaned_path, index_col=0, parse_dates=True)
    df = df.sort_index()

    first_zone = zones[0]

    # ---- feature columns for X scaler ----
    zone_feat_cols = []
    for z in zones:
        zone_feat_cols += [f'{z}_Load_Estimated'] + [f'{z}_{w}' for w in weather_cols[z]]

    temporal_cols = [f'{first_zone}_{t}' for t in
                     ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                      'dayofweek', 'is_weekend']]

    all_feature_cols = zone_feat_cols + temporal_cols

    # ---- fit scalers on training portion ----
    split_idx = int(len(df) * (1 - test_frac))

    x_scaler = StandardScaler()
    x_scaler.fit(df.iloc[:split_idx][all_feature_cols])
    df_scaled = df.copy()
    df_scaled[all_feature_cols] = x_scaler.transform(df[all_feature_cols])
    joblib.dump(x_scaler, xs_path)

    y_scalers: dict = {}
    load_scaled: dict = {}
    for z in zones:
        ys = StandardScaler()
        ys.fit(df.iloc[:split_idx][[f'{z}_Load']])
        load_scaled[z] = ys.transform(df[[f'{z}_Load']])[:, 0]
        joblib.dump(ys, ys_paths[z])
        y_scalers[z] = ys

    # ---- EPT dates / hours from first zone ----
    ept_dt    = pd.to_datetime(df[f'{first_zone}_Datetime_EPT'])
    ept_dates = ept_dt.dt.date.values
    ept_hours = ept_dt.dt.hour.values

    data_array     = df_scaled[all_feature_cols].values
    is_valid_arrs  = {z: df[f'{z}_is_valid'].values for z in zones}

    unique_days = np.unique(ept_dates)
    X_list, y_list, mask_list, ts_list = [], [], [], []

    for i in tqdm(range(len(unique_days) - 1), desc='Building Joint 3D Samples'):
        today, tomorrow = unique_days[i], unique_days[i + 1]

        if latest_info_hour <= 9:
            cutoff_date, cutoff_hour = today, latest_info_hour
        else:
            if i == 0:
                continue
            cutoff_date, cutoff_hour = unique_days[i - 1], latest_info_hour

        cutoff_rows = np.where(
            (ept_dates == cutoff_date) & (ept_hours == cutoff_hour)
        )[0]
        if len(cutoff_rows) == 0 or cutoff_rows[0] < lookback_hours:
            continue
        cutoff_pos = cutoff_rows[0]

        tmrw_pos = np.where(ept_dates == tomorrow)[0]

        # Build y: normalize each zone's load to 24h, then concat
        y_parts = []
        skip = False
        for z in zones:
            norm = _normalize_to_24h(load_scaled[z][tmrw_pos], ept_hours[tmrw_pos])
            if norm is None:
                skip = True
                break
            y_parts.append(norm)
        if skip:
            continue
        y_window = np.concatenate(y_parts).astype('float32')

        # Valid only when every zone has clean data for tomorrow
        is_seq_valid = all(is_valid_arrs[z][tmrw_pos].all() for z in zones)

        # X lookback window
        X_window = data_array[cutoff_pos - lookback_hours: cutoff_pos].copy()

        # Append tomorrow meta (EPT-based, from first zone)
        row = df.iloc[tmrw_pos[0]]
        tmrw_dow = row[f'{first_zone}_dayofweek']
        tmrw_meta = np.array([
            np.sin(2 * np.pi * tmrw_dow / 7),
            np.cos(2 * np.pi * tmrw_dow / 7),
            float(row[f'{first_zone}_is_weekend']),
            float(row[f'{first_zone}_is_holiday']),
        ], dtype='float32')
        X_window = np.concatenate(
            [X_window, np.tile(tmrw_meta, (lookback_hours, 1))], axis=1
        )

        X_list.append(X_window)
        y_list.append(y_window)
        mask_list.append(is_seq_valid)
        ts_list.append(tomorrow)

    X_3d        = np.array(X_list, dtype='float32')
    y_3d        = np.array(y_list, dtype='float32')
    mask_3d     = np.array(mask_list, dtype=bool)
    timestamps  = np.array(ts_list)

    np.save(x_path, X_3d)
    np.save(y_path, y_3d)
    np.save(mask_path, mask_3d)
    np.save(ts_path, timestamps)

    logger.info('Done. %d joint daily samples | valid: %d', len(X_3d), mask_3d.sum())
    return X_3d, y_3d, mask_3d, timestamps, y_scalers
