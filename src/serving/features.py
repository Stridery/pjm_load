"""
Assemble the ONE forecast sample for a target day, from a short recent window.

This is the training feature loop (src/feature_engine.build_timeseries_matrix) run for a single
day, with three deliberate differences and nothing else:
  1. no label — the target day has no metered load yet;
  2. scalers are LOADED, never fitted (a fit on 30 days would drift from training);
  3. the thermal climatology + threshold are the persisted train-fitted constants, not
     recomputed from the window (30 days cannot reproduce a 6-year climatology).
Every actual feature — the 168 h lookback, the 3-week macro block, the thermal-static block,
the 48 h forecast departures, the forecast-day calendar — is built by the SAME shared helpers
the trainer calls, so the assembled vector is byte-identical to training. verify_against_matrix
proves exactly that.
"""

import glob
import os

import joblib
import numpy as np
import pandas as pd

from src.config import WEATHER_COLS, MATRIX_DIR, TRANSFORMER_FEATURE_CONFIG, TREE_FEATURE_CONFIG
from src.feature_engine import _load_forecast, _forecast_for, FC_FEATURE_NAMES
from src.macro_features import compute_macro_features, MACRO_WINDOW_HOURS, MACRO_FEATURE_NAMES
from src.thermal_features import (
    add_thermal_sequence_cols, build_heat_streak, compute_thermal_static,
    THERMAL_SEQ_COLS, THERMAL_STATIC_NAMES,
)
from src.serving import thermal_refs


def _load_scaler(matrix_dir, pattern):
    hits = glob.glob(os.path.join(matrix_dir, pattern))
    if not hits:
        raise FileNotFoundError(f"No scaler matching '{pattern}' in {matrix_dir} — train first.")
    return joblib.load(sorted(hits)[0])


def _window_refs(df, ept_dates, unique_days, threshold):
    """The frame-local thermal state (heat_streak, day_index, doy) for THIS window, using the
    persisted threshold. build_thermal_references does this from the full history; here the
    window is enough because the longest heat streak (~21 d) fits inside a 30-day window."""
    temp = df['Temp_F'].values
    doy = pd.to_datetime(df['Datetime_EPT']).dt.dayofyear.values
    daily = pd.DataFrame({'t': temp, 'd': ept_dates}).groupby('d')['t'].mean()
    daily_mean_temp = daily.reindex(unique_days).values
    heat_streak = build_heat_streak(daily_mean_temp, threshold)
    day_index = {d: k for k, d in enumerate(unique_days)}
    return heat_streak, day_index, doy


def build_sequence_sample(window_df, target_day, matrix_dir=None):
    """Return (X_3d[1, 168, D], target_day) for the sequence models, or raise on a bad window.

    window_df : cleaned hourly frame (the recent ~30 days), UTC index, same columns as training.
    target_day: the EPT date to forecast (the day AFTER the window's last full day).
    """
    matrix_dir = matrix_dir or MATRIX_DIR
    lb = TRANSFORMER_FEATURE_CONFIG['lookback_hours']
    hh = TRANSFORMER_FEATURE_CONFIG['latest_info_hour']

    seq_scaler    = _load_scaler(matrix_dir, f'scaler_ts_lb{lb}_h{hh}.pkl')
    static_scaler = _load_scaler(matrix_dir, f'macro_scaler_lb{lb}_h{hh}.pkl')
    refs = thermal_refs.load(matrix_dir)
    climatology, threshold = refs['climatology'], refs['threshold']

    df = window_df.sort_index().copy()
    add_thermal_sequence_cols(df)                       # CDD_h/HDD_h/rolling — same as training

    ept = pd.to_datetime(df['Datetime_EPT'])
    ept_dates, ept_hours = ept.dt.date.values, ept.dt.hour.values
    unique_days = np.unique(ept_dates)

    feature_cols = ['Load_Estimated'] + WEATHER_COLS + THERMAL_SEQ_COLS
    data_array = seq_scaler.transform(df[feature_cols])  # training's scaler, not a new fit
    est_raw  = df['Load_Estimated'].values
    temp_raw = df['Temp_F'].values
    cdd_raw  = df['CDD_h'].values
    heat_streak, day_index, doy = _window_refs(df, ept_dates, unique_days, threshold)

    # Cutoff day: training forecasts unique_days[i+1] with the cutoff on unique_days[i] (the day
    # BEFORE the target) at latest_info_hour. So the cutoff is target-1 for hh<=9, target-2 above.
    # target_day must be present in the window (at least a calendar row) so we can index it.
    k = np.where(unique_days == target_day)[0]
    if len(k) == 0:
        raise ValueError(f"Target day {target_day} not in the window (need at least its calendar row).")
    k = k[0]
    cutoff_date = unique_days[k - 1] if hh <= 9 else unique_days[k - 2]
    rows = np.where((ept_dates == cutoff_date) & (ept_hours == hh))[0]
    if len(rows) == 0:
        raise ValueError(f"Target day {target_day}: no cutoff hour {hh} in the window.")
    cutoff_pos = rows[0]
    if cutoff_pos < max(lb, MACRO_WINDOW_HOURS):
        raise ValueError(f"Target day {target_day}: window too short before cutoff "
                         f"({cutoff_pos} h < {max(lb, MACRO_WINDOW_HOURS)} h needed).")

    X_window = data_array[cutoff_pos - lb:cutoff_pos]
    if np.isnan(X_window).any():
        raise ValueError(f"NaN inside the {lb} h lookback for {target_day} — fix the input data.")

    fc_raw = _forecast_for(_load_forecast(), target_day, temp_raw, ept_hours, cutoff_pos, lb, threshold)
    if fc_raw is None:
        raise ValueError(f"No complete 48 h forecast for {target_day} — cannot forecast it.")

    macro_raw   = compute_macro_features(est_raw, ept_hours, cutoff_pos)
    thermal_raw = compute_thermal_static(
        temp_raw, cdd_raw, doy, cutoff_pos, day_index[ept_dates[cutoff_pos - 1]],
        heat_streak, climatology)
    static_raw = np.concatenate([macro_raw, thermal_raw, fc_raw]).astype('float32')

    # Forecast-day calendar, broadcast across the window (same order as training).
    tmrw_pos = np.where(ept_dates == target_day)[0]
    row = df.iloc[tmrw_pos[0]]
    dow = row['dayofweek']
    tmrw_meta = np.array([row['month_sin'], row['month_cos'],
                          np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                          float(row['is_weekend']), float(row['is_holiday'])], dtype='float32')
    X_window = np.concatenate([X_window, np.tile(tmrw_meta, (lb, 1))], axis=1)

    static_scaled = static_scaler.transform(static_raw[None, :]).astype('float32')[0]
    X_window = np.concatenate([X_window, np.tile(static_scaled, (lb, 1))], axis=1)

    return X_window[None, :, :].astype('float32'), target_day


def build_tree_sample(window_df, target_day, matrix_dir=None):
    """Return a 1-row DataFrame (indexed by target_day) for the tree models — the flattened
    lookback + calendar + macro/thermal/forecast statics, UNSCALED (trees are scale-invariant).
    Mirrors src/feature_engine.build_or_load_matrix's per-sample dict, minus is_target_valid
    (dropped before prediction, exactly as ModelEvaluator does)."""
    matrix_dir = matrix_dir or MATRIX_DIR
    lb = TREE_FEATURE_CONFIG['lookback_hours']
    hh = TREE_FEATURE_CONFIG['latest_info_hour']
    climatology = None
    refs = thermal_refs.load(matrix_dir)
    climatology, threshold = refs['climatology'], refs['threshold']

    df = window_df.sort_index().copy()
    add_thermal_sequence_cols(df)

    ept = pd.to_datetime(df['Datetime_EPT'])
    ept_dates, ept_hours = ept.dt.date.values, ept.dt.hour.values
    unique_days = np.unique(ept_dates)

    feature_cols = ['Load_Estimated'] + WEATHER_COLS + THERMAL_SEQ_COLS
    data_array = df[feature_cols].values                 # raw, NOT scaled
    est_raw, temp_raw, cdd_raw = df['Load_Estimated'].values, df['Temp_F'].values, df['CDD_h'].values
    heat_streak, day_index, doy = _window_refs(df, ept_dates, unique_days, threshold)

    k = np.where(unique_days == target_day)[0]
    if len(k) == 0:
        raise ValueError(f"Target day {target_day} not in the window.")
    cutoff_date = unique_days[k[0] - 1] if hh <= 9 else unique_days[k[0] - 2]
    rows = np.where((ept_dates == cutoff_date) & (ept_hours == hh))[0]
    if len(rows) == 0:
        raise ValueError(f"Target day {target_day}: no cutoff hour {hh} in the window.")
    cutoff_pos = rows[0]
    if cutoff_pos < max(lb, MACRO_WINDOW_HOURS):
        raise ValueError(f"Target day {target_day}: window too short before cutoff.")

    past_window = data_array[cutoff_pos - lb:cutoff_pos]
    if np.isnan(past_window).any():
        raise ValueError(f"NaN inside the {lb} h lookback for {target_day}.")
    fc_raw = _forecast_for(_load_forecast(), target_day, temp_raw, ept_hours, cutoff_pos, lb, threshold)
    if fc_raw is None:
        raise ValueError(f"No complete 48 h forecast for {target_day}.")

    f = {}
    for j, col in enumerate(feature_cols):
        for t in range(lb):
            f[f'{col.lower()}_h{t}'] = past_window[t, j]

    row = df.iloc[np.where(ept_dates == target_day)[0][0]]
    dow = row['dayofweek']
    f.update({
        'tmrw_month_sin': row['month_sin'], 'tmrw_month_cos': row['month_cos'],
        'tmrw_dow_sin': np.sin(2 * np.pi * dow / 7), 'tmrw_dow_cos': np.cos(2 * np.pi * dow / 7),
        'tmrw_is_weekend': row['is_weekend'], 'tmrw_is_holiday': row['is_holiday'],
    })
    macro_raw = compute_macro_features(est_raw, ept_hours, cutoff_pos)
    for nm, val in zip(MACRO_FEATURE_NAMES, macro_raw):
        f[nm] = float(val)
    thermal_raw = compute_thermal_static(
        temp_raw, cdd_raw, doy, cutoff_pos, day_index[ept_dates[cutoff_pos - 1]],
        heat_streak, climatology)
    for nm, val in zip(THERMAL_STATIC_NAMES, thermal_raw):
        f[nm] = float(val)
    for nm, val in zip(FC_FEATURE_NAMES, fc_raw):
        f[nm] = float(val)

    return pd.DataFrame([f], index=pd.Index([target_day], name='timestamp'))


# ---------------------------------------------------------------------------
# Golden test — the assembled feature must equal what training baked into the matrix.
# ---------------------------------------------------------------------------

def verify_against_matrix(cleaned_path, matrix_dir=None, n_days=5, atol=1e-4):
    """Rebuild the sample for days the training 3D matrix already holds, from a window sliced out
    of the SAME cleaned data, and demand an exact match. Any drift between this assembler and the
    training loop shows up here as a non-zero diff."""
    matrix_dir = matrix_dir or MATRIX_DIR
    lb = TRANSFORMER_FEATURE_CONFIG['lookback_hours']
    hh = TRANSFORMER_FEATURE_CONFIG['latest_info_hour']
    X_train = np.load(os.path.join(matrix_dir, f'X_3d_lb{lb}_h{hh}.npy'))
    ts_train = np.load(os.path.join(matrix_dir, f'timestamps_3d_lb{lb}_h{hh}.npy'), allow_pickle=True)

    full = pd.read_csv(cleaned_path, index_col=0, parse_dates=True).sort_index()
    ept_full = pd.to_datetime(full['Datetime_EPT'])
    ept_dates_full = ept_full.dt.date.values

    worst, worst_day = 0.0, None
    probe = list(ts_train[-n_days:])
    pos = {d: k for k, d in enumerate(ts_train)}
    for d in probe:
        start = pd.Timestamp(d) - pd.Timedelta(days=35)   # a ~30-day window ending at the target
        window = full[(ept_full >= start) & (ept_dates_full <= d)]
        X, _ = build_sequence_sample(window, d, matrix_dir)
        diff = float(np.abs(X[0] - X_train[pos[d]]).max())
        if diff > worst:
            worst, worst_day = diff, d

    if worst > atol:
        raise AssertionError(
            f"Serving SEQ features differ from the training matrix: max |diff| = {worst:.3e} on "
            f"{worst_day} (tol {atol:.0e}). The assembler drifted from feature_engine.")
    print(f"Golden test SEQ: {len(probe)} day(s) rebuilt from a ~30-day window, "
          f"max |diff| vs training 3D matrix = {worst:.3e}  (<= {atol:.0e})  OK")

    # --- tree (2D) path: compare against the cached X_opt (minus is_target_valid) ---
    tlb = TREE_FEATURE_CONFIG['lookback_hours']
    thh = TREE_FEATURE_CONFIG['latest_info_hour']
    xopt_path = os.path.join(matrix_dir, f'X_opt_lb{tlb}_h{thh}.csv')
    if os.path.exists(xopt_path):
        X_opt = pd.read_csv(xopt_path, index_col=0)
        X_opt.index = pd.to_datetime(X_opt.index).date
        cols = [c for c in X_opt.columns if c != 'is_target_valid']
        tworst, tday = 0.0, None
        for d in probe:
            if d not in X_opt.index:
                continue
            start = pd.Timestamp(d) - pd.Timedelta(days=35)
            window = full[(ept_full >= start) & (ept_dates_full <= d)]
            row = build_tree_sample(window, d, matrix_dir)
            diff = float(np.abs(row[cols].values[0] - X_opt.loc[[d], cols].values[0]).max())
            if diff > tworst:
                tworst, tday = diff, d
        if tworst > atol:
            raise AssertionError(
                f"Serving TREE features differ from X_opt: max |diff| = {tworst:.3e} on {tday}.")
        print(f"Golden test TREE: max |diff| vs training 2D matrix = {tworst:.3e}  (<= {atol:.0e})  OK")
    return worst
