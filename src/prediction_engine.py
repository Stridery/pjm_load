# src/prediction_engine.py
"""
Feature construction for the days PJM has not verified yet.

Reads cleaned/predict.csv (every hour, no metered Load) and emits, for each day that
has no verified label, exactly the feature vector training would have built for it.

Two rules keep this honest, and both are load-bearing:

1. It never re-fits a scaler. The three StandardScalers are loaded from MATRIX_DIR
   exactly as training left them. Re-fitting on the forecast frame — which is 144 rows
   longer than the training frame — would shift every mean and std slightly, and the
   model would keep producing plausible, wrong numbers without a single error.

2. It never re-implements a feature. The window slicing, macro window, thermal
   references and calendar block all come from the same functions build_or_load_matrix
   and build_timeseries_matrix call. The loop below mirrors theirs day-for-day, which
   is what `verify_against_training_matrix()` proves.

The thermal references (heat threshold, day-of-year climatology) are fitted on the
TRAINING rows only. They are reproduced rather than persisted: predict.csv's first
n_train rows ARE the training CSV, row for row, so passing training's split_idx
selects the same physical rows and yields the same references. That is why predict.csv
keeps the full history instead of just the recent tail — a 21-day file cannot rebuild a
6-year climatology.
"""

import glob
import os

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.config import (
    WEATHER_COLS, MATRIX_DIR, PREDICT_PATH,
    TREE_FEATURE_CONFIG, TRANSFORMER_FEATURE_CONFIG,
)

# How far past the data the forecast reaches. This is not a tuning knob — it is the
# architectural ceiling. A sample cuts off at day D 00:00, reads the 168 h BEFORE that
# ([D-7 00:00, D-1 23:00]) and predicts D+1. So with real data through the last complete
# day L, the furthest cutoff is L+1 and the furthest target is L+2.
#
# Reaching L+2 costs nothing extra: the model never looks at the forecast day's weather, only
# its CALENDAR (month, day-of-week, weekend, holiday), and a calendar is knowable in advance.
# That is what makes a genuine day-ahead forecast possible with no weather forecast at all.
FORECAST_HORIZON_DAYS = 2
TIMEZONE = 'America/New_York'
from src.feature_engine import _normalize_to_24h
from src.macro_features import (
    compute_macro_features, MACRO_FEATURE_NAMES, MACRO_WINDOW_HOURS,
)
from src.thermal_features import (
    add_thermal_sequence_cols, build_thermal_references, compute_thermal_static,
    THERMAL_SEQ_COLS, THERMAL_STATIC_NAMES,
)


def _load_scaler(matrix_dir, pattern):
    hits = glob.glob(os.path.join(matrix_dir, pattern))
    if not hits:
        raise FileNotFoundError(
            f"No scaler matching '{pattern}' in {matrix_dir}. Train first — the forecast "
            f"must use the scalers training fitted, never a fresh fit on the forecast frame."
        )
    return joblib.load(sorted(hits)[0])


def _extend_horizon(df):
    """Append the hours out to L+2 so tomorrow can actually be forecast.

    predict.csv stops at the last hour PJM has published. Without these rows there is no
    position to put the cutoff on and no row to read the target day's calendar from, so the
    forecast could only ever reach days that had already happened — which is not a forecast.

    The new rows carry a CALENDAR and nothing else: load and weather stay NaN. That is safe
    precisely because compute_macro_features and compute_thermal_static read strictly BEFORE
    the cutoff, and the lookback window is data[cutoff-168 : cutoff], which excludes the
    cutoff row itself. Nothing ever reads a value from these rows. If that ever stops being
    true, the NaN guard in the builders below turns it into a crash rather than a wrong number.
    """
    ept = pd.to_datetime(df['Datetime_EPT'])
    per_day = ept.dt.date.value_counts()
    complete = sorted(d for d, n in per_day.items() if n >= 24)
    if not complete:
        raise ValueError("predict.csv holds no complete 24 h day — nothing can be forecast.")
    last_complete = complete[-1]

    end_ept = pd.Timestamp(last_complete) + pd.Timedelta(days=FORECAST_HORIZON_DAYS, hours=23)
    end_utc = end_ept.tz_localize(TIMEZONE, ambiguous=True, nonexistent='shift_forward'
                                  ).tz_convert('UTC')
    if end_utc <= df.index[-1]:
        return df                                    # already reaches the horizon

    full = pd.date_range(df.index[0], end_utc, freq='h', tz='UTC', name=df.index.name)
    df = df.reindex(full)

    new = df['Datetime_EPT'].isna()
    # Datetime_EPT round-trips through the CSV as text; write it back in the same format
    # rather than as datetimes, or the column silently splits into two dtypes.
    df.loc[new, 'Datetime_EPT'] = (df.index[new].tz_convert(TIMEZONE)
                                                .tz_localize(None)
                                                .strftime('%Y-%m-%d %H:%M:%S'))
    df.loc[new, 'has_label'] = 0

    # Calendar for the new rows, computed exactly as data_processor.clean_and_engineer does —
    # these are the only features the forecast day contributes, so they have to match.
    e = pd.to_datetime(df.loc[new, 'Datetime_EPT'])
    dow = e.dt.dayofweek
    holidays = USFederalHolidayCalendar().holidays(start=e.min(), end=e.max())
    df.loc[new, 'hour']       = e.dt.hour
    df.loc[new, 'month']      = e.dt.month
    df.loc[new, 'hour_sin']   = np.sin(2 * np.pi * e.dt.hour / 24)
    df.loc[new, 'hour_cos']   = np.cos(2 * np.pi * e.dt.hour / 24)
    df.loc[new, 'month_sin']  = np.sin(2 * np.pi * e.dt.month / 12)
    df.loc[new, 'month_cos']  = np.cos(2 * np.pi * e.dt.month / 12)
    df.loc[new, 'dayofweek']  = dow
    df.loc[new, 'is_weekend'] = (dow >= 5).astype(int)
    df.loc[new, 'is_holiday'] = e.dt.normalize().isin(holidays).astype(int)
    return df


def _prepare(predict_path):
    """Load predict.csv, extend it to the forecast horizon, add the per-hour thermal columns."""
    df = pd.read_csv(predict_path, index_col=0, parse_dates=True).sort_index()

    # len(training CSV): training is exactly the has_label==1 rows of this frame, and they are
    # a prefix of it. Read it BEFORE extending — the appended rows are unlabelled by
    # construction and would not change the count, but relying on that is asking for trouble.
    n_train_rows = int((df['has_label'] == 1).sum())

    df = _extend_horizon(df)
    add_thermal_sequence_cols(df)

    ept_dt    = pd.to_datetime(df['Datetime_EPT'])
    ept_dates = ept_dt.dt.date.values
    ept_hours = ept_dt.dt.hour.values
    unique_days = np.unique(ept_dates)

    # A forecast day is any day with no verified hour: the days PJM has published but not yet
    # verified, AND the two days past the end of the data that we just appended.
    labelled_by_day = pd.Series(df['has_label'].values, index=ept_dates).groupby(level=0).max()
    target_days = set(labelled_by_day.index[labelled_by_day == 0])

    return df, ept_dates, ept_hours, unique_days, n_train_rows, target_days


def _cutoff_pos(i, unique_days, ept_dates, ept_hours, latest_info_hour, min_history):
    """Position of the forecast cutoff, or None if this day cannot be forecast.

    Mirrors the cutoff logic in feature_engine's two builders exactly — same branch on
    latest_info_hour, same min_history guard.
    """
    today = unique_days[i]
    if latest_info_hour <= 9:
        cutoff_date, cutoff_hour = today, latest_info_hour
    else:
        if i == 0:
            return None
        cutoff_date, cutoff_hour = unique_days[i - 1], latest_info_hour

    rows = np.where((ept_dates == cutoff_date) & (ept_hours == cutoff_hour))[0]
    if len(rows) == 0 or rows[0] < min_history:
        return None
    return rows[0]


def build_tree_features(predict_path=None, only_days=None):
    """Flat (2D) forecast features for xgboost / lightgbm. Raw values — trees don't scale.

    Returns (X, preliminary) — both DataFrames indexed by forecast date. `preliminary` is
    the 24 h Load_Estimated of the forecast day: the near-real-time actual for dom, and
    the MIDATL regional aggregate (NOT comparable) for bge — see PREDICT_CONFIG.
    """
    predict_path = predict_path or PREDICT_PATH
    lookback_hours   = TREE_FEATURE_CONFIG['lookback_hours']
    latest_info_hour = TREE_FEATURE_CONFIG['latest_info_hour']

    df, ept_dates, ept_hours, unique_days, n_train_rows, target_days = _prepare(predict_path)
    if only_days is not None:
        target_days = set(only_days)

    feature_cols = ['Load_Estimated'] + WEATHER_COLS + THERMAL_SEQ_COLS
    data_array = df[feature_cols].values
    est_raw    = df['Load_Estimated'].values
    temp_raw   = df['Temp_F'].values
    cdd_raw    = df['CDD_h'].values

    min_history = max(lookback_hours, MACRO_WINDOW_HOURS)
    # Training's tree split_idx, so the heat threshold and climatology come out identical.
    split_idx = int(n_train_rows * (1 - TREE_FEATURE_CONFIG['test_frac']))
    _thr, heat_streak, climatology, day_index, doy = build_thermal_references(
        df, ept_dates, unique_days, split_idx)

    X_rows, prelim_rows = [], []
    for i in range(len(unique_days) - 1):
        tomorrow = unique_days[i + 1]
        if tomorrow not in target_days:
            continue

        cutoff_pos = _cutoff_pos(i, unique_days, ept_dates, ept_hours,
                                 latest_info_hour, min_history)
        if cutoff_pos is None:
            continue

        tmrw_pos = np.where(ept_dates == tomorrow)[0]
        prelim = _normalize_to_24h(est_raw[tmrw_pos], ept_hours[tmrw_pos])
        if prelim is None:
            continue    # not a whole day — skip rather than forecast a partial one

        past_window = data_array[cutoff_pos - lookback_hours : cutoff_pos]
        if np.isnan(past_window).any():
            raise ValueError(
                f"NaN inside the {lookback_hours} h lookback for {tomorrow}. A NaN here "
                f"turns the whole day's forecast into NaN silently — fix the input data."
            )

        f = {'timestamp': tomorrow}
        for j, col in enumerate(feature_cols):
            for k in range(lookback_hours):
                f[f'{col.lower()}_h{k}'] = past_window[k, j]

        tmrw_meta = df.iloc[tmrw_pos[0]]
        f.update({
            'tmrw_month_sin':  tmrw_meta['month_sin'],
            'tmrw_month_cos':  tmrw_meta['month_cos'],
            'tmrw_dow_sin':    np.sin(2 * np.pi * tmrw_meta['dayofweek'] / 7),
            'tmrw_dow_cos':    np.cos(2 * np.pi * tmrw_meta['dayofweek'] / 7),
            'tmrw_is_weekend': tmrw_meta['is_weekend'],
            'tmrw_is_holiday': tmrw_meta['is_holiday'],
        })

        macro_raw = compute_macro_features(est_raw, ept_hours, cutoff_pos)
        for nm, val in zip(MACRO_FEATURE_NAMES, macro_raw):
            f[nm] = float(val)
        thermal_raw = compute_thermal_static(
            temp_raw, cdd_raw, doy, cutoff_pos,
            day_index[ept_dates[cutoff_pos - 1]], heat_streak, climatology)
        for nm, val in zip(THERMAL_STATIC_NAMES, thermal_raw):
            f[nm] = float(val)
        # is_target_valid is NOT emitted: the training pipeline drops it before the model
        # ever sees it (model_evaluator._tree_test_split), so adding it here would hand
        # the model a column it was never trained on.

        X_rows.append(f)
        prelim_rows.append({'timestamp': tomorrow,
                            **{f'h{h}': prelim[h] for h in range(24)}})

    if not X_rows:
        return pd.DataFrame(), pd.DataFrame()
    return (pd.DataFrame(X_rows).set_index('timestamp'),
            pd.DataFrame(prelim_rows).set_index('timestamp'))


def build_sequence_features(predict_path=None, matrix_dir=None, only_days=None):
    """3D forecast features for transformer / lstm / moe / mstnn, scaled with the
    scalers TRAINING fitted.

    Returns (X_3d, timestamps, preliminary_df, y_scaler). y_scaler is returned because the
    models emit scaled output — the caller must inverse-transform with this exact object.
    """
    predict_path = predict_path or PREDICT_PATH
    matrix_dir   = matrix_dir or MATRIX_DIR
    lookback_hours   = TRANSFORMER_FEATURE_CONFIG['lookback_hours']
    latest_info_hour = TRANSFORMER_FEATURE_CONFIG['latest_info_hour']

    df, ept_dates, ept_hours, unique_days, n_train_rows, target_days = _prepare(predict_path)
    if only_days is not None:
        target_days = set(only_days)

    seq_scaler    = _load_scaler(matrix_dir, f'scaler_ts_lb{lookback_hours}_h{latest_info_hour}.pkl')
    static_scaler = _load_scaler(matrix_dir, f'macro_scaler_lb{lookback_hours}_h{latest_info_hour}.pkl')
    y_scaler      = _load_scaler(matrix_dir, f'y_scaler_lb{lookback_hours}_h{latest_info_hour}.pkl')

    feature_cols = ['Load_Estimated'] + WEATHER_COLS + THERMAL_SEQ_COLS
    data_array = seq_scaler.transform(df[feature_cols])     # training's scaler, not a new fit
    est_raw    = df['Load_Estimated'].values
    temp_raw   = df['Temp_F'].values
    cdd_raw    = df['CDD_h'].values

    min_history = max(lookback_hours, MACRO_WINDOW_HOURS)
    split_idx = int(n_train_rows * (1 - TRANSFORMER_FEATURE_CONFIG['test_frac']))
    _thr, heat_streak, climatology, day_index, doy = build_thermal_references(
        df, ept_dates, unique_days, split_idx)

    X_list, static_list, ts_list, prelim_rows = [], [], [], []
    for i in range(len(unique_days) - 1):
        tomorrow = unique_days[i + 1]
        if tomorrow not in target_days:
            continue

        cutoff_pos = _cutoff_pos(i, unique_days, ept_dates, ept_hours,
                                 latest_info_hour, min_history)
        if cutoff_pos is None:
            continue

        tmrw_pos = np.where(ept_dates == tomorrow)[0]
        prelim = _normalize_to_24h(est_raw[tmrw_pos], ept_hours[tmrw_pos])
        if prelim is None:
            continue

        X_window = data_array[cutoff_pos - lookback_hours : cutoff_pos]
        if np.isnan(X_window).any():
            raise ValueError(
                f"NaN inside the {lookback_hours} h lookback for {tomorrow}. A NaN here "
                f"turns the whole day's forecast into NaN silently — fix the input data."
            )

        macro_raw   = compute_macro_features(est_raw, ept_hours, cutoff_pos)
        thermal_raw = compute_thermal_static(
            temp_raw, cdd_raw, doy, cutoff_pos,
            day_index[ept_dates[cutoff_pos - 1]], heat_streak, climatology)
        static_list.append(np.concatenate([macro_raw, thermal_raw]))

        tmrw_row = df.iloc[tmrw_pos[0]]
        tmrw_dow = tmrw_row['dayofweek']
        tmrw_meta = np.array([
            tmrw_row['month_sin'],
            tmrw_row['month_cos'],
            np.sin(2 * np.pi * tmrw_dow / 7),
            np.cos(2 * np.pi * tmrw_dow / 7),
            float(tmrw_row['is_weekend']),
            float(tmrw_row['is_holiday']),
        ], dtype='float32')
        X_window = np.concatenate(
            [X_window, np.tile(tmrw_meta, (lookback_hours, 1))], axis=1)

        X_list.append(X_window)
        ts_list.append(tomorrow)
        prelim_rows.append({'timestamp': tomorrow,
                            **{f'h{h}': prelim[h] for h in range(24)}})

    if not X_list:
        return np.empty((0,)), np.array([]), pd.DataFrame(), y_scaler

    X_3d = np.array(X_list, dtype='float32')
    static_scaled = static_scaler.transform(
        np.array(static_list, dtype='float32')).astype('float32')
    static_bc = np.repeat(static_scaled[:, None, :], lookback_hours, axis=1)
    X_3d = np.concatenate([X_3d, static_bc], axis=2)

    return (X_3d, np.array(ts_list),
            pd.DataFrame(prelim_rows).set_index('timestamp'), y_scaler)


# ---------------------------------------------------------------------------
# Drift guard
# ---------------------------------------------------------------------------

def verify_against_training_matrix(matrix_dir=None, predict_path=None, n_days=5, atol=1e-4):
    """Rebuild features for days the TRAINING matrix already holds, and demand they match.

    This is the only thing standing between us and silent feature drift. The loop above is
    a second implementation of feature_engine's loop; if the two ever disagree — a column
    reordered, a scaler re-fitted, a window off by one — the forecast keeps running and
    keeps being wrong. Run this after any change to either side.
    """
    matrix_dir   = matrix_dir or MATRIX_DIR
    predict_path = predict_path or PREDICT_PATH
    lb = TRANSFORMER_FEATURE_CONFIG['lookback_hours']
    h  = TRANSFORMER_FEATURE_CONFIG['latest_info_hour']

    X_train = np.load(os.path.join(matrix_dir, f'X_3d_lb{lb}_h{h}.npy'))
    ts_train = np.load(os.path.join(matrix_dir, f'timestamps_3d_lb{lb}_h{h}.npy'),
                       allow_pickle=True)

    probe_days = list(ts_train[-n_days:])
    X_pred, ts_pred, _, _ = build_sequence_features(
        predict_path=predict_path, matrix_dir=matrix_dir, only_days=probe_days)

    if len(ts_pred) != len(probe_days):
        raise AssertionError(
            f"Rebuilt {len(ts_pred)} of {len(probe_days)} probe days — the forecast loop "
            f"is skipping days the training loop kept."
        )

    pos = {d: k for k, d in enumerate(ts_train)}
    worst, worst_day = 0.0, None
    for k, d in enumerate(ts_pred):
        diff = np.abs(X_pred[k] - X_train[pos[d]]).max()
        if diff > worst:
            worst, worst_day = float(diff), d

    if worst > atol:
        raise AssertionError(
            f"Forecast features do not match the training matrix: max |diff| = {worst:.3e} "
            f"on {worst_day} (tolerance {atol:.0e}). The two feature paths have drifted — "
            f"every forecast is suspect until this is zero."
        )
    print(f"Drift guard: {len(ts_pred)} day(s) rebuilt, max |diff| vs training matrix "
          f"= {worst:.3e}  (<= {atol:.0e})  OK")
    return worst
