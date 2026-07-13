"""Transformer residual-learning test (mirrors util/xgb_residual_test.py).

Instead of predicting the metered load directly, predict its DEVIATION from a naive
same-hour-last-week baseline, then add the baseline back at inference:

    baseline[h] = mean over the past 7 days of the PRELIMINARY (estimated) load
                  at clock-hour h                                  -> 24-dim profile
    train target: y_residual = y_metered - baseline
    inference   : pred_metered = model.predict(X) + baseline

Features are UNCHANGED (the same 3D matrix), the model is the same
TimeSeriesTransformer3D with the same TRANSFORMER_PARAMS and the same split
(+embargo) — only the TARGET changes.

Two things the 2D/XGBoost version did not have to handle:
  1. The 3D matrix is standardized, so the baseline is NOT read back out of it.
     It is rebuilt straight from the CLEANED hourly data (raw MW preliminary load),
     using the same cutoff rule as the matrix builder. No dependence on column order
     and no scaler inversion.
  2. The residual (hundreds of MW) is on a different scale from the standardized
     target the network expects, so it gets its OWN StandardScaler, fit on the
     TRAINING samples only. Predictions are inverse-transformed with that scaler
     before the baseline is added back.

The baseline is derived per-sample from that sample's own lookback window (nothing
is fit on the training set -> it cannot leak) and uses only the preliminary load,
which is available in real time.

NOTE: LDS/FDS (if enabled in TRANSFORMER_PARAMS) bin on the TARGET, which here is
the residual rather than the load. That follows from "same model, only the target
changes"; it is printed at startup so it is not a surprise.

Usage
  PJM_DATASET=dom python util/transformer_residual_test.py
  PJM_DATASET=dom python util/transformer_residual_test.py --baseline scalar
  PJM_DATASET=dom python util/transformer_residual_test.py --epochs 200

Model   -> models/<ds>/transformer_residual/<tag>/transformer_residual_best.pth
Results -> results/<ds>/evaluation/transformer_residual/<tag>/
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as cfg
from src.feature_engine import build_timeseries_matrix, _split_indices, apply_embargo
from src.models._eval_utils import EvalUtils
from src.models._seq_trainer import train_sequence
from src.models._utils import _make_run_dir
from src.models.transformer import TimeSeriesTransformer3D, predict as tr_predict

NAME = 'TRANSFORMER_RESIDUAL'
LOOKBACK = 168


def compute_baseline(cleaned_path, timestamps, mode='hourly',
                     lookback=LOOKBACK, latest_info_hour=0):
    """Build the baseline straight from the CLEANED hourly data (raw MW).

    Reads the preliminary (estimated) load and rebuilds each sample's own 168h
    lookback window with the same cutoff rule as build_timeseries_matrix.

    'hourly' : baseline[h] = mean of the past 7 days' estimated load at clock-hour h.
               The window ends at the midnight cutoff, so window position k sits at
               clock-hour k % 24 -> win[h::24] is exactly the 7 values for hour h.
               (DST shifts this by an hour a couple of times a year.)
    'scalar' : one number per day (mean of the whole window), broadcast to all 24h.

    Returns (N, 24) in MW, aligned row-for-row with `timestamps`.
    """
    df = pd.read_csv(cleaned_path, index_col=0, parse_dates=True).sort_index()
    ept = pd.to_datetime(df['Datetime_EPT'])
    dates = ept.dt.date.values
    hours = ept.dt.hour.values
    est = df['Load_Estimated'].values                   # preliminary load, raw MW
    unique_days = np.unique(dates)
    day_pos = {d: i for i, d in enumerate(unique_days)}

    target_dates = pd.to_datetime(timestamps).date
    base = np.empty((len(target_dates), 24), dtype='float64')

    for n, tmrw in enumerate(target_dates):
        i = day_pos[tmrw]                               # position of the TARGET day
        # same cutoff rule as the matrix builder
        cutoff_date = unique_days[i - 1] if latest_info_hour <= 9 else unique_days[i - 2]
        cutoff = np.where((dates == cutoff_date) & (hours == latest_info_hour))[0][0]
        win = est[cutoff - lookback:cutoff]              # (168,) raw MW
        if mode == 'scalar':
            base[n, :] = win.mean()
        else:
            for h in range(24):
                base[n, h] = win[h::24].mean()           # the 7 values at clock-hour h
    return base


def _metrics(tag, y_true, y_pred):
    t, p = y_true.flatten(), y_pred.flatten()
    print(f"  {tag:26s} MAPE {mean_absolute_percentage_error(t, p) * 100:6.2f}%  "
          f"MAE {mean_absolute_error(t, p):8.1f}  "
          f"RMSE {np.sqrt(mean_squared_error(t, p)):8.1f}  "
          f"ME {np.mean(p - t):+8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', default='hourly', choices=['hourly', 'scalar'])
    ap.add_argument('--epochs', type=int, default=None)
    args = ap.parse_args()

    params = dict(cfg.TRANSFORMER_PARAMS)
    if args.epochs:
        params['epochs'] = args.epochs
    fc = dict(cfg.TRANSFORMER_FEATURE_CONFIG)

    X, y_scaled, mask, ts = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)
    N, H = y_scaled.shape

    # --- baseline (MW), rebuilt from the cleaned data for these exact samples ---
    baseline = compute_baseline(cfg.CLEANED_PATH, ts, mode=args.baseline,
                                lookback=fc['lookback_hours'],
                                latest_info_hour=fc['latest_info_hour'])

    # --- metered target back to MW ---
    y_scaler = joblib.load(glob.glob(os.path.join(cfg.MATRIX_DIR, 'y_scaler_*.pkl'))[0])
    y_mw = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(N, H)

    residual_mw = y_mw - baseline                                   # (N, 24) MW

    # --- the same split (+embargo) the transformer itself uses ---
    train_pool, test_idx = _split_indices(N, fc['split_strategy'], fc['test_frac'], fc['random_state'])
    rtr, rval = _split_indices(len(train_pool), fc['val_strategy'], fc['val_frac'], fc['random_state'])
    train_idx, val_idx = train_pool[rtr], train_pool[rval]
    train_idx, val_idx = apply_embargo(train_idx, val_idx, cfg.EMBARGO_DAYS)
    fit_idx = train_idx[mask[train_idx]]         # denoised training samples, as in training

    # --- standardize the residual with a scaler fit on TRAINING samples only ---
    res_scaler = StandardScaler().fit(residual_mw[fit_idx].reshape(-1, 1))
    y_res_scaled = res_scaler.transform(residual_mw.reshape(-1, 1)).reshape(N, H).astype('float32')

    print(f"\nBaseline: {args.baseline} | train {len(fit_idx)} | val {len(val_idx)} | test {len(test_idx)}")
    print(f"Residual (MW): mean {residual_mw[fit_idx].mean():+.1f}  std {residual_mw[fit_idx].std():.1f}"
          f"  -> standardized by its own scaler (fit on train only)")
    if params.get('use_lds') or params.get('use_fds'):
        print(f"NOTE: use_lds={params.get('use_lds')} use_fds={params.get('use_fds')} — these bin on the "
              f"TARGET, which is now the residual, not the load.")

    # --- reference: how good is the naive baseline on its own? ---
    print("\n=== Naive baseline alone (no model) ===")
    _metrics('baseline / train', y_mw[fit_idx], baseline[fit_idx])
    _metrics('baseline / test',  y_mw[test_idx], baseline[test_idx])

    # --- train the SAME transformer, on the residual target ---
    print("\n--- Training transformer on the residual ---")
    train_sequence(
        TimeSeriesTransformer3D, 'transformer_residual', 'transformer_residual_best.pth',
        X, y_res_scaled, mask, params, fc, cfg.DATASET,
    )
    model_dir = _make_run_dir('models', 'transformer_residual', fc, cfg.DATASET,
                              use_lds=params.get('use_lds', False),
                              use_fds=params.get('use_fds', False))
    model_path = os.path.join(model_dir, 'transformer_residual_best.pth')

    # --- predict residual -> back to MW -> add the baseline back ---
    def predict_metered(idx):
        res_scaled = tr_predict(model_path, X[idx], params)                  # (n, 24)
        res_mw = res_scaler.inverse_transform(res_scaled.reshape(-1, 1)).reshape(len(idx), H)
        return res_mw + baseline[idx]

    pred_test  = predict_metered(test_idx)
    pred_train = predict_metered(fit_idx)

    print("\n=== Residual model (baseline added back) ===")
    _metrics('residual+base / train', y_mw[fit_idx], pred_train)
    _metrics('residual+base / test',  y_mw[test_idx], pred_test)

    # --- full evaluation suite, comparable to the plain transformer run ---
    run_tag = os.path.basename(model_dir)
    result_dir = os.path.join('results', cfg.DATASET, 'evaluation', 'transformer_residual', run_tag)
    train_df = EvalUtils.build_detailed_df(
        NAME, y_mw[fit_idx], pred_train, pd.to_datetime(ts[fit_idx]))
    EvalUtils.evaluate_one(
        NAME, y_mw[test_idx], pred_test, pd.to_datetime(ts[test_idx]), result_dir, train_df)
    print(f"\nResults saved to: {result_dir}")


if __name__ == '__main__':
    main()
