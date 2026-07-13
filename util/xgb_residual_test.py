"""XGBoost residual-learning test.

Instead of predicting the metered load directly, predict its DEVIATION from a
naive same-hour-last-week baseline, then add the baseline back at inference:

    baseline[h] = mean over the past 7 days of the PRELIMINARY (estimated) load
                  at clock-hour h                                    -> 24-dim profile
    train target: y_residual = y_metered - baseline
    inference   : pred_metered = model.predict(X) + baseline

Features are UNCHANGED (the same 2D matrix the normal XGBoost uses), and the model
is the same XGBRegressor with the same XGB_PARAMS — the only thing that changes is
the target. The baseline is derived per-sample from that sample's own lookback
window, so it involves no fitting on the training set and cannot leak. It also uses
only the preliminary load, which is available in real time.

The baseline's own accuracy is reported too, so you can see how much the model adds
on top of the naive predictor.

Usage
  PJM_DATASET=dom python util/xgb_residual_test.py
  PJM_DATASET=dom python util/xgb_residual_test.py --baseline scalar

Model   -> models/<ds>/xgboost_residual/<tag>/xgboost_residual_24_models.pkl
Results -> results/<ds>/evaluation/xgboost_residual/<tag>/
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as cfg
from src.feature_engine import build_or_load_matrix, get_train_test_split
from src.models._eval_utils import EvalUtils
from src.models._utils import _make_run_dir

NAME = 'XGB_RESIDUAL'
_LDS_KEYS = ['use_lds', 'lds_bin_width', 'lds_ks', 'lds_sigma', 'lds_min_freq_ratio']


def compute_baseline(X_opt, mode='hourly', lookback=168):
    """Per-sample baseline over the sample's own 168h lookback of preliminary load.

    'hourly' : baseline[h] = mean of the past 7 days' estimated load at clock-hour h.
               The window ends at the midnight cutoff, so lookback column k sits at
               clock-hour k % 24 -> each hour averages exactly 7 values (k = h, h+24,
               ..., h+144). (DST days shift this by an hour a couple of times a year.)
    'scalar' : one number per day (mean of the whole 168h window), broadcast to all 24h.
    """
    if mode == 'scalar':
        cols = [f'load_estimated_h{k}' for k in range(lookback)]
        flat = X_opt[cols].mean(axis=1)
        return pd.DataFrame({f'h{h}': flat for h in range(24)}, index=X_opt.index)

    base = pd.DataFrame(index=X_opt.index)
    for h in range(24):
        cols = [f'load_estimated_h{k}' for k in range(lookback) if k % 24 == h]
        base[f'h{h}'] = X_opt[cols].mean(axis=1)
    return base


def _metrics(tag, y_true, y_pred):
    t, p = y_true.flatten(), y_pred.flatten()
    print(f"  {tag:24s} MAPE {mean_absolute_percentage_error(t, p) * 100:6.2f}%  "
          f"MAE {mean_absolute_error(t, p):8.1f}  "
          f"RMSE {np.sqrt(mean_squared_error(t, p)):8.1f}  "
          f"ME {np.mean(p - t):+8.1f}")


def train_residual(X_train, y_train, base_train, params):
    xgb_params = {k: v for k, v in params.items() if k not in _LDS_KEYS}
    if torch.cuda.is_available():
        xgb_params['device'] = 'cuda'
    print(f"XGBoost device: {xgb_params.get('device', 'cpu')}")

    models = []
    for h in tqdm(range(24), desc="XGBoost (residual)"):
        y_res = y_train[f'h{h}'].values - base_train[f'h{h}'].values
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X_train, y_res)
        models.append(m)
    return models


def predict_metered(models, X, base):
    """pred_metered = model.predict(X) + baseline."""
    residual = np.array([m.predict(X) for m in models]).T      # (N, 24)
    return residual + base.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', default='hourly', choices=['hourly', 'scalar'])
    args = ap.parse_args()

    X_opt, y_opt = build_or_load_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)
    baseline = compute_baseline(X_opt, mode=args.baseline)

    # Same split (+embargo, +is_target_valid denoise) as the normal tree pipeline.
    X_train, y_train, X_test, y_test = get_train_test_split(X_opt, y_opt)
    base_train = baseline.loc[X_train.index]
    base_test  = baseline.loc[X_test.index]
    print(f"\nBaseline mode: {args.baseline} | Train {len(X_train)} | Test {len(X_test)}")

    # --- reference: how good is the naive baseline on its own? ---
    print("\n=== Naive baseline alone (no model) ===")
    _metrics('baseline / train', y_train.values, base_train.values)
    _metrics('baseline / test',  y_test.values,  base_test.values)

    print("\n--- Training XGBoost on the residual ---")
    models = train_residual(X_train, y_train, base_train, cfg.XGB_PARAMS)

    tf = cfg.TREE_FEATURE_CONFIG
    model_dir = _make_run_dir('models', 'xgboost_residual', tf, cfg.DATASET)
    save_path = os.path.join(model_dir, 'xgboost_residual_24_models.pkl')
    joblib.dump(models, save_path)
    print(f"Model saved to: {save_path}")

    # --- predict + add the baseline back ---
    pred_test  = predict_metered(models, X_test,  base_test)
    pred_train = predict_metered(models, X_train, base_train)

    print("\n=== Residual model (baseline added back) ===")
    _metrics('residual+base / train', y_train.values, pred_train)
    _metrics('residual+base / test',  y_test.values,  pred_test)

    # --- full evaluation suite, comparable to the plain XGBoost run ---
    run_tag = os.path.basename(model_dir)
    result_dir = os.path.join('results', cfg.DATASET, 'evaluation', 'xgboost_residual', run_tag)
    train_df = EvalUtils.build_detailed_df(
        NAME, y_train.values, pred_train, pd.to_datetime(y_train.index))
    EvalUtils.evaluate_one(
        NAME, y_test.values, pred_test, pd.to_datetime(y_test.index), result_dir, train_df)
    print(f"\nResults saved to: {result_dir}")


if __name__ == '__main__':
    main()
