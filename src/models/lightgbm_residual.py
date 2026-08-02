# src/models/lightgbm_residual.py
"""
LightGBM trained on the residual against a naive same-hour-last-week baseline.

Same 2D matrix, same LGBMRegressor, same LGBM_PARAMS as src/models/lightgbm.py — the only
thing that moves is the target (see src/models/_residual.py for the baseline). The baseline is
recovered from X's own load_estimated_h* columns, so predict()'s signature matches
lightgbm.predict exactly and this model drops into ModelEvaluator with no special-casing.
"""

import os

import joblib
import numpy as np
import lightgbm as lgb
from tqdm import tqdm

from src.models._eval_utils import EvalUtils
from src.models._residual import tree_baseline, print_metrics
from src.models._utils import _make_run_dir
from src.config import TREE_FEATURE_CONFIG, LGBM_RESIDUAL_PARAMS

NAME = 'LGBM_RESIDUAL'
MODEL_TYPE = 'lightgbm_residual'
FILENAME = 'lightgbm_residual_24_models.pkl'

# Flags this module consumes itself; LGBMRegressor would reject them.
_INTERNAL_KEYS = ['baseline']


def train(X_train, y_train, params=None, feature_cfg=None):
    print("\n--- Training LightGBM (residual target) ---")
    params = {**(params or LGBM_RESIDUAL_PARAMS)}
    feature_cfg = feature_cfg or TREE_FEATURE_CONFIG
    mode = params.get('baseline', 'hourly')

    lgbm_params = {k: v for k, v in params.items() if k not in _INTERNAL_KEYS}
    print("LightGBM device: cpu")
    print(f"Baseline mode:   {mode}")

    base = tree_baseline(X_train, mode, feature_cfg['lookback_hours'])

    models = []
    for h in tqdm(range(24), desc="LightGBM (residual)"):
        y_res = y_train[f'h{h}'].values - base[:, h]
        m = lgb.LGBMRegressor(**lgbm_params, n_estimators=1000)
        m.fit(X_train, y_res)
        models.append(m)

    model_dir = _make_run_dir('models', MODEL_TYPE, feature_cfg)
    save_path = os.path.join(model_dir, FILENAME)
    # The baseline mode rides along with the weights: predicting with a different mode than the
    # one trained on would add back a baseline the residuals were never measured from.
    joblib.dump({'models': models, 'baseline': mode, 'lookback': feature_cfg['lookback_hours']},
                save_path)
    print(f"Model saved to: {save_path}")


def predict(model_path, X):
    """Load, predict the residual, add the baseline back. Returns MW, shape (N, 24)."""
    bundle = joblib.load(model_path)
    if isinstance(bundle, list):     # weights from before the bundle format
        models, mode, lookback = bundle, 'hourly', TREE_FEATURE_CONFIG['lookback_hours']
    else:
        models, mode, lookback = bundle['models'], bundle['baseline'], bundle['lookback']

    residual = np.array([m.predict(X) for m in models]).T          # (N, 24)
    return residual + tree_baseline(X, mode, lookback)


def evaluate(model_path, X_test, y_true_np, timestamps, result_dir,
             X_train=None, y_true_train=None, timestamps_train=None):
    """Same signature as lightgbm.evaluate, plus the naive baseline's own score — without it
    there is no way to see whether the model is adding anything on top of the baseline."""
    y_pred = predict(model_path, X_test)

    bundle = joblib.load(model_path)
    mode = bundle['baseline'] if isinstance(bundle, dict) else 'hourly'
    base_test = tree_baseline(X_test, mode, TREE_FEATURE_CONFIG['lookback_hours'])
    print(f"\n=== {NAME}: model vs the baseline it leans on (test) ===")
    print_metrics('naive baseline alone', y_true_np, base_test)
    print_metrics('residual + baseline', y_true_np, y_pred)

    train_df = None
    if X_train is not None and y_true_train is not None:
        y_pred_train = predict(model_path, X_train)
        train_df = EvalUtils.build_detailed_df(NAME, y_true_train, y_pred_train, timestamps_train)
    EvalUtils.evaluate_one(NAME, y_true_np, y_pred, timestamps, result_dir, train_df)
