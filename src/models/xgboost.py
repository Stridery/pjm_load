import os
import numpy as np
import joblib
import torch
import xgboost as xgb
from tqdm import tqdm

from src.models._utils import _make_run_dir
from src.models._eval_utils import EvalUtils
from src.models._lds import compute_lds_weights
from src.config import TREE_FEATURE_CONFIG, XGB_PARAMS


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X_train, y_train, params=None, feature_cfg=None):
    print("\n--- Training XGBoost Experts ---")
    feature_cfg = feature_cfg or TREE_FEATURE_CONFIG

    # Copy so we can pop internal flags without mutating the caller's dict
    xgb_params = {**(params or XGB_PARAMS)}
    use_lds = xgb_params.pop('use_lds', False)

    model_dir = _make_run_dir('models', 'xgboost', feature_cfg, use_lds=use_lds)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        xgb_params['device'] = 'cuda'
    print(f"XGBoost device: {'cuda' if use_gpu else 'cpu'}")
    print(f"XGBoost LDS:    {'enabled' if use_lds else 'disabled'}")

    models = []
    for h in tqdm(range(24), desc="XGBoost"):
        y_h = y_train[f'h{h}'].values
        weights = compute_lds_weights(y_h) if use_lds else None
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_h, sample_weight=weights)
        models.append(model)

    save_path = os.path.join(model_dir, 'xgboost_24_models.pkl')
    joblib.dump(models, save_path)
    print(f"Model saved to: {save_path}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model_path, X):
    """Load saved models and return predictions (N, 24)."""
    models = joblib.load(model_path)
    return np.array([m.predict(X) for m in models]).T


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model_path, X_test, y_true_np, timestamps, result_dir,
             X_train=None, y_true_train=None, timestamps_train=None):
    """Predict then run the full evaluation suite. Pass X_train to include train subplot."""
    y_pred = predict(model_path, X_test)
    train_df = None
    if X_train is not None and y_true_train is not None:
        y_pred_train = predict(model_path, X_train)
        train_df = EvalUtils.build_detailed_df('XGBOOST', y_true_train, y_pred_train, timestamps_train)
    EvalUtils.evaluate_one('XGBOOST', y_true_np, y_pred, timestamps, result_dir, train_df)
