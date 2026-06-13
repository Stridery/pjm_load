import os
import numpy as np
import joblib
import torch
import xgboost as xgb
from tqdm import tqdm

from src.models._utils import _make_run_dir
from src.models._eval_utils import EvalUtils
from src.config import TREE_FEATURE_CONFIG, XGB_PARAMS


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X_train, y_train, params=None, feature_cfg=None):
    print("\n--- Training XGBoost Experts ---")
    feature_cfg = feature_cfg or TREE_FEATURE_CONFIG
    model_dir = _make_run_dir('models', 'xgboost', feature_cfg)

    use_gpu = torch.cuda.is_available()
    xgb_params = {**(params or XGB_PARAMS), 'device': 'cuda'} if use_gpu else (params or XGB_PARAMS)
    print(f"XGBoost device: {'cuda' if use_gpu else 'cpu'}")

    models = []
    for h in tqdm(range(24), desc="XGBoost"):
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train[f'h{h}'])
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

def evaluate(model_path, X_test, y_true_np, timestamps, result_dir):
    """Predict then run the full evaluation suite."""
    y_pred = predict(model_path, X_test)
    EvalUtils.evaluate_one('XGBOOST', y_true_np, y_pred, timestamps, result_dir)
