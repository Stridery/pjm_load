import os
import numpy as np
import joblib
import lightgbm as lgb
from tqdm import tqdm

from src.models._utils import _make_run_dir
from src.models._eval_utils import EvalUtils
from src.models._lds import compute_lds_weights
from src.config import TREE_FEATURE_CONFIG, LGBM_PARAMS


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X_train, y_train, params=None, feature_cfg=None):
    print("\n--- Training LightGBM Experts ---")
    feature_cfg = feature_cfg or TREE_FEATURE_CONFIG

    # Copy so we can pop internal flags without mutating the caller's dict
    _params = {**(params or LGBM_PARAMS)}
    use_lds = _params.pop('use_lds', False)

    model_dir = _make_run_dir('models', 'lightgbm', feature_cfg, use_lds=use_lds)

    print("LightGBM device: cpu")
    print(f"LightGBM LDS:    {'enabled' if use_lds else 'disabled'}")
    cat_features = ['today_dayofweek', 'tmrw_is_weekend']

    models = []
    for h in tqdm(range(24), desc="LightGBM"):
        y_h = y_train[f'h{h}'].values
        weights = compute_lds_weights(y_h) if use_lds else None
        model = lgb.LGBMRegressor(**_params, n_estimators=1000)
        model.fit(X_train, y_h, categorical_feature=cat_features, sample_weight=weights)
        models.append(model)

    save_path = os.path.join(model_dir, 'lightgbm_24_models.pkl')
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
        train_df = EvalUtils.build_detailed_df('LIGHTGBM', y_true_train, y_pred_train, timestamps_train)
    EvalUtils.evaluate_one('LIGHTGBM', y_true_np, y_pred, timestamps, result_dir, train_df)
