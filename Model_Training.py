import matplotlib.pyplot as plt
import src.config as cfg
import joblib
from src.feature_engine import build_or_load_matrix, get_train_test_split, build_timeseries_matrix
from src.model_trainer import PowerForecaster
import os

# --- Tree Models (XGBoost / LightGBM) ---
if cfg.TRAIN_CONFIG['xgboost'] or cfg.TRAIN_CONFIG['lightgbm']:
    X_opt, y_opt = build_or_load_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)
    X_train, y_train, X_test, y_test = get_train_test_split(X_opt, y_opt)
    forecaster = PowerForecaster(X_train, y_train, X_test, y_test)

    if cfg.TRAIN_CONFIG['xgboost']:
        xgb_preds = forecaster.train_xgboost(cfg.XGB_PARAMS)

    if cfg.TRAIN_CONFIG['lightgbm']:
        lgb_preds = forecaster.train_lightgbm(cfg.LGBM_PARAMS)

# --- Transformer ---
if cfg.TRAIN_CONFIG['transformer']:
    X_3d, y_3d, mask_3d, timestamps_3d = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)
    _lb = cfg.TRANSFORMER_FEATURE_CONFIG['lookback_hours']
    _h  = cfg.TRANSFORMER_FEATURE_CONFIG['latest_info_hour']
    scaler_ts = joblib.load(os.path.join(cfg.MATRIX_DIR, f'scaler_ts_lb{_lb}_h{_h}.pkl'))
    ts_forecaster = forecaster if (cfg.TRAIN_CONFIG['xgboost'] or cfg.TRAIN_CONFIG['lightgbm']) \
                    else PowerForecaster(None, None, None, None)
    transformer_3d_preds = ts_forecaster.train_transformer_3d(
        X_3d=X_3d,
        y_3d=y_3d,
        mask_3d=mask_3d,
        timestamps_3d=timestamps_3d,
        scaler_ts=scaler_ts,
        params=cfg.TRANSFORMER_PARAMS
    )