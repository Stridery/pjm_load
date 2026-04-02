import matplotlib.pyplot as plt
import src.config as cfg
import joblib
from src.feature_engine import build_or_load_matrix, get_train_test_split, build_timeseries_matrix
from src.model_trainer import PowerForecaster
import os

X_opt, y_opt = build_or_load_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)
X_train, y_train, X_test, y_test = get_train_test_split(X_opt, y_opt)


X_3d, y_3d, mask_3d, timestamps_3d= build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR, cfg.TRANSFORMER_PARAMS['seq_len'])
scaler_ts = joblib.load(os.path.join(cfg.MATRIX_DIR, 'scaler_ts.pkl'))

forecaster = PowerForecaster(X_train, y_train, X_test, y_test)
#xgb_preds = forecaster.train_xgboost(cfg.XGB_PARAMS)
#lgb_preds = forecaster.train_lightgbm(cfg.LGBM_PARAMS)

transformer_3d_preds = forecaster.train_transformer_3d(
    X_3d=X_3d,
    y_3d=y_3d,
    mask_3d=mask_3d,
    timestamps_3d=timestamps_3d,
    scaler_ts=scaler_ts,
    params=cfg.TRANSFORMER_PARAMS
)