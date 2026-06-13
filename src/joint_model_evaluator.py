# src/joint_model_evaluator.py
"""
Joint model evaluator: handles multi-zone predictions.

Each zone's 24-hour slice is inverse-transformed with its own y_scaler,
then evaluated independently using the existing _evaluate_one / plot logic.
"""

import glob
import os

import joblib
import numpy as np
import torch

from .feature_engine import _split_indices
from .model_evaluator import ModelEvaluator, plot_single_day
from .model_trainer import TimeSeriesTransformer3D, LSTMModel


class JointModelEvaluator:
    def __init__(self, eval_cfg, zones: list):
        """
        eval_cfg : same structure as EVAL_CONFIG (split_strategy, test_frac, ...)
        zones    : ordered list of zone names, e.g. ['dom', 'bge']
        """
        self.cfg    = eval_cfg
        self.zones  = zones
        self._base  = ModelEvaluator(eval_cfg)   # reuse plot / metric logic

        self.X_3d = self.y_3d = self.mask_3d = self.timestamps = None
        self.y_scalers: dict = {}

    # ------------------------------------------------------------------ #
    # Data loading                                                         #
    # ------------------------------------------------------------------ #

    def load_data(self):
        from src.config import (
            JOINT_ZONES, JOINT_CLEANED_PATH, JOINT_MATRIX_DIR,
            JOINT_FEATURE_CONFIG, _WEATHER_COLS,
        )
        from .joint_feature_engine import build_joint_cleaned, build_joint_timeseries_matrix

        # Build joint cleaned CSV if missing
        if not os.path.exists(JOINT_CLEANED_PATH):
            build_joint_cleaned(JOINT_ZONES)

        cfg = JOINT_FEATURE_CONFIG
        X, y, mask, ts, y_scalers = build_joint_timeseries_matrix(
            zones=JOINT_ZONES,
            weather_cols=_WEATHER_COLS,
            joint_cleaned_path=JOINT_CLEANED_PATH,
            matrix_dir=JOINT_MATRIX_DIR,
            lookback_hours=cfg['lookback_hours'],
            latest_info_hour=cfg['latest_info_hour'],
            test_frac=cfg['test_frac'],
        )
        self.X_3d       = X
        self.y_3d       = y
        self.mask_3d    = mask
        self.timestamps = ts
        self.y_scalers  = y_scalers

    # ------------------------------------------------------------------ #
    # Inverse transform per zone                                           #
    # ------------------------------------------------------------------ #

    def _inv(self, scaled_2d: np.ndarray, zone: str) -> np.ndarray:
        N, P = scaled_2d.shape
        return (
            self.y_scalers[zone]
            .inverse_transform(scaled_2d.flatten().reshape(-1, 1))
            .reshape(N, P)
        )

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def _predict(self, model_type: str, model_path: str, X_np: np.ndarray) -> np.ndarray:
        from src.config import JOINT_TRANSFORMER_PARAMS, JOINT_LSTM_PARAMS
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_features = X_np.shape[2]
        if model_type == 'transformer':
            model = TimeSeriesTransformer3D(num_features, JOINT_TRANSFORMER_PARAMS).to(device)
        else:
            model = LSTMModel(num_features, JOINT_LSTM_PARAMS).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            return model(torch.FloatTensor(X_np).to(device)).cpu().numpy()

    # ------------------------------------------------------------------ #
    # Evaluate all enabled models                                          #
    # ------------------------------------------------------------------ #

    def evaluate_all(self):
        from src.config import JOINT_DATASET
        cfg = self.cfg

        _, test_pos = _split_indices(
            len(self.X_3d),
            cfg['split_strategy'],
            cfg['test_frac'],
            cfg['random_state'],
        )
        X_te  = self.X_3d[test_pos]
        y_te  = self.y_3d[test_pos]
        ts_te = self.timestamps[test_pos]

        result_base = cfg.get('result_dir', f'results/{JOINT_DATASET}/evaluation')

        for model_type in ['transformer', 'lstm']:
            model_info = cfg['models'].get(model_type, {})
            if not model_info.get('enabled', 0):
                continue
            model_path = model_info['model_path']
            print(f'\n====== Loading JOINT {model_type.upper()} ======')
            y_pred_scaled = self._predict(model_type, model_path, X_te)

            val_tag = ''
            if 'val_strategy' in cfg:
                val_tag = f"_{cfg['val_strategy']}_val{cfg['val_frac']}"
            run_tag = f"{cfg['split_strategy']}_test{cfg['test_frac']}{val_tag}"

            for i, zone in enumerate(self.zones):
                start, end  = i * 24, (i + 1) * 24
                true_mw     = self._inv(y_te[:, start:end], zone)
                pred_mw     = self._inv(y_pred_scaled[:, start:end], zone)
                result_dir  = os.path.join(result_base, model_type, run_tag, zone)
                label       = f'joint_{model_type}_{zone.upper()}'
                print(f'\n====== {label} Error Analysis ======')
                self._base._evaluate_one(label, true_mw, pred_mw, ts_te, result_dir)

    # ------------------------------------------------------------------ #
    # Single-day plot                                                      #
    # ------------------------------------------------------------------ #

    def show_single_day(self, model_type: str, model_path: str, date_str: str):
        from src.config import JOINT_DATASET
        import pandas as pd

        target = pd.Timestamp(date_str).date()
        ts_dates = [pd.Timestamp(t).date() if not hasattr(t, 'year') else t
                    for t in self.timestamps]
        idx = np.where([d == target for d in ts_dates])[0]
        if len(idx) == 0:
            raise ValueError(f'{date_str} not found in joint matrix timestamps.')

        y_pred_scaled = self._predict(model_type, model_path, self.X_3d[idx])
        y_true_scaled = self.y_3d[idx]

        for i, zone in enumerate(self.zones):
            start, end  = i * 24, (i + 1) * 24
            true_mw     = self._inv(y_true_scaled[:, start:end], zone).flatten()
            pred_mw     = self._inv(y_pred_scaled[:, start:end], zone).flatten()
            save_path   = os.path.join(
                'results', JOINT_DATASET, 'singleday', model_type, f'{date_str}_{zone}.png'
            )
            plot_single_day(
                f'joint_{model_type}_{zone.upper()}', date_str,
                true_mw, pred_mw, save_path=save_path,
            )
