# src/model_evaluator.py
import glob
import os

import joblib
import numpy as np
import pandas as pd

from src.feature_engine import build_or_load_matrix, build_timeseries_matrix, _split_indices
from src.config import (
    CLEANED_PATH, MATRIX_DIR, DATASET,
    TRANSFORMER_PARAMS, LSTM_PARAMS,
)
from src.models import transformer as transformer_mod
from src.models import lstm as lstm_mod
from src.models import xgboost as xgboost_mod
from src.models import lightgbm as lightgbm_mod
from src.models._eval_utils import plot_single_day


class ModelEvaluator:
    def __init__(self, eval_cfg):
        self.cfg = eval_cfg
        self.X_opt = self.y_opt = None
        self.X_3d = self.y_3d = self.mask_3d = self.timestamps_3d = self.y_scaler = None

    # --- Data loading ---

    def _needs_tree(self):
        tree = ['xgboost', 'lightgbm']
        if any(self.cfg['models'][m]['enabled'] for m in tree):
            return True
        sd = self.cfg.get('single_day', {})
        return sd.get('enabled', 0) and sd.get('model') in tree

    def _needs_sequence(self):
        seq = ['transformer', 'lstm']
        if any(self.cfg['models'][m]['enabled'] for m in seq):
            return True
        sd = self.cfg.get('single_day', {})
        return sd.get('enabled', 0) and sd.get('model') in seq

    def load_data(self):
        if self._needs_tree():
            self.X_opt, self.y_opt = build_or_load_matrix(CLEANED_PATH, MATRIX_DIR)
        if self._needs_sequence():
            self.X_3d, self.y_3d, self.mask_3d, self.timestamps_3d = build_timeseries_matrix(
                CLEANED_PATH, MATRIX_DIR
            )
            y_scaler_files = glob.glob(os.path.join(MATRIX_DIR, 'y_scaler_*.pkl'))
            if not y_scaler_files:
                raise FileNotFoundError(f"No y_scaler file found in {MATRIX_DIR}")
            self.y_scaler = joblib.load(y_scaler_files[0])

    # --- Split helpers ---

    def _tree_test_split(self):
        _, test_pos = _split_indices(
            len(self.X_opt),
            self.cfg['split_strategy'],
            self.cfg['test_frac'],
            self.cfg['random_state'],
        )
        test_idx = self.X_opt.index[test_pos]
        X_test = self.X_opt.loc[test_idx].drop(columns=['is_target_valid'])
        y_test = self.y_opt.loc[test_idx]
        return X_test, y_test

    def _tree_train_split(self):
        train_pos, _ = _split_indices(
            len(self.X_opt),
            self.cfg['split_strategy'],
            self.cfg['test_frac'],
            self.cfg['random_state'],
        )
        train_idx = self.X_opt.index[train_pos]
        X_train = self.X_opt.loc[train_idx].drop(columns=['is_target_valid'])
        y_train = self.y_opt.loc[train_idx]
        return X_train, y_train

    def _seq_test_split(self):
        _, test_pos = _split_indices(
            len(self.X_3d),
            self.cfg['split_strategy'],
            self.cfg['test_frac'],
            self.cfg['random_state'],
        )
        return self.X_3d[test_pos], self.y_3d[test_pos], self.timestamps_3d[test_pos]

    def _seq_train_split(self):
        train_pos, _ = _split_indices(
            len(self.X_3d),
            self.cfg['split_strategy'],
            self.cfg['test_frac'],
            self.cfg['random_state'],
        )
        return self.X_3d[train_pos], self.y_3d[train_pos], self.timestamps_3d[train_pos]

    def _inverse_transform(self, scaled_2d):
        N, P = scaled_2d.shape
        return (
            self.y_scaler
            .inverse_transform(scaled_2d.flatten().reshape(-1, 1))
            .reshape(N, P)
        )

    # --- Public API ---

    def evaluate_all(self):
        result_base = self.cfg.get('result_dir', 'results/evaluation')

        tree_enabled = [m for m in ['xgboost', 'lightgbm'] if self.cfg['models'][m]['enabled']]
        if tree_enabled:
            X_test, y_test   = self._tree_test_split()
            X_train, y_train = self._tree_train_split()
            y_true_np        = y_test.values
            y_true_train_np  = y_train.values
            timestamps       = pd.to_datetime(y_test.index)
            timestamps_train = pd.to_datetime(y_train.index)

            for model_name in tree_enabled:
                print(f"\n====== Loading {model_name.upper()} ======")
                model_path = self.cfg['models'][model_name]['model_path']
                # Mirror the model's directory name (includes _lds suffix when applicable)
                run_tag    = os.path.basename(os.path.dirname(model_path))
                result_dir = os.path.join(result_base, model_name, run_tag)
                mod = xgboost_mod if model_name == 'xgboost' else lightgbm_mod
                mod.evaluate(model_path, X_test, y_true_np, timestamps, result_dir,
                             X_train=X_train, y_true_train=y_true_train_np,
                             timestamps_train=timestamps_train)

        seq_enabled = [m for m in ['transformer', 'lstm'] if self.cfg['models'][m]['enabled']]
        if seq_enabled:
            X_te, y_te_scaled, test_timestamps     = self._seq_test_split()
            X_tr, y_tr_scaled, train_timestamps    = self._seq_train_split()
            y_true_mw       = self._inverse_transform(y_te_scaled)
            y_true_train_mw = self._inverse_transform(y_tr_scaled)

            for model_name in seq_enabled:
                print(f"\n====== Loading {model_name.upper()} ======")
                model_path = self.cfg['models'][model_name]['model_path']
                # Mirror the model's directory name (includes _lds suffix when applicable)
                run_tag    = os.path.basename(os.path.dirname(model_path))
                result_dir = os.path.join(result_base, model_name, run_tag)
                mod    = transformer_mod if model_name == 'transformer' else lstm_mod
                params = TRANSFORMER_PARAMS if model_name == 'transformer' else LSTM_PARAMS
                mod.evaluate(model_path, X_te, y_true_mw, self.y_scaler,
                             test_timestamps, result_dir, params,
                             X_train=X_tr, y_true_train_mw=y_true_train_mw,
                             timestamps_train=train_timestamps)

    def show_single_day(self, model_name, model_path, date_str):
        from pathlib import Path

        if model_name in ['xgboost', 'lightgbm']:
            target = pd.to_datetime(date_str)
            if target not in self.X_opt.index:
                raise ValueError(f"{date_str} not found in 2D matrix index.")
            x_row    = self.X_opt.loc[[target]].drop(columns=['is_target_valid'])
            true_24h = self.y_opt.loc[target].values
            mod      = xgboost_mod if model_name == 'xgboost' else lightgbm_mod
            pred_24h = mod.predict(model_path, x_row).flatten()

        else:  # transformer / lstm
            target = pd.to_datetime(date_str)
            idx = np.where(pd.to_datetime(self.timestamps_3d) == target)[0]
            if len(idx) == 0:
                raise ValueError(f"{date_str} not found in 3D matrix timestamps.")
            mod    = transformer_mod if model_name == 'transformer' else lstm_mod
            params = TRANSFORMER_PARAMS if model_name == 'transformer' else LSTM_PARAMS
            pred_24h = self._inverse_transform(
                mod.predict(model_path, self.X_3d[idx], params)
            ).flatten()
            true_24h = self._inverse_transform(self.y_3d[idx]).flatten()

        model_subdir = os.path.join(*Path(model_path).parts[2:-1])
        save_path = os.path.join('results', DATASET, 'singleday', model_subdir, f'{date_str}.png')
        plot_single_day(model_name.upper(), date_str, true_24h, pred_24h, save_path=save_path)
