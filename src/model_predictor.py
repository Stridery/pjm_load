# src/model_predictor.py
"""
Forecast the days PJM has not verified yet, for every enabled model.

Mirrors ModelEvaluator: same config shape, same model registry, same run_tag → result_dir
convention. The difference is what it runs on — cleaned/predict.csv rather than the test
split — and that there is no verified label to score against.

Output, one file per model, dropped straight into that model's evaluation folder so a run's
forecast and its backtest live together:
    results/{DATASET}/evaluation/{model}/{run_tag}/{MODEL}_forecast.csv
        datetime, {MODEL}_pred [, preliminary_load, signed_error, abs_error, mape_pct]

The bracketed columns appear only when PREDICT_CONFIG['compare_to_preliminary'] is set —
i.e. only when the preliminary series is really this zone's load. For bge it is the MIDATL
regional aggregate, so the forecast ships without a truth column rather than with a fake one.
"""

import os

import numpy as np
import pandas as pd

from src.config import (
    DATASET, MATRIX_DIR, EVAL_CONFIG,
    TRANSFORMER_PARAMS, LSTM_PARAMS, MOE_TRANSFORMER_PARAMS, MSTNN_PARAMS,
    TRANSFORMER_RESIDUAL_PARAMS,
)
from src.prediction_engine import (
    build_tree_features, build_sequence_features, verify_against_training_matrix,
)
# One registry, shared with the evaluator — a model added there is forecastable here for
# free, and the two can never disagree about which models exist or how to drive them.
from src.model_evaluator import TREE_MODELS, SEQ_MODELS, TREE_MOD, SEQ_MOD

_SEQ_PARAMS = {'transformer': TRANSFORMER_PARAMS, 'lstm': LSTM_PARAMS,
               'moe_transformer': MOE_TRANSFORMER_PARAMS, 'mstnn': MSTNN_PARAMS,
               'transformer_residual': TRANSFORMER_RESIDUAL_PARAMS}


class ModelPredictor:

    def __init__(self, predict_cfg):
        self.cfg = predict_cfg
        self.X_tree = self.prelim_tree = None
        self.X_seq = self.ts_seq = self.prelim_seq = self.y_scaler = None

    def _enabled(self, names):
        return [m for m in names if self.cfg['models'].get(m, {}).get('enabled')]

    # --- Data ---

    def load_data(self, verify=True):
        """Build whatever the currently-enabled models need, once.

        Idempotent on purpose: the training pipeline reuses one predictor across all eight
        models, swapping cfg['models'] between them. Building only what is missing means the
        tree matrix is built when the first tree model asks for it and the sequence matrix
        when the first sequence model does — which also sidesteps an ordering trap, since the
        tree models train before the 3D matrix exists at all.

        `verify` re-derives days the training matrix already holds and asserts they match —
        cheap, and the only guard against silent feature drift.
        """
        path = self.cfg['predict_path']

        if self._enabled(SEQ_MODELS) and self.X_seq is None:
            if verify:
                verify_against_training_matrix(MATRIX_DIR, path)
            self.X_seq, self.ts_seq, self.prelim_seq, self.y_scaler = \
                build_sequence_features(path, MATRIX_DIR)

        if self._enabled(TREE_MODELS) and self.X_tree is None:
            self.X_tree, self.prelim_tree = build_tree_features(path)

        built = [len(x) for x in (self.ts_seq, self.X_tree) if x is not None]
        n = max(built) if built else 0
        if n == 0:
            print("No forecastable days: every day in predict.csv already has verified "
                  "metered load. Re-crawl once PJM publishes more preliminary data.")
        return n

    # --- Output ---

    def _write(self, model_name, model_path, timestamps, y_pred, prelim_df):
        # Straight into the model's evaluation folder — same run_tag convention as
        # ModelEvaluator — so a run's forecast sits beside its detailed_errors.csv.
        run_tag    = os.path.basename(os.path.dirname(model_path))
        result_dir = os.path.join(EVAL_CONFIG['result_dir'], model_name, run_tag)
        os.makedirs(result_dir, exist_ok=True)

        dates     = pd.to_datetime(pd.Series(timestamps)).repeat(24)
        hours     = np.tile(np.arange(24), len(timestamps))
        datetimes = dates.values + pd.to_timedelta(hours, unit='h')

        col = f'{model_name.upper()}_pred'
        df = pd.DataFrame({'datetime': datetimes, col: y_pred.flatten()})

        if self.cfg.get('compare_to_preliminary'):
            # The last two days reach PAST the data — a genuine day-ahead forecast, with no
            # preliminary load to score against yet. Their truth columns are NaN, and the MAPE
            # below is over the days that do have one. Say which is which; a single blended
            # number would quietly imply we had scored days we have not.
            df['preliminary_load'] = prelim_df.loc[list(timestamps)].values.flatten()
            df['signed_error'] = df[col] - df['preliminary_load']
            df['abs_error']    = df['signed_error'].abs()
            df['mape_pct']     = df['abs_error'] / df['preliminary_load'] * 100

            scored = int(prelim_df.loc[list(timestamps)].notna().any(axis=1).sum())
            ahead  = len(timestamps) - scored
            summary = (f"{scored} day(s) scored vs preliminary: MAPE {df['mape_pct'].mean():.2f}%"
                       f"  |  {ahead} day-ahead, no truth yet")
        else:
            summary = ("no truth column — this zone's preliminary series is a regional "
                       "aggregate, not its own load")

        out = os.path.join(result_dir, f'{model_name.upper()}_forecast.csv')
        df.to_csv(out, index=False)
        print(f"  {model_name:16s} {len(timestamps)} day(s) → {out}")
        print(f"  {'':16s} {summary}")
        return df

    # --- Run ---

    def predict_all(self):
        print(f"\n=== Forecasting [{DATASET}] "
              f"({'scored against preliminary' if self.cfg.get('compare_to_preliminary') else 'unscored'}) ===")

        for name in self._enabled(TREE_MODELS):
            model_path = self.cfg['models'][name]['model_path']
            mod = TREE_MOD[name]
            y_pred = mod.predict(model_path, self.X_tree)        # trees emit MW directly
            self._write(name, model_path, self.X_tree.index.values, y_pred, self.prelim_tree)

        for name in self._enabled(SEQ_MODELS):
            model_path = self.cfg['models'][name]['model_path']
            mod, params = SEQ_MOD[name], _SEQ_PARAMS[name]
            if name == 'moe_transformer':
                y_scaled = mod.predict(model_path, self.X_seq, self.ts_seq, params)
            else:
                y_scaled = mod.predict(model_path, self.X_seq, params)
            # Sequence models emit STANDARDIZED load. Invert with training's y_scaler —
            # a fresh scaler here would rescale every number and never raise.
            y_pred = self.y_scaler.inverse_transform(y_scaled)
            self._write(name, model_path, self.ts_seq, y_pred, self.prelim_seq)
