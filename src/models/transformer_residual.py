# src/models/transformer_residual.py
"""
The transformer, trained on the residual against a naive same-hour-last-week baseline.

Same 3D matrix, same TimeSeriesTransformer3D, same params and the same split as
src/models/transformer.py — only the target moves (see src/models/_residual.py).

Two things the tree version does not have to deal with:

  1. The 3D matrix is standardized, so the baseline cannot be read straight off it. It is
     recovered by inverting the sequence scaler on channel 0 (Load_Estimated) of the
     window — the same 168 h of raw-MW preliminary load the 2D path reads.

  2. The residual (hundreds of MW) is on a different scale from the standardized target the
     network expects, so it gets its OWN StandardScaler, fit on TRAINING samples only. That
     scaler is SAVED next to the weights: it is as much a part of the model as the weights
     are, and a model that cannot invert its own output is not a model.

predict() returns predictions standardized by the y_scaler — the same coordinate system
transformer.predict() returns. The whole residual round-trip (net -> res_scaler -> +baseline
-> y_scaler) happens inside, so every caller that already knows how to drive a sequence
model drives this one unchanged, including the forecast path.

NOTE: LDS/FDS, if enabled, bin on the TARGET — which here is the residual, not the load.
That follows from "same model, only the target changes"; it is printed at startup so it is
not a surprise.
"""

import glob
import os

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.models._eval_utils import EvalUtils
from src.models._residual import sequence_baseline, print_metrics
from src.models._seq_trainer import train_sequence
from src.models._utils import _make_run_dir
from src.models.transformer import TimeSeriesTransformer3D, predict as _net_predict
from src.config import (
    MATRIX_DIR, TRANSFORMER_FEATURE_CONFIG, TRANSFORMER_RESIDUAL_PARAMS, EMBARGO_DAYS,
)

NAME = 'TRANSFORMER_RESIDUAL'
MODEL_TYPE = 'transformer_residual'
FILENAME = 'transformer_residual_best.pth'
RES_SCALER = 'residual_scaler.pkl'      # lives beside the weights — part of the model


def _load_scaler(matrix_dir, pattern):
    hits = glob.glob(os.path.join(matrix_dir, pattern))
    if not hits:
        raise FileNotFoundError(f"No scaler matching '{pattern}' in {matrix_dir}")
    return joblib.load(sorted(hits)[0])


def _scalers(model_path):
    """The three scalers this model needs: sequence (for the baseline), residual (its own),
    and y (to hand results back in the coordinate system every caller expects)."""
    lb = TRANSFORMER_FEATURE_CONFIG['lookback_hours']
    h  = TRANSFORMER_FEATURE_CONFIG['latest_info_hour']
    seq_scaler = _load_scaler(MATRIX_DIR, f'scaler_ts_lb{lb}_h{h}.pkl')
    y_scaler   = _load_scaler(MATRIX_DIR, f'y_scaler_lb{lb}_h{h}.pkl')

    res_path = os.path.join(os.path.dirname(model_path), RES_SCALER)
    if not os.path.exists(res_path):
        raise FileNotFoundError(
            f"{res_path} is missing. The residual scaler is saved at train time and is "
            f"required to turn the network's output back into MW — without it the weights "
            f"are unusable."
        )
    return seq_scaler, joblib.load(res_path), y_scaler


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X_3d, y_3d, mask_3d, params=None, feature_cfg=None, dataset=None):
    """y_3d is the y_scaler-standardized metered load, exactly as the plain transformer
    receives it. It is brought back to MW here so the residual can be measured in MW."""
    from src.feature_engine import _split_indices, apply_embargo

    print("\n--- Training Transformer (residual target) ---")
    params = {**(params or TRANSFORMER_RESIDUAL_PARAMS)}
    fc = feature_cfg or TRANSFORMER_FEATURE_CONFIG
    mode = params.pop('baseline', 'hourly')
    N, H = y_3d.shape

    lb = fc['lookback_hours']
    hh = fc['latest_info_hour']
    seq_scaler = _load_scaler(MATRIX_DIR, f'scaler_ts_lb{lb}_h{hh}.pkl')
    y_scaler   = _load_scaler(MATRIX_DIR, f'y_scaler_lb{lb}_h{hh}.pkl')

    baseline = sequence_baseline(X_3d, seq_scaler, mode)                      # (N, 24) MW
    y_mw = y_scaler.inverse_transform(y_3d.reshape(-1, 1)).reshape(N, H)      # (N, 24) MW
    residual_mw = y_mw - baseline

    # The same split (+embargo, +mask denoise) the plain transformer trains on, so the
    # residual scaler sees exactly the samples the network will fit on and not one more.
    train_pool, _test = _split_indices(N, fc['split_strategy'], fc['test_frac'], fc['random_state'])
    rtr, rval = _split_indices(len(train_pool), fc['val_strategy'], fc['val_frac'], fc['random_state'])
    train_idx, val_idx = apply_embargo(train_pool[rtr], train_pool[rval], EMBARGO_DAYS)
    fit_idx = train_idx[mask_3d[train_idx]]

    res_scaler = StandardScaler().fit(residual_mw[fit_idx].reshape(-1, 1))
    y_res_scaled = res_scaler.transform(
        residual_mw.reshape(-1, 1)).reshape(N, H).astype('float32')

    print(f"Baseline mode:    {mode}")
    print(f"Residual (MW):    mean {residual_mw[fit_idx].mean():+.1f}  "
          f"std {residual_mw[fit_idx].std():.1f}  -> standardized by its own scaler "
          f"(fit on {len(fit_idx)} train samples only)")
    if params.get('use_lds') or params.get('use_fds'):
        print(f"NOTE: use_lds={params.get('use_lds')} use_fds={params.get('use_fds')} — these "
              f"bin on the TARGET, which here is the residual, not the load.")

    print_metrics('naive baseline alone (train)', y_mw[fit_idx], baseline[fit_idx])

    train_sequence(TimeSeriesTransformer3D, MODEL_TYPE, FILENAME,
                   X_3d, y_res_scaled, mask_3d, params, fc, dataset)

    model_dir = _make_run_dir('models', MODEL_TYPE, fc, dataset,
                              use_lds=params.get('use_lds', False),
                              use_fds=params.get('use_fds', False))
    joblib.dump({'scaler': res_scaler, 'baseline': mode},
                os.path.join(model_dir, RES_SCALER))
    print(f"Residual scaler saved to: {os.path.join(model_dir, RES_SCALER)}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model_path, X_np, params=None):
    """Returns y_scaler-standardized load, shape (N, 24) — same contract as
    transformer.predict, so ModelEvaluator and ModelPredictor need no special case."""
    params = params or TRANSFORMER_RESIDUAL_PARAMS
    seq_scaler, res_bundle, y_scaler = _scalers(model_path)
    res_scaler, mode = res_bundle['scaler'], res_bundle['baseline']

    res_scaled = _net_predict(model_path, X_np, params)                       # (N, 24)
    n, h = res_scaled.shape
    res_mw = res_scaler.inverse_transform(res_scaled.reshape(-1, 1)).reshape(n, h)
    load_mw = res_mw + sequence_baseline(X_np, seq_scaler, mode)

    # Back into y-standardized space: the caller inverse-transforms with y_scaler, so this
    # round-trip is exactly cancelled and the residual plumbing stays invisible to it.
    return y_scaler.transform(load_mw.reshape(-1, 1)).reshape(n, h)


def evaluate(model_path, X_test, y_true_mw, y_scaler, timestamps, result_dir,
             params=None, X_train=None, y_true_train_mw=None, timestamps_train=None):
    """Same signature as transformer.evaluate, plus the naive baseline's own score — the
    residual model is only worth its complexity if it beats the thing it leans on."""
    params = params or TRANSFORMER_RESIDUAL_PARAMS
    seq_scaler, res_bundle, _ = _scalers(model_path)

    y_pred_scaled = predict(model_path, X_test, params)
    n, h = y_pred_scaled.shape
    y_pred_mw = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(n, h)

    base_test = sequence_baseline(X_test, seq_scaler, res_bundle['baseline'])
    print(f"\n=== {NAME}: model vs the baseline it leans on (test) ===")
    print_metrics('naive baseline alone', y_true_mw, base_test)
    print_metrics('residual + baseline', y_true_mw, y_pred_mw)

    train_df = None
    if X_train is not None and y_true_train_mw is not None:
        p = predict(model_path, X_train, params)
        n2, h2 = p.shape
        y_pred_train_mw = y_scaler.inverse_transform(p.reshape(-1, 1)).reshape(n2, h2)
        train_df = EvalUtils.build_detailed_df(NAME, y_true_train_mw, y_pred_train_mw,
                                               timestamps_train)
    EvalUtils.evaluate_one(NAME, y_true_mw, y_pred_mw, timestamps, result_dir, train_df)
