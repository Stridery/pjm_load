"""
Run every trained model on the assembled forecast sample for one target day.

Reuses the evaluator's model registry (TREE_MODELS / SEQ_MODELS / *_MOD / SEQ_PARAMS) so serving
and evaluation can never disagree about which models exist or how to drive them. Feature parity
with training is guaranteed by src/serving/features (golden-tested); this layer only loads the
weights and calls each model's predict() with the convention its family needs:
    tree           predict(path, X_opt)                     -> MW directly
    sequence       predict(path, X_3d, params)              -> y-standardized, inverse below
    MoE sequence    predict(path, X_3d, timestamps, params) -> y-standardized, inverse below
Residual variants are transparent: their own predict() adds the baseline back internally.
"""

import glob
import os

import joblib
import numpy as np

from src.config import MODEL_ROOT, MATRIX_DIR, TRANSFORMER_FEATURE_CONFIG
from src.model_evaluator import TREE_MODELS, SEQ_MODELS, TREE_MOD, SEQ_MOD, SEQ_PARAMS
from src.serving.features import build_sequence_sample, build_tree_sample

_ALL_MOD = {**TREE_MOD, **SEQ_MOD}


def _weight_path(model_root, model):
    """The single saved-weight file for a model (each model has one run_tag dir)."""
    hits = (glob.glob(os.path.join(model_root, model, '*', f'{model}_*.pkl'))
            + glob.glob(os.path.join(model_root, model, '*', f'{model}_best.pth')))
    if not hits:
        raise FileNotFoundError(f"No weights for '{model}' under {model_root}/{model}/ — train it.")
    if len(hits) > 1:
        raise RuntimeError(f"Multiple weight files for '{model}': {hits}. Keep one run_tag.")
    return hits[0]


def available_models(model_root=None):
    """The models that actually have weights on disk, in registry order."""
    model_root = model_root or MODEL_ROOT
    out = []
    for m in TREE_MODELS + SEQ_MODELS:
        if glob.glob(os.path.join(model_root, m, '*', f'{m}_*.pkl')) or \
           glob.glob(os.path.join(model_root, m, '*', f'{m}_best.pth')):
            out.append(m)
    return out


def _y_scaler(matrix_dir):
    lb = TRANSFORMER_FEATURE_CONFIG['lookback_hours']
    hh = TRANSFORMER_FEATURE_CONFIG['latest_info_hour']
    hits = glob.glob(os.path.join(matrix_dir, f'y_scaler_lb{lb}_h{hh}.pkl'))
    if not hits:
        raise FileNotFoundError(f"No y_scaler in {matrix_dir} — train first.")
    return joblib.load(hits[0])


def forecast_day(window_df, target_day, models=None, model_root=None, matrix_dir=None):
    """{model: np.array(24) of MW} — every model's forecast for `target_day`.

    window_df must hold the recent ~30 days PLUS a (calendar-only is fine) row for target_day.
    """
    model_root = model_root or MODEL_ROOT
    matrix_dir = matrix_dir or MATRIX_DIR
    models = models or available_models(model_root)

    need_seq  = any(m in SEQ_MODELS for m in models)
    need_tree = any(m in TREE_MODELS for m in models)
    X_3d = build_sequence_sample(window_df, target_day, matrix_dir)[0] if need_seq else None
    X_opt = build_tree_sample(window_df, target_day, matrix_dir) if need_tree else None
    y_scaler = _y_scaler(matrix_dir) if need_seq else None
    ts = np.array([target_day])

    out = {}
    for m in models:
        path = _weight_path(model_root, m)
        mod = _ALL_MOD[m]
        if m in TREE_MODELS:
            pred = mod.predict(path, X_opt)                                  # (1, 24) MW
        elif m.startswith('moe_'):
            pred = y_scaler.inverse_transform(mod.predict(path, X_3d, ts, SEQ_PARAMS[m]))
        else:
            pred = y_scaler.inverse_transform(mod.predict(path, X_3d, SEQ_PARAMS[m]))
        out[m] = np.asarray(pred, dtype=float)[0]
    return out
