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
import re
import os

import joblib
import numpy as np

from src.config import (MODEL_ROOT, MATRIX_DIR, TRANSFORMER_FEATURE_CONFIG, MATRIX_TAG, TARGET_TAG,
                        matrix_tag_of)
from src.model_evaluator import TREE_MODELS, SEQ_MODELS, TREE_MOD, SEQ_MOD, SEQ_PARAMS
from src.serving.features import build_sequence_sample, build_tree_sample

_ALL_MOD = {**TREE_MOD, **SEQ_MOD}


def _weights_for(model_root, model):
    """Weight files for `model` from runs trained on the CURRENTLY configured feature set.

    Run dirs are filtered by their tag rather than globbed with '*', because retraining under
    a new feature set now leaves the previous run dir in place beside the new one (see
    src/models/_utils.run_tag). Globbing would pick up both — either raising "multiple weight
    files" or, if only the stale one survived, loading weights whose input width no longer
    matches the assembled sample.
    """
    out = []
    for d in sorted(glob.glob(os.path.join(model_root, model, '*'))):
        if not os.path.isdir(d) or matrix_tag_of(os.path.basename(d)) != MATRIX_TAG:
            continue
        out += glob.glob(os.path.join(d, f'{model}_*.pkl'))
        out += glob.glob(os.path.join(d, f'{model}_best.pth'))
    return out


_ENS_RE = re.compile(r'^(?P<base>.+?)_ens(?P<k>\d+)(?P<suffix>_residual)?$')


def _ens_parts(name):
    """('archformer_skip_residual', 5) for 'archformer_skip_ens5_residual'; (name, None) else.

    The count sits in the MIDDLE of the residual variants' names because the suffix is what
    selects the target mode and has to stay on the end for every other lookup in this module.
    """
    m = _ENS_RE.match(name)
    if not m:
        return name, None
    return m.group('base') + (m.group('suffix') or ''), int(m.group('k'))


def _seed_weights(model_root, model):
    """The per-seed weight files for `model`, oldest seed first.

    Written by a training run with PJM_SEED set (see _seq_trainer). They sit beside the
    unseeded `{model}_best.pth`, which stays the single-seed control, so the two never
    overwrite each other and the ensemble can be served next to its own member.
    """
    out = []
    for d in sorted(glob.glob(os.path.join(model_root, model, '*'))):
        if not os.path.isdir(d) or matrix_tag_of(os.path.basename(d)) != MATRIX_TAG:
            continue
        # `{model}_best_seed{N}.pth`: _seq_trainer appends the seed to the stem, and the stem
        # already ends in _best. _weights_for's exact `{model}_best.pth` glob does not match
        # these, which is what keeps ensemble members out of the single-model lookup.
        out += glob.glob(os.path.join(d, f'{model}_best_seed*.pth'))
    # Numeric, not lexical: seed10 must not sort before seed9.
    return sorted(out, key=lambda s: int(re.search(r'_seed(\d+)\.pth$', s).group(1)))


def _weight_path(model_root, model):
    """The single saved-weight file for a model (one run dir per feature set)."""
    hits = _weights_for(model_root, model)
    if not hits:
        raise FileNotFoundError(
            f"No weights for '{model}' under {model_root}/{model}/ matching feature-set tag "
            f"{MATRIX_TAG!r} — train it, or align CONGESTION_CONFIG['enabled'] with the "
            f"models you have.")
    if len(hits) > 1:
        raise RuntimeError(f"Multiple weight files for '{model}': {hits}. Keep one run dir "
                           f"per feature set.")
    return hits[0]


def available_models(model_root=None):
    """The models that have weights for the configured feature set, in registry order."""
    model_root = model_root or MODEL_ROOT
    return [m for m in TREE_MODELS + SEQ_MODELS if _weights_for(model_root, m)]


def _y_scaler(matrix_dir):
    lb = TRANSFORMER_FEATURE_CONFIG['lookback_hours']
    hh = TRANSFORMER_FEATURE_CONFIG['latest_info_hour']
    hits = glob.glob(os.path.join(matrix_dir, f'y_scaler_lb{lb}_h{hh}{MATRIX_TAG}{TARGET_TAG}.pkl'))
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

    def _one(base, path):
        mod = _ALL_MOD[base]
        if base in TREE_MODELS:
            return mod.predict(path, X_opt)                                  # (1, 24) MW
        if base.startswith('moe_'):
            return y_scaler.inverse_transform(mod.predict(path, X_3d, ts, SEQ_PARAMS[base]))
        return y_scaler.inverse_transform(mod.predict(path, X_3d, SEQ_PARAMS[base]))

    out = {}
    for m in models:
        # `{base}_ens{K}` asks for the mean of K per-seed weight files. Averaging the
        # PREDICTIONS, not the scores: by the triangle inequality the mean prediction's error
        # is at most the mean of the members' errors, so this is a model in its own right and
        # never worse than drawing one member at random. The name is not a registry entry --
        # it resolves to the same class and params as `{base}`, only with more weights.
        base, k = _ens_parts(m)
        if k is None:
            out[m] = np.asarray(_one(m, _weight_path(model_root, m)), dtype=float)[0]
            continue
        paths = _seed_weights(model_root, base)[:k]
        if len(paths) < k:
            raise FileNotFoundError(
                f"{m} needs {k} per-seed weight files for '{base}' under {model_root}, found "
                f"{len(paths)}. Train with PJM_SEED set, one run per seed.")
        members = [np.asarray(_one(base, p), dtype=float)[0] for p in paths]
        out[m] = np.mean(members, axis=0)
    return out
