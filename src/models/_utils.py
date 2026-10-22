import os

import numpy as np

from src.config import DATASET, MATRIX_TAG, TARGET_TAG


def run_tag(cfg):
    """The run's directory name: the split description, plus the feature-set tag.

    MATRIX_TAG is part of it so that models trained under different feature sets land in
    DIFFERENT directories instead of overwriting each other. Without it, turning the
    congestion block on and retraining would destroy the previous 16 models — and, worse,
    leave no way to A/B the two feature sets, since the comparison needs both on disk at
    once. It also keeps serving honest: a run dir's name now states which feature set its
    weights expect, and src/serving/predict refuses to load one that disagrees with config.
    """
    tag = f"{cfg['split_strategy']}_test{cfg['test_frac']}"
    if 'val_strategy' in cfg:
        tag += f"_{cfg['val_strategy']}_val{cfg['val_frac']}"
    return tag + MATRIX_TAG


def _make_run_dir(base, model_type, cfg, dataset=None):
    """Directory a run writes into: {base}/{dataset}{target}/{model_type}/{run tag}.

    Defaults to the configured DATASET; an explicit `dataset` still wins — that is how the
    joint models reach models/joint_dom_bge/.

    TARGET_TAG is appended for the same reason MATRIX_TAG is in run_tag(): a run under a
    different PJM_TARGET is a different model of a different series, and without the tag a
    price run silently OVERWRITES the load weights of the same architecture. It must be
    added here and not only in cfg.MODEL_ROOT, because this is the path the trainers
    actually save to while the evaluator resolves cfg.MODEL_ROOT — the two disagreeing is
    how the clobber goes unnoticed.
    """
    path = os.path.join(base, (dataset or DATASET) + TARGET_TAG, model_type, run_tag(cfg))
    os.makedirs(path, exist_ok=True)
    return path


def booster_matrix(X, models):
    """A DataFrame of flattened-window features as a contiguous float32 array, in the order
    the boosters were fitted on. Anything that is not a DataFrame is returned untouched.

    Serving hands the tree models one row of 3750 columns per target day, once per hour, and
    XGBoost's pandas path walks that frame column by column: pandas_transform_data was called
    540,000 times for three days of forecasts, and 24 boosters took 5,411 ms. The same values
    as an ndarray go through inplace_predict in 15 ms -- 360x, bit-identical output. At eight
    scored days nobody noticed; at ninety it is forty minutes of CI time.

    The column order comes from the fitted model's own feature_names_in_, not from the
    caller's frame. Dropping to a positional array removes the name check that was silently
    protecting us, so the order has to be re-established from the side that knows it --
    otherwise a builder that reordered its output would feed hour 3's column to hour 17's
    split and still return a plausible number.
    """
    if not hasattr(X, 'columns'):
        return X
    names = getattr(models[0], 'feature_names_in_', None)
    if names is not None:
        missing = [c for c in names if c not in X.columns]
        if missing:
            raise KeyError(f"{len(missing)} feature(s) the boosters were fitted on are absent "
                           f"from the sample, first few: {missing[:5]}")
        X = X[list(names)]
    return np.ascontiguousarray(X.to_numpy(dtype='float32'))
