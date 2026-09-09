import os
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
