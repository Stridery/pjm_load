import os
from src.config import DATASET


def _make_run_dir(base, model_type, cfg, dataset=None, use_lds=False, use_fds=False):
    tag = f"{cfg['split_strategy']}_test{cfg['test_frac']}"
    if 'val_strategy' in cfg:
        tag += f"_{cfg['val_strategy']}_val{cfg['val_frac']}"
    if use_lds:
        tag += '_lds'
    if use_fds:
        tag += '_fds'
    path = os.path.join(base, dataset or DATASET, model_type, tag)
    os.makedirs(path, exist_ok=True)
    return path
