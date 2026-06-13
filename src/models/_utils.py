import os
from src.config import DATASET


def _make_run_dir(base, model_type, cfg, dataset=None):
    tag = f"{cfg['split_strategy']}_test{cfg['test_frac']}"
    if 'val_strategy' in cfg:
        tag += f"_{cfg['val_strategy']}_val{cfg['val_frac']}"
    path = os.path.join(base, dataset or DATASET, model_type, tag)
    os.makedirs(path, exist_ok=True)
    return path
