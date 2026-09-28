# src/models/archformer_skip_residual.py
"""ARCHFormer_SKIP trained on the residual against a naive same-hour-last-week baseline.

Same variant, same params, same split as src/models/archformer_skip.py — only the target
moves. The residual machinery is shared (src/models/_residual.make_residual_model); this file
is just the registration, exactly like mstnn_residual.py.
"""

from src.models._residual import make_residual_model
from src.models.archformer import ARCHFormerNet
from src.models.archformer_skip import predict as _net_predict
from src.config import ARCHFORMER_FEATURE_CONFIG, ARCHFORMER_SKIP_RESIDUAL_PARAMS

train, predict, evaluate = make_residual_model(
    name='ARCHFORMER_SKIP_RESIDUAL',
    model_type='archformer_skip_residual',
    filename='archformer_skip_residual_best.pth',
    feature_cfg=ARCHFORMER_FEATURE_CONFIG,
    params_default=ARCHFORMER_SKIP_RESIDUAL_PARAMS,
    model_cls=ARCHFormerNet,
    base_predict=_net_predict,
    is_moe=False,
)
