# src/models/mstnn_residual.py
"""
MSTNN, trained on the residual against a naive same-hour-last-week baseline.

Same 3D matrix / same MSTNN / same params / same split as src/models/mstnn.py — only the
target moves. The residual machinery is shared (src/models/_residual.make_residual_model);
this file is just the registration.
"""

from src.models._residual import make_residual_model
from src.models.mstnn import MSTNN, predict as _net_predict
from src.config import MSTNN_FEATURE_CONFIG, MSTNN_RESIDUAL_PARAMS

train, predict, evaluate = make_residual_model(
    name='MSTNN_RESIDUAL',
    model_type='mstnn_residual',
    filename='mstnn_residual_best.pth',
    feature_cfg=MSTNN_FEATURE_CONFIG,
    params_default=MSTNN_RESIDUAL_PARAMS,
    model_cls=MSTNN,
    base_predict=_net_predict,
    is_moe=False,
)
