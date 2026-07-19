# src/models/moe_mstnn_residual.py
"""
MoE-MSTNN, trained on the residual against a naive same-hour-last-week baseline.

Same MoEMSTNN / same params / same split as src/models/moe_mstnn.py — only the target
moves. The residual machinery (including the MoE regime / per-expert evaluation) is shared
(src/models/_residual.make_residual_model); this file is just the registration.
"""

from src.models._residual import make_residual_model
from src.models.moe_mstnn import MoEMSTNN
from src.config import MOE_MSTNN_FEATURE_CONFIG, MOE_MSTNN_RESIDUAL_PARAMS

train, predict, evaluate = make_residual_model(
    name='MOE_MSTNN_RESIDUAL',
    model_type='moe_mstnn_residual',
    filename='moe_mstnn_residual_best.pth',
    feature_cfg=MOE_MSTNN_FEATURE_CONFIG,
    params_default=MOE_MSTNN_RESIDUAL_PARAMS,
    model_cls=MoEMSTNN,
    is_moe=True,
    expert_prefix='MOE_MSTNN',
)
