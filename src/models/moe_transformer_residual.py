# src/models/moe_transformer_residual.py
"""
The MoE transformer, trained on the residual against a naive same-hour-last-week baseline.

Same MoETransformer / same params / same split as src/models/moe_transformer.py — only the
target moves. The residual machinery (including the MoE regime / per-expert evaluation) is
shared (src/models/_residual.make_residual_model); this file is just the registration.
"""

from src.models._residual import make_residual_model
from src.models.moe_transformer import MoETransformer
from src.config import MOE_TRANSFORMER_FEATURE_CONFIG, MOE_TRANSFORMER_RESIDUAL_PARAMS

train, predict, evaluate = make_residual_model(
    name='MOE_TRANSFORMER_RESIDUAL',
    model_type='moe_transformer_residual',
    filename='moe_transformer_residual_best.pth',
    feature_cfg=MOE_TRANSFORMER_FEATURE_CONFIG,
    params_default=MOE_TRANSFORMER_RESIDUAL_PARAMS,
    model_cls=MoETransformer,
    is_moe=True,
    expert_prefix='MOE',
)
