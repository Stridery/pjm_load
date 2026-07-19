# src/models/transformer_residual.py
"""
The transformer, trained on the residual against a naive same-hour-last-week baseline.

Same 3D matrix / same TimeSeriesTransformer3D / same params / same split as
src/models/transformer.py — only the target moves. All the residual machinery lives in
src/models/_residual.make_residual_model; this file is just the registration.
"""

from src.models._residual import make_residual_model
from src.models.transformer import TimeSeriesTransformer3D, predict as _net_predict
from src.config import TRANSFORMER_FEATURE_CONFIG, TRANSFORMER_RESIDUAL_PARAMS

train, predict, evaluate = make_residual_model(
    name='TRANSFORMER_RESIDUAL',
    model_type='transformer_residual',
    filename='transformer_residual_best.pth',
    feature_cfg=TRANSFORMER_FEATURE_CONFIG,
    params_default=TRANSFORMER_RESIDUAL_PARAMS,
    model_cls=TimeSeriesTransformer3D,
    base_predict=_net_predict,
    is_moe=False,
)
