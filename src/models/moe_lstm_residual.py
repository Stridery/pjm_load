# src/models/moe_lstm_residual.py
"""
MoE-LSTM, trained on the residual against a naive same-hour-last-week baseline.

Same MoELSTM / same params / same split as src/models/moe_lstm.py — only the target moves.
The residual machinery (including the MoE regime / per-expert evaluation) is shared
(src/models/_residual.make_residual_model); this file is just the registration.
"""

from src.models._residual import make_residual_model
from src.models.moe_lstm import MoELSTM
from src.config import MOE_LSTM_FEATURE_CONFIG, MOE_LSTM_RESIDUAL_PARAMS

train, predict, evaluate = make_residual_model(
    name='MOE_LSTM_RESIDUAL',
    model_type='moe_lstm_residual',
    filename='moe_lstm_residual_best.pth',
    feature_cfg=MOE_LSTM_FEATURE_CONFIG,
    params_default=MOE_LSTM_RESIDUAL_PARAMS,
    model_cls=MoELSTM,
    is_moe=True,
    expert_prefix='MOE_LSTM',
)
