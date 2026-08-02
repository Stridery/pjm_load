# src/models/lstm_residual.py
"""
LSTM, trained on the residual against a naive same-hour-last-week baseline.

Same 3D matrix / same LSTM / same params / same split as src/models/lstm.py — only the target
moves. The residual machinery is shared (src/models/_residual.make_residual_model); this file
is just the registration.
"""

from src.models._residual import make_residual_model
from src.models.lstm import LSTMModel, predict as _net_predict
from src.config import LSTM_FEATURE_CONFIG, LSTM_RESIDUAL_PARAMS

train, predict, evaluate = make_residual_model(
    name='LSTM_RESIDUAL',
    model_type='lstm_residual',
    filename='lstm_residual_best.pth',
    feature_cfg=LSTM_FEATURE_CONFIG,
    params_default=LSTM_RESIDUAL_PARAMS,
    model_cls=LSTMModel,
    base_predict=_net_predict,
    is_moe=False,
)
