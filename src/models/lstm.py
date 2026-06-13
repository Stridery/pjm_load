import numpy as np
import torch
import torch.nn as nn

from src.models._seq_trainer import train_sequence
from src.models._eval_utils import EvalUtils
from src.config import LSTM_FEATURE_CONFIG, LSTM_PARAMS


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, num_features, params):
        super().__init__()
        hidden = params['hidden_size']
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden,
            num_layers=params['num_layers'],
            batch_first=True,
            dropout=params['dropout'] if params['num_layers'] > 1 else 0.0,
        )
        fc_hidden = params.get('fc_hidden', 128)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(fc_hidden, params['out_dim']),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc_out(out)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X_3d, y_3d, mask_3d, params=None, feature_cfg=None, dataset=None):
    print("\n--- Training LSTM ---")
    train_sequence(
        LSTMModel, 'lstm', 'lstm_best.pth',
        X_3d, y_3d, mask_3d,
        params or LSTM_PARAMS,
        feature_cfg or LSTM_FEATURE_CONFIG,
        dataset,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model_path, X_np, params):
    """Load a saved model and return scaled predictions (N, out_dim)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(X_np.shape[2], params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_np).to(device)).cpu().numpy()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model_path, X_test, y_true_mw, y_scaler, timestamps, result_dir,
             params=None):
    """Predict, inverse-transform, then run the full evaluation suite."""
    params = params or LSTM_PARAMS
    y_pred_scaled = predict(model_path, X_test, params)
    N, P = y_pred_scaled.shape
    y_pred_mw = (
        y_scaler
        .inverse_transform(y_pred_scaled.flatten().reshape(-1, 1))
        .reshape(N, P)
    )
    EvalUtils.evaluate_one('LSTM', y_true_mw, y_pred_mw, timestamps, result_dir)
