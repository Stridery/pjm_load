import math
import numpy as np
import torch
import torch.nn as nn

from src.models._seq_trainer import train_sequence
from src.models._eval_utils import EvalUtils
from src.config import TRANSFORMER_FEATURE_CONFIG, TRANSFORMER_PARAMS


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TimeSeriesTransformer3D(nn.Module):
    def __init__(self, num_features, params):
        super().__init__()
        d_model = params['d_model']
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=params['nhead'],
            dim_feedforward=d_model * 4, dropout=params['dropout'], batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=params['num_layers'])
        fc_hidden = params.get('fc_hidden', 128)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, fc_hidden),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(fc_hidden, params['out_dim'])
        )

    def encode(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x[:, -1, :]            # (batch, d_model)

    def decode(self, features):
        return self.fc_out(features)

    def forward(self, x):
        return self.decode(self.encode(x))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X_3d, y_3d, mask_3d, params=None, feature_cfg=None, dataset=None):
    print("\n--- Training Transformer ---")
    train_sequence(
        TimeSeriesTransformer3D, 'transformer', 'transformer_best.pth',
        X_3d, y_3d, mask_3d,
        params or TRANSFORMER_PARAMS,
        feature_cfg or TRANSFORMER_FEATURE_CONFIG,
        dataset,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model_path, X_np, params):
    """Load a saved model and return scaled predictions (N, out_dim)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TimeSeriesTransformer3D(X_np.shape[2], params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_np).to(device)).cpu().numpy()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model_path, X_test, y_true_mw, y_scaler, timestamps, result_dir,
             params=None, X_train=None, y_true_train_mw=None, timestamps_train=None):
    """Predict, inverse-transform, then run the full evaluation suite. Pass X_train to include train subplot."""
    params = params or TRANSFORMER_PARAMS
    y_pred_scaled = predict(model_path, X_test, params)
    N, P = y_pred_scaled.shape
    y_pred_mw = y_scaler.inverse_transform(y_pred_scaled.flatten().reshape(-1, 1)).reshape(N, P)

    train_df = None
    if X_train is not None and y_true_train_mw is not None:
        y_ptr = predict(model_path, X_train, params)
        N2, P2 = y_ptr.shape
        y_pred_train_mw = y_scaler.inverse_transform(y_ptr.flatten().reshape(-1, 1)).reshape(N2, P2)
        train_df = EvalUtils.build_detailed_df('TRANSFORMER', y_true_train_mw, y_pred_train_mw, timestamps_train)

    EvalUtils.evaluate_one('TRANSFORMER', y_true_mw, y_pred_mw, timestamps, result_dir, train_df)
