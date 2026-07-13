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

        # Static skip: per-timestep features (Load + weather) go through the LSTM; the
        # broadcast constants (forecast-day calendar + 3-week macro) bypass the sequence
        # and are concatenated to the final hidden state.
        self.n_seq = params.get('n_seq_features') or num_features
        self.n_static = num_features - self.n_seq
        self.enc_dim = hidden                        # learned representation (FDS calibrates this)
        self.feat_dim = hidden + self.n_static       # what the head consumes

        self.lstm = nn.LSTM(
            input_size=self.n_seq,
            hidden_size=hidden,
            num_layers=params['num_layers'],
            batch_first=True,
            dropout=params['dropout'] if params['num_layers'] > 1 else 0.0,
        )
        fc_hidden = params.get('fc_hidden', 128)
        self.fc_out = nn.Sequential(
            nn.Linear(self.feat_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(fc_hidden, params['out_dim']),
        )

    def encode(self, x):
        out, _ = self.lstm(x[:, :, :self.n_seq])
        z = out[:, -1, :]                            # (batch, hidden_size)
        if self.n_static > 0:
            z = torch.cat([z, x[:, 0, self.n_seq:]], dim=1)
        return z

    def decode(self, features):
        return self.fc_out(features)

    def forward(self, x):
        return self.decode(self.encode(x))


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
             params=None, X_train=None, y_true_train_mw=None, timestamps_train=None):
    """Predict, inverse-transform, then run the full evaluation suite. Pass X_train to include train subplot."""
    params = params or LSTM_PARAMS
    y_pred_scaled = predict(model_path, X_test, params)
    N, P = y_pred_scaled.shape
    y_pred_mw = y_scaler.inverse_transform(y_pred_scaled.flatten().reshape(-1, 1)).reshape(N, P)

    train_df = None
    if X_train is not None and y_true_train_mw is not None:
        y_ptr = predict(model_path, X_train, params)
        N2, P2 = y_ptr.shape
        y_pred_train_mw = y_scaler.inverse_transform(y_ptr.flatten().reshape(-1, 1)).reshape(N2, P2)
        train_df = EvalUtils.build_detailed_df('LSTM', y_true_train_mw, y_pred_train_mw, timestamps_train)

    EvalUtils.evaluate_one('LSTM', y_true_mw, y_pred_mw, timestamps, result_dir, train_df)
