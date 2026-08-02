# src/models/moe_lstm.py
"""
MoE-LSTM: the MoE regime head on top of the LSTM sequence encoder.

The MoE is "shared encoder + 12 regime expert heads + hard season routing". Here the shared
encoder is the LSTM's final hidden state (same as the vanilla LSTM) instead of the
transformer's attention or MSTNN's conv; the head is the exact same RegimeHead. So this is the
MoE transformer with its encoder swapped for an LSTM one.

Everything else — the season-routed training loop, predict, and the regime / per-expert
evaluation — is reused from moe_transformer.py via its model_cls hook. Only the encoder (this
file) and the config differ.
"""

import torch
import torch.nn as nn

from src.models._moe_head import RegimeHead
from src.models import moe_transformer as moe
from src.config import MOE_LSTM_PARAMS, MOE_LSTM_FEATURE_CONFIG

MODEL_TYPE = 'moe_lstm'
FILENAME = 'moe_lstm_best.pth'


class MoELSTM(nn.Module):
    def __init__(self, num_features, params):
        super().__init__()
        hidden    = params['hidden_size']
        out_dim   = params['out_dim']
        dropout   = params['dropout']
        fc_hidden = params.get('expert_fc_hidden', 64)
        self.out_dim = out_dim

        # Static skip: only the per-timestep features go through the LSTM; the broadcast
        # constants (calendar + macro + thermal-static) bypass it into the head.
        self.n_seq = params.get('n_seq_features') or num_features
        self.n_static = num_features - self.n_seq
        self.enc_dim = hidden                        # learned representation (FDS calibrates this)
        self.feat_dim = hidden + self.n_static       # what the expert heads consume

        self.lstm = nn.LSTM(
            input_size=self.n_seq,
            hidden_size=hidden,
            num_layers=params['num_layers'],
            batch_first=True,
            dropout=params['dropout'] if params['num_layers'] > 1 else 0.0,
        )

        # --- shared MoE head: 12 regime experts + hard season routing ---
        self.head = RegimeHead(self.feat_dim, out_dim, fc_hidden, dropout)

    def encode(self, x):
        out, _ = self.lstm(x[:, :, :self.n_seq])
        z = out[:, -1, :]                            # (batch, hidden_size)
        if self.n_static > 0:
            z = torch.cat([z, x[:, 0, self.n_seq:]], dim=1)   # (batch, feat_dim)
        return z

    def decode(self, z, season_idx):
        return self.head(z, season_idx)

    def forward(self, x, season_idx):
        return self.decode(self.encode(x), season_idx)


# ---------------------------------------------------------------------------
# Train / predict / evaluate — all delegate to the MoE machinery with model_cls
# ---------------------------------------------------------------------------

def train(X_3d, y_3d, mask_3d, timestamps_3d, params=None, feature_cfg=None, dataset=None):
    print("\n--- Training MoE-LSTM ---")
    moe.train(X_3d, y_3d, mask_3d, timestamps_3d,
              params or MOE_LSTM_PARAMS, feature_cfg or MOE_LSTM_FEATURE_CONFIG, dataset,
              model_type_name=MODEL_TYPE, save_name=FILENAME, model_cls=MoELSTM)


def predict(model_path, X_np, timestamps, params=None):
    return moe.predict(model_path, X_np, timestamps, params or MOE_LSTM_PARAMS, model_cls=MoELSTM)


def evaluate(model_path, X_test, y_true_mw, y_scaler, timestamps, result_dir,
             params=None, X_train=None, y_true_train_mw=None, timestamps_train=None):
    moe.evaluate(model_path, X_test, y_true_mw, y_scaler, timestamps, result_dir,
                 params=params or MOE_LSTM_PARAMS, X_train=X_train,
                 y_true_train_mw=y_true_train_mw, timestamps_train=timestamps_train,
                 model_cls=MoELSTM, name='MOE_LSTM', name_prefix='MOE_LSTM')
