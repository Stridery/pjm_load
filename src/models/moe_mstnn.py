# src/models/moe_mstnn.py
"""
MoE-MSTNN: the MoE regime head on top of the MSTNN multi-scale conv encoder.

The MoE is "shared encoder + 12 regime expert heads + hard season routing". Here the
shared encoder is MSTNN's multi-scale conv over the (day, hour) grid instead of the
transformer's attention; the head is the exact same RegimeHead. So this is the MoE
transformer with its encoder swapped for a conv one.

Everything else — the season-routed training loop (LDS / FDS / Stage-2), predict, and
the regime / per-expert evaluation — is reused from moe_transformer.py via its model_cls
hook. Only the encoder (this file) and the config differ.
"""

import torch
import torch.nn as nn

from src.models._moe_head import RegimeHead
from src.models.mstnn import _ConvBranch
from src.models import moe_transformer as moe
from src.config import MOE_MSTNN_PARAMS, MOE_MSTNN_FEATURE_CONFIG

MODEL_TYPE = 'moe_mstnn'
FILENAME = 'moe_mstnn_best.pth'


class MoEMSTNN(nn.Module):
    def __init__(self, num_features, params):
        super().__init__()
        lookback = params['lookback_hours']
        assert lookback % 24 == 0, "lookback_hours must be a whole number of days"
        self.days = lookback // 24
        self.hours = 24
        out_dim   = params['out_dim']
        channels  = params.get('mstnn_channels', 40)
        kernels   = params.get('mstnn_kernels', [[3, 3], [5, 5], [7, 3]])
        pool      = params.get('mstnn_pool', 'avg')
        dropout   = params['dropout']
        fc_hidden = params.get('expert_fc_hidden', 64)
        self.out_dim = out_dim

        # Static skip: only the per-timestep features go through the conv; the broadcast
        # constants (calendar + macro + thermal-static) bypass it into the head.
        self.n_seq = params.get('n_seq_features') or num_features
        self.n_static = num_features - self.n_seq

        self.branches = nn.ModuleList([
            _ConvBranch(self.n_seq, channels, tuple(k), pool) for k in kernels
        ])
        conv_dim = (len(kernels) * channels if pool == 'avg'
                    else len(kernels) * channels * (self.days // 2) * (self.hours // 2))
        self.enc_dim = conv_dim                      # learned representation (FDS calibrates this)
        self.feat_dim = conv_dim + self.n_static     # what the expert heads consume

        # --- shared MoE head: 12 regime experts + hard season routing ---
        self.head = RegimeHead(self.feat_dim, out_dim, fc_hidden, dropout)

    def encode(self, x):
        b = x.size(0)
        seq = x[:, :, :self.n_seq]
        grid = seq.reshape(b, self.days, self.hours, self.n_seq).permute(0, 3, 1, 2)   # (B, n_seq, days, 24)
        z = torch.cat([branch(grid) for branch in self.branches], dim=1)               # (B, conv_dim)
        if self.n_static > 0:
            z = torch.cat([z, x[:, 0, self.n_seq:]], dim=1)                            # (B, feat_dim)
        return z

    def decode(self, z, season_idx):
        return self.head(z, season_idx)

    def forward(self, x, season_idx):
        return self.decode(self.encode(x), season_idx)


# ---------------------------------------------------------------------------
# Train / predict / evaluate — all delegate to the MoE machinery with model_cls
# ---------------------------------------------------------------------------

def train(X_3d, y_3d, mask_3d, timestamps_3d, params=None, feature_cfg=None, dataset=None):
    print("\n--- Training MoE-MSTNN ---")
    moe.train(X_3d, y_3d, mask_3d, timestamps_3d,
              params or MOE_MSTNN_PARAMS, feature_cfg or MOE_MSTNN_FEATURE_CONFIG, dataset,
              model_type_name=MODEL_TYPE, save_name=FILENAME, model_cls=MoEMSTNN)


def predict(model_path, X_np, timestamps, params=None):
    return moe.predict(model_path, X_np, timestamps, params or MOE_MSTNN_PARAMS, model_cls=MoEMSTNN)


def evaluate(model_path, X_test, y_true_mw, y_scaler, timestamps, result_dir,
             params=None, X_train=None, y_true_train_mw=None, timestamps_train=None):
    moe.evaluate(model_path, X_test, y_true_mw, y_scaler, timestamps, result_dir,
                 params=params or MOE_MSTNN_PARAMS, X_train=X_train,
                 y_true_train_mw=y_true_train_mw, timestamps_train=timestamps_train,
                 model_cls=MoEMSTNN, name='MOE_MSTNN', name_prefix='MOE_MSTNN')
