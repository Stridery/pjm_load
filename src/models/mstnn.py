"""MSTNN — simplified Multi-Scale Temporal Neural Network for 24h load forecasting.

Reproduces the CORE idea of MSTNN (Song et al., IJCAI 2025) — a multi-scale
convolutional encoder over a periodically-reshaped time grid — adapted to a
single load series.

  Paper: reshape T = Y*M*D (year/month/day, 3 temporal scales) into a 3D tensor and
         apply parallel multi-size 3D conv filters (3x3x3, 5x5x3) → multi-scale
         periodic features; then a cross-stock Temporal Hypergraph Attention (THAN)
         mixes sector-level info across ~1000 stocks.
  Here:  the lookback is 168h = 7 days x 24 hours (2 temporal scales), so we reshape
         each sample to a (day, hour) grid with the F features as channels and apply
         parallel 2D conv branches of different kernel sizes (multi-scale receptive
         fields over the day/hour periodicity). THAN is DROPPED — there is a single
         entity (zone), so an inter-entity hypergraph is meaningless.

Plugs into the shared sequence trainer (LDS / Stage-2 / embargo / macro features /
split / eval all inherited). FDS is not wired (it needs a fixed encoder feat_dim).
"""

import torch
import torch.nn as nn

from src.models._seq_trainer import train_sequence
from src.models._eval_utils import EvalUtils
from src.config import MSTNN_FEATURE_CONFIG, MSTNN_PARAMS


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class _ConvBranch(nn.Module):
    """One multi-scale branch: same-padded conv over the (day, hour) grid → pool.

    pool='avg'    : global average pool → (out_ch,) per branch. Kills the giant
                    flatten→FC that otherwise memorizes on ~1.5k samples.
    pool='flatten': MaxPool2d(2) then flatten (the original, high-capacity form).
    """
    def __init__(self, in_ch, out_ch, kernel, pool='avg'):
        super().__init__()
        pad = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=pad)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1) if pool == 'avg' else nn.MaxPool2d(2)

    def forward(self, grid):
        return self.pool(self.act(self.conv(grid))).flatten(1)


class MSTNN(nn.Module):
    def __init__(self, num_features, params):
        super().__init__()
        lookback = params['lookback_hours']
        assert lookback % 24 == 0, "lookback_hours must be a whole number of days"
        self.days = lookback // 24
        self.hours = 24
        channels = params.get('mstnn_channels', 32)
        kernels  = params.get('mstnn_kernels', [[3, 3], [5, 5]])
        dropout  = params['dropout']

        # Static skip: the first n_seq features vary per timestep (Load + weather) and
        # go through the multi-scale conv over the (day, hour) grid; the rest (calendar
        # + macro) are broadcast constants — they bypass the conv and are concatenated
        # straight to the head, so the conv's capacity is spent only on temporal signal.
        self.n_seq = params.get('n_seq_features') or num_features
        self.n_static = num_features - self.n_seq

        pool = params.get('mstnn_pool', 'avg')
        self.branches = nn.ModuleList([
            _ConvBranch(self.n_seq, channels, tuple(k), pool) for k in kernels
        ])
        if pool == 'avg':
            flat = len(kernels) * channels                                      # global avg pool
        else:
            flat = len(kernels) * channels * (self.days // 2) * (self.hours // 2)   # flatten
        fc_hidden = params.get('fc_hidden', 128)
        self.fc_out = nn.Sequential(
            nn.Linear(flat + self.n_static, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, params['out_dim']),
        )

    def encode(self, x):
        # sequence features -> (day, hour) grid through the conv; static features bypass
        b = x.size(0)
        seq = x[:, :, :self.n_seq]                                      # (B, lookback, n_seq)
        grid = seq.reshape(b, self.days, self.hours, self.n_seq).permute(0, 3, 1, 2)  # (B, n_seq, days, 24)
        feat = torch.cat([branch(grid) for branch in self.branches], dim=1)           # (B, flat)
        if self.n_static > 0:
            static = x[:, 0, self.n_seq:]                              # (B, n_static), constant over window
            feat = torch.cat([feat, static], dim=1)
        return feat

    def decode(self, feat):
        return self.fc_out(feat)

    def forward(self, x):
        return self.decode(self.encode(x))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X_3d, y_3d, mask_3d, params=None, feature_cfg=None, dataset=None):
    print("\n--- Training MSTNN ---")
    train_sequence(
        MSTNN, 'mstnn', 'mstnn_best.pth',
        X_3d, y_3d, mask_3d,
        params or MSTNN_PARAMS,
        feature_cfg or MSTNN_FEATURE_CONFIG,
        dataset,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model_path, X_np, params):
    """Load a saved model and return scaled predictions (N, out_dim)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MSTNN(X_np.shape[2], params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_np).to(device)).cpu().numpy()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model_path, X_test, y_true_mw, y_scaler, timestamps, result_dir,
             params=None, X_train=None, y_true_train_mw=None, timestamps_train=None):
    """Predict, inverse-transform, then run the full evaluation suite."""
    params = params or MSTNN_PARAMS
    y_pred_scaled = predict(model_path, X_test, params)
    N, P = y_pred_scaled.shape
    y_pred_mw = y_scaler.inverse_transform(y_pred_scaled.flatten().reshape(-1, 1)).reshape(N, P)

    train_df = None
    if X_train is not None and y_true_train_mw is not None:
        y_ptr = predict(model_path, X_train, params)
        N2, P2 = y_ptr.shape
        y_pred_train_mw = y_scaler.inverse_transform(y_ptr.flatten().reshape(-1, 1)).reshape(N2, P2)
        train_df = EvalUtils.build_detailed_df('MSTNN', y_true_train_mw, y_pred_train_mw, timestamps_train)

    EvalUtils.evaluate_one('MSTNN', y_true_mw, y_pred_mw, timestamps, result_dir, train_df)
