"""Soft-gated Mixture-of-Experts Transformer.

The vanilla transformer's encoder, unchanged, with its single output MLP replaced by N expert
heads combined by a learned gate:

    z = encode(x)                       (d_model + n_static)
    g = softmax(W z + noise)            (n_experts)
    y = sum_i g_i * E_i(z)              (24)

WHAT THIS IS A CONTROL FOR
--------------------------
Two comparisons, and the point of the file is that both are clean because nothing else moves:

  vs transformer        same encoder, same recipe, same arm -> does a mixture help at all
  vs moe_transformer    same idea of regimes                -> is a learned route better than
                                                               the calendar route

moe_transformer's RegimeHead hard-routes on the target day's month and partitions the 24-hour
output between experts by hour band. That divides 1461 training days twelve ways: its experts
see 1,428-7,018 target scalars each against 35,064 for a single head, and it loses to the plain
transformer by 0.087 pp in residual mode and 0.827 pp in direct mode. Soft gating divides
nothing -- every expert sees every sample, weighted -- so the mixture's floor is an ensemble of
heads rather than a set of under-trained specialists.

NO TIMESTAMPS
-------------
The MoE models carry their own trainer because hard routing needs a per-sample season index
that `_seq_trainer`'s (X, y, w) signature cannot pass. The gate here reads z, which already
contains the static block -- tmrw_month_sin/cos, day-of-week, holiday, and the 31
forecast-weather departures. It therefore sees strictly more than the month index did, needs
no timestamp, and this file uses the shared trainer unchanged.

The load-balancing term reaches the loss through `_seq_trainer`'s `aux_loss` hook.
"""

import torch
import torch.nn as nn

from src.config import SMOE_TRANSFORMER_PARAMS, SMOE_TRANSFORMER_FEATURE_CONFIG
from src.models._eval_utils import EvalUtils
from src.models._moe_head import SoftMoEHead
from src.models._seq_trainer import train_sequence
from src.models.transformer import PositionalEncoding


class SoftMoETransformer3D(nn.Module):
    def __init__(self, num_features, params):
        super().__init__()
        d_model = params['d_model']
        self.n_seq = params.get('n_seq_features') or num_features
        self.n_static = num_features - self.n_seq
        self.enc_dim = d_model
        self.feat_dim = d_model + self.n_static

        self.input_projection = nn.Linear(self.n_seq, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=params['nhead'], dim_feedforward=d_model * 4,
            dropout=params['dropout'], batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=params['num_layers'])
        # What the ROUTER may look at. The experts always see all of z either way.
        #   'z'         the whole representation: encoder output + every static feature
        #   'calendar'  only the 6 broadcast calendar channels of the target day
        #               (month_sin/cos, dow_sin/cos, is_weekend, is_holiday), which sit at the
        #               front of the static block -- verified: channel n_seq correlates 1.0000
        #               with sin(2*pi*month/12). This is the softened form of moe_transformer's
        #               month router: still calendar-driven, but the boundaries are learned
        #               instead of written down, and no sample is excluded from any expert.
        gate_input = params.get('gate_input', 'z')
        if gate_input == 'calendar':
            gate_slice = (d_model, d_model + 6)
        elif gate_input == 'static':
            gate_slice = (d_model, self.feat_dim)
        elif gate_input == 'z':
            gate_slice = None
        else:
            raise ValueError(f"gate_input must be 'z' | 'calendar' | 'static', got {gate_input!r}")
        self.gate_input = gate_input
        self.head = SoftMoEHead(
            self.feat_dim, params['out_dim'],
            params.get('expert_fc_hidden', 128), params['dropout'],
            n_experts=params['n_experts'], gate_noise=params.get('gate_noise', 0.0),
            gate_slice=gate_slice,
        )
        self.balance_weight = float(params.get('balance_weight', 0.0))

    def encode(self, x):
        seq = x[:, :, :self.n_seq]
        h = self.transformer_encoder(self.pos_encoder(self.input_projection(seq)))
        z = h[:, -1, :]
        if self.n_static > 0:
            z = torch.cat([z, x[:, 0, self.n_seq:]], dim=1)
        return z

    def decode(self, features):
        return self.head(features)

    def forward(self, x):
        return self.decode(self.encode(x))

    def aux_loss(self, out, target):
        """Picked up by _seq_trainer. Keeps the gate from collapsing onto one expert."""
        if self.balance_weight <= 0:
            return out.new_zeros(())
        return self.balance_weight * self.head.balance_loss()

    def gate_summary(self):
        return self.head.gate_summary()


# ---------------------------------------------------------------------------

def train(X_3d, y_3d, mask_3d, params=None, feature_cfg=None, dataset=None):
    print("\n--- Training Soft-MoE Transformer ---")
    train_sequence(
        SoftMoETransformer3D, 'smoe_transformer', 'smoe_transformer_best.pth',
        X_3d, y_3d, mask_3d,
        params or SMOE_TRANSFORMER_PARAMS,
        feature_cfg or SMOE_TRANSFORMER_FEATURE_CONFIG,
        dataset,
    )


def predict(model_path, X_np, params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SoftMoETransformer3D(X_np.shape[2], params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_np).to(device)).cpu().numpy()


def gate_usage(model_path, X_np, params):
    """Mean gate weight per expert over X_np — the first thing to check after a run.

    All four near 0.25 means the gate never specialised and the model is an ensemble of heads.
    One near 1.0 means it collapsed and the balance weight is too low.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SoftMoETransformer3D(X_np.shape[2], params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        model(torch.FloatTensor(X_np).to(device))
    return model.gate_summary()


def evaluate(model_path, X_test, y_true_mw, y_scaler, timestamps, result_dir,
             params=None, X_train=None, y_true_train_mw=None, timestamps_train=None):
    params = params or SMOE_TRANSFORMER_PARAMS
    y_pred_scaled = predict(model_path, X_test, params)
    N, P = y_pred_scaled.shape
    y_pred_mw = y_scaler.inverse_transform(y_pred_scaled.flatten().reshape(-1, 1)).reshape(N, P)

    train_df = None
    if X_train is not None and y_true_train_mw is not None:
        y_ptr = predict(model_path, X_train, params)
        N2, P2 = y_ptr.shape
        y_pred_train_mw = y_scaler.inverse_transform(y_ptr.flatten().reshape(-1, 1)).reshape(N2, P2)
        train_df = EvalUtils.build_detailed_df('SMOE_TRANSFORMER', y_true_train_mw,
                                               y_pred_train_mw, timestamps_train)

    EvalUtils.evaluate_one('SMOE_TRANSFORMER', y_true_mw, y_pred_mw, timestamps,
                           result_dir, train_df)
