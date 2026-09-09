# src/models/archformer.py
"""GARCHFormer (ARCHFormer3) wired into this repo's sequence-model pipeline.

The architecture is Moyi Guo's, vendored unchanged in src/models/_garchformer/ (see
that package's docstring for what was and was not copied). This file is only the
adapter: it gives ARCHFormer3 the same train / predict / evaluate signatures the
other five sequence models have, so it shares the matrix, the split, the embargo,
the scalers, the evaluator and the scoreboard with them.

Interface match
---------------
Nothing had to be reshaped. ARCHFormer3 already consumes (B, L, F) and emits
(B, out_len), which is exactly this repo's 3-D tensor and 24-hour target:

    X_3d      (N, 168, C)      C = 68 / 76 / 81 / 89 depending on the feature arm
    forward   (B, 168, C) -> ((B, 24), (B, 24))       y_hat, sigma_sq
    y_3d      (N, 24)          standardized metered load

The wrapper's forward returns y_hat ALONE so that the shared trainer, the shared
evaluator and ModelPredictor need no special-casing; sigma_sq is stashed and
surfaced through `aux_loss`, the one hook added to _seq_trainer.

The whole tensor goes in
------------------------
Unlike the other five nets, this model is NOT given the static-skip treatment:
all C channels are fed to the PatchTokenizer, which is what the upstream design
does. The 54 static features are therefore presented as 168 identical copies
each, and the tokenizer's Linear(24*C -> 64) has to learn to collapse them. That
is a known inefficiency, not a bug -- it is kept because the instruction was to
run the author's architecture unmodified. A static-skip variant would need a
second head to concatenate into and is left as a separate experiment.

Patch geometry
--------------
patch_len = 24 divides the 168-hour lookback into 7 patches of one day each, so
one token is one DAY (contrast src/models/transformer.py, where one token is one
hour). L % patch_len == 0 is asserted by the tokenizer; 168 % 24 == 0 holds.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._seq_trainer import train_sequence
from src.models._eval_utils import EvalUtils
from src.models._garchformer.pointmodels import ARCHFormer3
from src.config import ARCHFORMER_FEATURE_CONFIG, ARCHFORMER_PARAMS

NAME = 'ARCHFORMER'
MODEL_TYPE = 'archformer'
FILENAME = 'archformer_best.pth'


# ---------------------------------------------------------------------------
# Architecture wrapper
# ---------------------------------------------------------------------------

class ARCHFormerNet(nn.Module):
    """ARCHFormer3 behind this repo's (num_features, params) constructor.

    `lambda_cons` reproduces ArchFormer3Trainer's variance-consistency term
    (upstream default 0.1). Setting it to 0 disables the variance path's
    training signal, which reduces the model to the ARCHFormer2 behaviour --
    same weights, the sigma head simply stops being supervised.
    """

    def __init__(self, num_features, params):
        super().__init__()
        lookback = params['lookback_hours']
        patch_len = params.get('patch_len', 24)
        assert lookback % patch_len == 0, \
            f"lookback_hours={lookback} must be divisible by patch_len={patch_len}"

        self.lambda_cons = float(params.get('lambda_cons', 0.1))
        self.net = ARCHFormer3(
            seq_length_enc=lookback,
            seq_length_dec=params.get('seq_length_dec', lookback),
            num_features=num_features,
            patch_len=patch_len,
            d_model=params['d_model'],
            num_heads=params['num_heads'],
            num_encoder_layers=params['num_encoder_layers'],
            num_decoder_layers=params['num_decoder_layers'],
            intermediate_dim=params['intermediate_dim'],
            out_len=params['out_dim'],
            dropout=params['dropout'],
        )
        self._sigma_sq = None

    def forward(self, x):
        y_hat, sigma_sq = self.net(x)
        # Stashed rather than returned: every caller in this repo expects a single
        # tensor, and a tuple would have to be unpacked in the trainer, the
        # evaluator, the residual factory and the serving path alike.
        self._sigma_sq = sigma_sq
        return y_hat

    def aux_loss(self, y_hat, y):
        """ArchFormer3Trainer's term: lambda * L1(sigma^2, (y - y_hat)^2).

        Read by _seq_trainer via hasattr(model, 'aux_loss'). Uses the sigma from
        the forward pass that produced `y_hat`, so there is no second forward.
        """
        if self.lambda_cons <= 0 or self._sigma_sq is None:
            return torch.zeros((), device=y_hat.device)
        # Not detached -- upstream ArchFormer3Trainer lets this term push the mean
        # path too, and that is part of the method rather than an oversight.
        eps_sq = (y - y_hat) ** 2
        return self.lambda_cons * F.l1_loss(self._sigma_sq, eps_sq)

    @torch.no_grad()
    def predict_sigma(self, x):
        """(B, 24) predicted variance for the target day, in standardized units^2.

        Not used by the point-forecast path; exposed because it is the model's
        distinguishing output and is what an interval forecast would be built on.
        """
        self.eval()
        self.forward(x)
        return self._sigma_sq


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X_3d, y_3d, mask_3d, params=None, feature_cfg=None, dataset=None):
    print("\n--- Training GARCHFormer (ARCHFormer3) ---")
    train_sequence(
        ARCHFormerNet, MODEL_TYPE, FILENAME,
        X_3d, y_3d, mask_3d,
        params or ARCHFORMER_PARAMS,
        feature_cfg or ARCHFORMER_FEATURE_CONFIG,
        dataset,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model_path, X_np, params):
    """Load a saved model and return scaled predictions (N, out_dim)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = params or ARCHFORMER_PARAMS
    model = ARCHFormerNet(X_np.shape[2], params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    out = []
    with torch.no_grad():
        # Chunked: the decoder holds (B, 168, d_model) activations per layer, and the
        # full 313-day test split at once is a needless spike on a CPU node.
        for i in range(0, len(X_np), 256):
            xb = torch.FloatTensor(X_np[i:i + 256]).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model_path, X_test, y_true_mw, y_scaler, timestamps, result_dir,
             params=None, X_train=None, y_true_train_mw=None, timestamps_train=None):
    """Predict, inverse-transform, then run the full evaluation suite."""
    params = params or ARCHFORMER_PARAMS
    y_pred_scaled = predict(model_path, X_test, params)
    N, P = y_pred_scaled.shape
    y_pred_mw = y_scaler.inverse_transform(y_pred_scaled.flatten().reshape(-1, 1)).reshape(N, P)

    train_df = None
    if X_train is not None and y_true_train_mw is not None:
        y_ptr = predict(model_path, X_train, params)
        N2, P2 = y_ptr.shape
        y_pred_train_mw = y_scaler.inverse_transform(y_ptr.flatten().reshape(-1, 1)).reshape(N2, P2)
        train_df = EvalUtils.build_detailed_df(NAME, y_true_train_mw, y_pred_train_mw, timestamps_train)

    EvalUtils.evaluate_one(NAME, y_true_mw, y_pred_mw, timestamps, result_dir, train_df)
