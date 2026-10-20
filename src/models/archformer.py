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
from src.models._garchformer.layers import (
    PatchTokenizer, PositionalEncoding, ARCHEncoder, MLPConnectionBlock,
    ARCHDecoder, InversePatchTokenizer, FinalProjectionHead,
)
from src.config import ARCHFORMER_FEATURE_CONFIG, ARCHFORMER_PARAMS

NAME = 'ARCHFORMER'
MODEL_TYPE = 'archformer'
FILENAME = 'archformer_best.pth'



# ---------------------------------------------------------------------------
# Static skip: the statics join at the decoder output, not at the tokenizer
# ---------------------------------------------------------------------------

class _SkipARCHFormer3(nn.Module):
    """ARCHFormer3 with the broadcast constants routed past the encoder-decoder stack.

    Upstream feeds every channel to the patch tokenizer. On this repo's tensor that means
    54 of the 76 channels are constants repeated 24 times inside each patch, so 71 % of the
    tokenizer's 1824 inputs are duplicate copies and the Linear(1824 -> 64) spends most of
    its 116,800 parameters re-reading them. Here only the n_seq time-varying channels are
    tokenised, and the statics are concatenated to every decoder token:

        x_dec        (B, N, d_model)
        static       (B, S)  ->  broadcast  (B, N, S)
        cat          (B, N, d_model + S)
        fuse         Linear(d_model + S -> d_model), ReLU, Dropout      <- the nonlinearity
        inverse tok  Linear(d_model -> patch_len * n_seq)   [vendored, shared with sigma]
        head         Linear(n_seq -> 1), take the last out_len positions

    The ReLU is load-bearing, and it is easy to leave out. Concatenating into a LINEAR layer
    is only addition: Linear([h || s]) = W_h h + W_s s + b, so the static contribution would
    not depend on h at all. Measured on an earlier draft of this class, with the fusion layer
    linear: perturbing the statics moved the forecast with std 0.22 across the 24 hours (so
    each static did get its own per-hour weight) but the effect of that perturbation was
    byte-identical on two samples with different sequences -- no interaction. With the ReLU
    the same test separates them.

    That is the same failure the PREVIOUS static skip had, one level worse: it emitted its
    own out_dim vector and added it to the finished y_hat, so 54 features collapsed to one
    offset per hour and the forecast could be shifted but never modulated. It scored 3.380
    against 3.242 for the same recipe with no skip at all.

    The variance path skips the fusion layer, so the statics never reach it. sigma^2
    estimates how volatile an hour is; that is a property of the series, and letting the
    calendar or a temperature forecast set it directly is a different model. Both paths
    share the inverse tokenizer and FinalProjectionHead, as upstream does.

    Every component here is the vendored class, unmodified -- only the wiring differs.
    """

    def __init__(self, seq_length_enc, seq_length_dec, n_seq, n_static, patch_len, d_model,
                 num_heads, num_encoder_layers, num_decoder_layers, intermediate_dim,
                 out_len, dropout=0.1):
        super().__init__()
        self.n_seq, self.n_static = n_seq, n_static
        n_patch_enc = seq_length_enc // patch_len
        n_patch_dec = seq_length_dec // patch_len

        self.tokenizer = PatchTokenizer(patch_len, n_seq, d_model)
        self.pos_encoder = PositionalEncoding(n_patch_enc, d_model)
        self.encoder = ARCHEncoder(d_model, num_heads, num_encoder_layers, dropout)
        self.connection_block = MLPConnectionBlock(n_patch_enc, n_patch_dec, d_model,
                                                   intermediate_dim)
        self.decoder = ARCHDecoder(d_model, num_heads, num_decoder_layers, dropout)
        # Fuse the statics into the day representation, with a nonlinearity, and come back
        # out at d_model -- so the vendored inverse tokenizer is used at its own width and
        # the variance path can share it.
        self.fuse = nn.Sequential(
            nn.Linear(d_model + n_static, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.inverse_tokenizer = InversePatchTokenizer(patch_len, n_seq, d_model)
        self.head = FinalProjectionHead(n_seq, out_len)

    def forward(self, x):
        seq = x[:, :, :self.n_seq]
        static = x[:, 0, self.n_seq:]                      # constant along t by construction

        h = self.pos_encoder(self.tokenizer(seq))
        h, mu_list, sigma_list = self.encoder(h)
        h = self.connection_block(h)
        h, _, sigma_dec_list = self.decoder(h, mu_list[-1], sigma_list[-1])

        s = static.unsqueeze(1).expand(-1, h.size(1), -1)  # (B, N, S)
        h_fused = self.fuse(torch.cat([h, s], dim=-1))     # (B, N, d_model)
        y_hat = self.head(self.inverse_tokenizer(h_fused))

        # sigma^2 skips the fusion entirely, so the statics never reach it.
        sigma_rec = self.inverse_tokenizer(sigma_dec_list[-1])
        sigma_sq = torch.abs(self.head(sigma_rec)) + 1e-6
        return y_hat, sigma_sq


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

        # Static skip, OFF by default so the shipped configuration stays the author's.
        # Setting n_seq_features turns it on: the time-varying channels go through the patch
        # tokenizer and the broadcast constants join at the decoder output instead
        # (_SkipARCHFormer3). Without it every channel is tokenised, as upstream does.
        self.n_seq = int(params.get('n_seq_features') or num_features)
        self.n_static = num_features - self.n_seq
        if self.n_static > 0:
            # Statics bypass the tokenizer and join at the decoder output -- see
            # _SkipARCHFormer3 for why, and for why the previous additive form was wrong.
            self.net = _SkipARCHFormer3(
                seq_length_enc=lookback,
                seq_length_dec=params.get('seq_length_dec', lookback),
                n_seq=self.n_seq,
                n_static=self.n_static,
                patch_len=patch_len,
                d_model=params['d_model'],
                num_heads=params['num_heads'],
                num_encoder_layers=params['num_encoder_layers'],
                num_decoder_layers=params['num_decoder_layers'],
                intermediate_dim=params['intermediate_dim'],
                out_len=params['out_dim'],
                dropout=params['dropout'],
            )
        else:
            # The author's model, every channel through the tokenizer. This is what
            # `archformer` and `archformer_hp` run.
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
        # Both nets take the full tensor and split it themselves (or not, when there is
        # nothing static to split off), so this wrapper does no slicing.
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
# Variant factory
# ---------------------------------------------------------------------------
# Three versions of this architecture ship side by side, so they land in the scoreboard as
# separate rows on one test set rather than as a claim about which is better:
#
#   archformer        the author's architecture exactly as published. 504,725 parameters on
#                     the _cong arm, all C channels through the patch tokenizer.
#   archformer_hp     the same code, smaller patches. patch_len 6 instead of 24, which cuts
#                     the tokenizer's input from 24xC to 6xC and raises the token count from
#                     7 to 28. A pure hyperparameter change -- no code path differs.
#   archformer_skip   the static block routed past the tokenizer to the head, the way the
#                     other six sequence models treat it. Only the wrapper differs; the
#                     vendored files are byte-identical in all three.
#
# Why smaller at all: his configs carry 5-28 feature columns and this repo hands the model
# 68-89, so Linear(24 x C -> 64) compresses 33:1 here against 6.8:1 on the config he wrote
# for this very dataset -- and 72% of that input is a static value repeated once per hour
# inside the patch. Both variants attack the same ratio from different sides. Measured on
# _cong (C=76): 504,725 -> 330,962 (hp) and -> 337,541 (skip).


def make_variant(*, name, model_type, filename, params_default):
    """(train, predict, evaluate) for one ARCHFormer variant.

    Everything except the parameter dict and the directory name is shared, so a variant
    cannot drift from the others in anything but the setting under test.
    """

    def train(X_3d, y_3d, mask_3d, params=None, feature_cfg=None, dataset=None):
        print(f"\n--- Training {name} ---")
        train_sequence(
            ARCHFormerNet, model_type, filename,
            X_3d, y_3d, mask_3d,
            params or params_default,
            feature_cfg or ARCHFORMER_FEATURE_CONFIG,
            dataset,
        )

    def predict(model_path, X_np, params=None):
        """Load a saved model and return scaled predictions (N, out_dim)."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ARCHFormerNet(X_np.shape[2], params or params_default).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        out = []
        with torch.no_grad():
            # Chunked: the decoder holds (B, L, d_model) activations per layer, and the full
            # 313-day test split at once is a needless spike on a CPU node.
            for i in range(0, len(X_np), 256):
                xb = torch.FloatTensor(X_np[i:i + 256]).to(device)
                out.append(model(xb).cpu().numpy())
        return np.concatenate(out, axis=0)

    def evaluate(model_path, X_test, y_true_mw, y_scaler, timestamps, result_dir,
                 params=None, X_train=None, y_true_train_mw=None, timestamps_train=None):
        """Predict, inverse-transform, then run the full evaluation suite."""
        params = params or params_default
        y_pred_scaled = predict(model_path, X_test, params)
        N, P = y_pred_scaled.shape
        y_pred_mw = y_scaler.inverse_transform(
            y_pred_scaled.flatten().reshape(-1, 1)).reshape(N, P)

        train_df = None
        if X_train is not None and y_true_train_mw is not None:
            y_ptr = predict(model_path, X_train, params)
            N2, P2 = y_ptr.shape
            y_pred_train_mw = y_scaler.inverse_transform(
                y_ptr.flatten().reshape(-1, 1)).reshape(N2, P2)
            train_df = EvalUtils.build_detailed_df(name, y_true_train_mw, y_pred_train_mw,
                                                   timestamps_train)
        EvalUtils.evaluate_one(name, y_true_mw, y_pred_mw, timestamps, result_dir, train_df)

    return train, predict, evaluate


train, predict, evaluate = make_variant(
    name=NAME, model_type=MODEL_TYPE, filename=FILENAME,
    params_default=ARCHFORMER_PARAMS,
)
