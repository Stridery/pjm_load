"""The network, the pinball objective, and the training scheme of Zhang et al. Fig. 1-2.

Architecture (their Fig. 1), in order:

    input -> dense -> PReLU -> dropout -> gaussian noise
          -> dense -> PReLU -> dropout -> gaussian noise
          -> dense(99)

Output width IS the quantile set, and the loss IS the quantile score of their eq. (1)-(2),
so one fit produces all 99 percentiles. That is the whole point of the family: "training
quantile forecasting models for different quantiles simultaneously has some advantages over
training quantile forecasting models separately", and it is also why a QRA-style method
needs 99 regressions where this needs one.

Training scheme (their Fig. 2). One `bad` counter, epochs since the best validation quantile
score, drives both rules: at `stop_patience` the run ends, and every `lr_patience` epochs
without improvement the learning rate is multiplied by `lr_decay` until it reaches
`min_lr`. The paper draws them as two separate diamonds off the same condition and states
they "serve for different purposes" (when to stop vs when to fine-tune); one counter with two
thresholds is that flowchart.

bQRNN is the paper's own control, not a strawman of ours: same PReLU MLP, same batch
training, same GPU, but none of the four additions — no dropout, no noise, no early stopping,
no learning-rate decay. Its row in Table II is NA for exactly those five hyperparameters. It
exists so the reported gain can be attributed to the training machinery rather than to the
architecture, and reproducing it is the only way to check the paper's central claim.


Standardisation, and a departure
--------------------------------
The paper normalises each sample to unit norm and THEN standardises by column. The first
step is not reproduced. Its feature vector mixes MW with degrees F, and dividing a sample by
its own L2 norm makes the scale of the target unrecoverable from the input row — two days
differing only in level become the same input. Column standardisation fitted on the training
rows and frozen is used instead: it is what every other model in this repo does, and it is
the arrangement the frozen-at-inference discipline elsewhere assumes.

The target is standardised too and inverted after prediction. The pinball loss is positively
homogeneous, so scaling y scales the loss without moving its minimiser, and quantiles are
equivariant under the affine inverse — the returned percentiles are exact, not approximate.
"""

import time

import numpy as np
import torch
import torch.nn as nn

# The paper's model set (its Table II). `kind` 'i' is the proposed iQRNN, 'b' the control.
# -ZD is "zero dropout": the pair at each width is what isolates dropout from network size,
# and the paper's finding is that dropout helps at 4000 and HURTS at 500, so both widths are
# needed for the claim to mean anything.
SPECS = (
    {'name': 'iQRNN-4000',    'kind': 'i', 'hidden': 4000, 'dropout': 0.5},
    {'name': 'iQRNN-4000-ZD', 'kind': 'i', 'hidden': 4000, 'dropout': 0.0},
    {'name': 'iQRNN-500',     'kind': 'i', 'hidden': 500,  'dropout': 0.5},
    {'name': 'iQRNN-500-ZD',  'kind': 'i', 'hidden': 500,  'dropout': 0.0},
    {'name': 'bQRNN-400',     'kind': 'b', 'hidden': 400,  'dropout': 0.0},
)

SPEC_BY_NAME = {s['name']: s for s in SPECS}


def resolve_device(setting):
    if setting == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return setting


class GaussianNoise(nn.Module):
    """Their eq. (4): additive zero-mean noise, training only.

    The paper's single largest ablation effect (its Fig. 9: SD 0 -> 0.3 moves the quantile
    score more than any other switch), borrowed from the finding that gradient noise helps
    very deep networks and applied here to a shallow one.
    """

    def __init__(self, sd):
        super().__init__()
        self.sd = float(sd)

    def forward(self, x):
        if self.training and self.sd > 0:
            return x + torch.randn_like(x) * self.sd
        return x

    def extra_repr(self):
        return f'sd={self.sd}'


class QRNN(nn.Module):
    """Two hidden layers of `hidden` units each -> `n_quantiles` outputs.

    Both hidden layers carry the same width on purpose. The paper sets them equal "because it
    excludes complex and precise tuning on the NN size", the point being that the method
    should not need a tuned architecture to work; keeping that makes its robustness claim
    testable here rather than assumed.
    """

    def __init__(self, n_in, n_quantiles, hidden, dropout=0.0, noise=0.0, n_layers=2):
        super().__init__()
        layers, d = [], n_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.PReLU(hidden)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            if noise > 0:
                layers.append(GaussianNoise(noise))
            d = hidden
        layers.append(nn.Linear(d, n_quantiles))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def pinball_loss(pred, target, levels):
    """Mean pinball loss over all quantiles and rows — their eq. (1) and (2).

        L = max( (q/100 - 1)(y - yhat_q),  (q/100)(y - yhat_q) )

    written as the max of the two branches exactly as the paper does, which is the same
    number as the sign-split form in src/quantile/qeval.pinball and avoids a branch.
    """
    d = target.unsqueeze(1) - pred                  # (B, Q)
    return torch.maximum(levels * d, (levels - 1.0) * d).mean()


def _standardise(train_block, *blocks):
    m = train_block.mean(axis=0)
    s = train_block.std(axis=0)
    s[s < 1e-9] = 1.0
    return m, s, [(b - m) / s for b in blocks]


def _batches(n, batch_size, generator):
    order = torch.randperm(n, generator=generator)
    for i in range(0, n, batch_size):
        yield order[i:i + batch_size]


def fit(spec, Xtr, ytr, Xva, yva, cfg, levels, seed, verbose=False):
    """Train one configuration on one seed. Returns (model, history dict).

    Everything downstream of the seed is deterministic given the same tensors: weight init,
    batch order, dropout masks and the noise draws all come from generators seeded here, so a
    repeated run reproduces bit for bit and the spread reported across runs is the spread the
    paper's Table IV reports, not scheduler jitter.
    """
    device = resolve_device(cfg['device'])
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    is_i = spec['kind'] == 'i'
    model = QRNN(Xtr.shape[1], len(levels), spec['hidden'], dropout=spec['dropout'],
                 noise=cfg['noise_sd'] if is_i else 0.0).to(device)

    xt = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    xv = torch.as_tensor(Xva, dtype=torch.float32, device=device)
    yv = torch.as_tensor(yva, dtype=torch.float32, device=device)
    lv = torch.as_tensor(levels, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    t0 = time.time()
    epoch = 0

    for epoch in range(1, cfg['max_epochs'] + 1):
        model.train()
        for idx in _batches(len(xt), cfg['batch_size'], gen):
            b = idx.to(device)
            opt.zero_grad(set_to_none=True)
            pinball_loss(model(xt[b]), yt[b], lv).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val = float(pinball_loss(model(xv), yv, lv))

        if val < best - 1e-9:
            best, best_epoch, bad = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if is_i:
            # Early stopping and the decay rule, both off `bad` (their Fig. 2).
            if bad >= cfg['stop_patience']:
                break
            if bad > 0 and bad % cfg['lr_patience'] == 0:
                for g in opt.param_groups:
                    g['lr'] = max(g['lr'] * cfg['lr_decay'], cfg['min_lr'])
        if verbose and epoch % 20 == 0:
            print(f"      epoch {epoch:4d}  val {val:.5f}  best {best:.5f} @{best_epoch}")

    # bQRNN has no early stopping, so it ends on whatever the last epoch produced — which is
    # the point of the control. iQRNN restores the best weights, as the paper's "utility to
    # restore the best configuration" does.
    if is_i and best_state is not None:
        model.load_state_dict(best_state)

    return model, {'epochs': epoch, 'best_epoch': best_epoch if is_i else epoch,
                   'val_pinball': best if is_i else val, 'seconds': time.time() - t0,
                   'n_params': sum(p.numel() for p in model.parameters())}


def predict(model, X, cfg, batch=8192):
    """(n, Q) quantile forecasts in standardised units."""
    device = resolve_device(cfg['device'])
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def crossing_rate(q_raw, step=None):
    """(fraction of hours with any crossing, mean CS) on the RAW, unsorted output.

    CS is the constraint score of Li et al. (2023) eq. (21) — sqrt(2*theta/N * sum of the
    squared positive part of q_i - q_{i+1}), where theta is the spacing between adjacent
    probability levels. For the 1st..99th percentiles that spacing is 1/100, not 1/99: the
    levels are evenly spaced on (0, 1) with both endpoints excluded.

    Reported because the paper claims the joint output layer suppresses crossing on its own
    ("adjacent output layer neurons use similar information", its Fig. 10), and that claim is
    only checkable BEFORE the output is sorted. Everything scored downstream uses the sorted
    array, since rearranging weakly improves the pinball loss and is what
    src/quantile/qra.py already does.
    """
    v = np.clip(q_raw[:-1] - q_raw[1:], 0, None)             # (Q-1, N)
    theta = step if step is not None else 1.0 / (q_raw.shape[0] + 1)
    return float((v > 0).any(axis=0).mean()), float(np.sqrt(2 * theta * (v ** 2).sum(0).mean()))


def run_spec(spec, X, y, train, valid, test, cfg, levels, seed, verbose=False):
    """One (configuration, seed): standardise, fit, predict, unscale.

    Returns ({'valid': (Q, n), 'test': (Q, n)} sorted, the same raw/unsorted, history).
    Standardisation constants come from the TRAINING rows only and are frozen for validation
    and test — the same fitted-on-train, frozen-at-inference arrangement the 16 point models
    get, and the reason the validation score here is a usable model-selection signal rather
    than a number the scaler has already seen.
    """
    _, _, (Xtr, Xva, Xte) = _standardise(X[train], X[train], X[valid], X[test])
    my, sy, (ytr, yva) = _standardise(y[train].reshape(-1, 1), y[train].reshape(-1, 1),
                                      y[valid].reshape(-1, 1))

    model, hist = fit(spec, Xtr, ytr.ravel(), Xva, yva.ravel(), cfg, levels, seed,
                      verbose=verbose)

    raw = {'valid': (predict(model, Xva, cfg) * sy[0] + my[0]).T,     # (Q, n), as qeval wants
           'test': (predict(model, Xte, cfg) * sy[0] + my[0]).T}
    return {k: np.sort(v, axis=0) for k, v in raw.items()}, raw, hist
