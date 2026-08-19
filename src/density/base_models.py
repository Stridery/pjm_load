"""The fifteen base density models.  Li et al. Section IV-A/B/C and V-B.

Five Gaussian process regressions, five quantile-regression neural networks, five
quantile-regression gradient boosters. GPR emits a Gaussian directly; the other two emit
deciles that src/density/gaussfit turns into one.

The point of the fifteen is DIVERSITY, not individual accuracy — the combination is what is
being tested, and a set of near-identical members would make the weights arbitrary. The
paper's three sources of diversity are reproduced:

    GPR   a different random 2000-point subsample of T1 each time. Not a stylistic choice:
          GPR inverts an n x n matrix, so the paper subsamples "to train GPR base models in
          relatively short time", and the subsample is what makes the five differ.
    QRNN  two architectures (with and without batch-norm; the second carries an L2 term
          instead, since batch-norm acts as the regulariser in the first) x different
          feature subsets.
    QRGB  one architecture, different max_depth / min_child_weight / subsample /
          colsample_bytree.

Standardisation is fit on T1 only and applied everywhere, for the two families that need it.
Trees do not, but they get it anyway so that every family sees the identical matrix and a
difference in results can never be a difference in preprocessing.
"""

import numpy as np

DECILES = np.arange(1, 10) / 10.0        # the paper converts "decile results" to Gaussians
GPR_SUBSAMPLE = 2000                     # their Section V-B


# Feature subsets used to diversify the neural nets (indices into features.FEATURE_NAMES).
SUBSETS = {
    'all': None,                                    # all 15
    'lags': list(range(11)),                        # the eleven lagged loads only
    'recent': [0, 1, 2, 3, 4, 5, 11, 12, 13, 14],   # the clustered short lags + calendar
}

# The fifteen specifications, in the paper's naming order (GPR1-5, NN1-5, XGB1-5).
SPECS = (
    [{'name': f'GPR{i + 1}', 'family': 'gpr', 'seed': 100 + i} for i in range(5)]
    + [{'name': 'NN1', 'family': 'qrnn', 'arch': 'bn', 'subset': 'all', 'hidden': 64, 'seed': 1},
       {'name': 'NN2', 'family': 'qrnn', 'arch': 'bn', 'subset': 'lags', 'hidden': 128, 'seed': 2},
       {'name': 'NN3', 'family': 'qrnn', 'arch': 'l2', 'subset': 'all', 'hidden': 64, 'seed': 3},
       {'name': 'NN4', 'family': 'qrnn', 'arch': 'l2', 'subset': 'all', 'hidden': 128, 'seed': 4},
       {'name': 'NN5', 'family': 'qrnn', 'arch': 'l2', 'subset': 'recent', 'hidden': 96, 'seed': 5}]
    + [{'name': 'XGB1', 'family': 'qrgb', 'max_depth': 4, 'min_child_weight': 1,
        'subsample': 0.9, 'colsample_bytree': 0.9, 'seed': 11},
       {'name': 'XGB2', 'family': 'qrgb', 'max_depth': 6, 'min_child_weight': 5,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 12},
       {'name': 'XGB3', 'family': 'qrgb', 'max_depth': 8, 'min_child_weight': 10,
        'subsample': 0.7, 'colsample_bytree': 0.7, 'seed': 13},
       {'name': 'XGB4', 'family': 'qrgb', 'max_depth': 3, 'min_child_weight': 20,
        'subsample': 0.9, 'colsample_bytree': 0.6, 'seed': 14},
       {'name': 'XGB5', 'family': 'qrgb', 'max_depth': 6, 'min_child_weight': 2,
        'subsample': 0.6, 'colsample_bytree': 1.0, 'seed': 15}]
)


# ---------------------------------------------------------------------------

def _standardise(Xtr, X):
    m, s = Xtr.mean(axis=0), Xtr.std(axis=0)
    s[s < 1e-9] = 1.0
    return (Xtr - m) / s, (X - m) / s


def _fit_gpr(spec, Xtr, ytr, Xp):
    """Gaussian process regression — the one family that emits a density natively."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

    rng = np.random.default_rng(spec['seed'])
    idx = rng.choice(len(Xtr), size=min(GPR_SUBSAMPLE, len(Xtr)), replace=False)
    Xs, Xps = _standardise(Xtr[idx], Xp)
    ym, ys = ytr[idx].mean(), ytr[idx].std()

    d = Xs.shape[1]
    kernel = (Matern(length_scale=np.ones(d), nu=1.5)          # ARD Matern 3/2 ...
              + RBF(length_scale=np.ones(d))                    # ... plus an RBF, as in V-B
              + WhiteKernel(noise_level=0.1))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                  n_restarts_optimizer=0, random_state=spec['seed'])
    gp.fit(Xs, (ytr[idx] - ym) / ys)

    # Chunked: K(X*, X) at 40k x 2k would otherwise be a 640 MB intermediate.
    mu = np.empty(len(Xps))
    sd = np.empty(len(Xps))
    for a in range(0, len(Xps), 4000):
        b = min(a + 4000, len(Xps))
        mu[a:b], sd[a:b] = gp.predict(Xps[a:b], return_std=True)
    return mu * ys + ym, np.maximum(sd * ys, 1e-6)


def _fit_qrnn(spec, Xtr, ytr, Xp):
    """Quantile-regression neural net: one MLP, nine decile outputs, pinball loss."""
    import torch
    import torch.nn as nn

    torch.manual_seed(spec['seed'])
    torch.set_num_threads(1)
    cols = SUBSETS[spec['subset']]
    Xtr_, Xp_ = (Xtr, Xp) if cols is None else (Xtr[:, cols], Xp[:, cols])
    Xs, Xps = _standardise(Xtr_, Xp_)
    ym, ys = ytr.mean(), ytr.std()

    h, d = spec['hidden'], Xs.shape[1]
    if spec['arch'] == 'bn':                       # batch-norm acts as the regulariser
        layers = [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(),
                  nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
                  nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, len(DECILES))]
        wd = 0.0
    else:                                          # no batch-norm, so an L2 term instead
        layers = [nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                  nn.Linear(h, h), nn.ReLU(), nn.Linear(h, len(DECILES))]
        wd = 1e-4
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)

    Xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor((ytr - ym) / ys, dtype=torch.float32)[:, None]
    taus = torch.tensor(DECILES, dtype=torch.float32)[None, :]
    n, bs = len(Xt), 256
    for _ in range(spec.get('epochs', 120)):
        perm = torch.randperm(n)
        for a in range(0, n, bs):
            j = perm[a:a + bs]
            if len(j) < 2:                          # BatchNorm needs more than one row
                continue
            e = yt[j] - net(Xt[j])
            loss = torch.maximum(taus * e, (taus - 1.0) * e).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        q = net(torch.tensor(Xps, dtype=torch.float32)).numpy().T   # (9, T)
    return np.sort(q, axis=0) * ys + ym                             # rearrange any crossing


def _fit_qrgb(spec, Xtr, ytr, Xp):
    """Quantile-regression gradient boosting — one booster, all nine deciles at once."""
    import xgboost as xgb
    Xs, Xps = _standardise(Xtr, Xp)
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=DECILES,
                         n_estimators=300, learning_rate=0.05, max_depth=spec['max_depth'],
                         min_child_weight=spec['min_child_weight'], subsample=spec['subsample'],
                         colsample_bytree=spec['colsample_bytree'], random_state=spec['seed'],
                         n_jobs=1, verbosity=0)
    m.fit(Xs, ytr)
    return np.sort(np.asarray(m.predict(Xps)).T, axis=0)            # (9, T)


_FIT = {'gpr': _fit_gpr, 'qrnn': _fit_qrnn, 'qrgb': _fit_qrgb}


def fit_one(task):
    """(name, mu, sigma) for one spec. Module-level so a process pool can pickle it."""
    from src.density.gaussfit import quantiles_to_gaussian
    from src.quantile._pool import single_threaded
    spec, Xtr, ytr, Xp = task
    with single_threaded():
        out = _FIT[spec['family']](spec, Xtr, ytr, Xp)
    if spec['family'] == 'gpr':
        return spec['name'], out[0], out[1]
    mu, sigma = quantiles_to_gaussian(DECILES, out)
    return spec['name'], mu, sigma
