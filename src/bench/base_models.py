"""The fifteen base density models, for the public datasets.  Li et al. Section IV-A/B/C, V-B.

A separate module from src/density/base_models.py rather than an import of it, for one
concrete reason: that module's feature subsets are POSITIONAL indices written against PJM's
fixed 15-column layout,

    'lags':   list(range(11))
    'recent': [0, 1, 2, 3, 4, 5, 11, 12, 13, 14]

and two of these datasets have 14 columns, not 15 (Sec. "Features" in src/bench/density.py:
the deepest lag leaves the lookback window on ercot and elia). Against 14 columns `lags` would
silently swallow `hour` as if it were a lagged load, and `recent` would raise IndexError on the
first fit. Neither is repairable from outside without editing the PJM module, which is not
ours to change.

Here the subsets are resolved BY NAME from the feature list, so they mean the same thing at any
width. Everything else - the three families, the five members each, every hyper-parameter, the
deciles, the standardisation, the sorting of crossed quantiles - is the PJM module's, carried
over unchanged, so the two ports remain the same model.

The point of the fifteen is DIVERSITY, not individual accuracy: the combination is what is
being tested, and near-identical members would make the weights arbitrary. The paper's three
sources of it:

    GPR   a different random 2000-point subsample of T1 each time. Not stylistic: GPR inverts
          an n x n matrix, so the paper subsamples "to train GPR base models in relatively
          short time", and the subsample is what makes the five differ.
    QRNN  two architectures (batch-norm, or L2 in its place) x three feature subsets x width.
    QRGB  one architecture, different max_depth / min_child_weight / subsample /
          colsample_bytree.

Standardisation is fit on T1 only and applied everywhere. Trees do not need it, but they get
it anyway so that every family sees the identical matrix and a difference in results can never
be a difference in preprocessing.
"""

import numpy as np

DECILES = np.arange(1, 10) / 10.0        # the paper converts "decile results" to Gaussians
GPR_SUBSAMPLE = 2000                     # their Section V-B
N_SHORT_LAGS = 5                         # the sub-daily block: the step itself and the 4 before
CALENDAR = ('hour', 'dayofweek', 'month', 'is_weekend')

SUBSET_NAMES = ('all', 'lags', 'recent')

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


def subset_cols(name, feature_names):
    """Column indices of a named subset, resolved from the feature names.

        all      every column
        lags     the lagged-target columns only, however many there are
        recent   the sub-daily block, the first daily lag, and the calendar columns

    `recent` is defined by the STRUCTURE of the lag list, not by a count: the five clustered
    short lags plus the first of the daily lags. On PJM's 15-column layout that resolves to
    exactly the hardcoded [0,1,2,3,4,5,11,12,13,14], which is the check that this is a
    re-expression of the same subset rather than a new one.
    """
    lags = [i for i, n in enumerate(feature_names) if n.startswith('lag')]
    cal = [i for i, n in enumerate(feature_names) if n in CALENDAR]
    if len(lags) + len(cal) != len(feature_names):
        raise ValueError(f"unclassifiable feature names: {feature_names}")
    if name == 'all':
        return None
    if name == 'lags':
        return lags
    if name == 'recent':
        return lags[:N_SHORT_LAGS + 1] + cal
    raise ValueError(f"unknown subset {name!r}; expected one of {SUBSET_NAMES}")


# ---------------------------------------------------------------------------

def _standardise(Xtr, X):
    m, s = Xtr.mean(axis=0), Xtr.std(axis=0)
    s[s < 1e-9] = 1.0
    return (Xtr - m) / s, (X - m) / s


def _fit_gpr(spec, Xtr, ytr, Xp):
    """Gaussian process regression - the one family that emits a density natively."""
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

    # Chunked: K(X*, X) at 180k x 2k would otherwise be a 2.9 GB intermediate on elia.
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
    Xs, Xps = _standardise(Xtr, Xp)          # columns already subset by the caller
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
    n, bs = len(Xt), spec.get('batch_size', 256)
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
        q = np.concatenate([net(torch.tensor(Xps[a:a + 20000], dtype=torch.float32)).numpy()
                            for a in range(0, len(Xps), 20000)]).T          # (9, T)
    return np.sort(q, axis=0) * ys + ym                                     # rearrange crossing


def _fit_qrgb(spec, Xtr, ytr, Xp):
    """Quantile-regression gradient boosting - one booster, all nine deciles at once."""
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
    """(name, mu, sigma) for one spec. Module-level so a process pool can pickle it.

    The caller has already applied the spec's column subset, so `Xtr` and `Xp` arrive at the
    width this model is meant to see. Doing it there rather than here keeps the subsetting
    where the feature names are, and keeps each task's pickled payload down to the columns it
    actually uses.
    """
    from src.density.gaussfit import quantiles_to_gaussian
    from src.quantile._pool import single_threaded
    spec, Xtr, ytr, Xp = task
    with single_threaded():
        out = _FIT[spec['family']](spec, Xtr, ytr, Xp)
    if spec['family'] == 'gpr':
        return spec['name'], out[0], out[1]
    mu, sigma = quantiles_to_gaussian(DECILES, out)
    return spec['name'], mu, sigma
