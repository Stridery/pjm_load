"""Baseline A — Li, Wang & Zhang (2020): combining density forecasts by minimising the CRPS.

The dataset-agnostic mathematics is imported unchanged from src/density: the probit fit that
turns deciles into Gaussians (gaussfit), the closed-form mixture CRPS (crps_gmm) and the
simplex QP that finds the weights (combine). Those take plain arrays and know nothing about
PJM.

The fifteen base specifications come from src/bench/base_models instead, because that one
module DID know about PJM: its neural-net feature subsets were POSITIONAL indices into a fixed
15-column layout, and two of these datasets have 14 columns. Against 14, the 'lags' subset
silently swallowed `hour` as if it were a lagged load and 'recent' raised IndexError. See that
module's docstring.

What is written here is only what the chapter's protocol changes — the feature set, the T1-T4 mapping, the
z-scored reporting space, and the three-level output.


The fifteen features
--------------------
The paper's own list, and nothing else: eleven lagged observations of the target plus hour of
day, day of week, month of year and a weekend flag. No exogenous variables at all. That is
Li's explicit choice, not a simplification here — the contribution being tested is the
combination step, so the base models are deliberately generic.

The paper's lags are 24, 25, 26, 27, 28, 48, 72, 96, 120, 144, 168 hours: the target step and
the four before it on the most recent fully observed day, then six daily lags out to a week.
Under this chapter's deadline the smallest legal lag is G+H-1 steps, which is larger than 24 h
on two of the three datasets, so the whole list is shifted by WHOLE DAYS until it clears that
floor. Shifting preserves the structure of the list; truncating it would not. Any lag pushed
past the far end of the lookback window, L-1+G, is then dropped:

    spain   [24, 168]   24 25 26 27 28 | 48 72 96 120 144 168          11, the paper verbatim
    ercot   [38, 182]   48 49 50 51 52 | 72 96 120 144 168             10, 192 falls outside
    elia    [99, 675]   192 196 200 204 208 | 288 384 480 576 672      10, 768 falls outside

So ercot and elia carry ten lagged features rather than eleven — fourteen features, not
fifteen. Both lose the same one, the far end of the weekly block, which is the least
informative member of the list.


The periods
-----------
Li splits into T1 (fit the base models), T2 (report them), T3 (optimise the weights) and T4
(test). The chapter supplies three ranges, so T2 and T3 are the two halves of the validation
days, in order:

    T1 = train days      T2 = first half of val      T3 = second half of val      T4 = test

T3 only has to pin down fifteen weights — the paper itself uses five weeks — so the split is
not delicate.
"""

import numpy as np

from src.bench import base_models as B
from src.density import combine as C
from src.density import crps_gmm as G
from src.density.gaussfit import mixture_quantiles

PAPER_SUBDAY_HOURS = (24, 25, 26, 27, 28)
PAPER_DAILY_HOURS = (48, 72, 96, 120, 144, 168)
CALENDAR_NAMES = ('hour', 'dayofweek', 'month', 'is_weekend')
LEVELS = np.array([0.05, 0.50, 0.95])
SCHEMES = ('CRPS-guided', 'MAPE-guided', 'Simple average', 'Best individual')


def lags_for(ds):
    """Lag list in STEPS for this dataset — the paper's hours, shifted and clipped."""
    hours = PAPER_SUBDAY_HOURS + PAPER_DAILY_HOURS
    base = np.array([h * ds.steps_per_hour for h in hours])
    shift = 0
    while base.min() + shift < ds.lag_min:          # shift by whole days, never truncate
        shift += ds.H
    lags = base + shift
    if lags.min() < ds.lag_min:
        raise ValueError(f"{ds.name}: no whole-day shift clears the lag floor")
    return [int(k) for k in lags if k <= ds.lag_max]


def feature_names(ds):
    return [f'lag{k}' for k in lags_for(ds)] + list(CALENDAR_NAMES)


def build(ds, day_starts, z):
    """(X, y) for every step of those days. `z` is the z-scored target over the whole record.

    Every step is its own sample: the lag floor guarantees each lag is observable for every
    step of the day, so there is no per-day assembly and no ragged edge.
    """
    steps = ds.steps_of(day_starts)
    lags = lags_for(ds)
    idx = ds.index[steps]
    cols = [z[steps - k] for k in lags]
    cols.append(idx.hour.values.astype(float) + idx.minute.values / 60.0)
    cols.append(idx.dayofweek.values.astype(float))
    cols.append(idx.month.values.astype(float))
    cols.append((idx.dayofweek.values >= 5).astype(float))
    return np.column_stack(cols), z[steps]


def specs_for_seed(seed, names=None):
    """The fifteen specifications with every internal seed offset, so a run is a real repeat:
    different GPR subsamples, different net initialisations, different booster subsampling."""
    out = []
    for s in B.SPECS:
        if names and s['name'] not in names:
            continue
        s = dict(s)
        s['seed'] = s['seed'] + 1000 * seed
        out.append(s)
    return out


def _mean_crps(w, Q, c, T):
    """Mean mixture CRPS from the summed quadratic terms — Li eq. (7), divided by T."""
    w = np.asarray(w, float)
    return float((w @ Q @ w + c @ w) / T)


def run(ds, seed=0, workers=None, models=None, grid=1500, verbose=True):
    """Fit, weight and score one seed. Returns a dict of results; writes nothing."""
    from src.quantile._pool import n_workers, run as prun

    sp = ds.splits()
    val = sp['val']
    half = len(val) // 2
    periods = {'T1': sp['train'], 'T2': val[:half], 'T3': val[half:], 'T4': sp['test']}

    mean, sd = ds.standardiser(sp['train'])
    z = (ds.y - mean) / sd

    X = {k: build(ds, v, z) for k, v in periods.items()}
    specs = specs_for_seed(seed, models)
    names = [s['name'] for s in specs]

    if verbose:
        print(f"  features {len(feature_names(ds))} = {len(lags_for(ds))} lags "
              f"{lags_for(ds)} + {list(CALENDAR_NAMES)}")
        for nm in B.SUBSET_NAMES:
            c = B.subset_cols(nm, feature_names(ds))
            n_c = len(feature_names(ds)) if c is None else len(c)
            print(f"    net subset {nm:7s} {n_c:2d} cols")
        for k, v in periods.items():
            print(f"  {k} {len(v):5d} days  {ds.index[v[0]].date()} -> "
                  f"{ds.index[v[-1] + ds.H - 1].date()}   ({len(v) * ds.H} steps)")

    # -- base models: fit on T1, predict on T2+T3+T4 ---------------------------------
    Xp = np.vstack([X[k][0] for k in ('T2', 'T3', 'T4')])
    cut = np.cumsum([0] + [len(X[k][1]) for k in ('T2', 'T3', 'T4')])
    w_ = n_workers(workers)
    if verbose:
        print(f"  fitting {len(specs)} base models on {min(w_, len(specs))} process(es) "
              f"({len(X['T1'][1])} training steps)")
    names_f = feature_names(ds)
    tasks = []
    for s in specs:
        cols = B.subset_cols(s.get('subset', 'all'), names_f)
        tr, pr = (X['T1'][0], Xp) if cols is None else (X['T1'][0][:, cols], Xp[:, cols])
        tasks.append((s, tr, X['T1'][1], pr))
    res = prun(B.fit_one, tasks, w_, label='base models' if verbose else None)
    mu_all = np.stack([r[1] for r in res])
    sg_all = np.stack([r[2] for r in res])
    part = {k: (mu_all[:, cut[i]:cut[i + 1]], sg_all[:, cut[i]:cut[i + 1]])
            for i, k in enumerate(('T2', 'T3', 'T4'))}

    # -- weights on T3 ---------------------------------------------------------------
    mu3, sg3 = part['T3']
    y3 = X['T3'][1]
    Q3, c3 = G.qc_matrices(mu3, sg3, y3)
    w_crps, info = C.solve_crps(Q3, c3, random_state=seed)
    # MAPE divides by the target, and the z-scored target crosses zero, so the competing
    # scheme is solved in PHYSICAL units. The weights are unchanged by that: they sum to one,
    # so the affine map commutes with the combination.
    w_mape = C.solve_mape(mu3 * sd + mean, y3 * sd + mean, random_state=seed)
    w_avg = C.simple_average(len(names))
    best_i = int(np.argmin(np.diag(Q3) + c3))             # one-hot w: mean CRPS = Q_ii + c_i
    w_best = np.eye(len(names))[best_i]

    # -- test on T4 ------------------------------------------------------------------
    mu4, sg4 = part['T4']
    y4 = X['T4'][1]
    Q4, c4 = G.qc_matrices(mu4, sg4, y4)
    T4 = len(y4)

    weights = {'CRPS-guided': w_crps, 'MAPE-guided': w_mape, 'Simple average': w_avg,
               'Best individual': w_best}
    scores, curves = {}, {}
    from src.bench import metrics as M
    for nm, w in weights.items():
        q = mixture_quantiles(w, mu4, sg4, LEVELS, grid=grid)
        scores[nm] = {**M.score(y4, q, LEVELS, scale=(mean, sd)),
                      'CRPS': _mean_crps(w, Q4, c4, T4)}
        curves[nm] = q

    # Every base model on its own, over T2 — the paper's Table I.
    mu2, sg2 = part['T2']
    Q2, c2 = G.qc_matrices(mu2, sg2, X['T2'][1])
    base_t2 = {nm: _mean_crps(np.eye(len(names))[i], Q2, c2, len(X['T2'][1]))
               for i, nm in enumerate(names)}
    base_t4 = {nm: _mean_crps(np.eye(len(names))[i], Q4, c4, T4) for i, nm in enumerate(names)}

    return {'dataset': ds.name, 'seed': seed, 'names': names, 'best_individual': names[best_i],
            'weights': {k: np.asarray(v, float) for k, v in weights.items()},
            'scores': scores, 'curves': curves, 'levels': LEVELS,
            'actual': y4, 'datetime': ds.index[ds.steps_of(periods['T4'])].values,
            'scale': (mean, sd), 'base_T2_crps': base_t2, 'base_T4_crps': base_t4,
            'qp_info': info, 'n_test_steps': T4}
