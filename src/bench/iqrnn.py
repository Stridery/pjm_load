"""Baseline B --- Zhang, Quan & Srinivasan (2019): an improved quantile regression NN.

ONE network trained on the summed pinball loss emits every quantile at once, instead of
fitting one model per quantile. The paper's contribution is not the architecture --- a
two-hidden-layer MLP with parametric ReLU --- but the training machinery brought over from
deep learning: batch training, early stopping, dropout, GAUSSIAN NOISE LAYERS and a
rule-based decaying learning rate.

The network, the loss and the schedule are imported unchanged from src/qrnn/model.py. They
take (X, y, masks, cfg) and know nothing about PJM, so there is nothing in them to port. What
is written here is the candidate pool, which is the one part of that paper that was written
against its own dataset.


The candidate pool
------------------
The paper's pool is calendar + holiday + load lags (1 d to 5 w) + temperature (now to 5 w),
1760 candidates, cut to about 100 by recursive feature elimination. Three things change.

**No temperature.** None of these datasets has it. Temperature is the paper's exogenous
driver known at the deadline, so each dataset's own exogenous columns take its place, under
the two availability rules the protocol gives:

    past_cols   realised values of other series, so only at lags inside the window
    known_cols  day-ahead forecasts describing the TARGET day, so at lag 0

**No five-week span.** The lookback window caps every lag at L-1+G, which is one week on all
three datasets. The paper's 1 d ... 5 w load block therefore becomes 1 d ... 1 w. This is
forced by the chapter's protocol, not by the data, and it is the single largest deviation
here.

**No holiday flag.** None of the three CSVs carries one, so the paper's third manual add-back
has nothing to point at and the add-back list is the remaining two.

The pool that results, against the paper's 1760:

    spain   145 target lags + 22 past x 7 echoes + 1 known + 10 calendar = 310
    ercot   145 target lags + 11 past x 7 echoes + 0 known + 10 calendar = 232
    elia    577 target lags +  0 past            + 4 known + 10 calendar = 591

past_cols get a daily-echo grid rather than the dense one the target gets. The paper can
afford a dense grid because it has ONE exogenous series; spain has 22, and a dense grid there
would be a 3000-column block that RFE would spend its whole budget inside.
"""

import hashlib
import json
import os

import numpy as np

from src.qrnn import model as Q

LEVELS = np.array([0.05, 0.50, 0.95])
N_ECHO = 7                      # daily echoes of each past column, after the first
CALENDAR_NAMES = ('hour', 'dayofweek', 'month', 'is_weekend',
                  'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos')
MANUAL_FEATURES = ('dayofweek', 'is_weekend')      # the paper's add-backs, minus is_holiday


def target_lags(ds):
    """Every step in the window --- the paper's dense treatment of its load block."""
    return list(range(ds.lag_min, ds.lag_max + 1))


def past_lags(ds):
    """The first legal lag, then daily echoes of it, all inside the window."""
    first = ds.lag_min
    aligned = -(-first // ds.H) * ds.H              # ceil to a whole number of days
    lags = [first] + [aligned + k * ds.H for k in range(N_ECHO)]
    return sorted({k for k in lags if ds.lag_min <= k <= ds.lag_max})


def candidate_names(ds):
    """Pool column names, in build() order."""
    return ([f'y_lag{k}' for k in target_lags(ds)]
            + [f'{c}_lag{k}' for c in ds.past_cols for k in past_lags(ds)]
            + [f'{c}_lag0' for c in ds.known_cols]
            + list(CALENDAR_NAMES))


def build(ds, day_starts, z):
    """(X, y) over every step of those days. float32 --- elia's full pool is 848 MB in f32.

    `z` is the z-scored target over the whole record; the exogenous columns are left in their
    own units here and standardised later with everything else, on training rows only.
    """
    steps = ds.steps_of(day_starts)
    names = candidate_names(ds)
    X = np.empty((len(steps), len(names)), dtype=np.float32)
    j = 0
    for k in target_lags(ds):
        X[:, j] = z[steps - k]; j += 1
    pl = past_lags(ds)
    for c in ds.past_cols:
        v = ds.column(c)
        for k in pl:
            X[:, j] = v[steps - k]; j += 1
    for c in ds.known_cols:
        X[:, j] = ds.column(c)[steps]; j += 1

    idx = ds.index[steps]
    hour = idx.hour.values.astype(float) + idx.minute.values / 60.0
    dow = idx.dayofweek.values.astype(float)
    month = idx.month.values.astype(float)
    for col in [hour, dow, month, (dow >= 5).astype(float),
                np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12)]:
        X[:, j] = col; j += 1
    assert j == len(names), (j, len(names))
    return X, z[steps]


def n_features_for(ds, cfg):
    """How many features RFE keeps.

    The paper keeps 100 of 1760, about 6%. These pools are 232 to 591, so keeping 100 would
    be keeping a third to a half --- RFE would barely be selecting. The rule is therefore
    `min(paper's 100, a third of the pool)`: it reproduces the paper's number wherever the
    pool is large enough to make it a real cut, and shrinks it where it is not.
    """
    if cfg.get('n_features'):
        return int(cfg['n_features'])
    return int(min(cfg.get('n_features_max', 100), len(candidate_names(ds)) // 3))


# ---------------------------------------------------------------------------

def _signature(ds, cfg, n_rows, n_keep):
    payload = json.dumps({'dataset': ds.name, 'pool': len(candidate_names(ds)),
                          'window': [ds.lag_min, ds.lag_max], 'n_rows': int(n_rows),
                          'n_keep': int(n_keep), 'manual': list(MANUAL_FEATURES),
                          'step': cfg['rfe_step'], 'rows': cfg['rfe_rows'],
                          'trees': cfg['rfe_trees'], 'criterion': cfg['rfe_criterion'],
                          'seed': cfg['random_state']}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16], payload


def select(ds, Xtr, ytr, cfg, cache_path=None, refresh=False, verbose=True):
    """(indices, names) of the retained features. Cached --- RFE is the slow step.

    Selection sees TRAINING ROWS ONLY. Letting it read the validation days would choose the
    features by the same data the network is stopped on.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import RFE

    names = candidate_names(ds)
    n_keep = n_features_for(ds, cfg)
    sig, payload = _signature(ds, cfg, len(Xtr), n_keep)

    if cache_path and os.path.exists(cache_path) and not refresh:
        cached = json.load(open(cache_path))
        if cached.get('signature') == sig:
            keep = cached['features']
            if verbose:
                print(f"  loaded {len(keep)} cached features (--refresh-rfe to rebuild)")
            return np.array([names.index(n) for n in keep]), keep

    rng = np.random.default_rng(cfg['random_state'])
    tr = np.arange(len(Xtr))
    if len(tr) > cfg['rfe_rows']:
        tr = np.sort(rng.choice(tr, size=cfg['rfe_rows'], replace=False))
    if verbose:
        print(f"  RFE: {len(names)} candidates -> {n_keep} on {len(tr)} rows "
              f"({cfg['rfe_trees']} trees, {cfg['rfe_criterion']}, step {cfg['rfe_step']:.0%})")

    forest = RandomForestRegressor(n_estimators=cfg['rfe_trees'],
                                   criterion=cfg['rfe_criterion'], max_depth=12,
                                   max_features=0.3, min_samples_leaf=20, n_jobs=-1,
                                   random_state=cfg['random_state'])
    rfe = RFE(forest, n_features_to_select=n_keep, step=cfg['rfe_step'])
    rfe.fit(Xtr[tr], ytr[tr])

    keep = [n for n, k in zip(names, rfe.support_) if k]
    added = [n for n in MANUAL_FEATURES if n not in keep]
    keep = keep + added
    if verbose:
        print(f"  selected {len(keep) - len(added)} + {len(added)} manual = {len(keep)}"
              + (f"   (added {', '.join(added)})" if added else ""))
        print(f"  kept by group: {summarise(ds, keep)}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        json.dump({'signature': sig, 'spec': json.loads(payload), 'features': keep,
                   'n_candidates': len(names)}, open(cache_path, 'w'), indent=1)
    return np.array([names.index(n) for n in keep]), keep


def summarise(ds, keep):
    """{group: count} of the retained features."""
    known = {f'{c}_lag0' for c in ds.known_cols}
    out = {'target lags': 0, 'past cols': 0, 'known cols': 0, 'calendar': 0}
    for n in keep:
        out['target lags' if n.startswith('y_lag') else
            'known cols' if n in known else
            'calendar' if n in CALENDAR_NAMES else 'past cols'] += 1
    return out


def batch_size_for(ds, cfg):
    """The paper's batch is 168 hourly samples --- one week, chosen because "the weekly
    pattern of electric load is considerably significant". On a 15-minute series one week is
    672 steps, so the batch follows the granularity rather than the integer."""
    if cfg.get('batch_size'):
        return int(cfg['batch_size'])
    return int(cfg.get('batch_hours', 168) * ds.steps_per_hour)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

DEFAULT_CFG = {
    'spec': 'iQRNN-500',        # one of src.qrnn.model.SPECS
    'n_features': None,         # None -> n_features_for()'s rule
    'n_features_max': 100,      # the paper's count, used where the pool supports it
    'rfe_step': 0.35,
    'rfe_rows': 12000,
    'rfe_trees': 40,
    'rfe_criterion': 'absolute_error',
    'batch_size': None,         # None -> batch_hours * steps_per_hour
    'batch_hours': 168,         # the paper's one week
    'max_epochs': 200,
    'lr': 1e-3,
    'min_lr': 1e-5,
    'lr_decay': 0.4,
    'lr_patience': 5,
    'stop_patience': 20,
    'noise_sd': 0.3,
    'device': 'auto',
    'random_state': 42,
}


def run(ds, seed=0, cfg=None, cache_path=None, refresh_rfe=False, verbose=True):
    """Fit and score one seed. Returns a dict; writes only the RFE cache.

    The validation days do two jobs the paper gives to two separate stages: early stopping,
    and whatever configuration choice is made. The paper tunes on a validation year and then
    RE-TRAINS on train+validation with an internal 90/10 split. Retraining on the selection
    set would leave nothing held out to stop on, and the chapter's three periods do not
    provide a fourth, so the two are collapsed here as they are in the PJM port.
    """
    from src.bench import metrics as M

    cfg = {**DEFAULT_CFG, **(cfg or {})}
    sp = ds.splits()
    mean, sd = ds.standardiser(sp['train'])
    z = (ds.y - mean) / sd

    Xtr, ytr = build(ds, sp['train'], z)
    Xva, yva = build(ds, sp['val'], z)
    Xte, yte = build(ds, sp['test'], z)
    if verbose:
        print(f"  pool {len(candidate_names(ds))} candidates; matrices "
              f"{Xtr.shape} / {Xva.shape} / {Xte.shape} float32 "
              f"({(Xtr.nbytes + Xva.nbytes + Xte.nbytes) / 2**30:.2f} GB)")

    keep, names = select(ds, Xtr, ytr, cfg, cache_path=cache_path, refresh=refresh_rfe,
                         verbose=verbose)
    Xtr, Xva, Xte = Xtr[:, keep], Xva[:, keep], Xte[:, keep]

    spec = Q.SPEC_BY_NAME[cfg['spec']]
    run_cfg = {**cfg, 'batch_size': batch_size_for(ds, cfg)}
    if verbose:
        print(f"  {spec['name']}: {Xtr.shape[1]} -> {spec['hidden']} -> {spec['hidden']} "
              f"-> {len(LEVELS)}   batch {run_cfg['batch_size']}  seed {seed}")

    # The target is already z-scored, so run_spec's own standardisation is a near-identity
    # and the returned quantiles come back in the reporting units.
    n = len(ytr) + len(yva) + len(yte)
    X = np.vstack([Xtr, Xva, Xte])
    y = np.concatenate([ytr, yva, yte])
    tr_m = np.zeros(n, bool); tr_m[:len(ytr)] = True
    va_m = np.zeros(n, bool); va_m[len(ytr):len(ytr) + len(yva)] = True
    te_m = np.zeros(n, bool); te_m[len(ytr) + len(yva):] = True

    sorted_q, raw_q, hist = Q.run_spec(spec, X, y, tr_m, va_m, te_m, run_cfg, LEVELS,
                                       seed, verbose=False)
    q = sorted_q['test']
    frac, cs = Q.crossing_rate(raw_q['test'], step=1.0 / (len(LEVELS) + 1))
    scores = M.score(yte, q, LEVELS, scale=(mean, sd))
    if verbose:
        print(f"  epochs {hist['epochs']} (best {hist['best_epoch']}), "
              f"{hist['seconds'] / 60:.1f} min, {hist['n_params']:,} params, "
              f"val pinball {hist['val_pinball']:.5f}; crossing {100 * frac:.2f}% of steps")

    return {'dataset': ds.name, 'seed': seed, 'spec': spec['name'], 'features': names,
            'scores': scores, 'curves': {spec['name']: q}, 'levels': LEVELS,
            'actual': yte, 'datetime': ds.index[ds.steps_of(sp['test'])].values,
            'scale': (mean, sd), 'history': hist,
            'crossing_frac': float(frac), 'crossing_cs': float(cs),
            'n_test_steps': len(yte), 'groups': summarise(ds, names)}
