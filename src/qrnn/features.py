"""The candidate pool and the recursive feature elimination that cuts it to 103.

Zhang et al. Section III-B lists the pool exactly:

    "1760 features ... include date and time, holiday information (whether the day is a
     holiday or not), lag load data, and temperature data. The lag load data are from one
     day ago to five weeks ago, whereas the temperature data are from the current time to
     five weeks ago. Out of the 1760 features, 100 features are selected"

then "some of the features, such as the weekday and holiday information, are manually added
into the selected features ... eliminated because the number of weekends and holidays is
relatively small in the dataset", giving 103 per sample.

Reproduced here as 1645 candidates -> 100 by RFE -> 103 with the manual add-backs. The two
places the pool differs from the paper are both forced:

    load lags start at 48 h, not 24. The repo's cutoff is the target day's previous
        midnight, so target hour k sits 25+k hours after the last observed load and a lag
        valid for all 24 hours must be >= 48. Lag 24 is never observable, lags 25-28 only
        for k < 4. Same shift, same reason, as src/density/features.py.
    temperature only. The paper's weather is one averaged temperature series and nothing
        else; the cleaned CSV carries fourteen weather columns, and adding them would make
        this a different model from the published one. Temp_F alone also keeps the feature
        set on the same footing as the QRA and scenario baselines.

One candidate is deliberately absent: a linear TREND. The MLR baselines need one because a
linear model has no other way to carry DOM's data-centre growth (see QUANTILE_CONFIG), but
here the load-lag block states the current level outright, and a monotone trend column takes
values in the test period that never occurred in training — which is the one input shape a
neural network cannot extrapolate.


Why RFE with a random forest on MAE
-----------------------------------
The paper's reasoning is transcribed rather than improved on: a quantile regression forest
scored by pinball would match the downstream loss, but "the training of quanregForest is
time-consuming ... RFE requires repeated and massive training of regression models", so it
substitutes MAE-trained random forests as "a practical compromise", noting MAE is the
pinball loss at the 50th percentile. Two further compromises are ours and are recorded in
the cache file: RFE runs on a row subsample and drops a fixed FRACTION per round rather than
one feature at a time. Eliminating 1545 features one at a time is 1545 forest fits.
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd

LOAD_LAG_MIN = 48                     # the blind spot; the paper's 24 is not observable here
LOAD_LAG_MAX = 840                    # five weeks, as published
TEMP_LAG_MIN = 0                      # realised temperature as the forecast proxy
TEMP_LAG_MAX = 840

LOAD_LAGS = tuple(range(LOAD_LAG_MIN, LOAD_LAG_MAX + 1))
TEMP_LAGS = tuple(range(TEMP_LAG_MIN, TEMP_LAG_MAX + 1))

# "date and time, holiday information". The raw codes are the paper's; the cyclic pairs are
# this repo's usual encoding of the same three fields, offered alongside so RFE can pick
# whichever form carries more signal rather than having the choice made for it.
CALENDAR_NAMES = ['hour', 'dayofweek', 'month', 'is_weekend', 'is_holiday',
                  'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']

MIN_HISTORY = max(LOAD_LAG_MAX, TEMP_LAG_MAX)


def candidate_names(load_lag_min=LOAD_LAG_MIN):
    """The candidate column names, in build() order. 1645 at the default lag floor."""
    return ([f'load_lag{h}' for h in range(int(load_lag_min), LOAD_LAG_MAX + 1)]
            + [f'temp_lag{h}' for h in TEMP_LAGS]
            + list(CALENDAR_NAMES))


def build(ept, load, temp, hol, load_lag_min=LOAD_LAG_MIN):
    """(X, y, rows) over every hour with a complete candidate set.

    One sample is ONE TARGET HOUR, which is the paper's granularity and also what keeps the
    sample count in its regime: 1942 usable days x 24 h is ~46.6k samples against its 42,984.
    Feeding whole days instead would leave ~1.5k samples for a 24x99 output and overfit
    immediately.

    `rows` indexes back into the cleaned frame so the caller can slice train/validation/test
    without re-deriving timestamps. Lags are taken by POSITION, which is exact on DST days:
    the cleaned CSV is built off UTC, so consecutive rows are consecutive real hours and a
    positional lag is an elapsed-time lag.
    """
    load = np.asarray(load, dtype=float)
    temp = np.asarray(temp, dtype=float)
    hol = np.asarray(hol, dtype=float)
    rows = np.arange(MIN_HISTORY, len(load))

    names = candidate_names(load_lag_min)
    X = np.empty((len(rows), len(names)), dtype=np.float32)
    j = 0
    for h in range(int(load_lag_min), LOAD_LAG_MAX + 1):
        X[:, j] = load[rows - h]
        j += 1
    for h in TEMP_LAGS:
        X[:, j] = temp[rows - h]
        j += 1

    e = ept[rows]
    hour = e.hour.values.astype(float)
    dow = e.dayofweek.values.astype(float)
    month = e.month.values.astype(float)
    for col in [hour, dow, month, (dow >= 5).astype(float), hol[rows],
                np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12)]:
        X[:, j] = col
        j += 1
    assert j == len(names)

    return X, load[rows], rows


def split_masks(ept_rows, test_start, val_days):
    """(train, valid, test) boolean masks over the sample rows.

    `test` is the shared tail-0.16 period every baseline in this repo is scored on. `valid`
    is the `val_days` days immediately before it and does two jobs the paper gives to two
    separate stages: early stopping, and choosing between network configurations. Collapsing
    the two is a deviation — the paper tunes on a validation year and then RE-TRAINS on
    train+validation with an internal 90/10 split for early stopping. Retraining on the
    selection set would leave nothing held out to stop on, and at this record length a second
    disjoint year cannot be spared.
    """
    test_start = pd.Timestamp(test_start)
    val_start = test_start - pd.Timedelta(days=int(val_days))
    test = ept_rows >= test_start
    valid = (ept_rows >= val_start) & (~test)
    train = ept_rows < val_start
    if not train.any() or not valid.any():
        raise SystemExit(f"val_days={val_days} leaves no training rows before {val_start}")
    return train, valid, test


# ---------------------------------------------------------------------------
# Recursive feature elimination
# ---------------------------------------------------------------------------

def _signature(zone, n_rows, n_select, manual, step, sub_rows, trees, crit, seed, lag_min,
               first, last):
    payload = json.dumps({'zone': zone, 'n_rows': int(n_rows), 'n_select': int(n_select),
                          'manual': list(manual), 'step': float(step),
                          'sub_rows': int(sub_rows), 'trees': int(trees), 'criterion': crit,
                          'seed': int(seed),
                          'load_lags': [int(lag_min), LOAD_LAG_MAX],
                          'temp_lags': [TEMP_LAG_MIN, TEMP_LAG_MAX],
                          'calendar': CALENDAR_NAMES, 'first': str(first), 'last': str(last)},
                         sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16], payload


def select(X, y, train_mask, cfg, zone, ept_rows, cache_path=None, refresh=False, verbose=True):
    """(indices, names) of the ~103 retained features. Cached — RFE is the slow step.

    Selection sees TRAINING ROWS ONLY. Letting it read the validation year would pick the
    features by the same data the network is stopped on; letting it read the test period
    would be leakage of the ordinary kind.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import RFE

    names = candidate_names(cfg['load_lag_min'])
    sig, payload = _signature(zone, int(train_mask.sum()), cfg['n_features'],
                              cfg['manual_features'], cfg['rfe_step'], cfg['rfe_rows'],
                              cfg['rfe_trees'], cfg['rfe_criterion'], cfg['random_state'],
                              cfg['load_lag_min'],
                              ept_rows[train_mask][0], ept_rows[train_mask][-1])

    if cache_path and os.path.exists(cache_path) and not refresh:
        cached = json.load(open(cache_path))
        if cached.get('signature') == sig:
            keep = cached['features']
            if verbose:
                print(f"  loaded {len(keep)} cached features (--refresh-rfe to rebuild)")
            return np.array([names.index(n) for n in keep]), keep

    tr = np.where(train_mask)[0]
    rng = np.random.default_rng(cfg['random_state'])
    if len(tr) > cfg['rfe_rows']:
        tr = np.sort(rng.choice(tr, size=cfg['rfe_rows'], replace=False))
    if verbose:
        print(f"  RFE: {len(names)} candidates -> {cfg['n_features']} on {len(tr)} rows "
              f"({cfg['rfe_trees']} trees, {cfg['rfe_criterion']}, step {cfg['rfe_step']:.0%})")

    forest = RandomForestRegressor(n_estimators=cfg['rfe_trees'], criterion=cfg['rfe_criterion'],
                                   max_depth=12, max_features=0.3, min_samples_leaf=20,
                                   n_jobs=-1, random_state=cfg['random_state'])
    rfe = RFE(forest, n_features_to_select=cfg['n_features'], step=cfg['rfe_step'])
    rfe.fit(X[tr], y[tr])

    keep = [n for n, k in zip(names, rfe.support_) if k]
    added = [n for n in cfg['manual_features'] if n not in keep]
    keep = keep + added                      # the paper's manual add-backs, appended in order
    if verbose:
        print(f"  selected {len(keep) - len(added)} + {len(added)} manual = {len(keep)}"
              + (f"   (added {', '.join(added)})" if added else ""))

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        json.dump({'signature': sig, 'spec': json.loads(payload), 'features': keep,
                   'n_candidates': len(names)}, open(cache_path, 'w'), indent=1)
        if verbose:
            print(f"  cached -> {cache_path}")

    return np.array([names.index(n) for n in keep]), keep


def summarise(keep):
    """{group: count} of the retained features — the one-line read of what RFE kept."""
    out = {'load lags': 0, 'temperature': 0, 'calendar': 0}
    for n in keep:
        out['load lags' if n.startswith('load_lag') else
            'temperature' if n.startswith('temp_lag') else 'calendar'] += 1
    return out
