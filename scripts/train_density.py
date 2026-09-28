"""
Probabilistic baseline 2 — Li, Wang & Zhang (2020), IEEE TSG 11(2):1679-1690.
Combining probability density forecasts by minimising the CRPS.

    PJM_DATASET=dom python scripts/train_density.py
    PJM_DATASET=dom python scripts/train_density.py --workers 8 --models GPR1 NN1 XGB1

Parallel to the other two baselines in every structural respect: reads exactly one input
(data/{zone}/cleaned/cleaned_pjm_load_weather.csv), writes to results/{zone}/density/, and
touches neither the matrix cache, the 16 saved point models, nor the serving path.

    T1  train the fifteen base models
    T2  report them (the paper's Table I, first three rows)
    T3  optimise the combination weights
    T4  the shared test period, reported here

Three weighting schemes are compared, as in the paper: CRPS-guided (the contribution),
MAPE-guided, and simple average. The best individual base model on T3 is carried alongside,
because the paper's claim is that the combination beats it.

Horizon: data through day D-1, target day D+1 — this repo's cutoff, which is also the 16
point models'. It is 24 h behind a D-1 -> D issue time, and the paper's lag list is shifted by 24
to stay observable (see src/density/features.py).
"""

# Entry points live in scripts/ but import the package from the repo root, and running
# `python scripts/foo.py` puts scripts/ on sys.path rather than the root. Prepend the root so
# `import src...` resolves from any working directory.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import os

import numpy as np
import pandas as pd

import src.config as cfg
from src.density import base_models as B
from src.density import combine as C
from src.density import crps_gmm as G
from src.density import features as F
from src.density.gaussfit import mixture_mean, mixture_quantiles
from src.quantile._pool import n_workers, run
from src.quantile.qeval import (QUANTILES, coverage_report, pinball, point_metrics, winkler)
from src.quantile.frame import load_frame, split_at_day_boundary

EVAL_LEVELS = QUANTILES / 100.0          # the grid pinball/Winkler/coverage are read off


def _score(y, mu, sigma, w):
    """Every metric for one weighting, on one period."""
    q = mixture_quantiles(w, mu, sigma, EVAL_LEVELS, grid=1500)
    pm = point_metrics(y, mixture_mean(w, mu))
    return {'CRPS': float(G.crps(w, mu, sigma, y).mean()),
            'MAPE': pm['MAPE'], 'MAE': pm['MAE'],
            'pinball': float(pinball(y, q).mean()),
            'winkler50': winkler(y, q, 50), 'winkler90': winkler(y, q, 90)}, q


def main():
    dc = dict(cfg.DENSITY_CONFIG)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--models', nargs='*', default=None, help='subset of base model names')
    ap.add_argument('--refresh', action='store_true', help='ignore the cached base forecasts')
    a = ap.parse_args()
    if a.workers is not None:
        dc['n_workers'] = a.workers
    if a.models:
        dc['models'] = a.models

    df, ept, load, temp, hol, valid = load_frame()
    split = split_at_day_boundary(ept, dc['test_frac'])
    test_start = ept[split].normalize()

    X, y, rows = F.build(df, ept, load)
    ept_r, valid_r = ept[rows], valid[rows]
    m1, m2, m3, m4 = F.split_indices(ept_r, dc['t2_days'], dc['t3_days'], test_start)
    m1 = m1 & (valid_r == 1)                      # same 3-sigma mask the point models use

    print(f"=== density-combination baseline | zone {cfg.DATASET} ===")
    print(f"  features    {len(F.FEATURE_NAMES)} = 11 lags {F.LAGS} + {F.CALENDAR_NAMES}")
    print(f"  T1 train    {ept_r[m1][0].date()} -> {ept_r[m1][-1].date()}   ({m1.sum()} h)")
    print(f"  T2 report   {ept_r[m2][0].date()} -> {ept_r[m2][-1].date()}   ({m2.sum()} h)")
    print(f"  T3 weights  {ept_r[m3][0].date()} -> {ept_r[m3][-1].date()}   ({m3.sum()} h)")
    print(f"  T4 test     {ept_r[m4][0].date()} -> {ept_r[m4][-1].date()}   ({m4.sum()} h)")

    specs = [s for s in B.SPECS if dc['models'] is None or s['name'] in dc['models']]
    out_dir = dc['result_dir']
    os.makedirs(out_dir, exist_ok=True)

    # -- Base models: fit on T1, predict on T2+T3+T4 --------------------------------
    pred_mask = m2 | m3 | m4
    cache = os.path.join(out_dir, 'base_densities.npz')
    sig = json.dumps({'zone': cfg.DATASET, 'specs': [s['name'] for s in specs],
                      'lags': list(F.LAGS), 'n_train': int(m1.sum()),
                      'first': str(ept_r[pred_mask][0]), 'last': str(ept_r[pred_mask][-1])},
                     sort_keys=True)

    mu = sigma = None
    if os.path.exists(cache) and not a.refresh:
        z = np.load(cache, allow_pickle=True)
        if str(z['signature']) == sig:
            mu, sigma = z['mu'], z['sigma']
            print(f"\n  loaded {mu.shape[0]} cached base densities (--refresh to rebuild)")
    if mu is None:
        w = n_workers(dc['n_workers'])
        print(f"\n--- fitting {len(specs)} base models on {min(w, len(specs))} process(es) ---")
        tasks = [(s, X[m1], y[m1], X[pred_mask]) for s in specs]
        res = run(B.fit_one, tasks, w, label='base models')
        mu = np.stack([r[1] for r in res])
        sigma = np.stack([r[2] for r in res])
        np.savez_compressed(cache, mu=mu, sigma=sigma, signature=np.array(sig))
        print(f"  cached -> {cache}")

    # Re-index the three evaluation periods inside the prediction block.
    sub = {k: v[pred_mask] for k, v in {'2': m2, '3': m3, '4': m4}.items()}
    names = [s['name'] for s in specs]

    # -- Base model report on T2 (their Table I, first three rows) ------------------
    rows_t2 = []
    for i, nm in enumerate(names):
        one = np.zeros(len(names)); one[i] = 1.0
        sc, _ = _score(y[m2], mu[:, sub['2']], sigma[:, sub['2']], one)
        rows_t2.append({'model': nm, **sc})
    t2 = pd.DataFrame(rows_t2)
    t2.to_csv(os.path.join(out_dir, 'base_models_T2.csv'), index=False)
    print(f"\n--- base models on T2 ---")
    print(f"  {'model':8s}{'MAPE':>8s}{'MAE':>9s}{'CRPS':>9s}")
    for r in t2.itertuples():
        print(f"  {r.model:8s}{r.MAPE:7.2f}%{r.MAE:9.1f}{r.CRPS:9.1f}")

    # -- Weights on T3 --------------------------------------------------------------
    print(f"\n--- optimising weights on T3 ---")
    Q, c = G.qc_matrices(mu[:, sub['3']], sigma[:, sub['3']], y[m3])
    w_crps, info = C.solve_crps(Q, c, n_starts=dc['n_starts'], random_state=dc['random_state'])
    w_mape = C.solve_mape(mu[:, sub['3']], y[m3], random_state=dc['random_state'])
    w_avg = C.simple_average(len(names))
    print(f"  Q_new smallest eigenvalue {info['min_eig_Qnew']:+.4g}  -> "
          f"{'convex, solution is global' if info['certified_global'] else 'NOT certified convex'}")
    print(f"  CRPS-guided selects {info['n_selected']}/{len(names)} models")

    # The best single model on T3 — the benchmark the paper's claim is against.
    t3_crps = [float(G.crps(np.eye(len(names))[i], mu[:, sub['3']], sigma[:, sub['3']],
                            y[m3]).mean()) for i in range(len(names))]
    best_i = int(np.argmin(t3_crps))
    w_best = np.eye(len(names))[best_i]

    pd.DataFrame({'model': names, 'crps_weight': w_crps, 'mape_weight': w_mape,
                  'T3_crps': t3_crps}).to_csv(os.path.join(out_dir, 'weights.csv'), index=False)
    sel = [f"{n} {v:.2f}" for n, v in zip(names, w_crps) if v > 0.01]
    print(f"  CRPS weights: {'  '.join(sel)}")

    # -- Test on T4 -----------------------------------------------------------------
    schemes = {'CRPS-guided': w_crps, 'MAPE-guided': w_mape, 'Simple average': w_avg,
               f'Best individual ({names[best_i]})': w_best}
    res, curves = {}, {}
    for nm, w in schemes.items():
        res[nm], curves[nm] = _score(y[m4], mu[:, sub['4']], sigma[:, sub['4']], w)

    tbl = pd.DataFrame([{'model': k, **v} for k, v in res.items()])
    tbl.to_csv(os.path.join(out_dir, 'test_scores.csv'), index=False)
    print(f"\n{'=' * 78}\nTEST | zone {cfg.DATASET} | {int(m4.sum())} h "
          f"({ept_r[m4][0].date()} -> {ept_r[m4][-1].date()})\n{'=' * 78}")
    print(f"{'scheme':26s}{'CRPS':>9s}{'MAPE':>8s}{'MAE':>9s}{'pinball':>9s}"
          f"{'Wink50':>9s}{'Wink90':>9s}")
    for r in tbl.itertuples():
        print(f"{r.model:26s}{r.CRPS:9.1f}{r.MAPE:7.2f}%{r.MAE:9.1f}{r.pinball:9.1f}"
              f"{r.winkler50:9.0f}{r.winkler90:9.0f}")

    cov = pd.DataFrame([{'model': k, **cc} for k, q in curves.items()
                        for cc in coverage_report(y[m4], q)])
    cov.to_csv(os.path.join(out_dir, 'coverage.csv'), index=False)
    print(f"\n{'scheme':26s}" + "".join(f"{c:>9s}" for c in ['50%', '80%', '90%', '95%', '98%']))
    for k, q in curves.items():
        print(f"{k:26s}" + "".join(f"{c['empirical_pct']:8.1f}%"
                                   for c in coverage_report(y[m4], q)))

    # Same two breakdowns the other baselines write, so the shared figures work here too.
    pd.DataFrame({'quantile': QUANTILES,
                  **{k: pinball(y[m4], q).mean(axis=1) for k, q in curves.items()}}
                 ).to_csv(os.path.join(out_dir, 'pinball_by_quantile.csv'), index=False)

    mon = pd.DatetimeIndex(ept_r[m4]).to_period('M').astype(str).values
    mrows = []
    for mth in pd.unique(mon):
        s_ = mon == mth
        r = {'month': mth, 'hours': int(s_.sum()), 'mean_load': float(y[m4][s_].mean())}
        for k, q in curves.items():
            lo = q[np.where(QUANTILES == 5)[0][0]][s_]
            hi = q[np.where(QUANTILES == 95)[0][0]][s_]
            r[f'{k}_pinball'] = float(pinball(y[m4][s_], q[:, s_]).mean())
            r[f'{k}_cover90'] = float(100 * ((y[m4][s_] >= lo) & (y[m4][s_] <= hi)).mean())
            r[f'{k}_width90'] = float(np.mean(hi - lo))
        mrows.append(r)
    pd.DataFrame(mrows).to_csv(os.path.join(out_dir, 'monthly.csv'), index=False)

    np.savez_compressed(os.path.join(out_dir, 'test_quantiles.npz'),
                        actual=y[m4], datetime=ept_r[m4].values,
                        labels=np.array(list(curves), dtype=object),
                        **{f'cfg{i}': q for i, q in enumerate(curves.values())})
    print(f"\n-> {out_dir}")


if __name__ == '__main__':
    main()
