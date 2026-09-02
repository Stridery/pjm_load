"""
Probabilistic baseline 4 — Zhang, Quan & Srinivasan (2019), IEEE TSG 10(4):4425-4434.
An improved quantile regression neural network: one network, all 99 percentiles.

    PJM_DATASET=dom python train_qrnn.py
    PJM_DATASET=bge python train_qrnn.py --models iQRNN-500 bQRNN-400 --n-runs 3
    PJM_DATASET=dom python train_qrnn.py --lag-min 24        # QRA's issue time, for the
                                                             # strict head-to-head
    PJM_DATASET=dom python train_qrnn.py --size-sweep --noise-sweep
    PJM_DATASET=dom python -m src.qrnn.plots                 # figures from what the run wrote

Parallel to train_quantile.py / train_qra.py / train_density.py in every structural respect:
reads exactly one input (data/{zone}/cleaned/cleaned_pjm_load_weather.csv), writes to
results/{zone}/qrnn/, and touches neither the matrix cache, the 16 saved point models, nor
the serving path. Tuning lives in QRNN_CONFIG in src/config.py.

Reports:

    iQRNN-*    the proposed network, at two widths x {dropout, zero dropout}
    bQRNN-*    the paper's own control: same MLP and batch training, none of the four
               additions (noise, dropout, early stopping, LR decay). Without this row the
               reported gain cannot be attributed to the training machinery.
    QRA        folded in from results/{zone}/qra/ if present — this is the comparison the
               paper's headline claim is about.
    density    folded in from results/{zone}/density/ if present.

Every configuration is trained `n_runs` times from different seeds and reported as mean and
SD, because the paper's robustness argument is about SPREAD: its bQRNN scores an SD of 0.789
against iQRNN's 0.008, and a single run cannot show that.


Horizons in one table
---------------------
The folded-in rows do not all answer the same question and the table says so rather than
letting a reader assume otherwise:

    this run    data through (D-1) 23:00 -> day D+1     (--lag-min 48, the default)
    QRA         data through (D-1) 23:00 -> day D       (a day nearer)
    density     data through (D-1) 23:00 -> day D+1     (same as this run)

All of them, including this one, use REALISED temperature where a forecast would be, which
is what the paper does ("actual temperatures ... as a proxy for temperature forecasts") and
what makes the four comparable with each other. None of them is comparable with the 16 point
models, which read archived real forecasts.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import src.config as cfg
from src.qrnn import features as F
from src.qrnn import model as M
from src.quantile.qeval import (QUANTILES, coverage_report, crps_from_quantiles, pinball,
                                point_metrics, winkler)
from train_quantile import load_frame, split_at_day_boundary

LEVELS = QUANTILES / 100.0
MEDIAN = int(np.where(QUANTILES == 50)[0][0])
SCORE_KEYS = ('pinball', 'winkler50', 'winkler90', 'CRPS',
              'median_MAPE', 'median_MAE', 'median_RMSE', 'median_ME')


def _scores(y, q):
    """Every metric the other three baselines report, plus the CRPS they share.

    Every row in this table — including the folded-in ones — is scored by THIS function, from
    quantile curves, rather than by copying the number each baseline reported for itself. It
    costs a little: the density baseline computes its CRPS in closed form from the Gaussian
    mixture, and reading it off a 99-point quantile grid instead loses the tails beyond the
    1st and 99th percentiles (~0.15% on dom). One consistent method across the column is
    worth more than each row being individually optimal, because the column is what gets read
    as a comparison.
    """
    med = q[MEDIAN]
    pm = point_metrics(y, med)
    return {'pinball': float(pinball(y, q).mean()),
            'winkler50': winkler(y, q, 50), 'winkler90': winkler(y, q, 90),
            'CRPS': float(crps_from_quantiles(LEVELS, q, y).mean()),
            'median_MAPE': pm['MAPE'], 'median_MAE': pm['MAE'], 'median_RMSE': pm['RMSE'],
            'median_ME': float(np.mean(med - y))}


def _key(idx):
    """Duplicate-aware datetime key. EPT is naive local time, so the fall-back day carries
    01:00 twice and a plain index is not unique; disambiguating by occurrence is what lets
    two runs be aligned without silently dropping or doubling an hour."""
    t = pd.Series(pd.DatetimeIndex(idx))
    return pd.MultiIndex.from_arrays([t, t.groupby(t).cumcount()])


def _fold_in(npz_path, y_test, when_test, horizon, res, curves, only=None):
    """Add another baseline's saved quantile curves to the table, on the shared hours.

    Reads the test_quantiles.npz that every baseline here writes, so nothing has to be
    refitted and no assumption is made about how the other run produced its numbers. Test
    periods differ slightly between baselines (they drop different leading rows), so the
    intersection is scored and its size printed rather than assumed to be everything.
    """
    if not os.path.exists(npz_path):
        return
    z = np.load(npz_path, allow_pickle=True)
    ka, kb = _key(pd.to_datetime(z['datetime'])), _key(when_test)
    common = ka.intersection(kb)
    if not len(common):
        return
    a, b = ka.get_indexer(common), kb.get_indexer(common)

    # Two layouts are in the wild and both are read rather than one of them being rewritten:
    # train_quantile / train_density store `labels` alongside cfg0..cfgN, train_qra stores
    # each curve under its own (paren-stripped) name. Touching either writer to unify them
    # would mean re-running those baselines to reuse their results, which is the opposite of
    # the point.
    if 'labels' in z:
        found = [(str(lab), f'cfg{i}') for i, lab in enumerate(z['labels'])]
    else:
        found = [(k, k) for k in z.files if k not in ('actual', 'datetime')]

    taken = []
    for lab, key in found:
        if only is not None and lab not in only:
            continue
        res[lab] = {**_scores(y_test[b], z[key][:, a]), 'horizon': horizon, 'runs': 1,
                    'pinball_sd': 0.0, 'shared_hours': len(common)}
        curves[lab] = None                    # scored on a subset; not drawn on shared axes
        taken.append(lab)
    if taken:
        print(f"  folded in {', '.join(taken)} from "
              f"{os.path.basename(os.path.dirname(npz_path))} on {len(common)} shared hours "
              f"({100 * len(common) / len(y_test):.0f}% of this test period)")


# ---------------------------------------------------------------------------

def _run_config(spec, X, y, tr, va, te, qc, seed0, n_runs, label=None, verbose=False):
    """`n_runs` independent trainings of one configuration. Returns (rows, best curves).

    The curves kept for the figures are the MEDIAN run by test pinball, not the best one:
    reporting the best of n and the mean of n in the same table would be two different
    estimators, and the median run is the one the mean actually describes.
    """
    rows, keep = [], []
    for r in range(n_runs):
        q, raw, hist = M.run_spec(spec, X, y, tr, va, te, qc, LEVELS, seed0 + r,
                                  verbose=verbose)
        sc = _scores(y[te], q['test'])
        frac, cs = M.crossing_rate(raw['test'])
        rows.append({'model': label or spec['name'], 'seed': seed0 + r, **sc,
                     'val_pinball': hist['val_pinball'], 'epochs': hist['epochs'],
                     'best_epoch': hist['best_epoch'], 'seconds': hist['seconds'],
                     'n_params': hist['n_params'], 'crossed_frac': frac, 'CS': cs})
        keep.append(q)
        print(f"    seed {seed0 + r}  pinball {sc['pinball']:8.2f}  Wink90 {sc['winkler90']:8.0f}"
              f"  {hist['epochs']:4d} ep ({hist['best_epoch']:4d} best)"
              f"  {hist['seconds']:6.1f}s")
    order = np.argsort([r['pinball'] for r in rows])
    return rows, keep[order[len(order) // 2]]


def main():
    qc = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.QRNN_CONFIG.items()}
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='*', default=None,
                    help='subset of src.qrnn.model.SPECS (default: whatever QRNN_CONFIG enables)')
    ap.add_argument('--n-runs', type=int, default=None)
    ap.add_argument('--lag-min', type=int, default=None,
                    help='shortest load lag: 48 = this repo\'s cutoff, 24 = QRA\'s issue time')
    ap.add_argument('--device', default=None, choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--refresh-rfe', action='store_true', help='rebuild the feature selection')
    ap.add_argument('--size-sweep', action='store_true',
                    help="network-size robustness, their Fig. 3/8 (adds ~7x2 short runs)")
    ap.add_argument('--noise-sweep', action='store_true',
                    help="Gaussian-noise SD sweep, their Fig. 9 (adds ~5 short runs)")
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    for k, v in [('n_runs', a.n_runs), ('load_lag_min', a.lag_min), ('device', a.device)]:
        if v is not None:
            qc[k] = v

    names = a.models or [n for n, on in qc['models'].items() if on]
    unknown = set(names) - set(M.SPEC_BY_NAME)
    if unknown:
        raise SystemExit(f"unknown model(s) {sorted(unknown)}. "
                         f"Valid: {sorted(M.SPEC_BY_NAME)}")
    specs = [M.SPEC_BY_NAME[n] for n in names]

    # A non-default lag floor is a different QUESTION, not a different setting, so it gets its
    # own directory instead of overwriting the comparable run — the same arrangement
    # MATRIX_TAG gives the congestion A/B and `_ctx{L}` gives the foundation sweep. Without
    # it the two arms would silently clobber each other and only the last one would exist.
    out_dir = qc['result_dir'] + ('' if qc['load_lag_min'] == 48 else f"_lag{qc['load_lag_min']}")
    os.makedirs(out_dir, exist_ok=True)
    device = M.resolve_device(qc['device'])

    # -- data -----------------------------------------------------------------------
    df, ept, load, temp, hol, valid = load_frame()
    split = split_at_day_boundary(ept, qc['test_frac'])
    test_start = ept[split].normalize()

    X, y, rows = F.build(ept, load, temp, hol, load_lag_min=qc['load_lag_min'])
    ept_r, valid_r = ept[rows], valid[rows]
    m_tr, m_va, m_te = F.split_masks(ept_r, test_start, qc['val_days'])
    m_tr = m_tr & (valid_r == 1)              # the same 3-sigma mask the point models use

    horizon = 'D-1 -> D+1' if qc['load_lag_min'] >= 48 else 'D-1 -> D'
    print(f"=== iQRNN baseline | zone {cfg.DATASET} | {device} ===")
    print(f"  record     {ept[0]} -> {ept[-1]}   ({len(ept)} h)")
    print(f"  candidates {X.shape[1]}  (load lags {qc['load_lag_min']}-{F.LOAD_LAG_MAX}, "
          f"temp lags {F.TEMP_LAG_MIN}-{F.TEMP_LAG_MAX}, {len(F.CALENDAR_NAMES)} calendar)")
    print(f"  horizon    {horizon}   |   temperature is REALISED (the paper's proxy)")
    print(f"  train      {ept_r[m_tr][0].date()} -> {ept_r[m_tr][-1].date()}  ({m_tr.sum()} h)")
    print(f"  valid      {ept_r[m_va][0].date()} -> {ept_r[m_va][-1].date()}  ({m_va.sum()} h)")
    print(f"  test       {ept_r[m_te][0].date()} -> {ept_r[m_te][-1].date()}  ({m_te.sum()} h)")

    # -- feature selection ----------------------------------------------------------
    print(f"\n--- recursive feature elimination ---")
    idx, keep = F.select(X, y, m_tr, qc, cfg.DATASET, ept_r,
                         cache_path=os.path.join(out_dir, 'rfe_features.json'),
                         refresh=a.refresh_rfe)
    print("  kept: " + "  ".join(f"{k} {v}" for k, v in F.summarise(keep).items()))
    X = np.ascontiguousarray(X[:, idx])       # drop the candidate pool; it is ~320 MB

    # -- train ----------------------------------------------------------------------
    y_te, when_te = y[m_te], ept_r[m_te]
    all_rows, curves, res = [], {}, {}
    for spec in specs:
        print(f"\n--- {spec['name']}  ({spec['hidden']} units, dropout {spec['dropout']}, "
              f"noise {qc['noise_sd'] if spec['kind'] == 'i' else 0.0}, "
              f"{qc['n_runs']} runs) ---")
        rws, best = _run_config(spec, X, y, m_tr, m_va, m_te, qc, qc['random_state'],
                                qc['n_runs'], verbose=a.verbose)
        all_rows += rws
        curves[spec['name']] = best['test']
        # Every score in the table is the MEAN over the independent trainings, so no column
        # is a different estimator from its neighbour. The curves kept for the figures are
        # the median run's; mixing a best-of-n curve with a mean-of-n table would let the
        # picture and the number disagree.
        s = pd.DataFrame(rws)
        res[spec['name']] = {**{k: float(s[k].mean()) for k in SCORE_KEYS},
                             'pinball_sd': float(s.pinball.std(ddof=0)),
                             'horizon': horizon, 'runs': len(rws), 'shared_hours': len(y_te)}

    pd.DataFrame(all_rows).to_csv(os.path.join(out_dir, 'runs.csv'), index=False)

    # -- the other baselines, if they have been run for this zone --------------------
    print()
    _fold_in(os.path.join(cfg.QRA_CONFIG['result_dir'], 'test_quantiles.npz'),
             y_te, when_te, 'D-1 -> D', res, curves, only={'QRA', 'Vanilla(scenario)'})
    _fold_in(os.path.join(cfg.DENSITY_CONFIG['result_dir'], 'test_quantiles.npz'),
             y_te, when_te, 'D-1 -> D+1', res, curves, only={'CRPS-guided'})

    # -- report ---------------------------------------------------------------------
    tbl = pd.DataFrame([{'model': k, **v} for k, v in res.items()])
    tbl.to_csv(os.path.join(out_dir, 'test_scores.csv'), index=False)

    print(f"\n{'=' * 96}\nTEST | zone {cfg.DATASET} | {len(y_te)} h "
          f"({when_te[0].date()} -> {when_te[-1].date()})\n{'=' * 96}")
    print(f"{'model':22s}{'pinball':>10s}{'(sd)':>8s}{'Wink50':>9s}{'Wink90':>9s}{'CRPS':>9s}"
          f"{'MAPE':>8s}{'runs':>6s}  {'horizon':<12s}")
    for r in tbl.sort_values('pinball').itertuples():
        note = '' if r.shared_hours == len(y_te) else f'  [{r.shared_hours} h]'
        print(f"{r.model:22s}{r.pinball:10.2f}{r.pinball_sd:8.2f}{r.winkler50:9.0f}"
              f"{r.winkler90:9.0f}{r.CRPS:9.1f}{r.median_MAPE:7.2f}%{r.runs:6d}  "
              f"{r.horizon:<12s}{note}")

    drawn = {k: v for k, v in curves.items() if v is not None}
    cov = pd.DataFrame([{'model': k, **c} for k, q in drawn.items()
                        for c in coverage_report(y_te, q)])
    cov.to_csv(os.path.join(out_dir, 'coverage.csv'), index=False)
    print(f"\n{'model':22s}" + "".join(f"{c:>9s}" for c in ['50%', '80%', '90%', '95%', '98%']))
    for k, q in drawn.items():
        print(f"{k:22s}" + "".join(f"{c['empirical_pct']:8.1f}%" for c in coverage_report(y_te, q)))

    # Does the joint output layer suppress crossing on its own, as the paper's Fig. 10
    # argues? Measured on the RAW output, before the sort that every score above uses.
    cr = pd.DataFrame(all_rows).groupby('model')[['crossed_frac', 'CS']].agg(['mean', 'std'])
    cr.columns = ['_'.join(c) for c in cr.columns]
    cr.reset_index().to_csv(os.path.join(out_dir, 'crossing.csv'), index=False)
    print(f"\n{'model':22s}{'hours with a crossing':>24s}{'CS (unsorted)':>16s}")
    for r in cr.reset_index().itertuples():
        print(f"{r.model:22s}{100 * r.crossed_frac_mean:23.1f}%{r.CS_mean:16.2f}")

    # -- shared diagnostics: the same three files the other baselines write ----------
    pd.DataFrame({'quantile': QUANTILES,
                  **{k: pinball(y_te, q).mean(axis=1) for k, q in drawn.items()}}
                 ).to_csv(os.path.join(out_dir, 'pinball_by_quantile.csv'), index=False)

    mon = pd.DatetimeIndex(when_te).to_period('M').astype(str).values
    mrows = []
    for mth in pd.unique(mon):
        s_ = mon == mth
        r = {'month': mth, 'hours': int(s_.sum()), 'mean_load': float(y_te[s_].mean())}
        for k, q in drawn.items():
            lo = q[np.where(QUANTILES == 5)[0][0]][s_]
            hi = q[np.where(QUANTILES == 95)[0][0]][s_]
            r[f'{k}_pinball'] = float(pinball(y_te[s_], q[:, s_]).mean())
            r[f'{k}_cover90'] = float(100 * ((y_te[s_] >= lo) & (y_te[s_] <= hi)).mean())
            r[f'{k}_width90'] = float(np.mean(hi - lo))
        mrows.append(r)
    pd.DataFrame(mrows).to_csv(os.path.join(out_dir, 'monthly.csv'), index=False)

    np.savez_compressed(os.path.join(out_dir, 'test_quantiles.npz'),
                        actual=y_te, datetime=when_te.values,
                        labels=np.array(list(drawn), dtype=object),
                        **{f'cfg{i}': q for i, q in enumerate(drawn.values())})

    # -- optional sweeps: the paper's own robustness figures -------------------------
    if a.size_sweep:
        _size_sweep(X, y, m_tr, m_va, m_te, qc, out_dir)
    if a.noise_sweep:
        _noise_sweep(X, y, m_tr, m_va, m_te, qc, out_dir)

    print(f"\n-> {out_dir}")


def _size_sweep(X, y, tr, va, te, qc, out_dir, sizes=(200, 300, 400, 500, 600, 700, 800)):
    """Their Fig. 3 / Fig. 8: score against network size, for iQRNN and the bQRNN control.

    This is the paper's practicality claim — "a reliable and practical NN for forecasting
    should be robust with a large range of NN sizes" — and it is the one result that cannot
    be read off a single well-tuned run. Both curves are needed: the claim is not that iQRNN
    is flat, it is that iQRNN is flat WHERE bQRNN is not.
    """
    runs = max(2, qc['n_runs'] // 2)
    print(f"\n--- network-size sweep | {len(sizes)} sizes x 2 models x {runs} runs ---")
    out = []
    for kind, base in [('i', 'iQRNN-ZD'), ('b', 'bQRNN')]:
        for h in sizes:
            spec = {'name': f'{base}-{h}', 'kind': kind, 'hidden': h, 'dropout': 0.0}
            rws, _ = _run_config(spec, X, y, tr, va, te, qc, qc['random_state'], runs,
                                 label=f'{base}-{h}')
            for r in rws:
                out.append({'family': base, 'hidden': h, **r})
    pd.DataFrame(out).to_csv(os.path.join(out_dir, 'size_sweep.csv'), index=False)
    print(f"  -> {os.path.join(out_dir, 'size_sweep.csv')}")


def _noise_sweep(X, y, tr, va, te, qc, out_dir, sds=(0.0, 0.1, 0.2, 0.3, 0.4)):
    """Their Fig. 9: quantile score against the Gaussian-noise SD.

    Run on the zero-dropout 500 network on purpose, which is what the paper does — with
    dropout on, two regularisers move at once and neither can be attributed.
    """
    runs = max(2, qc['n_runs'] // 2)
    print(f"\n--- Gaussian-noise sweep | {len(sds)} levels x {runs} runs ---")
    spec = M.SPEC_BY_NAME['iQRNN-500-ZD']
    out = []
    for sd in sds:
        rws, _ = _run_config(spec, X, y, tr, va, te, {**qc, 'noise_sd': sd},
                             qc['random_state'], runs, label=f'SD={sd}')
        for r in rws:
            out.append({'noise_sd': sd, **r})
    pd.DataFrame(out).to_csv(os.path.join(out_dir, 'noise_sweep.csv'), index=False)
    print(f"  -> {os.path.join(out_dir, 'noise_sweep.csv')}")


if __name__ == '__main__':
    main()
