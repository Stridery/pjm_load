"""
Probabilistic baseline 2 — Liu, Nowotarski, Hong & Weron (2017), IEEE TSG 8(2):730-737.
Quantile Regression Averaging on sister forecasts.

    PJM_DATASET=dom python train_qra.py
    PJM_DATASET=bge python train_qra.py --sister-refit-every 7      # faster, less faithful

Parallel to train_quantile.py in every structural respect: reads exactly one input
(data/{zone}/cleaned/cleaned_pjm_load_weather.csv), writes to results/{zone}/qra/, and
touches neither the matrix cache, the 16 saved point models, nor the serving path.

Reports the paper's method on this data:

    QRA        quantile regression of actual load on the m best sisters.
    Vanilla    the scenario baseline — produced by train_quantile.py, folded in if present,
               so the two papers land in one table.

The paper's other three benchmarks (Direct, Ind, BI) are not implemented: only the method
itself is shipped.

Timeline (dom; bge shifts by ~2 weeks):

    2021-01-01 ....... 2024-02 ......... 2024-08-30 ....... 2025-08-30 ....... 2026-07-20
        data starts    sister forecasts   validation year    test year            end
                       begin (L runway)   selects (J,D),m,L  reported here

Temperature: the realised series is used where a forecast would be, which the paper states
it does too ("the actual future temperatures were used as a proxy for temperature
forecasts... a common practice in load forecasting"). Every number below is therefore
ex-post in the weather, and NOT comparable to the ex-ante scenario baseline without saying so
— which is precisely the comparison the paper's own Vanilla row makes.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import src.config as cfg
from src.quantile import qra as Q
from src.quantile._pool import n_workers, run
from src.quantile import sisters as S
from src.quantile.qeval import (QUANTILES, coverage_report, pinball, point_metrics,
                                quantile_score, winkler)
from train_quantile import load_frame, split_at_day_boundary


def _scores(y, q):
    return {'pinball': float(pinball(y, q).mean()),
            'winkler50': winkler(y, q, 50), 'winkler90': winkler(y, q, 90)}


def _day_slices(day_list, ept, rows):
    """(lo, hi) per day in the OUTPUT row coordinate system (rows are day-concatenated)."""
    lens = pd.Series(ept[rows].normalize()).value_counts().sort_index()
    lens = [int(lens[pd.Timestamp(d).normalize()]) for d in day_list]
    edges = np.r_[0, np.cumsum(lens)]
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(day_list))]


def main():
    qc = dict(cfg.QRA_CONFIG)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sister-refit-every', type=int, default=None)
    ap.add_argument('--qra-refit-every-val', type=int, default=None)
    ap.add_argument('--qra-refit-every-test', type=int, default=None)
    ap.add_argument('--max-lag', type=int, default=None)
    ap.add_argument('--max-dma', type=int, default=None)
    ap.add_argument('--workers', type=int, default=None,
                    help='process-pool size (default: the SLURM allocation, capped at 8)')
    ap.add_argument('--refresh-sisters', action='store_true',
                    help='ignore the cached stage-1 forecasts and re-run them')
    a = ap.parse_args()
    for k, v in [('sister_refit_every', a.sister_refit_every),
                 ('qra_refit_every_val', a.qra_refit_every_val),
                 ('qra_refit_every_test', a.qra_refit_every_test),
                 ('max_lag', a.max_lag), ('max_dma', a.max_dma),
                 ('n_workers', a.workers)]:
        if v is not None:
            qc[k] = v

    df, ept, load, temp, hol, valid = load_frame()
    split = split_at_day_boundary(ept, qc['test_frac'])
    test_start = ept[split].normalize()
    val_start = test_start - pd.Timedelta(days=qc['val_days'])
    L_max = max(qc['L_grid'])
    sis_start = val_start - pd.Timedelta(days=L_max)      # runway for the widest window

    print(f"=== QRA baseline | zone {cfg.DATASET} ===")
    print(f"  base model  Vanilla eq.(1) + recency (no trend/macro term, as in the paper)")
    print(f"  record      {ept[0]} -> {ept[-1]}")
    print(f"  sisters from{sis_start.date()}   validation {val_start.date()}   "
          f"test {test_start.date()} -> {ept[-1].date()}")
    need = sis_start - pd.DateOffset(years=max(S.TRAIN_YEARS))
    if need < ept[0]:
        raise SystemExit(f"Need history from {need.date()} for a {max(S.TRAIN_YEARS)}y window; "
                         f"record starts {ept[0].date()}. Shorten val_days or L_grid.")
    print(f"  earliest estimation window starts {need.date()} "
          f"({(need - ept[0]).days} d of margin)")

    # -- Sisters ------------------------------------------------------------------
    print(f"\n--- selecting recency orders on the validation year ---")
    sel = S.select_recency(ept, load, temp, valid, val_start, test_start,
                           n_keep=qc['sisters_per_window'],
                           max_lag=qc['max_lag'], max_dma=qc['max_dma'])
    specs = S.sister_specs(sel)

    print(f"\n--- rolling sister forecasts (refit every {qc['sister_refit_every']} d) ---")
    days = [d.date() for d in pd.date_range(sis_start, ept[-1].normalize(), freq='D')]

    # Stage 1 is by far the longest step and depends on nothing downstream, so it is cached.
    # The signature covers everything that changes its output; a mismatch re-runs rather than
    # silently serving forecasts built under a different specification.
    out_dir = qc['result_dir']
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, 'sister_forecasts.npz')
    sig = json.dumps({'zone': cfg.DATASET, 'specs': [[t, list(r)] for t, r in specs],
                      'first': str(days[0]), 'last': str(days[-1]),
                      'refit_every': qc['sister_refit_every']}, sort_keys=True)

    fc = None
    if os.path.exists(cache) and not a.refresh_sisters:
        z = np.load(cache, allow_pickle=True)
        if str(z['signature']) == sig:
            fc, rows = z['fc'], z['rows']
            days = [pd.Timestamp(d).date() for d in z['days']]
            print(f"  loaded {fc.shape[0]} cached sister forecasts from {cache}")
            print(f"  (delete it or pass --refresh-sisters to rebuild)")
        else:
            print("  cached sister forecasts do not match this configuration — rebuilding")
    if fc is None:
        fc, rows, days = S.rolling_forecasts(ept, load, temp, valid, specs, days,
                                             refit_every=qc['sister_refit_every'],
                                             workers=qc.get('n_workers'))
        np.savez_compressed(cache, fc=fc, rows=rows,
                            days=np.array([np.datetime64(d) for d in days]),
                            signature=np.array(sig))
        print(f"  cached -> {cache}")
    y = load[rows]
    slices = _day_slices(days, ept, rows)
    val_from = days.index(val_start.date())
    test_from = days.index(test_start.date())
    print(f"  {fc.shape[0]} sisters x {fc.shape[1]} h;  sister MAE over the test period: "
          + "  ".join(f"{np.mean(np.abs(fc[i, slices[test_from][0]:] - y[slices[test_from][0]:])):.0f}"
                      for i in range(fc.shape[0])))

    pd.DataFrame([{'sister': i + 1, 'train_years': ty, 'lag_J': r[0], 'dma_D': r[1]}
                  for i, (ty, r) in enumerate(specs)]).to_csv(
        os.path.join(out_dir, 'sisters.csv'), index=False)

    # -- Validation sweep: choose (m, L) for QRA and L for the benchmarks -----------
    print(f"\n--- validation sweep (refit every {qc['qra_refit_every_val']} d) ---")
    yv = y[slices[val_from][0]:slices[test_from][0]]
    vsl = slice(slices[val_from][0], slices[test_from][0])
    w = n_workers(qc.get('n_workers'))
    tasks = [(fc, y, slices, val_from, test_from, vsl, m, L, qc['qra_refit_every_val'])
             for L in qc['L_grid'] for m in qc['m_grid']]
    print(f"  {len(tasks)} (m, L) configurations on {min(w, len(tasks))} process(es)")
    sweep = run(Q.score_config, tasks, w, label='configs')
    for L in qc['L_grid']:
        print(f"  L={L:3d}: best QRA pinball "
              f"{min(r['pinball'] for r in sweep if r['L'] == L):.2f}")
    sw = pd.DataFrame(sweep)
    sw.to_csv(os.path.join(out_dir, 'validation_sweep.csv'), index=False)

    def pick(prefix):
        sub = sw[sw.model.str.startswith(prefix)]
        return sub.loc[sub.pinball.idxmin()]

    bq = pick('QRA')
    print(f"  selected  QRA m={int(bq.m)}, L={int(bq.L)}")

    # -- Test period ---------------------------------------------------------------
    print(f"\n--- test period (refit every {qc['qra_refit_every_test']} d) ---")
    tsl = slice(slices[test_from][0], None)
    yt = y[tsl]
    res, curves = {}, {}
    curves['QRA'] = Q.qra(fc, y, slices, test_from, m=int(bq.m), L=int(bq.L),
                          refit_every=qc['qra_refit_every_test'])[:, tsl]
    for k, q in curves.items():
        res[k] = _scores(yt, q)

    # The scenario baseline, if train_quantile.py has been run for this zone. Its test period
    # starts a little earlier, so it is scored on the intersection — stated, not silently.
    sc_dir = os.path.join(cfg.QUANTILE_CONFIG['result_dir'], 'vanilla')
    if os.path.exists(os.path.join(sc_dir, 'scenario_preds.npy')):
        from src.quantile.qeval import empirical_percentiles
        P = np.load(os.path.join(sc_dir, 'scenario_preds.npy'))
        sh = np.load(os.path.join(sc_dir, 'scenario_shifts.npy'))
        when = pd.to_datetime(np.load(os.path.join(sc_dir, 'test_datetime.npy')))
        g = pd.read_csv(os.path.join(sc_dir, 'grid_shifted.csv'), index_col=0)
        g.columns = [int(c) for c in g.columns]
        n_, k_ = g.stack().idxmin()
        mid = int(np.where(sh == 0)[0][0])
        sub = P[-int(k_):, mid - int(n_):mid + int(n_) + 1, :].reshape(-1, len(when))
        # EPT is naive local time, so the fall-back day carries 01:00 TWICE and a plain
        # index is not unique. Disambiguate by occurrence before aligning the two runs.
        def _key(idx):
            t = pd.Series(pd.DatetimeIndex(idx))
            return pd.MultiIndex.from_arrays([t, t.groupby(t).cumcount()])
        ka, kb = _key(when), _key(ept[rows][tsl])
        common = ka.intersection(kb)
        if len(common):
            a, b = ka.get_indexer(common), kb.get_indexer(common)
            res['Vanilla(scenario)'] = _scores(yt[b], empirical_percentiles(sub[:, a]))
            print(f"  scenario baseline folded in on {len(common)} shared hours "
                  f"({100 * len(common) / len(yt):.0f}% of the test period)")

    order = ['QRA', 'Vanilla(scenario)']
    tbl = pd.DataFrame([{'model': k, **res[k]} for k in order if k in res])
    tbl.to_csv(os.path.join(out_dir, 'test_scores.csv'), index=False)

    print(f"\n{'=' * 70}\nTEST | zone {cfg.DATASET} | {len(yt)} h "
          f"({ept[rows][tsl][0].date()} -> {ept[rows][tsl][-1].date()})\n{'=' * 70}")
    print(f"{'model':20s}{'pinball':>10s}{'Winkler50':>12s}{'Winkler90':>12s}")
    for r in tbl.itertuples():
        print(f"{r.model:20s}{r.pinball:10.2f}{r.winkler50:12.1f}{r.winkler90:12.1f}")

    cov = pd.DataFrame([{'model': k, **c} for k in curves for c in coverage_report(yt, curves[k])])
    cov.to_csv(os.path.join(out_dir, 'coverage.csv'), index=False)
    print(f"\n{'model':20s}" + "".join(f"{c:>10s}" for c in ['50%', '80%', '90%', '95%', '98%']))
    for k in curves:
        e = [c['empirical_pct'] for c in coverage_report(yt, curves[k])]
        print(f"{k:20s}" + "".join(f"{v:9.1f}%" for v in e))

    # -- Detailed diagnostics ------------------------------------------------------
    # Where does the score come from? A single pinball number hides which quantiles, which
    # months, and how wide the intervals had to be to get there.
    pd.DataFrame({'quantile': QUANTILES,
                  **{k: pinball(yt, q).mean(axis=1) for k, q in curves.items()}}
                 ).to_csv(os.path.join(out_dir, 'pinball_by_quantile.csv'), index=False)

    when_t = ept[rows][tsl]
    mon = pd.Series(when_t).dt.to_period('M').astype(str).values
    rows_m = []
    for mth in pd.unique(mon):
        sel = mon == mth
        r = {'month': mth, 'hours': int(sel.sum()), 'mean_load': float(yt[sel].mean())}
        for k, q in curves.items():
            lo = q[np.where(QUANTILES == 5)[0][0]][sel]
            hi = q[np.where(QUANTILES == 95)[0][0]][sel]
            r[f'{k}_pinball'] = float(pinball(yt[sel], q[:, sel]).mean())
            r[f'{k}_cover90'] = float(100 * ((yt[sel] >= lo) & (yt[sel] <= hi)).mean())
            r[f'{k}_width90'] = float(np.mean(hi - lo))
        rows_m.append(r)
    pd.DataFrame(rows_m).to_csv(os.path.join(out_dir, 'monthly.csv'), index=False)

    # Sister diversity: the premise of the whole method is that disagreement between the
    # sisters stands in for uncertainty. Measuring how far that premise holds here is the
    # single most useful diagnostic for reading the Direct benchmark's score.
    fct = fc[:, tsl]
    err = yt[None, :] - fct
    C = np.corrcoef(err)
    iu = np.triu_indices_from(C, 1)
    pd.DataFrame([{'n_sisters': fct.shape[0],
                   'sister_dispersion_std_MW': float(fct.std(axis=0).mean()),
                   'actual_error_std_MW': float(err.std()),
                   'dispersion_shortfall_x': float(err.std() / fct.std(axis=0).mean()),
                   'sister_error_corr_median': float(np.median(C[iu])),
                   'sister_error_corr_min': float(C[iu].min()),
                   'sister_MAE_best': float(np.abs(err).mean(axis=1).min()),
                   'sister_MAE_worst': float(np.abs(err).mean(axis=1).max())}]
                 ).to_csv(os.path.join(out_dir, 'sister_diagnostics.csv'), index=False)
    print(f"\n  sister dispersion {fct.std(axis=0).mean():.0f} MW vs actual error std "
          f"{err.std():.0f} MW  ({err.std() / fct.std(axis=0).mean():.1f}x shortfall);  "
          f"error corr median {np.median(C[iu]):.3f}")

    np.savez_compressed(os.path.join(out_dir, 'test_quantiles.npz'),
                        actual=yt, datetime=ept[rows][tsl].values,
                        **{k.replace('(', '_').replace(')', ''): v for k, v in curves.items()})
    print(f"\n-> {out_dir}")


if __name__ == '__main__':
    main()
