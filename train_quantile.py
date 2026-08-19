"""
Probabilistic load-forecasting baselines — Xie & Hong (2018), IEEE TSG 9(3):1680-1687.

    PJM_DATASET=dom python train_quantile.py
    PJM_DATASET=bge python train_quantile.py --models vanilla
    PJM_DATASET=dom python train_quantile.py --n-max 10 --no-bootstrap

Fits the paper's MLR point models on the cleaned hourly record, feeds them temperature
scenarios generated from past weather years, and scores the resulting percentiles with the
pinball loss. Reads exactly one input — data/{zone}/cleaned/cleaned_pjm_load_weather.csv —
and writes to results/{zone}/quantile/. Nothing here touches the matrix cache, the saved
point-model weights, or the daily serving path.

Runtime is a few minutes on CPU; no GPU, no sbatch.


The one structural decision worth knowing
-----------------------------------------
Point-forecasting a scenario is the expensive step, and a (year, shift) scenario does not
depend on k or n — those only decide WHICH scenarios go into a percentile. So every
(year, shift) pair is forecast ONCE into a cube, and each of the ~84 grid cells is a slice
of that cube. Re-forecasting per cell would be ~40x the work for identical numbers.
"""

import argparse
import os

import numpy as np
import pandas as pd

import src.config as cfg
from src.quantile import scenarios as sc
from src.quantile.mlr import MLR
from src.quantile.qeval import (QUANTILES, coverage_report, empirical_percentiles,
                                pinball, point_metrics, quantile_score, winkler)

# Rows of scenario temperature prepended to the evaluation window so that hong2014's
# recency terms (t-1..t-3, and the 24 h mean) at the first test hour are lagged off real
# scenario hours rather than off a back-filled edge.
WARMUP_HOURS = 48


def load_frame():
    """The cleaned hourly record: (ept, load, temp, is_holiday, is_valid)."""
    df = pd.read_csv(cfg.CLEANED_PATH, index_col=0, parse_dates=True).sort_index()
    ept = pd.DatetimeIndex(pd.to_datetime(df['Datetime_EPT']))
    return (df, ept, df['Load'].values.astype(float), df['Temp_F'].values.astype(float),
            df['is_holiday'].values.astype(float), df['is_valid'].values.astype(int))


def split_at_day_boundary(ept, test_frac):
    """Tail split, snapped back to an EPT midnight so no partial day sits at either end."""
    split = len(ept) - int(len(ept) * test_frac)
    while split > 0 and ept[split].hour != 0:
        split -= 1
    return split


def usable_scenario_years(ept, test_start_day, requested=None):
    """Complete calendar years of weather that end strictly before the test period.

    The paper keeps its 1976-2005 scenario history clear of the 2006-2011 test years;
    drawing scenarios from the test period itself would be leakage. A partial year is
    excluded too — it would give different target dates different numbers of scenarios and
    make "k years of history" mean different things on different days.
    """
    counts = pd.Series(ept.year).value_counts()
    years = [int(y) for y in sorted(counts.index)
             if counts[y] >= 8700 and pd.Timestamp(year=int(y) + 1, month=1, day=1) <= test_start_day]
    if requested is not None:
        missing = set(requested) - set(years)
        if missing:
            raise ValueError(f"scenario_years {sorted(missing)} are not complete pre-test years "
                             f"(usable: {years})")
        years = sorted(requested)
    if not years:
        raise SystemExit("No complete weather year precedes the test period — cannot build scenarios.")
    return years


def estimation_window_start(ept, split, train_years):
    """First row of the estimation window: `train_years` back from the split, or 0 for None.

    The paper fixes its rolling estimation window at three years. That is not a detail here:
    growth enters these models through ONE linear Trend coefficient (there is no GSP index
    for a PJM zone), so a window long enough to span a change in growth rate fits the average
    slope and under-forecasts the recent end. DOM, growing on data-centre demand, shows it
    plainly — see QUANTILE_CONFIG['train_years'].
    """
    if not train_years:
        return 0
    cutoff = ept[split] - pd.DateOffset(years=int(train_years))
    return int(np.searchsorted(ept.values, np.datetime64(cutoff)))


def run_model(name, ept, load, temp, hol, valid, split, train_start, years, qcfg, out_root,
              pool_ept=None, pool_temp=None, do_bootstrap=True):
    print(f"\n{'=' * 68}\n{name.upper()}\n{'=' * 68}")
    pool_ept = ept if pool_ept is None else pool_ept
    pool_temp = temp if pool_temp is None else pool_temp
    train_mask = np.zeros(len(ept), dtype=bool)
    train_mask[train_start:split] = True
    train_mask &= (valid == 1)          # drop the 3-sigma outliers, as the point models do

    model = MLR.fit(name, ept, temp, load, train_mask, is_holiday=hol)
    print(f"  design: {model.n_cal} calendar + {len(model.beta) - model.n_cal} temperature "
          f"= {len(model.beta)} columns   |   fit on {int(train_mask.sum())} h")

    # Evaluation window = test rows, preceded by a warmup the recency lags can reach into.
    ev0 = max(split - WARMUP_HOURS, 0)
    ept_ev, hol_ev = ept[ev0:], hol[ev0:]
    keep = slice(split - ev0, None)                      # drops the warmup after predicting
    y_true = load[split:]
    cal_ev = model.calendar_contrib(ept_ev, hol_ev)      # scenario-independent — hoisted

    # Deterministic reference: the same model driven by the temperature that actually
    # occurred. This is the floor the scenario forecasts are working against — without it
    # there is no way to tell model error from temperature uncertainty in the quantile score.
    perfect = model.predict_with_temp(ept_ev, temp[ev0:], cal_contrib=cal_ev)[keep]
    pm = point_metrics(y_true, perfect)
    # Scored as a degenerate predictive distribution (all 99 quantiles collapsed onto the
    # point). That is the FLOOR for this model: everything above it in the scenario results
    # is the price of not knowing tomorrow's temperature. Without it the quantile scores
    # below are unanchored numbers.
    qs_floor = quantile_score(y_true, perfect.reshape(1, -1))
    pm['QS_degenerate'] = qs_floor
    print(f"  perfect-weather point forecast: MAPE {pm['MAPE']:.2f}%  MAE {pm['MAE']:.1f}  "
          f"RMSE {pm['RMSE']:.1f}")
    print(f"  perfect-weather degenerate QS (temperature known exactly): {qs_floor:.2f}")

    # --- Scenario cube: forecast every (year, shift) pair exactly once -------------
    n_max = qcfg['n_max']
    pool = sc.build_pool(pool_ept, pool_temp, years)
    temp_cube, shifts = sc.fixed_and_shifted(ept_ev, pool, years, n_max)
    mid = int(np.where(shifts == 0)[0][0])

    pred = np.empty((len(years), len(shifts), len(y_true)), dtype='float32')
    for a in range(len(years)):
        for b in range(len(shifts)):
            pred[a, b] = model.predict_with_temp(
                ept_ev, temp_cube[a, b], cal_contrib=cal_ev)[keep]
    del temp_cube
    print(f"  scenario cube: {len(years)} year(s) x {len(shifts)} shift(s) "
          f"= {pred.shape[0] * pred.shape[1]} point forecasts over {len(y_true)} h")

    # --- Fixed-date over EVERY k (their Fig. 2 / Fig. 3) --------------------------
    # Cheap, and it is the curve the paper's headline finding lives on: the quantile score
    # falls with the length of history and flattens once there is enough of it.
    fixed = pd.Series({k: quantile_score(y_true, pred[-k:, mid, :])
                       for k in range(1, len(years) + 1)}, name='fixed_date')
    fixed.index.name = 'k'

    # --- k x n grid (their Table I / Table III), on the paper's own k columns ------
    ks = [k for k in qcfg['k_grid_shifted'] if k <= len(years)] or [len(years)]
    grid = pd.DataFrame(index=pd.Index(range(0, n_max + 1), name='n'),
                        columns=pd.Index(ks, name='k'), dtype=float)
    for k in ks:
        for n in range(0, n_max + 1):
            sub = pred[-k:, mid - n:mid + n + 1, :]      # most recent k years, shifts +/-n
            grid.loc[n, k] = quantile_score(y_true, sub.reshape(-1, len(y_true)))

    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    grid.to_csv(os.path.join(out_dir, 'grid_shifted.csv'))
    fixed.to_csv(os.path.join(out_dir, 'fixed_date_by_k.csv'))
    model.save(os.path.join(out_dir, f'{name}.pkl'))
    # Which years each k actually draws on. With a 4-year pool a single atypical year moves
    # a low-k column a lot, so a k column is not interpretable without this.
    pd.DataFrame({'k': list(fixed.index),
                  'years': [f'{years[-k]}-{years[-1]}' for k in fixed.index]}
                 ).to_csv(os.path.join(out_dir, 'scenario_years.csv'), index=False)

    best_n = grid.idxmin()
    _fk = list(fixed.index)
    print(f"\n  fixed-date by k:     " + "  ".join(
        f"k={k}: {fixed[k]:7.2f}" for k in _fk if k in (1, 2, 3, 4, 5, 10, 15, 20, 30) or k == _fk[-1]))
    if len(fixed) > 1:
        halve = [k for k in _fk if fixed[k] <= fixed[1] - 0.95 * (fixed[1] - fixed.min())]
        print(f"    95% of the total fixed-date improvement is reached by k={min(halve)}"
              f"  ({fixed[1]:.1f} -> {fixed[min(halve)]:.1f}, floor {fixed.min():.1f} at k={fixed.idxmin()})")
    print(f"  best shifted-date:   " + "  ".join(
        f"k={k}: {grid[k].min():7.2f} (n={best_n[k]})" for k in grid.columns))
    for k in grid.columns:
        rot = int(np.ceil(30 / k - 1))                    # the paper's eq. (3) rule of thumb
        if rot <= n_max:
            print(f"    rule of thumb k={k} -> n={rot}: {grid.loc[rot, k]:7.2f}"
                  f"   (best {grid[k].min():7.2f} at n={best_n[k]})")
        else:
            print(f"    rule of thumb k={k} -> n={rot}: beyond n_max={n_max}, not evaluated")

    # --- Block bootstrap ----------------------------------------------------------
    boot = None
    if do_bootstrap:
        boot = pd.Series(index=grid.columns, dtype=float, name='bootstrap')
        for k in grid.columns:
            temps = sc.bootstrap(ept_ev, pool, years[-k:],
                                 block_days=qcfg['bootstrap_block_days'],
                                 n_scenarios=qcfg['bootstrap_scenarios'],
                                 random_state=qcfg['random_state'])
            bp = np.array([model.predict_with_temp(ept_ev, t, cal_contrib=cal_ev)[keep]
                           for t in temps])
            boot[k] = quantile_score(y_true, bp)
        # fixed spans every k; bootstrap only the grid's. Align on the grid, or the
        # union index silently pads bootstrap with NaN and the paired plot breaks.
        cmp_df = pd.DataFrame({'fixed_date': fixed.reindex(boot.index), 'bootstrap': boot})
        cmp_df.to_csv(os.path.join(out_dir, 'bootstrap_vs_fixed.csv'))
        print(f"\n  bootstrap (l={qcfg['bootstrap_scenarios']}, m={qcfg['bootstrap_block_days']}d): "
              + "  ".join(f"k={k}: {boot[k]:7.2f}" for k in boot.index))
        print("  vs fixed-date:       " + "  ".join(
            f"k={k}: {boot[k] - fixed[k]:+7.2f}" for k in boot.index))

    # --- Full evaluation of the two configurations worth reporting ----------------
    # The grid has hundreds of cells; only two are results. `shifted-date` is the method at
    # its selected optimum, `fixed-date` is the simplest defensible setting (no shifting, all
    # available history) and the one a practitioner would reach for first. Both are scored on
    # exactly the footing the QRA baseline uses, so the two papers land in one table.
    nb, kb = grid.stack().idxmin()
    configs = {
        f'shifted-date (k={int(kb)}, n={int(nb)})':
            pred[-int(kb):, mid - int(nb):mid + int(nb) + 1, :].reshape(-1, len(y_true)),
        f'fixed-date (k={len(years)})': pred[-len(years):, mid, :],
    }
    curves = {name: empirical_percentiles(ens) for name, ens in configs.items()}

    rows = []
    for name, q in curves.items():
        med = q[np.where(QUANTILES == 50)[0][0]]      # the ensemble median IS this method's
        pmm = point_metrics(y_true, med)              # own point forecast — not the diagnostic
        rows.append({'model': name, 'pinball': float(pinball(y_true, q).mean()),
                     'winkler50': winkler(y_true, q, 50), 'winkler90': winkler(y_true, q, 90),
                     'median_MAPE': pmm['MAPE'], 'median_MAE': pmm['MAE'],
                     'median_RMSE': pmm['RMSE'],
                     'median_ME': float(np.mean(med - y_true)),
                     'n_scenarios': int(configs[name].shape[0])})
    rows.append({'model': 'perfect-weather point (diagnostic, not a forecast)',
                 'pinball': qs_floor, 'winkler50': winkler(y_true, empirical_percentiles(
                     perfect.reshape(1, -1)), 50),
                 'winkler90': winkler(y_true, empirical_percentiles(perfect.reshape(1, -1)), 90),
                 'median_MAPE': pm['MAPE'], 'median_MAE': pm['MAE'], 'median_RMSE': pm['RMSE'],
                 'median_ME': float(np.mean(perfect - y_true)), 'n_scenarios': 1})
    scores = pd.DataFrame(rows)
    scores.to_csv(os.path.join(out_dir, 'test_scores.csv'), index=False)

    print(f"\n  {'configuration':38s}{'pinball':>9s}{'Wink50':>9s}{'Wink90':>9s}"
          f"{'MAPE':>8s}{'MAE':>9s}{'ME':>9s}")
    for r in scores.itertuples():
        print(f"  {r.model:38s}{r.pinball:9.1f}{r.winkler50:9.0f}{r.winkler90:9.0f}"
              f"{r.median_MAPE:7.2f}%{r.median_MAE:9.0f}{r.median_ME:+9.0f}")

    pd.DataFrame([{'model': n, **c} for n, q in curves.items()
                  for c in coverage_report(y_true, q)]).to_csv(
        os.path.join(out_dir, 'coverage.csv'), index=False)
    print(f"\n  {'configuration':38s}" + "".join(f"{c:>9s}" for c in
                                                 ['50%', '80%', '90%', '95%', '98%']))
    for n, q in curves.items():
        print(f"  {n:38s}" + "".join(f"{c['empirical_pct']:8.1f}%"
                                     for c in coverage_report(y_true, q)))

    pd.DataFrame({'quantile': QUANTILES,
                  **{n: pinball(y_true, q).mean(axis=1) for n, q in curves.items()}}
                 ).to_csv(os.path.join(out_dir, 'pinball_by_quantile.csv'), index=False)

    when_t = pd.DatetimeIndex(ept[split:])
    mon = when_t.to_period('M').astype(str).values
    mrows = []
    for mth in pd.unique(mon):
        sel = mon == mth
        r = {'month': mth, 'hours': int(sel.sum()), 'mean_load': float(y_true[sel].mean())}
        for n, q in curves.items():
            lo = q[np.where(QUANTILES == 5)[0][0]][sel]
            hi = q[np.where(QUANTILES == 95)[0][0]][sel]
            r[f'{n}_pinball'] = float(pinball(y_true[sel], q[:, sel]).mean())
            r[f'{n}_cover90'] = float(100 * ((y_true[sel] >= lo) & (y_true[sel] <= hi)).mean())
            r[f'{n}_width90'] = float(np.mean(hi - lo))
        mrows.append(r)
    pd.DataFrame(mrows).to_csv(os.path.join(out_dir, 'monthly.csv'), index=False)

    np.savez_compressed(os.path.join(out_dir, 'test_quantiles.npz'),
                        actual=y_true, datetime=when_t.values,
                        **{f'cfg{i}': q for i, q in enumerate(curves.values())},
                        labels=np.array(list(curves), dtype=object))

    # The raw material for the plots: actual load, the whole scenario cube, and the EPT
    # clock the two are indexed on (needed to slice a peak week out of them).
    np.save(os.path.join(out_dir, 'test_actual.npy'), y_true)
    np.save(os.path.join(out_dir, 'test_datetime.npy'), ept[split:].values)
    np.save(os.path.join(out_dir, 'scenario_preds.npy'), pred.astype('float32'))
    np.save(os.path.join(out_dir, 'scenario_shifts.npy'), shifts)
    pd.Series({'perfect_weather_' + m: v for m, v in pm.items()}).to_csv(
        os.path.join(out_dir, 'point_reference.csv'), header=False)
    print(f"\n  -> {out_dir}")
    return {'model': name, 'grid': grid, 'fixed': fixed, 'bootstrap': boot,
            'point': pm, 'test_hours': len(y_true)}


def main():
    qcfg = dict(cfg.QUANTILE_CONFIG)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', nargs='*', default=None,
                    help="subset to run (default: whatever is enabled in QUANTILE_CONFIG)")
    ap.add_argument('--n-max', type=int, default=None, help="shifted-date sweep upper bound")
    ap.add_argument('--no-bootstrap', action='store_true')
    args = ap.parse_args()
    if args.n_max is not None:
        qcfg['n_max'] = args.n_max

    models = args.models or [m for m, on in qcfg['models'].items() if on]

    df, ept, load, temp, hol, valid = load_frame()
    split = split_at_day_boundary(ept, qcfg['test_frac'])
    train_start = estimation_window_start(ept, split, qcfg['train_years'])

    # Scenario pool source. The long ERA5 temperature store (fetch_scenario_weather.py) is
    # used when present; without it the pool can only come from the load record, which caps
    # k at 4 and makes the paper's central k-sweep unanswerable.
    from fetch_scenario_weather import load_long_temperature
    long = load_long_temperature(cfg.DATASET)
    if long is None:
        pool_ept, pool_temp = ept, temp
        src = 'cleaned CSV — run fetch_scenario_weather.py to lift the k cap'
    else:
        pool_ept, pool_temp = long
        src = f'weather_long {pool_ept[0].year}-{pool_ept[-1].year}'
    years = usable_scenario_years(pool_ept, ept[split].normalize(), qcfg['scenario_years'])

    print(f"=== Probabilistic baselines | zone {cfg.DATASET} ===")
    print(f"  record   {ept[0]} -> {ept[-1]}   ({len(ept)} h)")
    print(f"  train    {ept[train_start]} -> {ept[split - 1]}   ({split - train_start} h, "
          f"window = {qcfg['train_years'] or 'all'} y)")
    print(f"  test     {ept[split]} -> {ept[-1]}   ({len(ept) - split} h, "
          f"{(len(ept) - split) / 24:.0f} d)")
    print(f"  scenario pool  {src}")
    print(f"  scenario years {years[0]}-{years[-1]} (k_max={len(years)}),  "
          f"shifted-date n = 0..{qcfg['n_max']} on k={[k for k in qcfg['k_grid_shifted'] if k <= len(years)]}")

    out_root = qcfg['result_dir']
    results = [run_model(m, ept, load, temp, hol, valid, split, train_start, years, qcfg,
                         out_root, pool_ept=pool_ept, pool_temp=pool_temp,
                         do_bootstrap=not args.no_bootstrap)
               for m in models]

    print(f"\n{'=' * 68}\nSUMMARY | zone {cfg.DATASET}\n{'=' * 68}")
    print(f"{'model':10s} {'perfect-wx MAPE':>16s} {'QS floor':>9s} {'QS fixed kmax':>14s} "
          f"{'QS best':>9s} {'at (k,n)':>9s} {'wx cost':>9s}")
    for r in results:
        g = r['grid']
        kbest, nbest = g.stack().idxmin()[::-1]
        floor = r['point']['QS_degenerate']
        print(f"{r['model']:10s} {r['point']['MAPE']:15.2f}% {floor:9.1f} "
              f"{r['fixed'][g.columns[-1]]:14.1f} {g.values.min():9.1f} "
              f"{f'({kbest},{nbest})':>9s} {g.values.min() / floor:8.2f}x")
    os.makedirs(out_root, exist_ok=True)
    pd.DataFrame([{'model': r['model'], 'perfect_weather_MAPE': r['point']['MAPE'],
                   'qs_degenerate_floor': r['point']['QS_degenerate'],
                   'qs_fixed_kmax': r['fixed'][r['grid'].columns[-1]],
                   'qs_best': r['grid'].values.min(),
                   'test_hours': r['test_hours']} for r in results]
                 ).to_csv(os.path.join(out_root, 'summary.csv'), index=False)
    print(f"\n-> {out_root}/summary.csv")


if __name__ == '__main__':
    main()
