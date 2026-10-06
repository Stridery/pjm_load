"""Read the GARCHFormer grid cells back into one table and pick the best cell per dataset.

    python scripts/bench_garchformer_collect.py                      # all three
    python scripts/bench_garchformer_collect.py --dataset spain
    python scripts/bench_garchformer_collect.py --metric RMSE        # select on something else

Reads every `results/bench/{dataset}/garchformer/cells/*.json` and writes, per dataset:

    grid.csv          one row per (cell, seed) -- EVERY combination's metrics, kept whole
    grid_summary.csv  one row per cell, mean and sample sd over the seeds, ranked
    best.json         the selected cell and its scores

SELECTION IS ON VALIDATION, SCORING IS ON TEST, and they are different columns. Ranking 64
cells by their test MAPE and then reporting that MAPE is selection on the test set: the
winner's number is the maximum of 64 draws, which is optimistic by however much the spread
is, and on this grid the spread is not small. So `val_MAPE` (mean over seeds) picks the
cell and `test_MAPE` is what gets reported for it. Both are in the table for every cell,
so the size of that gap is visible rather than assumed.

Completeness is checked against `src.bench.garchformer.cells()`, the same definition the
condor itemdata is generated from, so a dataset that is missing cells says so instead of
quietly ranking a partial grid.
"""

# Entry points live in scripts/ but import the package from the repo root, and running
# `python scripts/foo.py` puts scripts/ on sys.path rather than the root. Prepend the root so
# `import src...` resolves from any working directory.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from src.bench.data import DATASETS

KEYS = ['d_model', 'num_heads', 'layers', 'lambda_cons']
# Reported for both splits. MAPE first: it is the selection metric and the one metric here
# that is in physical units and therefore comparable across the three datasets without
# further thought.
METRICS = ['MAPE', 'SMAPE', 'MAE', 'RMSE', 'MSE', 'Pinball', 'PICP', 'MPIW', 'Winkler']


def read_cells(dataset, out_root):
    d = os.path.join(out_root, dataset, 'garchformer', 'cells')
    rows = []
    for p in sorted(glob.glob(os.path.join(d, '*.json'))):
        with open(p) as fh:
            rows.append(json.load(fh))
    return pd.DataFrame(rows), d


def summarise(df, metric):
    """One row per cell: mean over seeds, plus the sample sd of the selection metric.

    ddof=1 for the same reason src/bench/metrics.aggregate uses it -- the seeds are a sample
    of the run-to-run variation, and at two or three of them the population form understates
    the spread by a factor that is not cosmetic. One seed gives nan, not a fabricated 0.
    """
    cols = [f'{p}_{m}' for p in ('val', 'test') for m in METRICS]
    cols = [c for c in cols if c in df.columns]
    g = df.groupby(KEYS, as_index=False)
    out = g[cols + ['n_params', 'seconds', 'best_epoch']].mean()
    out['seeds'] = g.size()['size'].to_numpy()
    for p in ('val', 'test'):
        c = f'{p}_{metric}'
        if c in df.columns:
            out[f'{c}_sd'] = g[c].std(ddof=1)[c].to_numpy()
    return out.sort_values(f'val_{metric}').reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', nargs='*', default=list(DATASETS), choices=DATASETS)
    ap.add_argument('--metric', default='MAPE', choices=METRICS,
                    help='selected on its VALIDATION value (default MAPE)')
    ap.add_argument('--out', default='results/bench')
    ap.add_argument('--top', type=int, default=10, help='rows to print per dataset')
    a = ap.parse_args()

    from src.bench.garchformer import cells, skipped

    expected = {(c['d_model'], c['num_heads'], c['layers'], c['lambda_cons'])
                for c in cells()}
    sk = skipped()
    print(f"grid: {len(expected)} cells per dataset per seed"
          + (f"  ({len(sk)} (d_model, heads) pair(s) dropped as indivisible: {sk})"
             if sk else ""))

    best_all = {}
    for name in a.dataset:
        df, cdir = read_cells(name, a.out)
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        if df.empty:
            print(f"  no cells under {cdir}")
            continue

        out_dir = os.path.dirname(cdir)
        df = df.sort_values(KEYS + ['seed']).reset_index(drop=True)
        df.to_csv(os.path.join(out_dir, 'grid.csv'), index=False)

        got = {tuple(r) for r in df[KEYS].to_numpy()}
        missing = sorted(expected - {(int(d), int(h), int(l), float(m)) for d, h, l, m in got})
        seeds = sorted(df['seed'].unique().tolist())
        print(f"  {len(df)} runs over {len(got)}/{len(expected)} cells, seeds {seeds}")
        if missing:
            print(f"  INCOMPLETE -- {len(missing)} cell(s) absent, the ranking below is "
                  f"over what exists:")
            for m in missing[:12]:
                print(f"      d_model={m[0]:<4d} heads={m[1]:<3d} layers={m[2]} lambda={m[3]}")
            if len(missing) > 12:
                print(f"      ... and {len(missing) - 12} more")

        s = summarise(df, a.metric)
        s.to_csv(os.path.join(out_dir, 'grid_summary.csv'), index=False)

        vm, tm = f'val_{a.metric}', f'test_{a.metric}'
        print(f"\n  ranked by {vm} (mean over seeds); test shown but NOT used to rank\n")
        hdr = (f"    {'d_model':>7s} {'heads':>5s} {'lyr':>3s} {'lambda':>6s} "
               f"{'params':>9s} {vm:>10s} {'sd':>7s} {tm:>10s} {'min':>6s}")
        print(hdr)
        print('    ' + '-' * (len(hdr) - 4))
        for _, r in s.head(a.top).iterrows():
            sd = r.get(f'{vm}_sd', float('nan'))
            print(f"    {int(r['d_model']):>7d} {int(r['num_heads']):>5d} "
                  f"{int(r['layers']):>3d} {r['lambda_cons']:>6.2f} "
                  f"{int(r['n_params']):>9,d} {r[vm]:>10.4f} "
                  f"{'   --  ' if np.isnan(sd) else format(sd, '>7.4f')} "
                  f"{r[tm]:>10.4f} {r['seconds'] / 60:>6.1f}")

        b = s.iloc[0]
        best = {'dataset': name, 'selected_on': vm, 'n_cells_ranked': int(len(s)),
                'n_cells_expected': len(expected), 'incomplete': bool(missing),
                'seeds': seeds,
                **{k: (int(b[k]) if k != 'lambda_cons' else float(b[k])) for k in KEYS},
                'n_params': int(b['n_params']),
                **{f'{p}_{m}': float(b[f'{p}_{m}'])
                   for p in ('val', 'test') for m in METRICS if f'{p}_{m}' in b}}
        with open(os.path.join(out_dir, 'best.json'), 'w') as fh:
            json.dump(best, fh, indent=1, sort_keys=True)
        best_all[name] = best

        print(f"\n  BEST  d_model={best['d_model']} num_heads={best['num_heads']} "
              f"layers={best['layers']} lambda_cons={best['lambda_cons']:g}"
              f"   ({best['n_params']:,} params)")
        print(f"        val  {a.metric} {best[vm]:.4f}   ->   "
              f"test {a.metric} {best[tm]:.4f}")
        print(f"  wrote grid.csv, grid_summary.csv, best.json under {out_dir}/")

    if len(best_all) > 1:
        print(f"\n{'=' * 78}\nbest cell per dataset (selected on val_{a.metric})\n{'=' * 78}")
        print(f"  {'dataset':8s} {'d_model':>7s} {'heads':>5s} {'lyr':>3s} {'lambda':>6s} "
              f"{'val':>9s} {'test':>9s}")
        for n, b in best_all.items():
            print(f"  {n:8s} {b['d_model']:>7d} {b['num_heads']:>5d} {b['layers']:>3d} "
                  f"{b['lambda_cons']:>6.2f} {b[f'val_{a.metric}']:>9.4f} "
                  f"{b[f'test_{a.metric}']:>9.4f}")


if __name__ == '__main__':
    main()
