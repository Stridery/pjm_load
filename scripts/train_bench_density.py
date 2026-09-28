"""Baseline A on the public datasets — Li, Wang & Zhang (2020), density combination.

    python scripts/train_bench_density.py --dataset spain
    python scripts/train_bench_density.py --dataset ercot --seeds 0 1 2 --workers 8
    python scripts/train_bench_density.py --dataset elia --models GPR1 NN1 XGB1   # smoke test

Reads data/quantile/{dataset}_config.json and the CSV it names; writes results/bench/
{dataset}/density/ and touches nothing else. The PJM baselines are unaffected.

Each seed re-seeds the whole pipeline — GPR subsamples, network initialisation, booster
subsampling and the multi-start QP — so the spread across seeds is the real run-to-run
spread, not the spread of one number. The table reports the mean and the SAMPLE standard
deviation (ddof=1) over the seeds, for every scheme and every metric; per_seed_scores.csv
holds the individual runs.

Metrics are on the z-scored target except MAPE and SMAPE, which are computed in physical
units. See src/bench/metrics.py.
"""

# Entry points live in scripts/ but import the package from the repo root, and running
# `python scripts/foo.py` puts scripts/ on sys.path rather than the root. Prepend the root so
# `import src...` resolves from any working directory.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import os

import numpy as np
import pandas as pd

from src.bench import density as D
from src.bench import metrics as M
from src.bench.data import DATASETS, load

ORDER = ['Pinball', 'PICP', 'MPIW', 'Winkler', 'MAE', 'RMSE', 'MSE', 'MAPE', 'SMAPE', 'CRPS']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=DATASETS)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--models', nargs='*', default=None, help='subset of base model names')
    ap.add_argument('--out', default='results/bench')
    a = ap.parse_args()

    ds = load(a.dataset)
    out_dir = os.path.join(a.out, a.dataset, 'density')
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== density combination | {ds.describe()} ===")
    runs = []
    for s in a.seeds:
        print(f"\n--- seed {s} ---")
        runs.append(D.run(ds, seed=s, workers=a.workers, models=a.models))
        r = runs[-1]
        print(f"  QP: Q_new smallest eigenvalue {r['qp_info']['min_eig_Qnew']:+.4g} -> "
              f"{'global' if r['qp_info']['certified_global'] else 'local only'}; "
              f"{r['qp_info']['n_selected']}/{len(r['names'])} models kept; "
              f"best individual {r['best_individual']}")

    # -- the table -------------------------------------------------------------------
    rows = []
    for scheme in D.SCHEMES:
        agg = M.aggregate([r['scores'][scheme] for r in runs])
        rows.append({'scheme': scheme,
                     **{k: agg[k][0] for k in ORDER},
                     **{f'{k}_sd': agg[k][1] for k in ORDER}})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(out_dir, 'test_scores.csv'), index=False)

    print(f"\n{'=' * 96}")
    print(f"TEST | {a.dataset} | {runs[0]['n_test_steps']} steps | "
          f"seeds {a.seeds} | mean and SAMPLE sd (ddof=1) | "
          f"standardized units except MAPE/SMAPE")
    print('=' * 96)
    multi = len(a.seeds) > 1
    print(f"{'scheme':17s}" + ''.join(f"{k:>12s}" for k in ORDER))
    for r in rows:
        print(f"{r['scheme']:17s}" + ''.join(f"{r[k]:12.4f}" for k in ORDER))
        if multi:                       # every scheme gets its own spread, not just the first
            print(f"{'  +- sd':17s}" + ''.join(f"{r[k + '_sd']:12.4f}" for k in ORDER))

    # Which base model won on T3 in each seed. Averaging the 'Best individual' row over seeds
    # is only meaningful if it is the same model each time, so say when it is not.
    won = [r['best_individual'] for r in runs]
    print(f"\nbest individual per seed: {', '.join(f'{s_}:{w}' for s_, w in zip(a.seeds, won))}"
          + ('' if len(set(won)) == 1 else
             '   <- NOT the same model every seed; that row averages different models'))

    # Per-seed scores, so the spread can be inspected rather than taken on trust.
    pd.DataFrame([{'seed': r['seed'], 'scheme': k, **v}
                  for r in runs for k, v in r['scores'].items()]
                 ).to_csv(os.path.join(out_dir, 'per_seed_scores.csv'), index=False)

    # -- weights and base models ------------------------------------------------------
    W = pd.DataFrame({'model': runs[0]['names']})
    for scheme in ('CRPS-guided', 'MAPE-guided'):
        W[scheme] = np.mean([r['weights'][scheme] for r in runs], axis=0)
    W['T2_crps'] = [np.mean([r['base_T2_crps'][n] for r in runs]) for n in runs[0]['names']]
    W['T4_crps'] = [np.mean([r['base_T4_crps'][n] for r in runs]) for n in runs[0]['names']]
    W.to_csv(os.path.join(out_dir, 'weights_and_base_models.csv'), index=False)
    print(f"\n{'model':9s}{'w(CRPS)':>10s}{'w(MAPE)':>10s}{'T2 CRPS':>10s}{'T4 CRPS':>10s}")
    for r in W.itertuples():
        print(f"{r.model:9s}{r._2:10.3f}{r._3:10.3f}{r.T2_crps:10.4f}{r.T4_crps:10.4f}")

    # The paper's claim, stated as a number: does the combination beat its best member?
    comb = tbl.loc[tbl.scheme == 'CRPS-guided', 'CRPS'].iloc[0]
    best = tbl.loc[tbl.scheme == 'Best individual', 'CRPS'].iloc[0]
    print(f"\ncombination vs best individual on T4 CRPS: {comb:.4f} vs {best:.4f} "
          f"({100 * (best - comb) / best:+.2f}% )")

    # -- curves, for the figures ------------------------------------------------------
    r0 = runs[0]
    np.savez_compressed(os.path.join(out_dir, 'test_quantiles.npz'),
                        actual=r0['actual'], datetime=r0['datetime'], levels=r0['levels'],
                        scale=np.array(r0['scale']),
                        labels=np.array(list(r0['curves']), dtype=object),
                        **{f'cfg{i}': q for i, q in enumerate(r0['curves'].values())})
    print(f"\n-> {out_dir}")


if __name__ == '__main__':
    main()
