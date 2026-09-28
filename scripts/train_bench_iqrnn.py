"""Baseline B on the public datasets --- Zhang, Quan & Srinivasan (2019), iQRNN.

    python scripts/train_bench_iqrnn.py --dataset spain
    python scripts/train_bench_iqrnn.py --dataset elia --seeds 0 1 2 --spec iQRNN-500

Reads data/quantile/{dataset}_config.json and the CSV it names; writes results/bench/
{dataset}/iqrnn/ and touches nothing else. The PJM baselines are unaffected.

Three seeds by default, reported as mean and SAMPLE standard deviation (ddof=1). A seed drives
weight initialisation, batch order, dropout masks and the Gaussian-noise draws, so a repeated
run reproduces bit for bit and the spread across seeds is the spread the paper's Table IV
reports rather than scheduler jitter. Feature selection is NOT re-seeded: RFE runs once per
dataset and is cached, because the paper selects features once and then trains repeatedly.

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

from src.bench import iqrnn as I
from src.bench import metrics as M
from src.bench.data import DATASETS, load
from src.qrnn.model import SPECS

ORDER = ['Pinball', 'PICP', 'MPIW', 'Winkler', 'MAE', 'RMSE', 'MSE', 'MAPE', 'SMAPE']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=DATASETS)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--spec', default=I.DEFAULT_CFG['spec'],
                    choices=[s['name'] for s in SPECS])
    ap.add_argument('--n-features', type=int, default=None,
                    help='override the RFE target (default: min(100, pool/3))')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--refresh-rfe', action='store_true')
    ap.add_argument('--out', default='results/bench')
    a = ap.parse_args()

    ds = load(a.dataset)
    out_dir = os.path.join(a.out, a.dataset, 'iqrnn')
    os.makedirs(out_dir, exist_ok=True)
    cfg = {'spec': a.spec, 'device': a.device, 'n_features': a.n_features}

    print(f"=== iQRNN | {ds.describe()} ===")
    runs = []
    for s in a.seeds:
        print(f"\n--- seed {s} ---")
        runs.append(I.run(ds, seed=s, cfg=cfg, refresh_rfe=a.refresh_rfe and s == a.seeds[0],
                          cache_path=os.path.join(out_dir, 'features.json')))

    agg = M.aggregate([r['scores'] for r in runs])
    row = {'dataset': a.dataset, 'spec': a.spec, 'seeds': len(a.seeds),
           **{k: agg[k][0] for k in ORDER}, **{f'{k}_sd': agg[k][1] for k in ORDER}}
    pd.DataFrame([row]).to_csv(os.path.join(out_dir, 'test_scores.csv'), index=False)
    pd.DataFrame([{'seed': r['seed'], **r['scores'], 'epochs': r['history']['epochs'],
                   'minutes': r['history']['seconds'] / 60,
                   'val_pinball': r['history']['val_pinball'],
                   'crossing_pct': 100 * r['crossing_frac']} for r in runs]
                 ).to_csv(os.path.join(out_dir, 'per_seed_scores.csv'), index=False)

    print(f"\n{'=' * 96}")
    print(f"TEST | {a.dataset} | {runs[0]['n_test_steps']} steps | {a.spec} | "
          f"seeds {a.seeds} | mean and SAMPLE sd (ddof=1) | "
          f"standardized units except MAPE/SMAPE")
    print('=' * 96)
    print(f"{'':17s}" + ''.join(f"{k:>12s}" for k in ORDER))
    print(f"{a.spec:17s}" + ''.join(f"{row[k]:12.4f}" for k in ORDER))
    if len(a.seeds) > 1:
        print(f"{'  +- sd':17s}" + ''.join(f"{row[k + '_sd']:12.4f}" for k in ORDER))

    print(f"\nfeatures kept: {runs[0]['groups']}")
    print(f"crossing on the RAW output: "
          + ', '.join(f"seed {r['seed']} {100 * r['crossing_frac']:.2f}%" for r in runs)
          + "   (everything scored above is sorted)")

    r0 = runs[0]
    np.savez_compressed(os.path.join(out_dir, 'test_quantiles.npz'),
                        actual=r0['actual'], datetime=r0['datetime'], levels=r0['levels'],
                        scale=np.array(r0['scale']),
                        labels=np.array([a.spec], dtype=object), cfg0=r0['curves'][a.spec])
    print(f"\n-> {out_dir}")


if __name__ == '__main__':
    main()
