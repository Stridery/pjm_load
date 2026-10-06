"""One GARCHFormer grid cell on one public dataset. Trains, scores, writes a single JSON.

    python scripts/bench_garchformer.py --dataset ercot --d-model 64 --num-heads 16 \
                                        --layers 2 --lambda-cons 0.1 --seed 0

ONE CELL PER PROCESS, on purpose. The grid is 64 cells per dataset and the slowest of them
is about nine minutes, so the unit that gets queued is one training run: an evicted job costs
minutes rather than hours, and 192 of them fill a pool that would hand one long process a
single slot. `condor/make_bench_gf_jobs.sh` generates the list and `condor/BenchGarch.sub`
submits it.

No data is fetched, no matrix is cached and NO WEIGHTS ARE SAVED. The data is fixed
(`data/quantile/`), the architecture comes from the four grid values, and the only artifact
is the cell's own metrics file:

    results/bench/{dataset}/garchformer/cells/d{d}_h{h}_L{l}_lam{lam}_s{seed}.json

One file per cell rather than one shared CSV, because 192 jobs appending to the same file
over NFS interleave their rows. The write is to a .tmp and then an os.replace, which is
atomic on one filesystem, so a killed job leaves either the old file or the new one and never
half a row. `scripts/bench_garchformer_collect.py` reads the directory back into one table.

An existing cell file is treated as done and the process exits 0 without training, so a
resubmission after a partial run only picks up what is missing. `--force` retrains it.
"""

# Entry points live in scripts/ but import the package from the repo root, and running
# `python scripts/foo.py` puts scripts/ on sys.path rather than the root. Prepend the root so
# `import src...` resolves from any working directory.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import os

from src.bench.data import DATASETS, load


def lam_tag(lam):
    """0.05 -> 'lam0p05'. A '.' in a filename is legal but makes the stem ambiguous."""
    return 'lam' + f'{float(lam):g}'.replace('.', 'p').replace('-', 'm')


def cell_tag(cell, seed):
    return (f"d{int(cell['d_model'])}_h{int(cell['num_heads'])}"
            f"_L{int(cell['layers'])}_{lam_tag(cell['lambda_cons'])}_s{int(seed)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=DATASETS)
    ap.add_argument('--d-model', type=int, required=True)
    ap.add_argument('--num-heads', type=int, required=True,
                    help='encoder and decoder share this count')
    ap.add_argument('--layers', type=int, required=True,
                    help='used for BOTH num_encoder_layers and num_decoder_layers')
    ap.add_argument('--lambda-cons', type=float, required=True,
                    help='variance-consistency weight; 0 leaves the sigma head unsupervised')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--out', default='results/bench')
    ap.add_argument('--force', action='store_true', help='retrain even if the cell exists')
    a = ap.parse_args()

    cell = {'d_model': a.d_model, 'num_heads': a.num_heads,
            'layers': a.layers, 'lambda_cons': a.lambda_cons}

    # Refuse the illegal (d_model, num_heads) pairs LOUDLY. Upstream's
    # generate_valid_param_combinations drops them with a bare `continue`, so a grid that
    # asked for 72 cells silently runs 64 and the count in the log is the only trace. Exit 2
    # names the cell, and condor records it as a failure rather than a success with no row.
    if a.d_model % a.num_heads:
        print(f"ILLEGAL CELL: d_model={a.d_model} % num_heads={a.num_heads} != 0. "
              f"Both ArchQAttentionBlock and MuCrossAttentionBlock assert divisibility.")
        return 2

    out_dir = os.path.join(a.out, a.dataset, 'garchformer', 'cells')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, cell_tag(cell, a.seed) + '.json')
    if os.path.exists(path) and not a.force:
        print(f"already done, skipping: {path}")
        return 0

    # Imported here and not at module scope: `src.bench.garchformer` pulls in
    # `src.models.__init__`, which costs ~6 s of xgboost and lightgbm. The two exits above
    # should not pay for it.
    from src.bench import garchformer as G

    ds = load(a.dataset)
    print(f"=== garchformer | {ds.describe()}")
    params = {}
    if a.epochs is not None:
        params['epochs'] = a.epochs
    if a.patience is not None:
        params['patience'] = a.patience

    row = G.run(ds, cell, params=params or None, seed=a.seed, device=a.device)
    row['cell'] = cell_tag(cell, a.seed)

    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(row, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)

    print(f"  epochs {row['epochs_run']} (best {row['best_epoch']}), "
          f"{row['seconds'] / 60:.1f} min, {row['n_params']:,} params")
    print(f"  val  MAPE {row['val_MAPE']:7.3f}  MAE {row['val_MAE']:.4f}  "
          f"RMSE {row['val_RMSE']:.4f}  PICP {row['val_PICP']:5.1f}")
    print(f"  test MAPE {row['test_MAPE']:7.3f}  MAE {row['test_MAE']:.4f}  "
          f"RMSE {row['test_RMSE']:.4f}  PICP {row['test_PICP']:5.1f}")
    print(f"  -> {path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
