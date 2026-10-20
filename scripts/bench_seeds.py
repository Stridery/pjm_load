"""Seed benchmark for the shipped ARCHFormer variants: how reliable is one number, and what
does averaging several give.

    PJM_DATASET=dom PJM_CONGESTION=1 PJM_LMP=0 python scripts/bench_seeds.py
    PJM_DATASET=bge PJM_CONGESTION=1 PJM_LMP=0 python scripts/bench_seeds.py --mode residual
    PJM_DATASET=dom PJM_CONGESTION=1 PJM_LMP=0 python scripts/bench_seeds.py --seeds 3

Zone and feature arm come from the usual environment variables, like every other driver here.

WHY THIS IS NOT scripts/tune_archformer.py
----------------------------------
That script answers "which recipe": many configurations, one seed each. This one answers "is
the number real": two shipped variants, many seeds each, plus the ensemble. Keeping them apart
keeps each CSV one shape -- a sweep row has a `config` and no ensemble, a seed row has a `seed`
and belongs to one -- and it lets this script point at cfg.ARCHFORMER_HP_PARAMS itself rather
than rebuilding the recipe from overrides, so what gets measured is what ships.

WHAT IT MEASURES, AND WHY IT MATTERS
------------------------------------
ARCHFormer's seed spread is sd 0.115 pp over a range of 3.097-3.416 (dom _cong, residual). The
gap to the best single model in the scoreboard is 0.22 pp. So a single run cannot rank it, and
the scoreboard's archformer rows -- like every other row there -- are one draw from that
distribution. transformer_residual, for contrast, has sd 0.029: the problem is specific to this
architecture's 504,725 parameters over 1461 training days, not general.

The ensemble is the finding. Averaging 5 seeds' predictions scores 2.993, better than the seed
mean by 0.287 pp and better than every single-seed run of every other model. That gain is large
*because* the seed variance is large -- there is more to average away.

`archformer_skip` is registered but not in the default list. It is the one variant whose
WIRING differs -- the 54 broadcast constants skip the patch tokenizer and join at the decoder
output through a fused ReLU -- so averaging it in with the two recipe variants would answer
neither question. Run it with `--variant archformer_skip` when that is what is being asked.
Its earlier additive form scored 3.380 against hp's 3.242; that implementation is gone.

WHAT IT WRITES
--------------
    results/{zone}{TARGET_TAG}/_tuning/seed_study_{mode}{MATRIX_TAG}.csv
        one row per (variant, seed), plus one `seed=ens` row per variant.

    results/{zone}{TARGET_TAG}/evaluation/{variant}_ens{N}/{run}/...
        the ensemble prediction put through the ordinary evaluator, so _scoreboard.collect
        picks it up with no special case and the metrics file has the same shape as every
        other run's. The directory name carries `_ens{N}` because that row is NOT comparable
        to a single-seed row and the table has to say so.

No model weights are saved. A seed study is a measurement; the configuration that wins is meant
to be re-run through scripts/Model_Training.py afterwards.
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

import src.config as cfg
from src.feature_engine import build_timeseries_matrix
from src.models._eval_utils import EvalUtils
from src.models.archformer import ARCHFormerNet
from tune_archformer import SEED, load_split, targets_for, train_one

# The two variants worth repeating, by the name their results directory uses.
VARIANTS = {
    'archformer':      cfg.ARCHFORMER_PARAMS,      # Moyi Guo's recipe, the baseline
    'archformer_hp':   cfg.ARCHFORMER_HP_PARAMS,   # same architecture, this dataset's recipe
    # hp's recipe, different WIRING: the 54 broadcast constants skip the patch tokenizer and
    # join at the decoder output through a fused ReLU. 345,103 parameters against 504,725,
    # the saving taken out of the two patch projections. Not in the default list -- it is the
    # one variant whose architecture differs, so it is opted into rather than swept past.
    'archformer_skip': cfg.ARCHFORMER_SKIP_PARAMS,
}

# Benchmarked unless --variant says otherwise. skip stays out: a seed study is for comparing
# recipes on one architecture, and mixing a rewired model into that average would answer
# neither question.
DEFAULT_VARIANTS = ('archformer', 'archformer_hp')


def run_dir_name(mode):
    """The split tag the shipped runs use, so the ensemble lands beside its single-seed kin."""
    fcg = cfg.ARCHFORMER_FEATURE_CONFIG
    return (f"tail_test{fcg['test_frac']}_tail_val{fcg['val_frac']}{cfg.MATRIX_TAG}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', nargs='*', default=['direct', 'residual'],
                    choices=['direct', 'residual'])
    ap.add_argument('--variant', nargs='*', default=list(DEFAULT_VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument('--seeds', type=int, default=cfg.ARCHFORMER_PARAMS['n_seeds'],
                    help='how many seeds; the default lives in ARCHFORMER_PARAMS')
    ap.add_argument('--epochs', type=int, default=None,
                    help="override each variant's own epoch budget (default: use it)")
    ap.add_argument('--no-eval', action='store_true',
                    help='skip writing the ensemble into evaluation/, CSV only')
    args = ap.parse_args()

    D = load_split()
    # timestamps are not in load_split's npy reads; the builder is memoised, so this is cheap.
    _, _, _, timestamps = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)
    ts_test = timestamps[D['test']]

    C, n_seq = D['X'].shape[2], cfg.N_SEQ_FEATURES
    out_dir = os.path.join(cfg.RESULT_ROOT, '_tuning')
    os.makedirs(out_dir, exist_ok=True)
    seeds = [SEED + i for i in range(args.seeds)]
    print(f"zone {cfg.DATASET}{cfg.TARGET_TAG}  arm {cfg.MATRIX_TAG!r}  C={C} "
          f"(n_seq {n_seq}, static {C - n_seq})  "
          f"train {len(D['train'])} val {len(D['val'])} test {len(D['test'])}")
    print(f"variants {args.variant}  modes {args.mode}  seeds {seeds}\n")

    for mode in args.mode:
        y_space, to_mw, truth_mw = targets_for(mode, D)
        truth_test = truth_mw[D['test']]
        suffix = '_residual' if mode == 'residual' else ''
        print(f"{'=' * 96}\nMODE {mode.upper()}\n{'=' * 96}", flush=True)
        rows = []
        for vname in args.variant:
            params = dict(VARIANTS[vname])
            epochs = args.epochs or params['epochs']
            preds, mets = [], []
            for sd in seeds:
                m, p = train_one(ARCHFormerNet, params, D, y_space, to_mw, truth_mw,
                                 epochs, seed=sd, return_pred=True)
                preds.append(p); mets.append(m)
                rows.append({'variant': vname + suffix, 'mode': mode, 'arm': cfg.MATRIX_TAG,
                             'seed': sd, **m})
                print(f"  {vname + suffix:24s} seed {sd:<4d} MAPE {m['MAPE']:6.3f}  "
                      f"peak {m['peak_MAPE']:6.3f}  ep{m['best_epoch']:3d}  {m['seconds']:4d}s",
                      flush=True)

            mp = np.array([m['MAPE'] for m in mets])
            ens = np.mean(preds, axis=0)
            from tune_archformer import metrics as _metrics
            em = _metrics(truth_test, ens)
            rows.append({'variant': vname + suffix, 'mode': mode, 'arm': cfg.MATRIX_TAG,
                         'seed': 'ens', **em, 'n_params': mets[0]['n_params']})
            print(f"  {vname + suffix:24s} mean      MAPE {mp.mean():6.3f}  +/-{mp.std(ddof=1):5.3f}"
                  f"   [{mp.min():.3f}, {mp.max():.3f}]")
            print(f"  {vname + suffix:24s} ENS({len(seeds)})    MAPE {em['MAPE']:6.3f}  "
                  f"peak {em['peak_MAPE']:6.3f}   gain {mp.mean() - em['MAPE']:+.3f} pp\n",
                  flush=True)

            if not args.no_eval:
                model_dir = f'{vname}{suffix}_ens{len(seeds)}'
                rdir = os.path.join(cfg.RESULT_ROOT, 'evaluation', model_dir, run_dir_name(mode))
                EvalUtils.evaluate_one(model_dir.upper(), truth_test, ens, ts_test, rdir)
                # Say what this row is. Without it the scoreboard shows an ensemble beside
                # single-seed runs with nothing to tell them apart but the directory name.
                mpath = os.path.join(rdir, f'{model_dir.upper()}_metrics.txt')
                if os.path.exists(mpath):
                    with open(mpath, 'a') as fh:
                        fh.write(f"\n--- seed ensemble ---\n"
                                 f"variant: {vname}{suffix}\n"
                                 f"seeds: {len(seeds)} ({', '.join(map(str, seeds))})\n"
                                 f"per-seed MAPE: {', '.join(f'{x:.3f}' for x in mp)}\n"
                                 f"per-seed mean: {mp.mean():.3f}  sd: {mp.std(ddof=1):.3f}\n"
                                 f"NOT comparable to a single-seed row: this is the mean of "
                                 f"{len(seeds)} predictions.\n")
                print(f"  -> {rdir}", flush=True)

        df = pd.DataFrame(rows)
        path = os.path.join(out_dir, f'seed_study_{mode}{cfg.MATRIX_TAG}.csv')
        df.to_csv(path, index=False)
        print(f"  -> {path}\n", flush=True)


if __name__ == '__main__':
    main()
