"""Soft-gated MoE vs hard calendar routing vs no mixture at all — measurement, no weights.

    PJM_DATASET=dom PJM_CONGESTION=1 PJM_LMP=0 python scripts/bench_smoe.py
    PJM_DATASET=dom PJM_CONGESTION=1 PJM_LMP=0 python scripts/bench_smoe.py --mode residual --seeds 3

THE QUESTION
------------
moe_transformer routes on the target day's month and gives each expert a fixed slice of the
24-hour output. That divides 1461 training days twelve ways -- its experts see 1,428 to 7,018
target scalars each against 35,064 for a single head -- and it loses to the plain transformer
in every comparison in this repo: 0.087 pp in residual mode, 0.827 pp in direct mode.

The mainstream formulation (Jacobs et al. 1991; Shazeer et al. 2017) divides nothing:

    y = sum_i g_i(z) * E_i(z),   g = softmax(W z)

Every expert outputs all 24 hours and receives gradient from every sample, weighted by its
gate. Two variants differ ONLY in what the router is allowed to look at:

    smoe_z     gate reads all of z -- encoder output plus every static feature
    smoe_cal   gate reads only the 6 target-day calendar channels, which is the softened form
               of moe_transformer's month rule: still calendar-driven, but the boundaries are
               learned rather than written down, and no sample is withheld from any expert

WHAT TO READ FIRST, AND IT IS NOT THE MAPE
------------------------------------------
The gate usage per expert. With n_experts=4:

    all ~0.25      the gate never specialised. The model is an ensemble of heads over a shared
                   encoder, and the premise -- that different regimes want different predictors
                   -- is not supported on this data, however the split is drawn.
    one ~1.0       collapsed; balance_weight is too low and the mixture is one wide head.
    spread, and    the premise holds. Only then does comparing MAPE against moe_transformer
    structured     say anything about hard-vs-learned routing.

A gate that stays uniform is a real result, not a failure: unlike hard routing, a soft mixture
that declines to specialise degenerates to an average of its experts rather than to a set of
under-trained specialists, so its floor is at worst the plain model.

WHAT IT WRITES
--------------
    results/{zone}{TARGET_TAG}/evaluation/{variant}/{run}/    via the ordinary evaluator
    results/{zone}{TARGET_TAG}/_tuning/smoe_{mode}{MATRIX_TAG}.csv

No model weights are saved.
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
import torch

import src.config as cfg
from src.feature_engine import build_timeseries_matrix
from src.models._eval_utils import EvalUtils
from src.models.smoe_transformer import SoftMoETransformer3D
from src.models.transformer import TimeSeriesTransformer3D
from tune_archformer import SEED, load_split, targets_for, train_one, metrics

BASE = cfg.SMOE_TRANSFORMER_PARAMS


def variants():
    """(name, class, params). The two controls matter as much as the two treatments.

    transformer_wide exists because smoe carries ~54k more parameters than the plain head, all
    of them in the extra experts. Without a single-head model at matching width, a win could
    just be capacity. fc_hidden 480 puts it within a few percent of smoe's 173,872.
    """
    plain = dict(cfg.TRANSFORMER_PARAMS)
    wide = {**plain, 'fc_hidden': 480}
    return [
        ('transformer',       TimeSeriesTransformer3D, plain),
        ('transformer_wide',  TimeSeriesTransformer3D, wide),
        ('smoe_z',            SoftMoETransformer3D,   {**BASE, 'gate_input': 'z'}),
        ('smoe_cal',          SoftMoETransformer3D,   {**BASE, 'gate_input': 'calendar'}),
    ]


def gate_usage(net, X, idx, chunk=256):
    """Mean gate weight per expert over the given rows, plus the per-sample max weight.

    The mean says whether any expert is ignored overall; the mean of the per-sample max says
    whether individual days are routed sharply. Both near 1/n means no routing is happening;
    a high per-sample max with a flat mean means the gate does pick, just not the same expert
    every time -- which is the outcome the whole idea is betting on.
    """
    if not hasattr(net, 'head'):
        return None, None
    net.eval()
    gs = []
    with torch.no_grad():
        for i in range(0, len(idx), chunk):
            net(torch.FloatTensor(X[idx[i:i + chunk]]))
            gs.append(net.head._last_gate.cpu().numpy())
    g = np.concatenate(gs, axis=0)
    return g.mean(axis=0), float(g.max(axis=1).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', nargs='*', default=['direct', 'residual'],
                    choices=['direct', 'residual'])
    ap.add_argument('--seeds', type=int, default=cfg.ARCHFORMER_PARAMS['n_seeds'],
                    help="how many seeds; the default lives in ARCHFORMER_PARAMS so a\n"
                         "seed count means the same thing in every bench driver")
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--only', nargs='*', default=None,
                    help='subset of variant names to run')
    ap.add_argument('--no-eval', action='store_true',
                    help='CSV only; do not write into evaluation/')
    args = ap.parse_args()

    D = load_split()
    _, _, _, timestamps = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)
    ts_test = timestamps[D['test']]
    seeds = [SEED + i for i in range(args.seeds)]
    out_dir = os.path.join(cfg.RESULT_ROOT, '_tuning')
    os.makedirs(out_dir, exist_ok=True)

    todo = [v for v in variants() if args.only is None or v[0] in args.only]
    print(f"zone {cfg.DATASET}{cfg.TARGET_TAG}  arm {cfg.MATRIX_TAG!r}  C={D['X'].shape[2]} "
          f"(n_seq {cfg.N_SEQ_FEATURES})  train {len(D['train'])} val {len(D['val'])} "
          f"test {len(D['test'])}")
    print(f"variants {[v[0] for v in todo]}  modes {args.mode}  seeds {seeds}\n")

    fcg = cfg.TRANSFORMER_FEATURE_CONFIG
    run_dir = f"tail_test{fcg['test_frac']}_tail_val{fcg['val_frac']}{cfg.MATRIX_TAG}"

    for mode in args.mode:
        y_space, to_mw, truth_mw = targets_for(mode, D)
        truth_test = truth_mw[D['test']]
        suffix = '_residual' if mode == 'residual' else ''
        print(f"{'=' * 100}\nMODE {mode.upper()}\n{'=' * 100}", flush=True)
        rows = []
        for vname, Cls, params in todo:
            epochs = args.epochs or params['epochs']
            preds, mets, last_net = [], [], None
            for sd in seeds:
                m, p, net = train_one(Cls, params, D, y_space, to_mw, truth_mw, epochs,
                                      seed=sd, return_pred=True, return_net=True)
                preds.append(p); mets.append(m); last_net = net
                rows.append({'variant': vname + suffix, 'mode': mode, 'arm': cfg.MATRIX_TAG,
                             'seed': sd, **m})
                print(f"  {vname + suffix:26s} seed {sd:<4d} MAPE {m['MAPE']:6.3f}  "
                      f"peak {m['peak_MAPE']:6.3f}  ep{m['best_epoch']:3d}  {m['seconds']:4d}s",
                      flush=True)

            mp = np.array([m['MAPE'] for m in mets])
            ens = np.mean(preds, axis=0)
            em = metrics(truth_test, ens)
            rows.append({'variant': vname + suffix, 'mode': mode, 'arm': cfg.MATRIX_TAG,
                         'seed': 'ens', **em, 'n_params': mets[0]['n_params']})
            print(f"  {vname + suffix:26s} mean      MAPE {mp.mean():6.3f}  "
                  f"+/-{mp.std(ddof=1):5.3f}   ENS {em['MAPE']:6.3f}   "
                  f"{mets[0]['n_params']:,}p", flush=True)

            # Gate diagnostics from the LAST seed's trained net. Read from one seed, not
            # averaged: expert identity is arbitrary up to permutation, so expert 2 of one run
            # and expert 2 of another are unrelated and their gate vectors must not be blended.
            if vname.startswith('smoe'):
                gmean, gsharp = gate_usage(last_net, D['X'], D['test'])
                rows[-1]['gate_mean'] = np.round(gmean, 4).tolist()
                rows[-1]['gate_sharpness'] = round(gsharp, 4)
                flat = 1.0 / params['n_experts']
                verdict = ('uniform -> no specialisation' if gmean.max() < flat * 1.25 else
                           'collapsed -> raise balance_weight' if gmean.max() > 0.85 else
                           'specialised')
                print(f"  {'':26s} gate mean {np.round(gmean, 3)}  "
                      f"per-sample max {gsharp:.3f}  [{verdict}]", flush=True)

            if not args.no_eval:
                # `_ens{N}`, like bench_seeds writes: what lands here is the mean of N seeds'
                # predictions, not a run. Without the suffix the `transformer` control would
                # overwrite the shipped single-seed result under the same name and the
                # scoreboard would then compare an ensemble against single runs with nothing
                # marking the difference.
                mdir = f'{vname}{suffix}_ens{len(seeds)}'
                rdir = os.path.join(cfg.RESULT_ROOT, 'evaluation', mdir, run_dir)
                EvalUtils.evaluate_one(mdir.upper(), truth_test, ens, ts_test, rdir)
                mpath = os.path.join(rdir, f'{mdir.upper()}_metrics.txt')
                if os.path.exists(mpath):
                    with open(mpath, 'a') as fh:
                        fh.write(f"\n--- seed ensemble ---\nvariant: {vname}{suffix}\n"
                                 f"seeds: {len(seeds)} ({', '.join(map(str, seeds))})\n"
                                 f"per-seed MAPE: {', '.join(f'{x:.3f}' for x in mp)}\n"
                                 f"per-seed mean: {mp.mean():.3f}  sd: {mp.std(ddof=1):.3f}\n"
                                 f"NOT comparable to a single-seed row.\n")

        df = pd.DataFrame(rows)
        path = os.path.join(out_dir, f'smoe_{mode}{cfg.MATRIX_TAG}.csv')
        df.to_csv(path, index=False)
        print(f"\n  -> {path}\n", flush=True)


if __name__ == '__main__':
    main()
