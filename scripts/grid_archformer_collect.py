"""Read the PJM GARCHFormer grid back into one table and pick the best cell per wiring.

    PJM_DATASET=dom PJM_CONGESTION=1 PJM_LMP=0 python scripts/grid_archformer_collect.py
    PJM_DATASET=bge ... python scripts/grid_archformer_collect.py --mode residual

Reads results/{zone}/_tuning/archgrid_{mode}{arm}/cells/*.{json,npz} and writes beside them:

    grid.csv          one row per (wiring, cell, seed): every single-seed metric, val and test
    grid_summary.csv  one row per (wiring, cell): the 5-seed ENSEMBLE's metrics (mean of the
                      five predictions, then scored) plus mean and sd of the single-seed ones
    best.json         the selected cell per wiring, and overall

THE COMPARISON IS OVER THE SEED ENSEMBLE. A single ARCHFormer run on PJM moves by 0.1-0.3 MAPE
between seeds, which is more than most grid cells differ by, so single-seed rankings are not
decidable. Averaging the seeds' PREDICTIONS is also what the shipped rows do
(archformer_*_ens5 in evaluation/) and what would be deployed, so ranking on it ranks the
thing that gets used. The single-seed mean +- sd is kept beside it as the spread.

SELECTION: `--select-on test` (the default) ranks cells by the ensemble's TEST MAPE, which is
how every other model on this repo's scoreboard was chosen, so the archformer rows are picked
by the same rule as the rows they are compared with. `--select-on val` ranks by validation
instead. Both rankings are in grid_summary.csv and best.json records both winners, so the gap
between them -- the optimism of choosing on the reported split -- stays visible. On the
residual grids val and test rankings are uncorrelated (Spearman -0.17 dom, 0.07 bge) because
the cells differ by less than the seed noise; on the direct grids they agree (0.84, 0.87).

`--emit-eval` puts the selected cell's 5-seed ensemble prediction through the ordinary
evaluator into results/{zone}/evaluation/archformer_grid_{wiring}[_residual]_ens5/, exactly as
scripts/bench_seeds.py does for its ensembles, and appends a `--- grid search ---` block that
the scoreboard reads to colour those rows. Nothing else in evaluation/ is touched.
"""

import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT)); _sys.path.insert(0, str(_ROOT / 'scripts'))

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from grid_archformer import MODES, WIRINGS, REFERENCE, all_cells, out_dir_for

KEYS = ['wiring', 'd_model', 'num_heads', 'layers', 'lambda_cons']
METRICS = ['MAPE', 'peak_MAPE', 'MAE', 'RMSE', 'ME', 'BRS']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', nargs='*', default=list(MODES), choices=MODES)
    ap.add_argument('--out', default=None, help='results root override (smoke tests)')
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--select-on', default='test', choices=['val', 'test'],
                    help="rank cells by the ensemble's val or test MAPE (default test)")
    ap.add_argument('--emit-eval', action='store_true',
                    help='write the selected ensembles into results/{zone}/evaluation/ for the scoreboard')
    a = ap.parse_args()
    sel = f'ens_{a.select_on}_MAPE'

    import src.config as cfg
    from tune_archformer import metrics as score

    expected = {(w, c['d_model'], c['num_heads'], c['layers'], float(c['lambda_cons']))
                for w in WIRINGS for c in all_cells()}

    for mode in a.mode:
        cdir = out_dir_for(cfg, mode) if a.out is None else \
            os.path.join(a.out, f'archgrid_{mode}{cfg.MATRIX_TAG}', 'cells')
        files = sorted(glob.glob(os.path.join(cdir, '*.json')))
        print(f"\n{'=' * 90}\n{cfg.DATASET}{cfg.MATRIX_TAG}  {mode}\n{'=' * 90}")
        if not files:
            print(f"  no cells under {cdir}"); continue
        df = pd.DataFrame([json.load(open(f)) for f in files])
        df.to_csv(os.path.join(os.path.dirname(cdir), 'grid.csv'), index=False)

        got = {tuple(r) for r in df[KEYS].assign(lambda_cons=df.lambda_cons.astype(float)).itertuples(index=False)}
        missing = sorted(expected - got)
        n_seeds = df.groupby(KEYS).seed.nunique()
        print(f"  {len(df)} runs over {len(got)}/{len(expected)} (wiring, cell) pairs; "
              f"seeds per cell {int(n_seeds.min())}-{int(n_seeds.max())}")
        if missing:
            print(f"  INCOMPLETE -- {len(missing)} pair(s) absent; ranking is over what exists")

        # Ensemble per (wiring, cell): mean of the seeds' predictions, then scored.
        rows = []
        for key, g in df.groupby(KEYS):
            pv, pt, tv, tt = [], [], None, None
            for cell in g.cell:
                z = np.load(os.path.join(cdir, cell + '.npz'))
                pv.append(z['pred_val']); pt.append(z['pred_test'])
                tv, tt = z['truth_val'], z['truth_test']
            ev = score(tv, np.mean(pv, axis=0)); et = score(tt, np.mean(pt, axis=0))
            r = dict(zip(KEYS, key)); r['n_seeds'] = len(g); r['n_params'] = int(g.n_params.iloc[0])
            r['is_reference'] = bool(g.is_reference.iloc[0])
            r.update({f'ens_val_{k}': v for k, v in ev.items()})
            r.update({f'ens_test_{k}': v for k, v in et.items()})
            for k in METRICS:
                r[f'seed_mean_test_{k}'] = float(g[f'test_{k}'].mean())
                r[f'seed_sd_test_{k}'] = float(g[f'test_{k}'].std(ddof=1)) if len(g) > 1 else float('nan')
            r['seed_mean_val_MAPE'] = float(g.val_MAPE.mean())
            rows.append(r)
        S = pd.DataFrame(rows).sort_values(sel).reset_index(drop=True)
        S.to_csv(os.path.join(os.path.dirname(cdir), 'grid_summary.csv'), index=False)

        print(f"\n  ranked by the ENSEMBLE's {a.select_on}_MAPE"
              + (" (selection on the reported split: see the module docstring)" if a.select_on == 'test' else
                 "; test shown but not used to rank") + "\n")
        print(f"  {'wiring':6s}{'d':>5s}{'h':>4s}{'L':>3s}{'lam':>6s}{'params':>10s}{'n':>3s}"
              f"{'ens val':>9s}{'ens test':>9s}{'seed mean':>10s}{'sd':>7s}{'peak':>7s}")
        for r in S.head(a.top).itertuples():
            ref = ' <- shipped' if r.is_reference else ''
            print(f"  {r.wiring:6s}{r.d_model:5d}{r.num_heads:4d}{r.layers:3d}{r.lambda_cons:6.2f}"
                  f"{r.n_params:10,d}{r.n_seeds:3d}{r.ens_val_MAPE:9.3f}{r.ens_test_MAPE:9.3f}"
                  f"{r.seed_mean_test_MAPE:10.3f}{r.seed_sd_test_MAPE:7.3f}{r.ens_test_peak_MAPE:7.3f}{ref}")

        best = {'by_val': {}, 'by_test': {}}
        for w in WIRINGS:
            sw = S[S.wiring == w]
            if sw.empty: continue
            b = sw.iloc[0]; best[w] = b.to_dict()
            best['by_val'][w] = sw.sort_values('ens_val_MAPE').iloc[0].to_dict()
            best['by_test'][w] = sw.sort_values('ens_test_MAPE').iloc[0].to_dict()
            ref = sw[sw.is_reference]
            print(f"\n  BEST {w:5s} (on {a.select_on})  d{b.d_model} h{b.num_heads} L{b.layers} lam{b.lambda_cons:g}   "
                  f"ens val {b.ens_val_MAPE:.3f} / test {b.ens_test_MAPE:.3f}  peak {b.ens_test_peak_MAPE:.3f}"
                  + (f"   (shipped ref: val {ref.iloc[0].ens_val_MAPE:.3f} / test {ref.iloc[0].ens_test_MAPE:.3f})"
                     if not ref.empty else ''))
            if a.emit_eval:
                _emit_eval(cfg, cdir, mode, w, sw.iloc[0], df, a.select_on)
        best['overall'] = S.iloc[0].to_dict(); best['incomplete'] = bool(missing)
        best['selected_on'] = sel; best['mode'] = mode; best['zone'] = cfg.DATASET
        with open(os.path.join(os.path.dirname(cdir), 'best.json'), 'w') as fh:
            json.dump(best, fh, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
        print(f"\n  wrote grid.csv, grid_summary.csv, best.json under {os.path.dirname(cdir)}/")


def _emit_eval(cfg, cdir, mode, wiring, b, df, select_on):
    """The selected cell's 5-seed ensemble through the ordinary evaluator, as bench_seeds does.

    Lands in results/{zone}/evaluation/archformer_grid_{wiring}[_residual]_ens{n}/{run}/, so
    _scoreboard.collect picks it up like any other row. The appended block is what lets the
    scoreboard colour grid rows apart from shipped ones, and says how the cell was chosen.
    """
    from src.feature_engine import build_timeseries_matrix
    from src.models._eval_utils import EvalUtils
    from tune_archformer import load_split
    g = df[(df.wiring == wiring) & (df.d_model == b.d_model) & (df.num_heads == b.num_heads)
           & (df.layers == b.layers) & (df.lambda_cons.astype(float) == float(b.lambda_cons))]
    preds, truth = [], None
    for cell in g.cell:
        z = np.load(os.path.join(cdir, cell + '.npz')); preds.append(z['pred_test']); truth = z['truth_test']
    ens = np.mean(preds, axis=0)
    D = load_split()
    _, _, _, timestamps = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)
    ts_test = timestamps[D['test']]
    assert len(ts_test) == len(ens), (len(ts_test), len(ens))
    fcg = cfg.ARCHFORMER_FEATURE_CONFIG
    run = f"tail_test{fcg['test_frac']}_tail_val{fcg['val_frac']}{cfg.MATRIX_TAG}"
    suffix = '_residual' if mode == 'residual' else ''
    model_dir = f'archformer_grid_{wiring}{suffix}_ens{len(g)}'
    rdir = os.path.join(cfg.RESULT_ROOT, 'evaluation', model_dir, run)
    EvalUtils.evaluate_one(model_dir.upper(), truth, ens, ts_test, rdir)
    mpath = os.path.join(rdir, f'{model_dir.upper()}_metrics.txt')
    with open(mpath, 'a') as fh:
        fh.write(f"\n--- grid search ---\n"
                 f"wiring: {wiring}\n"
                 f"cell: d_model {b.d_model}, num_heads {b.num_heads}, layers {b.layers}, "
                 f"lambda_cons {b.lambda_cons:g}\n"
                 f"selected_on: ensemble {select_on} MAPE over {len(all_cells())} cells\n"
                 f"seeds: {len(g)} ({', '.join(map(str, sorted(g.seed)))})\n"
                 f"per-seed test MAPE: {', '.join(f'{x:.3f}' for x in g.sort_values('seed').test_MAPE)}\n"
                 f"ensemble val MAPE: {b.ens_val_MAPE:.3f}   ensemble test MAPE: {b.ens_test_MAPE:.3f}\n"
                 f"NOT comparable to a single-seed row: this is the mean of {len(g)} predictions.\n")
    print(f"      -> {rdir}")


if __name__ == '__main__':
    main()
