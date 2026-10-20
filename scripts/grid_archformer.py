"""One GARCHFormer grid cell on the PJM point-forecast task. Trains, scores val and test, writes
one JSON (metrics) and one NPZ (predictions), exits.

    PJM_DATASET=dom PJM_CONGESTION=1 PJM_LMP=0 python scripts/grid_archformer.py \
        --mode residual --wiring skip --d-model 64 --num-heads 16 --layers 2 --lambda-cons 0.1 --seed 0

The same 64-cell grid as src/bench/garchformer.py (d_model x num_heads x layers x lambda_cons),
run TWICE on this repo's tensor:

    hp     every channel through the patch tokenizer -- ARCHFormer3 as the author wired it,
           with this dataset's training recipe (ARCHFORMER_HP_PARAMS)
    skip   the 22 time-varying channels through the tokenizer, the 54 broadcast statics fused
           at the decoder output -- _SkipARCHFormer3 (ARCHFORMER_SKIP_PARAMS)

Everything but the four grid values and the wiring is the shipped hp recipe: huber / 5e-4 /
wd 1e-4 / dropout 0.2, 50 epochs, patience 15, patch_len 24, intermediate_dim 32. One extra
REFERENCE cell per wiring, 64 / 4 heads / 2+2 / 0.1, is the configuration currently shipped;
num_heads=4 is not in the grid, and without that row the grid optimum could not be read
against the scoreboard.

Data, split, targets and the training loop are scripts/tune_archformer.py's, imported --
load_split (tail-0.16 test, 0.1 val, 14-day embargo), targets_for (direct / residual) and
train_one (seeded, early stopping on val, chunked predict). Nothing there is modified; the
val score is computed here from the returned network.

Both validation and test PREDICTIONS are saved (MW, float32), not just metrics, because the
comparison the collector reports is over 5 seeds AVERAGED AS AN ENSEMBLE -- the mean of the
five predictions, which is what wins on PJM (scripts/bench_seeds.py) and what would be
shipped -- alongside the mean and sd of the single-seed metrics. An ensemble cannot be formed
from metrics after the fact.

No model weights are written. Output is one cell per file,

    results/{zone}/_tuning/archgrid_{mode}{arm}/cells/{wiring}_d{d}_h{h}_L{l}_lam{..}_s{seed}.{json,npz}

under _tuning/ and not evaluation/, so _scoreboard.collect never picks a grid row up as a
shipped run. An existing cell is skipped (exit 0), so a partial run is resumed by resubmitting.
"""

import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT))
_sys.path.insert(0, str(_ROOT / 'scripts'))      # tune_archformer lives beside this file

import argparse
import json
import os

WIRINGS = ('hp', 'skip')
MODES = ('direct', 'residual')
REFERENCE = {'d_model': 64, 'num_heads': 4, 'layers': 2, 'lambda_cons': 0.1}   # shipped


def all_cells():
    """The bench grid plus the shipped reference, in a stable order."""
    from src.bench.garchformer import cells          # ONE grid definition, shared with the bench
    out = list(cells())
    if REFERENCE not in out:
        out.append(dict(REFERENCE))
    return out


def lam_tag(lam):
    return 'lam' + f'{float(lam):g}'.replace('.', 'p').replace('-', 'm')


def cell_tag(wiring, cell, seed):
    return (f"{wiring}_d{int(cell['d_model'])}_h{int(cell['num_heads'])}"
            f"_L{int(cell['layers'])}_{lam_tag(cell['lambda_cons'])}_s{int(seed)}")


def out_dir_for(cfg, mode):
    return os.path.join(cfg.RESULT_ROOT, '_tuning', f'archgrid_{mode}{cfg.MATRIX_TAG}', 'cells')


def params_for(cfg, wiring, cell):
    base = cfg.ARCHFORMER_HP_PARAMS if wiring == 'hp' else cfg.ARCHFORMER_SKIP_PARAMS
    return {**base,
            'd_model': int(cell['d_model']), 'num_heads': int(cell['num_heads']),
            'num_encoder_layers': int(cell['layers']), 'num_decoder_layers': int(cell['layers']),
            'lambda_cons': float(cell['lambda_cons'])}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', required=True, choices=MODES)
    ap.add_argument('--wiring', required=True, choices=WIRINGS)
    ap.add_argument('--d-model', type=int, required=True)
    ap.add_argument('--num-heads', type=int, required=True)
    ap.add_argument('--layers', type=int, required=True)
    ap.add_argument('--lambda-cons', type=float, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=None, help='override the recipe (smoke tests)')
    ap.add_argument('--out', default=None, help='override the results root (smoke tests)')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    if a.d_model % a.num_heads:
        print(f"ILLEGAL CELL: d_model={a.d_model} % num_heads={a.num_heads} != 0")
        return 2

    import src.config as cfg
    cell = {'d_model': a.d_model, 'num_heads': a.num_heads,
            'layers': a.layers, 'lambda_cons': a.lambda_cons}
    out_dir = out_dir_for(cfg, a.mode) if a.out is None else \
        os.path.join(a.out, f'archgrid_{a.mode}{cfg.MATRIX_TAG}', 'cells')
    os.makedirs(out_dir, exist_ok=True)
    tag = cell_tag(a.wiring, cell, a.seed)
    path = os.path.join(out_dir, tag + '.json')
    if os.path.exists(path) and os.path.exists(path[:-5] + '.npz') and not a.force:
        print(f"already done, skipping: {path}")
        return 0

    import numpy as np
    import torch
    from tune_archformer import load_split, targets_for, train_one, metrics
    from src.models.archformer import ARCHFormerNet
    torch.set_num_threads(1)         # one thread per cell; the sbatch runs eight cells at once

    D = load_split()
    y_space, to_mw, truth_mw = targets_for(a.mode, D)
    params = params_for(cfg, a.wiring, cell)
    epochs = a.epochs or params['epochs']
    print(f"=== archgrid | zone {cfg.DATASET} arm {cfg.MATRIX_TAG!r} | {a.mode} {a.wiring} | "
          f"d{a.d_model} h{a.num_heads} L{a.layers} lam{a.lambda_cons:g} seed {a.seed} | "
          f"C={D['X'].shape[2]} n_seq={params.get('n_seq_features', 'all')} | "
          f"train {len(D['train'])} val {len(D['val'])} test {len(D['test'])}")

    m_test, pred_test, net = train_one(ARCHFormerNet, params, D, y_space, to_mw, truth_mw,
                                       epochs, seed=a.seed, return_pred=True, return_net=True)

    # Validation score, same metrics, from the network train_one hands back (best epoch loaded).
    net.eval()
    with torch.no_grad():
        pv = np.concatenate([net(torch.FloatTensor(D['X'][D['val'][i:i + 256]])).numpy()
                             for i in range(0, len(D['val']), 256)], axis=0)
    pred_val = to_mw(pv, D['val'])
    m_val = metrics(truth_mw[D['val']], pred_val)

    row = {'zone': cfg.DATASET, 'arm': cfg.MATRIX_TAG, 'mode': a.mode, 'wiring': a.wiring,
           **cell, 'seed': a.seed, 'cell': tag, 'is_reference': cell == REFERENCE,
           'n_params': m_test['n_params'], 'best_epoch': m_test['best_epoch'],
           'seconds': m_test['seconds'], 'epochs_budget': epochs,
           **{f'val_{k}': float(v) for k, v in m_val.items()},
           **{f'test_{k}': float(v) for k, v in m_test.items()
              if k not in ('n_params', 'best_epoch', 'seconds')}}

    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(row, fh, indent=1, sort_keys=True)
    np.savez_compressed(path[:-5] + '.npz.tmp.npz',
                        pred_val=pred_val.astype('float32'), pred_test=pred_test.astype('float32'),
                        truth_val=truth_mw[D['val']].astype('float32'),
                        truth_test=truth_mw[D['test']].astype('float32'))
    os.replace(path[:-5] + '.npz.tmp.npz', path[:-5] + '.npz')
    os.replace(tmp, path)

    print(f"  {row['n_params']:,} params | epochs {row['best_epoch']}/{epochs} | {row['seconds']} s")
    print(f"  val  MAPE {m_val['MAPE']:.3f}  peak {m_val['peak_MAPE']:.3f}  MAE {m_val['MAE']:.0f}")
    print(f"  test MAPE {m_test['MAPE']:.3f}  peak {m_test['peak_MAPE']:.3f}  MAE {m_test['MAE']:.0f}")
    print(f"  -> {path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
