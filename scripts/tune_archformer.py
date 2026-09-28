"""
ARCHFormer hyperparameter sweep — three groups, both targets modes, any zone/arm.

    PJM_DATASET=dom PJM_CONGESTION=1 PJM_LMP=0 python scripts/tune_archformer.py
    PJM_DATASET=bge PJM_CONGESTION=1 PJM_LMP=0 python scripts/tune_archformer.py --mode residual
    PJM_DATASET=dom PJM_CONGESTION=1 PJM_LMP=0 python scripts/tune_archformer.py --group 2 --epochs 60

Zone and feature arm come from the usual environment variables, like every other driver here;
there is no second spelling of them to get out of sync with src/config.py.

WHY THIS SWEEP EXISTS
---------------------
The architecture's hyperparameters are Moyi Guo's, chosen on datasets of 5-28 feature columns
(his widest config is `spain`, 27). This repo hands it 68-89 channels. PatchTokenizer is a
single Linear(patch_len x C -> d_model), so the compression it has to perform went from

    24 x 18 = 432 -> 64      6.8 : 1      (his pjm_zh config, built for THIS dataset)
    24 x 89 = 2136 -> 64    33.4 : 1      (here)

and 72% of those 2136 inputs are static features repeated once per hour inside the patch --
a kind of column his pipeline does not have at all (utils/data_utils.py reads every column
per row). So the question is not whether his defaults are good; it is whether they were ever
aimed at this shape.

THE THREE GROUPS
----------------
  0  baseline   his architecture exactly as shipped: patch_len 24, d_model 64, no static skip.
  1  pure HP    patch_len x d_model grid. No code change of any kind -- this is the group that
                answers "fine-tune the hyperparameters" literally.
  2  static skip  the time-varying channels go through the tokenizer, the broadcast constants
                get their own out_dim head. The vendored files stay byte-identical; only the
                wrapper changes.
  3  recipe     the TRAINING knobs, one at a time and then together. Four of them differ from
                every other net in this repo, all in the same direction:

                    loss       mse    vs huber
                    lr         2e-3   vs 3e-4      <- 6.7x higher
                    weight_decay 0    vs 1e-4
                    dropout    0.1    vs 0.3       <- 3x less

                They are the author's, and they were reasonable on his data. Here the
                validation curve bounces rather than descends (0.2601 -> 0.2163 -> 0.2008 ->
                0.2299), which is what too high a learning rate looks like, and the model
                peaks at epoch 8 not because it has too many parameters -- group 1 shrank it
                by a third and got WORSE -- but because it stops making progress.

Both TARGET MODES are run unless --mode says otherwise:
  direct    regress the target itself
  residual  regress the deviation from the same-hour-last-week baseline and add it back,
            which is what src/models/archformer_residual.py ships. Identical machinery,
            reproduced here in memory so the sweep writes no weights.

WHAT IT WRITES
--------------
    results/{zone}{TARGET_TAG}/_tuning/archformer_{mode}{MATRIX_TAG}.csv

Deliberately NOT under evaluation/, so _scoreboard.collect's glob never picks a sweep row up
as if it were a shipped run. No model weights are saved at all: a sweep is a measurement, and
the winning configuration is meant to be re-run through scripts/Model_Training.py afterwards.
"""

# Entry points live in scripts/ but import the package from the repo root, and running
# `python scripts/foo.py` puts scripts/ on sys.path rather than the root. Prepend the root so
# `import src...` resolves from any working directory.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import src.config as cfg
from src.feature_engine import _split_indices, apply_embargo
from src.models._residual import sequence_baseline
from src.models._seq_trainer import make_criterion
from src.models.archformer import ARCHFormerNet
from src.models.transformer import TimeSeriesTransformer3D

SEED = 42          # default; --seeds overrides


# ---------------------------------------------------------------------------
# The grids
# ---------------------------------------------------------------------------

def grids(n_seq):
    """(group, label, param-overrides) for every configuration the sweep runs."""
    out = [(0, 'pl=24 d=64  base', {})]
    for pl in (24, 12, 8, 6):
        for d in (64, 128):
            if (pl, d) != (24, 64):
                out.append((1, f'pl={pl:<2d} d={d:<3d} base', {'patch_len': pl, 'd_model': d}))
    for pl in (24, 12, 8, 6):
        for d in (64, 128):
            out.append((2, f'pl={pl:<2d} d={d:<3d} SKIP',
                        {'patch_len': pl, 'd_model': d, 'n_seq_features': n_seq}))
    # One knob at a time, then the repo's full recipe, then a middle setting. Changing them
    # together first would say the recipe matters without saying WHICH part of it does.
    HUBER = {'loss': 'huber', 'huber_delta': 1.0}
    for lab, over in [
        ('huber only',        HUBER),
        ('lr 3e-4 only',      {'learning_rate': 3e-4}),
        ('dropout 0.3 only',  {'dropout': 0.3}),
        ('wd 1e-4 only',      {'weight_decay': 1e-4}),
        ('repo recipe (all4)', {**HUBER, 'learning_rate': 3e-4,
                                'weight_decay': 1e-4, 'dropout': 0.3}),
        ('middle: hub 5e-4',  {**HUBER, 'learning_rate': 5e-4,
                               'weight_decay': 1e-4, 'dropout': 0.2}),
    ]:
        out.append((3, f'{lab:<19s}', over))
    return out


# ---------------------------------------------------------------------------
# Data, shared by every run so the comparison is on one split
# ---------------------------------------------------------------------------

def load_split():
    tag = cfg.MATRIX_TAG + cfg.TARGET_TAG
    d = cfg.MATRIX_DIR
    X = np.load(f'{d}X_3d_lb168_h0{tag}.npy')
    y = np.load(f'{d}y_3d_lb168_h0{tag}.npy')
    mask = np.load(f'{d}mask_3d_lb168_h0{tag}.npy')
    seq_scaler = joblib.load(f'{d}scaler_ts_lb168_h0{tag}.pkl')
    y_scaler = joblib.load(f'{d}y_scaler_lb168_h0{tag}.pkl')

    fcg = cfg.ARCHFORMER_FEATURE_CONFIG
    n = len(X)
    pool, test = _split_indices(n, fcg['split_strategy'], fcg['test_frac'], fcg['random_state'])
    rtr, rval = _split_indices(len(pool), fcg['val_strategy'], fcg['val_frac'], fcg['random_state'])
    train, val = apply_embargo(pool[rtr], pool[rval], cfg.EMBARGO_DAYS)
    return dict(X=X, y=y, mask=mask, seq_scaler=seq_scaler, y_scaler=y_scaler,
                train=train[mask[train]], val=val, test=test)


def targets_for(mode, D):
    """(y_train_space, to_mw) for the mode.

    direct    the net regresses the standardized target; to_mw is the y scaler's inverse.
    residual  the net regresses the standardized DEVIATION from the same-hour-last-week
              baseline, and to_mw undoes the residual scaler and adds the baseline back --
              the identical arrangement src/models/_residual.make_residual_model ships,
              including fitting the residual scaler on the training rows ONLY.
    """
    ysc, y = D['y_scaler'], D['y']
    N, H = y.shape
    y_mw = ysc.inverse_transform(y.reshape(-1, 1)).reshape(N, H)
    if mode == 'direct':
        return y, (lambda p, idx: ysc.inverse_transform(p.reshape(-1, 1)).reshape(p.shape)), y_mw

    base = sequence_baseline(D['X'], D['seq_scaler'], 'hourly')
    res_mw = y_mw - base
    rsc = StandardScaler().fit(res_mw[D['train']].reshape(-1, 1))
    y_res = rsc.transform(res_mw.reshape(-1, 1)).reshape(N, H).astype('float32')

    def to_mw(p, idx):
        r = rsc.inverse_transform(p.reshape(-1, 1)).reshape(p.shape)
        return r + base[idx]

    return y_res, to_mw, y_mw


# ---------------------------------------------------------------------------
# Metrics — the scoreboard's, recomputed here so nothing is written to disk
# ---------------------------------------------------------------------------

def metrics(truth, pred):
    t, p = truth.ravel(), pred.ravel()
    thr = np.quantile(t, 0.9)
    k = t >= thr
    bins = pd.qcut(t, 10, labels=False, duplicates='drop')
    brs = np.polyfit(pd.Series(t).groupby(bins).mean().values,
                     pd.Series(p - t).groupby(bins).mean().values, 1)[0]
    return {'MAPE': float(np.mean(np.abs((t - p) / t)) * 100),
            'peak_MAPE': float(np.mean(np.abs((t[k] - p[k]) / t[k])) * 100),
            'MAE': float(np.mean(np.abs(t - p))),
            'RMSE': float(np.sqrt(np.mean((t - p) ** 2))),
            'ME': float(np.mean(p - t)),
            'BRS': float(brs)}


# ---------------------------------------------------------------------------
# One configuration
# ---------------------------------------------------------------------------

def train_one(Cls, params, D, y_train_space, to_mw, truth_mw, epochs, seed=SEED,
              return_pred=False, return_net=False):
    """Train once and score the test split.

    `return_pred` also hands back the test prediction in target units, which is what
    scripts/bench_seeds.py averages into an ensemble -- metrics alone cannot be averaged into one.
    `return_net` hands back the trained network, with the best checkpoint already loaded, so a
    caller can inspect internals that no metric captures -- scripts/bench_smoe.py reads the gate
    weights out of it. No weights reach disk either way."""
    torch.manual_seed(seed)
    net = Cls(D['X'].shape[2], params)
    opt = torch.optim.AdamW(net.parameters(), lr=params['learning_rate'],
                            weight_decay=params['weight_decay'])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=10)
    crit, vcrit = make_criterion(params), make_criterion(params)

    Xt = torch.FloatTensor(D['X'][D['train']]); Yt = torch.FloatTensor(y_train_space[D['train']])
    Xv = torch.FloatTensor(D['X'][D['val']]);   Yv = torch.FloatTensor(y_train_space[D['val']])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, Yt),
                                         batch_size=params['batch_size'], shuffle=True)
    best, state, best_ep, bad = np.inf, None, 0, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        for bx, by in loader:
            opt.zero_grad()
            out = net(bx)
            loss = crit(out, by)
            if hasattr(net, 'aux_loss'):
                loss = loss + net.aux_loss(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = vcrit(net(Xv), Yv).item()
        sch.step(v)
        if v < best:
            best, state, best_ep, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, ep, 0
        else:
            bad += 1
            if bad >= params['early_stop_patience']:
                break
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        # Chunked: the decoder holds (B, L, d_model) activations per layer and the whole test
        # split at once is a needless spike on a CPU node.
        p = np.concatenate([net(torch.FloatTensor(D['X'][D['test'][i:i + 256]])).numpy()
                            for i in range(0, len(D['test']), 256)], axis=0)
    m = metrics(truth_mw[D['test']], to_mw(p, D['test']))
    m.update(best_epoch=best_ep, n_params=sum(q.numel() for q in net.parameters()),
             seconds=round(time.time() - t0))
    extras = ()
    if return_pred:
        extras += (to_mw(p, D['test']),)
    if return_net:
        extras += (net,)
    return (m,) + extras if extras else m


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', nargs='*', default=['direct', 'residual'],
                    choices=['direct', 'residual'])
    ap.add_argument('--group', nargs='*', type=int, default=[0, 1, 2, 3],
                    choices=[0, 1, 2, 3])
    ap.add_argument('--epochs', type=int, default=cfg.ARCHFORMER_PARAMS['epochs'])
    ap.add_argument('--seeds', nargs='*', type=int, default=[SEED],
                    help="run every configuration once per seed and report mean +/- sd. The "
                         "effect sizes being chased here (~0.1 pp) are the size of the seed "
                         "spread, so a single run cannot separate them.")
    ap.add_argument('--reference', action='store_true',
                    help="also train the vanilla transformer, as the number to beat")
    args = ap.parse_args()

    D = load_split()
    C, n_seq = D['X'].shape[2], cfg.N_SEQ_FEATURES
    out_dir = os.path.join(cfg.RESULT_ROOT, '_tuning')
    os.makedirs(out_dir, exist_ok=True)
    print(f"zone {cfg.DATASET}{cfg.TARGET_TAG}  arm {cfg.MATRIX_TAG!r}  C={C} (n_seq {n_seq}, "
          f"static {C - n_seq})  train {len(D['train'])} val {len(D['val'])} test {len(D['test'])}")
    print(f"epochs {args.epochs}  groups {args.group}  modes {args.mode}\n")

    todo = [g for g in grids(n_seq) if g[0] in args.group]
    for mode in args.mode:
        y_space, to_mw, truth_mw = targets_for(mode, D)
        print(f"{'=' * 104}\nMODE {mode.upper()}\n{'=' * 104}", flush=True)
        rows = []
        if args.reference:
            m = train_one(TimeSeriesTransformer3D, cfg.TRANSFORMER_PARAMS, D, y_space, to_mw,
                          truth_mw, cfg.TRANSFORMER_PARAMS['epochs'])
            rows.append({'group': -1, 'config': 'transformer (reference)', **m})
            print(f"  {'transformer (reference)':24s} MAPE {m['MAPE']:6.3f}  peak {m['peak_MAPE']:6.3f}"
                  f"  MAE {m['MAE']:7.1f}  BRS {m['BRS']:+.4f}  ep{m['best_epoch']:3d}"
                  f"  {m['n_params']:9,d}p  {m['seconds']:4d}s", flush=True)
        for group, label, over in todo:
            per_seed = []
            for sd in args.seeds:
                m = train_one(ARCHFormerNet, {**cfg.ARCHFORMER_PARAMS, **over}, D, y_space,
                              to_mw, truth_mw, args.epochs, seed=sd)
                rows.append({'group': group, 'config': label, 'seed': sd, **m})
                per_seed.append(m)
            mp = np.array([m['MAPE'] for m in per_seed])
            pk = np.array([m['peak_MAPE'] for m in per_seed])
            spread = (f"  +/-{mp.std(ddof=1):5.3f}" if len(mp) > 1 else "")
            print(f"  [{group}] {label:20s} MAPE {mp.mean():6.3f}{spread}  peak {pk.mean():6.3f}"
                  f"  MAE {np.mean([m['MAE'] for m in per_seed]):7.1f}"
                  f"  BRS {np.mean([m['BRS'] for m in per_seed]):+.4f}"
                  f"  ep{int(np.mean([m['best_epoch'] for m in per_seed])):3d}"
                  f"  n={len(mp)}", flush=True)

        df = pd.DataFrame(rows).sort_values('MAPE').reset_index(drop=True)
        path = os.path.join(out_dir, f'archformer_{mode}{cfg.MATRIX_TAG}.csv')
        df.to_csv(path, index=False)
        print(f"\n  best by MAPE: {df.iloc[0]['config']}  ({df.iloc[0]['MAPE']:.3f} %)")
        best_peak = df.sort_values('peak_MAPE').iloc[0]
        print(f"  best by peak: {best_peak['config']}  ({best_peak['peak_MAPE']:.3f} %)")
        print(f"  -> {path}\n", flush=True)


if __name__ == '__main__':
    main()
