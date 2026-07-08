"""Train a standalone vanilla transformer on ONE MoE expert's slice.

Example: MOE_SUMMER_LOW = summer days (Jun/Jul/Aug), predicting only hours 0-5.
This isolates how well a dedicated single-regime model does, as a reference point
for that expert inside the full MoE.

The split matches the MoE exactly (global tail 0.16/0.19 on ALL days, then
filtered to the expert's season), so the test days are the same ones the MoE's
per-expert eval reports on — directly comparable.

Pipeline: build matrix -> filter to (season, hours) -> train -> predict ->
evaluate (full EvalUtils suite, restricted to the expert's hours) -> save.

The model, LDS, FDS and 2-stage calibration are all driven by TRANSFORMER_PARAMS
in src/config.py (use_lds / use_fds / stage2_epochs / stage2_mode / ...), exactly
like the vanilla transformer. Toggle them there.

Usage:
  python test/test_single_expert.py                          # summer / low
  python test/test_single_expert.py --season winter --expert peak
  python test/test_single_expert.py --season summer --expert low --epochs 200
  PJM_DATASET=dom python test/test_single_expert.py          # pick dataset

Results -> results/<dataset>/single_expert/<season>__<expert>/
Model   -> models/<dataset>/single_expert/<season>__<expert>/best.pth
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as cfg
from src.feature_engine import build_timeseries_matrix, _split_indices, apply_embargo
from src.models.moe_transformer import season_indices
from src.models.transformer import TimeSeriesTransformer3D, predict as tr_predict
from src.models._eval_utils import EvalUtils
from src.models._lds import compute_lds_weights
from src.models._fds import FDSModule
from src.models._seq_trainer import run_stage2


def _season_filter(idx, season_of_all, season_i):
    """Keep only the indices whose day belongs to the target season."""
    return idx[season_of_all[idx] == season_i]


def _train(Xtr, ytr, Xva, yva, params, device, save_path):
    """Stage-1 (L1 + optional LDS/FDS) with early stopping, then optional Stage-2.

    LDS / FDS / 2-stage are all toggled by the transformer config
    (use_lds / use_fds / stage2_epochs), mirroring src/models/_seq_trainer.py.
    """
    use_lds = params.get('use_lds', False)
    use_fds = params.get('use_fds', False)

    model = TimeSeriesTransformer3D(Xtr.shape[2], params).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'],
                            weight_decay=params['weight_decay'])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=10)

    # --- LDS per-day sample weights (representative = mean scaled load) ---
    if use_lds:
        day_rep = ytr.mean(axis=1)
        w = compute_lds_weights(
            day_rep, bin_width=params.get('lds_bin_width', 200.0),
            ks=params.get('lds_ks', 5), sigma=params.get('lds_sigma', 2.0),
            min_freq_ratio=params.get('lds_min_freq_ratio', 0.05))
        print(f"LDS: enabled | weight range [{w.min():.3f}, {w.max():.3f}]")
    else:
        w = np.ones(len(ytr), dtype=np.float32)

    X_tr = torch.FloatTensor(Xtr)
    y_tr = torch.FloatTensor(ytr)
    loader = DataLoader(TensorDataset(X_tr, y_tr, torch.FloatTensor(w)),
                        batch_size=params['batch_size'], shuffle=True)
    Xva_t = torch.FloatTensor(Xva).to(device)
    yva_t = torch.FloatTensor(yva).to(device)

    # --- FDS module (calibrates the encoder representation) ---
    fds, fds_start = None, 0
    if use_fds:
        fds_start = params.get('fds_start_epoch', 5)
        fds = FDSModule(feature_dim=params['d_model'], bin_width=params.get('fds_bin_width', 200.0),
                        ks=params.get('fds_ks', 5), sigma=params.get('fds_sigma', 2.0),
                        momentum=params.get('fds_momentum', 0.1))
        print(f"FDS: enabled | feat_dim={params['d_model']} | start_epoch={fds_start}")

    train_crit = nn.L1Loss(reduction='none')
    val_crit = nn.L1Loss()
    patience = params.get('early_stop_patience', 50)
    best, no_improve = float('inf'), 0

    # ---------------- Stage 1 — L1 + LDS + FDS ---------------- #
    for ep in range(params['epochs']):
        model.train(); tl = 0.0
        for bx, by, bw in loader:
            bx, by, bw = bx.to(device), by.to(device), bw.to(device)
            opt.zero_grad()
            if use_fds:
                features_raw = model.encode(bx)
                by_rep = by.mean(dim=1)
                if ep >= fds_start and fds._ready:
                    features = fds.calibrate(features_raw, by_rep)
                else:
                    features = features_raw
                pred = model.decode(features)
            else:
                pred = model(bx)
            loss = (train_crit(pred, by) * bw.unsqueeze(1)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            if use_fds:
                fds.collect(features_raw, by_rep)
        if use_fds:
            fds.update_and_smooth()

        model.eval()
        with torch.no_grad():
            vl = val_crit(model(Xva_t), yva_t).item()
        sch.step(vl)
        if vl < best:
            best, no_improve = vl, 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
        if (ep + 1) % 10 == 0:
            tag = ''
            if use_fds:
                tag = f' | FDS {"active" if (fds._ready and ep >= fds_start) else "warmup"}'
            print(f"Epoch {ep+1:03d} | Train {tl/len(loader):.4f} | Val {vl:.4f} "
                  f"| LR {opt.param_groups[0]['lr']:.6f}{tag}")
        if no_improve >= patience:
            print(f"Early stopping at epoch {ep+1}")
            break
    print(f"Best Stage-1 val L1: {best:.4f} -> {save_path}")

    # ---------------- Stage 2 — pluggable calibration of fc_out ---------------- #
    run_stage2(model, X_tr, y_tr, params, device, save_path)   # head_key='fc_out', season=None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', default='summer', choices=list(cfg.REGIME_MAP))
    ap.add_argument('--expert', default='low', help="expert name within the season (low/peak/high)")
    ap.add_argument('--epochs', type=int, default=cfg.TRANSFORMER_PARAMS['epochs'])
    ap.add_argument('--patience', type=int, default=cfg.TRANSFORMER_PARAMS['early_stop_patience'])
    ap.add_argument('--lr', type=float, default=cfg.TRANSFORMER_PARAMS['learning_rate'])
    args = ap.parse_args()

    season, expert = args.season, args.expert
    if expert not in cfg.REGIME_MAP[season]:
        raise SystemExit(f"expert '{expert}' not in season '{season}': "
                         f"choose {list(cfg.REGIME_MAP[season])}")
    hours = cfg.REGIME_MAP[season][expert]
    name = f'EXPERT_{season}_{expert}'.upper()
    print(f"=== {name} | dataset={cfg.DATASET} | hours={hours} ({len(hours)}h) ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X, y, mask, ts = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)
    season_of_all = season_indices(ts)
    season_i = cfg.SEASON_ORDER.index(season)

    # Same split as the MoE: global tail 0.16/0.19, then filter to this season.
    fc = cfg.MOE_TRANSFORMER_FEATURE_CONFIG
    train_pool, test_idx = _split_indices(len(X), fc['split_strategy'], fc['test_frac'], fc['random_state'])
    rtr, rval = _split_indices(len(train_pool), fc['val_strategy'], fc['val_frac'], fc['random_state'])
    train_idx, val_idx = train_pool[rtr], train_pool[rval]
    train_idx, val_idx = apply_embargo(train_idx, val_idx, cfg.EMBARGO_DAYS)   # match MoE embargo

    tr_i = _season_filter(train_idx, season_of_all, season_i)
    tr_i = tr_i[mask[tr_i]]                                  # denoise training days only
    va_i = _season_filter(val_idx, season_of_all, season_i)
    te_i = _season_filter(test_idx, season_of_all, season_i)
    trpool_i = _season_filter(train_pool, season_of_all, season_i)   # for the train subplot
    print(f"{season} days -> train {len(tr_i)} | val {len(va_i)} | test {len(te_i)}")
    if len(tr_i) == 0 or len(va_i) == 0 or len(te_i) == 0:
        raise SystemExit("Empty train/val/test for this season under the current split.")

    # Dedicated transformer: out_dim = number of hours this expert owns.
    params = dict(cfg.TRANSFORMER_PARAMS)
    params.update(out_dim=len(hours), epochs=args.epochs,
                  early_stop_patience=args.patience, learning_rate=args.lr)

    def hslice(idx):
        return y[idx][:, hours]                              # (N, n_hours), scaled

    save_dir = os.path.join('models', cfg.DATASET, 'single_expert', f'{season}__{expert}')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best.pth')

    print("\n--- Training ---")
    _train(X[tr_i], hslice(tr_i), X[va_i], hslice(va_i), params, device, save_path)

    # --- Predict + inverse-transform to MW ---
    y_scaler = joblib.load(glob.glob(os.path.join(cfg.MATRIX_DIR, 'y_scaler_*.pkl'))[0])

    def inv(a):
        return y_scaler.inverse_transform(a.flatten().reshape(-1, 1)).reshape(a.shape)

    def scatter(a):
        """(N, n_hours) compact -> (N, 24) with values placed at the expert's hours."""
        full = np.zeros((len(a), 24), dtype=float)
        full[:, hours] = a
        return full

    pred_te = inv(tr_predict(save_path, X[te_i], params))
    true_te = inv(hslice(te_i))
    pred_tr = inv(tr_predict(save_path, X[trpool_i], params))
    true_tr = inv(hslice(trpool_i))

    # --- Evaluate on the expert's hours (full EvalUtils suite) ---
    result_dir = os.path.join('results', cfg.DATASET, 'single_expert', f'{season}__{expert}')
    train_df = EvalUtils.build_detailed_df(name, scatter(true_tr), scatter(pred_tr),
                                           ts[trpool_i], hours=hours)
    EvalUtils.evaluate_one(name, scatter(true_te), scatter(pred_te), ts[te_i],
                           result_dir, train_df=train_df, hours=hours)
    print(f"\nResults saved to: {result_dir}")
    print(f"Model saved to:   {save_path}")


if __name__ == '__main__':
    main()
