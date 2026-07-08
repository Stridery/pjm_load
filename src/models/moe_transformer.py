"""Mixture-of-Experts Transformer for regime-aware 24h load forecasting.

Design (hard routing + shared encoder + regime expert heads):
  - A single Transformer encoder (identical to the vanilla transformer) turns the
    168h lookback sequence into one day representation z (d_model).
  - The output head is a Mixture-of-Experts. Routing is two-axis and fully
    deterministic (no gating network):
        season  -> per-sample, from the target day's month  (SEASON_ORDER)
        hour    -> per-output-index, fixed ownership          (REGIME_MAP)
    Each expert = (season group, a fixed set of hour indices). The 8 experts
    (summer x3, winter x3, shoulder x2) each specialise in one physical regime.
  - One forward pass trains all experts: for a day of known season, its experts
    each predict their owned hours; results scatter into the 24-vector; the usual
    L1 loss on 24 hours routes each hour's gradient to its expert + the shared
    encoder.

The vanilla transformer / LSTM are untouched; this lives in its own file with a
dedicated (simplified) trainer because it needs the per-sample season route,
which the shared _seq_trainer signature (X, y, w) cannot carry.
"""

import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.feature_engine import _split_indices, apply_embargo
from src.config import EMBARGO_DAYS
from src.models._utils import _make_run_dir
from src.models._eval_utils import EvalUtils
from src.models._lds import compute_lds_weights
from src.models._fds import FDSModule
from src.models._seq_trainer import run_stage2
from src.config import (
    REGIME_MAP, SEASON_ORDER, MONTH_TO_SEASON,
    MOE_TRANSFORMER_PARAMS, MOE_TRANSFORMER_FEATURE_CONFIG,
)


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def season_indices(timestamps):
    """Map an array of daily timestamps -> season route index in SEASON_ORDER."""
    months = pd.to_datetime(timestamps).month
    return np.array(
        [SEASON_ORDER.index(MONTH_TO_SEASON[int(m)]) for m in months],
        dtype=np.int64,
    )


def _validate_regime_map(out_dim):
    """Each season's hour lists must be disjoint and tile 0..out_dim-1."""
    for season, experts in REGIME_MAP.items():
        hours = [h for hrs in experts.values() for h in hrs]
        if sorted(hours) != list(range(out_dim)):
            raise ValueError(
                f"REGIME_MAP['{season}'] hours {sorted(hours)} do not tile "
                f"0..{out_dim - 1} exactly (check for gaps/overlaps)."
            )


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def _expert_head(d_model, fc_hidden, n_out, dropout):
    return nn.Sequential(
        nn.Linear(d_model, fc_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc_hidden, n_out),
    )


class MoETransformer(nn.Module):
    def __init__(self, num_features, params):
        super().__init__()
        d_model   = params['d_model']
        out_dim   = params['out_dim']
        fc_hidden = params.get('expert_fc_hidden', 64)
        dropout   = params['dropout']
        self.out_dim = out_dim
        _validate_regime_map(out_dim)

        # --- shared encoder (same as vanilla transformer) ---
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=params['nhead'],
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=params['num_layers'])

        # --- expert heads, grouped by season; fixed iteration order per season ---
        self.experts = nn.ModuleDict()
        self._season_expert_names = {}     # season -> [expert names] (build order)
        for season in SEASON_ORDER:
            names = list(REGIME_MAP[season].keys())
            self._season_expert_names[season] = names
            concat_hours = []
            for name in names:
                hrs = REGIME_MAP[season][name]
                self.experts[f'{season}__{name}'] = _expert_head(d_model, fc_hidden, len(hrs), dropout)
                concat_hours.extend(hrs)
            # inv_perm[h] = column of the season's concatenated expert output holding hour h
            inv_perm = [concat_hours.index(h) for h in range(out_dim)]
            self.register_buffer(f'inv_perm_{season}', torch.tensor(inv_perm, dtype=torch.long))

    def encode(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x[:, -1, :]                       # (batch, d_model)

    def _season_forward(self, z, season):
        """Full 24-vector prediction as if the whole batch were this season."""
        cols = [self.experts[f'{season}__{n}'](z) for n in self._season_expert_names[season]]
        cat = torch.cat(cols, dim=1)             # (batch, out_dim) in expert order
        return cat[:, getattr(self, f'inv_perm_{season}')]   # reorder to hour 0..out_dim-1

    def decode(self, z, season_idx):
        # Compute every season's head, then hard-select each sample's own season.
        # Unselected season outputs receive no gradient (sliced away from the loss).
        stacked = torch.stack([self._season_forward(z, s) for s in SEASON_ORDER], dim=0)  # (S,B,24)
        batch_ar = torch.arange(z.size(0), device=z.device)
        return stacked[season_idx, batch_ar]     # (batch, out_dim)

    def forward(self, x, season_idx):
        return self.decode(self.encode(x), season_idx)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X_3d, y_3d, mask_3d, timestamps_3d, params=None, feature_cfg=None, dataset=None):
    """Dedicated trainer: same split/LDS/early-stop scheme as _seq_trainer, but
    threads the per-sample season route through the DataLoader and forward pass."""
    print("\n--- Training MoE Transformer ---")
    params      = params or MOE_TRANSFORMER_PARAMS
    feature_cfg = feature_cfg or MOE_TRANSFORMER_FEATURE_CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GPU Acceleration: {device}")

    use_lds   = params.get('use_lds', False)
    model_dir = _make_run_dir('models', 'moe_transformer', feature_cfg, dataset, use_lds=use_lds)
    save_path = os.path.join(model_dir, 'moe_transformer_best.pth')

    strategy     = feature_cfg['split_strategy']
    test_frac    = feature_cfg['test_frac']
    val_strategy = feature_cfg['val_strategy']
    val_frac     = feature_cfg['val_frac']
    random_state = feature_cfg['random_state']

    season_all = season_indices(timestamps_3d)

    train_pool_idx, _test_idx = _split_indices(len(X_3d), strategy, test_frac, random_state)
    rel_train_idx, rel_val_idx = _split_indices(len(train_pool_idx), val_strategy, val_frac, random_state)
    train_idx = train_pool_idx[rel_train_idx]
    val_idx   = train_pool_idx[rel_val_idx]
    train_idx, val_idx = apply_embargo(train_idx, val_idx, EMBARGO_DAYS)   # 2-week gap at split boundaries

    # denoise training samples via validity mask
    train_mask = mask_3d[train_idx]
    X_tr_np = X_3d[train_idx][train_mask]
    y_tr_np = y_3d[train_idx][train_mask]
    s_tr_np = season_all[train_idx][train_mask]
    print(f"Train after denoising: {len(X_tr_np)} | Val: {len(val_idx)} | Test: {len(_test_idx)}")

    # --- LDS per-day sample weights (same as _seq_trainer) ---
    if use_lds:
        day_rep = y_tr_np.mean(axis=1)
        lds_w = compute_lds_weights(
            day_rep,
            bin_width      = params.get('lds_bin_width', 200.0),
            ks             = params.get('lds_ks', 5),
            sigma          = params.get('lds_sigma', 2.0),
            min_freq_ratio = params.get('lds_min_freq_ratio', 0.05),
        )
        print(f"LDS: enabled | weight range [{lds_w.min():.3f}, {lds_w.max():.3f}]")
    else:
        lds_w = np.ones(len(y_tr_np), dtype=np.float32)

    X_tr = torch.FloatTensor(X_tr_np)
    y_tr = torch.FloatTensor(y_tr_np)
    s_tr = torch.LongTensor(s_tr_np)
    w_tr = torch.FloatTensor(lds_w)

    X_val = torch.FloatTensor(X_3d[val_idx]).to(device)
    y_val = torch.FloatTensor(y_3d[val_idx]).to(device)
    s_val = torch.LongTensor(season_all[val_idx]).to(device)

    loader = DataLoader(TensorDataset(X_tr, y_tr, s_tr, w_tr),
                        batch_size=params['batch_size'], shuffle=True)
    model = MoETransformer(num_features=X_3d.shape[2], params=params).to(device)

    # --- FDS: calibrates the shared encoder representation (feat_dim = d_model) ---
    use_fds   = params.get('use_fds', False)
    fds       = None
    fds_start = 0
    if use_fds:
        fds_start = params.get('fds_start_epoch', 5)
        fds = FDSModule(
            feature_dim=params['d_model'],
            bin_width=params.get('fds_bin_width', 200.0),
            ks=params.get('fds_ks', 5),
            sigma=params.get('fds_sigma', 2.0),
            momentum=params.get('fds_momentum', 0.1),
        )
        print(f"FDS: enabled | feat_dim={params['d_model']} | bin_width={fds.bin_width} "
              f"| start_epoch={fds_start} | momentum={fds.momentum}")

    train_criterion = nn.L1Loss(reduction='none')
    val_criterion   = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'],
                            weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss     = float('inf')
    epochs_no_improve = 0
    patience          = params.get('early_stop_patience', 30)

    # ------------------------------------------------------------------ #
    # Stage 1 — full training with LDS + FDS                              #
    # ------------------------------------------------------------------ #
    for epoch in range(params['epochs']):
        model.train()
        train_loss = 0.0
        for bx, by, bs, bw in loader:
            bx, by, bs, bw = bx.to(device), by.to(device), bs.to(device), bw.to(device)
            optimizer.zero_grad()

            if use_fds:
                # encode / (calibrate) / decode so FDS can intervene on z, then route by season
                features_raw = model.encode(bx)             # (batch, d_model)
                by_rep       = by.mean(dim=1)
                if epoch >= fds_start and fds._ready:
                    features = fds.calibrate(features_raw, by_rep)
                else:
                    features = features_raw
                pred = model.decode(features, bs)
            else:
                pred = model(bx, bs)

            loss = (train_criterion(pred, by) * bw.unsqueeze(1)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

            if use_fds:
                fds.collect(features_raw, by_rep)

        if use_fds:
            fds.update_and_smooth()

        model.eval()
        with torch.no_grad():
            v_loss = val_criterion(model(X_val, s_val), y_val).item()
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0:
            fds_tag = ''
            if use_fds:
                fds_tag = f' | FDS: {"active" if (fds._ready and epoch >= fds_start) else "warmup"}'
            print(f"Epoch {epoch+1:03d} | Train: {train_loss/len(loader):.4f} | "
                  f"Val: {v_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}{fds_tag}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at Epoch {epoch+1}")
            break

    print(f"Best Stage 1 MoE model saved to: {save_path}")

    # ------------------------------------------------------------------ #
    # Stage 2 — pluggable calibration on the expert heads (head_key='experts') #
    # ------------------------------------------------------------------ #
    run_stage2(model, X_tr, y_tr, params, device, save_path, season=s_tr, head_key='experts')


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model_path, X_np, timestamps, params):
    """Load a saved model and return scaled predictions (N, out_dim)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MoETransformer(X_np.shape[2], params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    season_idx = torch.LongTensor(season_indices(timestamps)).to(device)
    with torch.no_grad():
        return model(torch.FloatTensor(X_np).to(device), season_idx).cpu().numpy()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _regime_breakdown(y_true_mw, y_pred_mw, timestamps, result_dir):
    """Per-expert (season x hour-set) MAPE — the regime view the professor wants."""
    from sklearn.metrics import mean_absolute_percentage_error
    season_idx = season_indices(timestamps)
    rows = []
    for s_i, season in enumerate(SEASON_ORDER):
        sel = np.where(season_idx == s_i)[0]
        if len(sel) == 0:
            continue
        for name, hours in REGIME_MAP[season].items():
            yt = y_true_mw[np.ix_(sel, hours)].flatten()
            yp = y_pred_mw[np.ix_(sel, hours)].flatten()
            mape = mean_absolute_percentage_error(yt, yp) * 100
            rows.append({'season': season, 'expert': name,
                         'n_days': len(sel), 'n_hours': len(hours),
                         'MAPE_pct': round(mape, 3)})
    df = pd.DataFrame(rows)
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, 'MOE_TRANSFORMER_regime_mape.csv')
    df.to_csv(path, index=False)
    print("\n====== MoE Regime Breakdown (test) ======")
    print(df.to_string(index=False))
    print(f"Regime MAPE saved to: {path}")


def _predict_mw(model_path, X_np, timestamps, params, y_scaler):
    scaled = predict(model_path, X_np, timestamps, params)
    N, P = scaled.shape
    return y_scaler.inverse_transform(scaled.flatten().reshape(-1, 1)).reshape(N, P)


def _evaluate_experts(y_true_mw, y_pred_mw, timestamps, result_dir,
                      y_true_train_mw=None, y_pred_train_mw=None, timestamps_train=None):
    """Re-run the full evaluation suite for each of the 8 experts, on the days of
    its season restricted to the hours it owns, into experts/<season>__<name>/."""
    season_te = season_indices(timestamps)
    have_train = (y_true_train_mw is not None and y_pred_train_mw is not None
                  and timestamps_train is not None)
    season_tr = season_indices(timestamps_train) if have_train else None

    for s_i, season in enumerate(SEASON_ORDER):
        sel = np.where(season_te == s_i)[0]
        if len(sel) == 0:
            print(f"\n[expert eval] skip season '{season}': no {season} days in this test split")
            continue
        for expert, hours in REGIME_MAP[season].items():
            name    = f'MOE_{season}_{expert}'.upper()
            sub_dir = os.path.join(result_dir, 'experts', f'{season}__{expert}')
            train_df = None
            if have_train:
                selt = np.where(season_tr == s_i)[0]
                if len(selt):
                    train_df = EvalUtils.build_detailed_df(
                        name, y_true_train_mw[selt], y_pred_train_mw[selt],
                        timestamps_train[selt], hours=hours)
            EvalUtils.evaluate_one(name, y_true_mw[sel], y_pred_mw[sel],
                                   timestamps[sel], sub_dir, train_df=train_df, hours=hours)


def evaluate(model_path, X_test, y_true_mw, y_scaler, timestamps, result_dir,
             params=None, X_train=None, y_true_train_mw=None, timestamps_train=None):
    """Predict, inverse-transform, run the full eval suite + per-regime breakdown +
    a full per-expert evaluation (one subfolder per expert)."""
    params = params or MOE_TRANSFORMER_PARAMS
    y_pred_mw = _predict_mw(model_path, X_test, timestamps, params, y_scaler)

    have_train = X_train is not None and y_true_train_mw is not None
    y_pred_train_mw = (_predict_mw(model_path, X_train, timestamps_train, params, y_scaler)
                       if have_train else None)

    train_df = (EvalUtils.build_detailed_df('MOE_TRANSFORMER', y_true_train_mw,
                                            y_pred_train_mw, timestamps_train)
                if have_train else None)

    # --- whole-model evaluation (all 24h) ---
    EvalUtils.evaluate_one('MOE_TRANSFORMER', y_true_mw, y_pred_mw, timestamps, result_dir, train_df)
    _regime_breakdown(y_true_mw, y_pred_mw, timestamps, result_dir)

    # --- per-expert evaluation (each expert's season-days x owned-hours) ---
    _evaluate_experts(y_true_mw, y_pred_mw, timestamps, result_dir,
                      y_true_train_mw, y_pred_train_mw, timestamps_train)
