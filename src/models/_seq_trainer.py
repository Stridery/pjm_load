"""Shared training loop for all sequence models (Transformer, LSTM)."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.feature_engine import _split_indices
from src.models._utils import _make_run_dir
from src.models._lds import compute_lds_weights
from src.models._fds import FDSModule


def train_sequence(model_cls, model_type_name, save_name,
                   X_3d, y_3d, mask_3d, params, feature_cfg, dataset=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nGPU Acceleration: {device}")

    use_lds = params.get('use_lds', False)
    use_fds = params.get('use_fds', False)
    model_dir = _make_run_dir('models', model_type_name, feature_cfg, dataset,
                              use_lds=use_lds, use_fds=use_fds)

    random_state = feature_cfg['random_state']
    strategy     = feature_cfg['split_strategy']
    test_frac    = feature_cfg['test_frac']
    val_strategy = feature_cfg['val_strategy']
    val_frac     = feature_cfg['val_frac']

    train_pool_idx, test_idx = _split_indices(len(X_3d), strategy, test_frac, random_state)
    rel_train_idx, rel_val_idx = _split_indices(len(train_pool_idx), val_strategy, val_frac, random_state)
    train_idx = train_pool_idx[rel_train_idx]
    val_idx   = train_pool_idx[rel_val_idx]

    train_mask = mask_3d[train_idx]
    X_tr_np = X_3d[train_idx][train_mask]
    y_tr_np = y_3d[train_idx][train_mask]       # (N_valid, out_dim), scaled
    print(f"Train after denoising: {len(X_tr_np)} samples | Val: {len(val_idx)} | Test: {len(test_idx)}")

    X_val = torch.FloatTensor(X_3d[val_idx])
    y_val = torch.FloatTensor(y_3d[val_idx])

    # --- LDS sample weights (per day, based on mean scaled load) ---
    if use_lds:
        day_rep = y_tr_np.mean(axis=1)
        lds_w   = compute_lds_weights(day_rep)
        print(f"LDS: enabled | weight range [{lds_w.min():.3f}, {lds_w.max():.3f}]")
    else:
        lds_w = np.ones(len(y_tr_np), dtype=np.float32)

    X_tr = torch.FloatTensor(X_tr_np)
    y_tr = torch.FloatTensor(y_tr_np)
    w_tr = torch.FloatTensor(lds_w)

    loader = DataLoader(TensorDataset(X_tr, y_tr, w_tr), batch_size=params['batch_size'], shuffle=True)
    model  = model_cls(num_features=X_3d.shape[2], params=params).to(device)

    # --- FDS module ---
    fds         = None
    fds_start   = 0
    if use_fds:
        feat_dim = params.get('hidden_size') or params.get('d_model')
        if feat_dim is None:
            raise ValueError("FDS requires 'hidden_size' (LSTM) or 'd_model' (Transformer) in params.")
        fds_start = params.get('fds_start_epoch', 5)
        fds = FDSModule(
            feature_dim=feat_dim,
            n_bins=params.get('fds_n_bins', 50),
            ks=params.get('fds_ks', 5),
            sigma=params.get('fds_sigma', 2.0),
        )
        print(f"FDS: enabled | feat_dim={feat_dim} | n_bins={fds.n_bins} | start_epoch={fds_start}")

    # Per-sample loss for LDS weighting; val uses plain mean for unbiased early-stop
    train_criterion = nn.L1Loss(reduction='none')
    val_criterion   = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    save_path           = os.path.join(model_dir, save_name)
    best_val_loss       = float('inf')
    epochs_no_improve   = 0
    early_stop_patience = params.get('early_stop_patience', 30)

    for epoch in range(params['epochs']):
        model.train()
        train_loss = 0

        for bx, by, bw in loader:
            bx, by, bw = bx.to(device), by.to(device), bw.to(device)
            optimizer.zero_grad()

            if use_fds:
                # Separate encode / (calibrate) / decode so FDS can intervene
                features_raw = model.encode(bx)                 # (batch, feat_dim)
                by_rep       = by.mean(dim=1)                   # representative label per day

                if epoch >= fds_start and fds._ready:
                    features = fds.calibrate(features_raw, by_rep)
                else:
                    features = features_raw

                pred = model.decode(features)
            else:
                pred = model(bx)

            # LDS-weighted L1 loss: (batch, out_dim) × (batch, 1) → scalar
            loss = (train_criterion(pred, by) * bw.unsqueeze(1)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

            # Accumulate RAW features (pre-calibration) for FDS stats update
            if use_fds:
                fds.collect(features_raw, by_rep)

        # Update FDS running statistics once per epoch
        if use_fds:
            fds.update_and_smooth()

        # Validation — always uses the plain forward pass (no FDS)
        model.eval()
        with torch.no_grad():
            v_loss = val_criterion(model(X_val.to(device)), y_val.to(device)).item()
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

        if epochs_no_improve >= early_stop_patience:
            print(f"\nEarly stopping at Epoch {epoch+1}")
            break

    print(f"Best model saved to: {save_path}")
