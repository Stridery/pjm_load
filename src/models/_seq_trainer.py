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


def _pinball_calibration(model, X_tr, y_tr, params, device):
    """Stage 2 — load-dependent pinball loss calibration of fc_out.

    q(load_pct) = 0.5 + (q_max - 0.5) * load_pct ** p
    loss = mean( max(q*u, (q-1)*u) ),  u = y_true - y_pred  (u>0 = under-predict)

    routing='pred': load_pct from frozen Stage-1 predictions — deployable, no
                    true labels needed at inference; correction baked into fc_out.
    routing='true': uses true load quantiles — diagnostic upper bound only.
    """
    q_max    = params.get('stage2_q_max', 0.9)
    p_exp    = params.get('stage2_p', 2.0)
    routing  = params.get('stage2_routing', 'pred')
    epochs   = params.get('stage2_epochs', 5)
    lr       = params.get('stage2_lr', 1e-3)
    batch_sz = params.get('batch_size', 32)

    for name, param in model.named_parameters():
        param.requires_grad = ('fc_out' in name)
    n_frozen  = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
    n_tunable = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    print(f"\n--- Stage 2 [pinball] ---")
    print(f"Frozen: {n_frozen} | Tunable (fc_out): {n_tunable} | "
          f"routing={routing} | q_max={q_max} | p={p_exp} | lr={lr}")

    # Collect Stage-1 predictions for quantile routing (no grad, frozen backbone)
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X_tr), batch_sz):
            chunks.append(model(X_tr[i:i + batch_sz].to(device)).cpu())
    stage1_pred = torch.cat(chunks, dim=0)          # (N, 24), scaled

    route_vals = y_tr.numpy() if routing == 'true' else stage1_pred.numpy()
    ref    = np.sort(route_vals.flatten())
    pct_np = np.clip(
        np.searchsorted(ref, route_vals) / len(ref), 0.0, 1.0,
    ).astype(np.float32)                            # (N, 24)
    print(f"load_pct [{pct_np.min():.3f}, {pct_np.max():.3f}] → "
          f"q [{0.5:.3f}, {0.5 + (q_max - 0.5):.3f}]")

    s2_loader = DataLoader(
        TensorDataset(X_tr, y_tr, torch.FloatTensor(pct_np)),
        batch_size=batch_sz, shuffle=True,
    )
    optimizer = optim.Adam(
        [p for n, p in model.named_parameters() if 'fc_out' in n], lr=lr,
    )
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, by, bpct in s2_loader:
            bx, by, bpct = bx.to(device), by.to(device), bpct.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            u    = by - pred
            q    = 0.5 + (q_max - 0.5) * bpct ** p_exp
            loss = torch.max(q * u, (q - 1) * u).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Stage2 Epoch {epoch + 1:02d}/{epochs} | Pinball: {epoch_loss / len(s2_loader):.5f}")

    for param in model.parameters():
        param.requires_grad = True
    model.eval()
    print("Stage 2 complete.")


def _bft_calibration(model, X_tr, y_tr, params, device):
    """Stage 2 — Balanced Fine-Tuning (BFT) via bin resampling of fc_out.

    Constructs a temporary balanced dataset by:
      - Using daily peak load (max over 24h) as the representative label
      - Equal-frequency (quantile) binning into n_bins bins → each bin initially holds ~N/n_bins samples
      - Under-sampling high-frequency (common-load) bins
      - Over-sampling low-frequency (peak-load) bins
    Target per bin = median count of non-empty bins → symmetric truncation/expansion.

    Loss: pure nn.MSELoss() — no routing, no quantile guidance.
    Inference: plain forward pass; correction is baked into fc_out weights.
    """
    n_bins   = params.get('stage2_bft_n_bins', 50)
    epochs   = params.get('stage2_epochs', 5)
    lr       = params.get('stage2_lr', 1e-3)
    batch_sz = params.get('batch_size', 32)

    for name, param in model.named_parameters():
        param.requires_grad = ('fc_out' in name)
    n_frozen  = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
    n_tunable = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    print(f"\n--- Stage 2 [bft] ---")
    print(f"Frozen: {n_frozen} | Tunable (fc_out): {n_tunable} | n_bins={n_bins} | lr={lr}")

    # Daily peak load as representative label (better captures extreme events)
    y_np    = y_tr.numpy()             # (N, 24)
    day_rep = y_np.max(axis=1)         # (N,)

    # Equal-frequency (quantile) binning: each bin holds ~equal number of samples
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges     = np.quantile(day_rep, quantiles)
    edges     = np.unique(edges)                   # collapse ties (may reduce effective n_bins)
    bin_idx   = np.clip(np.digitize(day_rep, edges[1:-1]), 0, len(edges) - 2)  # (N,)
    n_bins    = len(edges) - 1                     # update to actual bin count after dedup

    bin_counts = np.bincount(bin_idx, minlength=n_bins)  # n_bins already updated above
    nonempty   = bin_counts[bin_counts > 0]
    target_n   = int(np.median(nonempty))      # balanced target per bin

    rng = np.random.RandomState(42)
    parts = []
    for b in range(n_bins):
        idx_b = np.where(bin_idx == b)[0]
        n_b   = len(idx_b)
        if n_b == 0:
            continue
        chosen = rng.choice(idx_b, size=target_n, replace=(n_b < target_n))
        parts.append(chosen)

    bal_idx = np.concatenate(parts)
    rng.shuffle(bal_idx)

    X_bal = X_tr[bal_idx]
    y_bal = y_tr[bal_idx]

    n_nonempty = int((bin_counts > 0).sum())
    print(f"Original: {len(X_tr)} | Balanced: {len(X_bal)} "
          f"({target_n}/bin × {n_nonempty} non-empty bins) "
          f"| under-sampled: {int((bin_counts > target_n).sum())} bins "
          f"| over-sampled: {int(((bin_counts > 0) & (bin_counts < target_n)).sum())} bins")

    bal_loader = DataLoader(
        TensorDataset(X_bal, y_bal), batch_size=batch_sz, shuffle=True,
    )
    optimizer = optim.Adam(
        [p for n, p in model.named_parameters() if 'fc_out' in n], lr=lr,
    )
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, by in bal_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Stage2 Epoch {epoch + 1:02d}/{epochs} | MSE: {epoch_loss / len(bal_loader):.4f}")

    for param in model.parameters():
        param.requires_grad = True
    model.eval()
    print("Stage 2 complete.")


def post_hoc_calibration(model, train_loader, optimizer, epochs=5, device='cuda'):
    """Stage 2: freeze backbone, fine-tune fc_out head with unweighted MSELoss.

    Stage 1 (aggressive LDS/FDS) maximises peak-feature extraction but
    systematically over-predicts ordinary samples (bias shift).  Stage 2
    corrects this by re-fitting only the final regression head on the true
    data distribution, with no sample reweighting.

    Args:
        model:        Trained sequence model (Transformer or LSTM).
        train_loader: DataLoader yielding (X, y, w) — sample weights w are ignored.
        optimizer:    Optimizer pre-built for fc_out params only (lr ~ 1e-4).
        epochs:       Number of fine-tuning epochs (5–10 is usually sufficient).
        device:       Torch device string.
    """
    for name, param in model.named_parameters():
        param.requires_grad = ('fc_out' in name)

    n_frozen  = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
    n_tunable = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    print(f"\n--- Stage 2: Post-hoc Calibration ---")
    print(f"Frozen: {n_frozen} param tensors | Tunable (fc_out): {n_tunable} param tensors")

    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, by, _ in train_loader:          # ignore LDS sample weights
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)                     # plain forward — no FDS, no LDS weighting
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Stage2 Epoch {epoch + 1:02d}/{epochs} | MSE: {epoch_loss / len(train_loader):.4f}")

    for param in model.parameters():
        param.requires_grad = True

    model.eval()
    print("Stage 2 complete.")


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
        lds_w   = compute_lds_weights(
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
            bin_width=params.get('fds_bin_width', 200.0),
            ks=params.get('fds_ks', 5),
            sigma=params.get('fds_sigma', 2.0),
            momentum=params.get('fds_momentum', 0.1),
        )
        print(f"FDS: enabled | feat_dim={feat_dim} | bin_width={fds.bin_width} "
              f"| start_epoch={fds_start} | momentum={fds.momentum}")

    # Per-sample loss for LDS weighting; val uses plain mean for unbiased early-stop
    train_criterion = nn.L1Loss(reduction='none')
    val_criterion   = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    save_path           = os.path.join(model_dir, save_name)
    best_val_loss       = float('inf')
    epochs_no_improve   = 0
    early_stop_patience = params.get('early_stop_patience', 30)

    # ------------------------------------------------------------------ #
    # Stage 1 — full training with LDS + FDS                              #
    # ------------------------------------------------------------------ #
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

    print(f"Best Stage 1 model saved to: {save_path}")

    # ------------------------------------------------------------------ #
    # Stage 2 — pluggable calibration (stage2_mode: 'mse' | 'pinball')  #
    # ------------------------------------------------------------------ #
    stage2_epochs = params.get('stage2_epochs', 0)
    if stage2_epochs > 0:
        mode = params.get('stage2_mode', 'pinball')
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        if mode == 'pinball':
            _pinball_calibration(model, X_tr, y_tr, params=params, device=device)
        elif mode == 'bft':
            _bft_calibration(model, X_tr, y_tr, params=params, device=device)
        elif mode == 'mse':
            stage2_opt = optim.AdamW(
                [p for n, p in model.named_parameters() if 'fc_out' in n],
                lr=params.get('stage2_lr', 1e-3),
            )
            post_hoc_calibration(model, loader, stage2_opt, epochs=stage2_epochs, device=device)
        else:
            raise ValueError(f"Unknown stage2_mode {mode!r}. Use 'mse', 'bft', or 'pinball'.")
        torch.save(model.state_dict(), save_path)
        print(f"Stage 2 [{mode}] model saved to: {save_path}")
