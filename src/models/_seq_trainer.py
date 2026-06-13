"""Shared training loop for all sequence models (Transformer, LSTM)."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.feature_engine import _split_indices
from src.models._utils import _make_run_dir


def train_sequence(model_cls, model_type_name, save_name,
                   X_3d, y_3d, mask_3d, params, feature_cfg, dataset=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nGPU Acceleration: {device}")

    model_dir = _make_run_dir('models', model_type_name, feature_cfg, dataset)

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
    X_tr = torch.FloatTensor(X_3d[train_idx][train_mask])
    y_tr = torch.FloatTensor(y_3d[train_idx][train_mask])
    print(f"Train after denoising: {len(X_tr)} samples | Val: {len(val_idx)} | Test: {len(test_idx)}")

    X_val = torch.FloatTensor(X_3d[val_idx])
    y_val = torch.FloatTensor(y_3d[val_idx])

    loader    = DataLoader(TensorDataset(X_tr, y_tr), batch_size=params['batch_size'], shuffle=True)
    model     = model_cls(num_features=X_3d.shape[2], params=params).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    save_path           = os.path.join(model_dir, save_name)
    best_val_loss       = float('inf')
    epochs_no_improve   = 0
    early_stop_patience = params.get('early_stop_patience', 30)

    for epoch in range(params['epochs']):
        model.train()
        train_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(X_val.to(device)), y_val.to(device)).item()
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:03d} | Train: {train_loss/len(loader):.4f} | Val: {v_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if epochs_no_improve >= early_stop_patience:
            print(f"\nEarly stopping at Epoch {epoch+1}")
            break

    print(f"Best model saved to: {save_path}")
