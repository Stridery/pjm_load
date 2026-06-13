# src/model_trainer.py
import numpy as np
from src.feature_engine import _split_indices
from src.config import TRANSFORMER_FEATURE_CONFIG, TREE_FEATURE_CONFIG, LSTM_FEATURE_CONFIG, DATASET
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import math
import os
import joblib


def _make_run_dir(base, model_type, cfg, dataset=None):
    tag = f"{cfg['split_strategy']}_test{cfg['test_frac']}"
    if 'val_strategy' in cfg:
        tag += f"_{cfg['val_strategy']}_val{cfg['val_frac']}"
    path = os.path.join(base, dataset or DATASET, model_type, tag)
    os.makedirs(path, exist_ok=True)
    return path


class PowerForecaster:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
        self.xgb_models = []
        self.lgb_models = []

    def train_xgboost(self, params):
        print("\n--- Training XGBoost Experts ---")
        model_dir = _make_run_dir('models', 'xgboost', TREE_FEATURE_CONFIG)

        use_gpu = torch.cuda.is_available()
        xgb_params = {**params, 'device': 'cuda'} if use_gpu else params
        print(f"XGBoost device: {'cuda' if use_gpu else 'cpu'}")

        self.xgb_models = []
        for h in tqdm(range(24), desc="XGBoost"):
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(self.X_train, self.y_train[f'h{h}'])
            self.xgb_models.append(model)

        save_path = os.path.join(model_dir, 'xgboost_24_models.pkl')
        joblib.dump(self.xgb_models, save_path)
        print(f"Model saved to: {save_path}")

    def train_lightgbm(self, params):
        print("\n--- Training LightGBM Experts ---")
        model_dir = _make_run_dir('models', 'lightgbm', TREE_FEATURE_CONFIG)

        print("LightGBM device: cpu")
        cat_features = ['today_dayofweek', 'tmrw_is_weekend']
        self.lgb_models = []

        for h in tqdm(range(24), desc="LightGBM"):
            model = lgb.LGBMRegressor(**params, n_estimators=1000)
            model.fit(self.X_train, self.y_train[f'h{h}'], categorical_feature=cat_features)
            self.lgb_models.append(model)

        save_path = os.path.join(model_dir, 'lightgbm_24_models.pkl')
        joblib.dump(self.lgb_models, save_path)
        print(f"Model saved to: {save_path}")

    def train_transformer_3d(self, X_3d, y_3d, mask_3d, params,
                              feature_cfg=None, dataset=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nGPU Acceleration: {device}")
        _cfg = feature_cfg or TRANSFORMER_FEATURE_CONFIG
        model_dir = _make_run_dir('models', 'transformer', _cfg, dataset)

        random_state = _cfg['random_state']
        strategy     = _cfg['split_strategy']
        test_frac    = _cfg['test_frac']
        val_strategy = _cfg['val_strategy']
        val_frac     = _cfg['val_frac']

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
        model     = TimeSeriesTransformer3D(num_features=X_3d.shape[2], params=params).to(device)
        criterion = nn.L1Loss()
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        save_path       = os.path.join(model_dir, 'transformer_best.pth')
        best_val_loss   = float('inf')
        epochs_no_improve = 0
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

    def train_lstm_3d(self, X_3d, y_3d, mask_3d, params,
                       feature_cfg=None, dataset=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nGPU Acceleration: {device}")
        _cfg = feature_cfg or LSTM_FEATURE_CONFIG
        model_dir = _make_run_dir('models', 'lstm', _cfg, dataset)

        random_state = _cfg['random_state']
        strategy     = _cfg['split_strategy']
        test_frac    = _cfg['test_frac']
        val_strategy = _cfg['val_strategy']
        val_frac     = _cfg['val_frac']

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
        model     = LSTMModel(num_features=X_3d.shape[2], params=params).to(device)
        criterion = nn.L1Loss()
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        save_path         = os.path.join(model_dir, 'lstm_best.pth')
        best_val_loss     = float('inf')
        epochs_no_improve = 0
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


class TimeSeriesTransformer3D(nn.Module):
    def __init__(self, num_features, params):
        super().__init__()
        d_model = params['d_model']
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=params['nhead'],
            dim_feedforward=d_model*4, dropout=params['dropout'], batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=params['num_layers'])
        fc_hidden = params.get('fc_hidden', 128)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, fc_hidden),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(fc_hidden, params['out_dim'])
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        return self.fc_out(x)


class LSTMModel(nn.Module):
    def __init__(self, num_features, params):
        super().__init__()
        hidden = params['hidden_size']
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden,
            num_layers=params['num_layers'],
            batch_first=True,
            dropout=params['dropout'] if params['num_layers'] > 1 else 0.0,
        )
        fc_hidden = params.get('fc_hidden', 128)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(fc_hidden, params['out_dim']),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc_out(out)
