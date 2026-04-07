# src/model_trainer.py
import numpy as np
from src.feature_engine import _split_indices
from src.config import TRANSFORMER_FEATURE_CONFIG, TREE_FEATURE_CONFIG
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import math
import os
import joblib
import matplotlib.pyplot as plt


def _make_run_dir(base, model_type, cfg):
    """根据 config 参数生成并创建实验子目录。"""
    tag = f"{cfg['split_strategy']}_test{cfg['test_frac']}"
    if 'val_strategy' in cfg:
        tag += f"_{cfg['val_strategy']}_val{cfg['val_frac']}"
    path = os.path.join(base, model_type, tag)
    os.makedirs(path, exist_ok=True)
    return path


class PowerForecaster:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.xgb_models = []
        self.lgb_models = []
        
        self.xgb_preds = None
        self.lgb_preds = None
        self.transformer_preds = None

    def _plot_error_distributions(self, model_name, hourly_mape, dow_mape, dom_mape, run_dir):
        plt.figure(figsize=(15, 12))
        
        # ----------------------------------------
        # [1] Hourly MAPE Plot
        # ----------------------------------------
        plt.subplot(3, 1, 1)
        hours = range(24)
        plt.bar(hours, hourly_mape, color='#4C72B0', edgecolor='black', alpha=0.8)
        plt.title(f'{model_name} - Hourly MAPE (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day (0-23)', fontsize=12)
        plt.ylabel('MAPE (%)', fontsize=12)
        plt.xticks(hours) 
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # ----------------------------------------
        # [2] Day of Week MAPE Plot
        # ----------------------------------------
        plt.subplot(3, 1, 2)
        days_labels = ['Mon (0)', 'Tue (1)', 'Wed (2)', 'Thu (3)', 'Fri (4)', 'Sat (5)', 'Sun (6)']
        dow_values = [dow_mape.get(i, 0) for i in range(7)] 
        plt.bar(days_labels, dow_values, color='#55A868', edgecolor='black', alpha=0.8)
        plt.title(f'{model_name} - Day of Week MAPE (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Day of Week', fontsize=12)
        plt.ylabel('MAPE (%)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # ----------------------------------------
        # [3] Day of Month MAPE Plot
        # ----------------------------------------
        plt.subplot(3, 1, 3)
        dom_idx = sorted(dom_mape.index)
        dom_values = [dom_mape[d] for d in dom_idx]
        plt.bar(dom_idx, dom_values, color='#C44E52', edgecolor='black', alpha=0.8)
        plt.title(f'{model_name} - Day of Month MAPE (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Day of Month (1-31)', fontsize=12)
        plt.ylabel('MAPE (%)', fontsize=12)
        plt.xticks(range(1, 32)) # 强制显示所有的 1-31 刻度
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()

        save_path = os.path.join(run_dir, f'{model_name}_error_dashboard.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Error distribution plots saved to: {save_path}")

    def _plot_best_worst_days(self, model_name, detailed_df, daily_mape, run_dir):
        worst_days = list(daily_mape.head(3).index)
        best_days = list(daily_mape.tail(3).index[::-1])
        hours = range(24)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{model_name} — Best & Worst 3 Days', fontsize=16, fontweight='bold')

        for col, date in enumerate(worst_days):
            ax = axes[0, col]
            day_data = detailed_df[detailed_df['date'] == date].sort_values('datetime')
            mape_val = daily_mape.loc[date]
            ax.plot(hours, day_data['true_load'].values, label='True', color='#4C72B0', linewidth=2)
            ax.plot(hours, day_data[f'{model_name}_pred'].values, label='Pred', color='#C44E52', linewidth=2, linestyle='--')
            ax.set_title(f'Worst #{col+1}: {date}\nMAPE: {mape_val:.2f}%', fontsize=11)
            ax.set_xlabel('Hour'); ax.set_ylabel('Load (MW)')
            ax.legend(); ax.grid(linestyle='--', alpha=0.6)

        for col, date in enumerate(best_days):
            ax = axes[1, col]
            day_data = detailed_df[detailed_df['date'] == date].sort_values('datetime')
            mape_val = daily_mape.loc[date]
            ax.plot(hours, day_data['true_load'].values, label='True', color='#4C72B0', linewidth=2)
            ax.plot(hours, day_data[f'{model_name}_pred'].values, label='Pred', color='#55A868', linewidth=2, linestyle='--')
            ax.set_title(f'Best #{col+1}: {date}\nMAPE: {mape_val:.2f}%', fontsize=11)
            ax.set_xlabel('Hour'); ax.set_ylabel('Load (MW)')
            ax.legend(); ax.grid(linestyle='--', alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(run_dir, f'{model_name}_best_worst_days.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Best/worst day plots saved to: {save_path}")

    def _evaluate_and_save(self, model_name, y_true_df, y_pred_np, run_dir):
        print(f"\n====== {model_name} Error Analysis ======")
        

        mape = mean_absolute_percentage_error(y_true_df.values, y_pred_np)
        mae = mean_absolute_error(y_true_df.values, y_pred_np)
        rmse = np.sqrt(mean_squared_error(y_true_df.values, y_pred_np))
        print(f"Global -> MAPE: {mape*100:.2f}% | MAE: {mae:.2f} | RMSE: {rmse:.2f}")


        hourly_mape = [mean_absolute_percentage_error(y_true_df.iloc[:, h], y_pred_np[:, h]) * 100
                       for h in range(24)]

        dates = pd.to_datetime(y_true_df.index).repeat(24)
        hours = np.tile(np.arange(24), len(y_true_df))
        datetimes = dates + pd.to_timedelta(hours, unit='h')

        detailed_df = pd.DataFrame({
            'datetime': datetimes,
            'true_load': y_true_df.values.flatten(),
            f'{model_name}_pred': y_pred_np.flatten()
        })
        detailed_df['datetime'] = pd.to_datetime(detailed_df['datetime'])
        detailed_df['abs_error'] = np.abs(detailed_df['true_load'] - detailed_df[f'{model_name}_pred'])
        detailed_df['mape_pct'] = (detailed_df['abs_error'] / detailed_df['true_load']) * 100
        detailed_df['day_of_week'] = detailed_df['datetime'].dt.dayofweek
        detailed_df['day_of_month'] = detailed_df['datetime'].dt.day
        detailed_df['date'] = detailed_df['datetime'].dt.date

        dow_mape = detailed_df.groupby('day_of_week')['mape_pct'].mean()
        dom_mape = detailed_df.groupby('day_of_month')['mape_pct'].mean()


        daily_mape = detailed_df.groupby('date')['mape_pct'].mean().sort_values(ascending=False)

        save_path = os.path.join(run_dir, f'{model_name}_detailed_errors.csv')
        detailed_df.drop(columns=['date']).to_csv(save_path, index=False)

        self._plot_error_distributions(model_name, hourly_mape, dow_mape, dom_mape, run_dir)
        self._plot_best_worst_days(model_name, detailed_df, daily_mape, run_dir)
        print(f"Detailed predictions and errors saved to: {save_path}\n")


    def _evaluate_and_save_3d(self, model_name, y_true_np, y_pred_np, test_timestamps, run_dir):
        print(f"\n====== {model_name} Error Analysis ======")
        

        mape = mean_absolute_percentage_error(y_true_np.flatten(), y_pred_np.flatten())
        mae = mean_absolute_error(y_true_np.flatten(), y_pred_np.flatten())
        rmse = np.sqrt(mean_squared_error(y_true_np.flatten(), y_pred_np.flatten()))
        print(f"Global -> MAPE: {mape*100:.2f}% | MAE: {mae:.2f} | RMSE: {rmse:.2f}")


        hourly_mape = [mean_absolute_percentage_error(y_true_np[:, h], y_pred_np[:, h]) * 100
                       for h in range(24)]

        dates = pd.to_datetime(test_timestamps).repeat(24)
        hours = np.tile(np.arange(24), len(test_timestamps))
        datetimes = dates + pd.to_timedelta(hours, unit='h')

        detailed_df = pd.DataFrame({
            'datetime': datetimes,
            'true_load': y_true_np.flatten(),
            f'{model_name}_pred': y_pred_np.flatten()
        })
        detailed_df['datetime'] = pd.to_datetime(detailed_df['datetime'])
        detailed_df['abs_error'] = np.abs(detailed_df['true_load'] - detailed_df[f'{model_name}_pred'])
        detailed_df['mape_pct'] = (detailed_df['abs_error'] / detailed_df['true_load']) * 100
        detailed_df['day_of_week'] = detailed_df['datetime'].dt.dayofweek
        detailed_df['day_of_month'] = detailed_df['datetime'].dt.day
        detailed_df['date'] = detailed_df['datetime'].dt.date

        dow_mape = detailed_df.groupby('day_of_week')['mape_pct'].mean()
        dom_mape = detailed_df.groupby('day_of_month')['mape_pct'].mean()


        daily_mape = detailed_df.groupby('date')['mape_pct'].mean().sort_values(ascending=False)

        save_path = os.path.join(run_dir, f'{model_name}_detailed_errors.csv')
        detailed_df.drop(columns=['date']).to_csv(save_path, index=False)

        self._plot_error_distributions(model_name, hourly_mape, dow_mape, dom_mape, run_dir)
        self._plot_best_worst_days(model_name, detailed_df, daily_mape, run_dir)
        print(f"Detailed predictions and errors saved to: {save_path}\n")

    def train_xgboost(self, params):
        print("\n--- Training XGBoost Experts ---")
        model_dir  = _make_run_dir('models',  'xgboost', TREE_FEATURE_CONFIG)
        result_dir = _make_run_dir('results', 'xgboost', TREE_FEATURE_CONFIG)

        use_gpu = torch.cuda.is_available()
        xgb_params = {**params, 'device': 'cuda'} if use_gpu else params
        print(f"XGBoost device: {'cuda' if use_gpu else 'cpu'}")
        self.xgb_models = []
        for h in tqdm(range(24), desc="XGBoost"):
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(self.X_train, self.y_train[f'h{h}'])
            self.xgb_models.append(model)

        joblib.dump(self.xgb_models, os.path.join(model_dir, 'xgboost_24_models.pkl'))

        self.xgb_preds = np.array([m.predict(self.X_test) for m in self.xgb_models]).T
        mape = mean_absolute_percentage_error(self.y_test, self.xgb_preds)
        print(f"XGBoost MAPE: {mape*100:.2f}%")
        self._evaluate_and_save("XGBoost", self.y_test, self.xgb_preds, result_dir)
        return self.xgb_preds

    def train_lightgbm(self, params):
        print("\n--- Training LightGBM Experts ---")
        model_dir  = _make_run_dir('models',  'lightgbm', TREE_FEATURE_CONFIG)
        result_dir = _make_run_dir('results', 'lightgbm', TREE_FEATURE_CONFIG)

        print(f"LightGBM device: cpu (LightGBM GPU requires custom build, using CPU)")
        lgb_params = params
        self.lgb_models = []
        cat_features = ['today_dayofweek', 'tmrw_is_weekend']

        for h in tqdm(range(24), desc="LightGBM"):
            model = lgb.LGBMRegressor(**lgb_params, n_estimators=1000)
            model.fit(self.X_train, self.y_train[f'h{h}'], categorical_feature=cat_features)
            self.lgb_models.append(model)

        joblib.dump(self.lgb_models, os.path.join(model_dir, 'lightgbm_24_models.pkl'))

        self.lgb_preds = np.array([m.predict(self.X_test) for m in self.lgb_models]).T
        mape = mean_absolute_percentage_error(self.y_test, self.lgb_preds)
        print(f"LightGBM MAPE: {mape*100:.2f}%")
        self._evaluate_and_save("LightGBM", self.y_test, self.lgb_preds, result_dir)
        return self.lgb_preds
    

    def train_transformer_3d(self, X_3d, y_3d, mask_3d, timestamps_3d, scaler_ts, params):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nGPU Acceleration: {device}")
        model_dir  = _make_run_dir('models',  'transformer', TRANSFORMER_FEATURE_CONFIG)
        result_dir = _make_run_dir('results', 'transformer', TRANSFORMER_FEATURE_CONFIG)

        random_state  = params.get('random_state', 42)
        strategy      = TRANSFORMER_FEATURE_CONFIG['split_strategy']
        test_frac     = TRANSFORMER_FEATURE_CONFIG['test_frac']
        val_strategy  = TRANSFORMER_FEATURE_CONFIG['val_strategy']
        val_frac      = TRANSFORMER_FEATURE_CONFIG['val_frac']

        # 第一步：切 test
        train_pool_idx, test_idx = _split_indices(len(X_3d), strategy, test_frac, random_state)

        # 第二步：在 train pool 内切 val
        rel_train_idx, rel_val_idx = _split_indices(len(train_pool_idx), val_strategy, val_frac, random_state)
        train_idx = train_pool_idx[rel_train_idx]
        val_idx   = train_pool_idx[rel_val_idx]

        test_timestamps = timestamps_3d[test_idx]
        X_te       = torch.FloatTensor(X_3d[test_idx])
        y_te_scaled = y_3d[test_idx]

        # 训练集过滤无效样本
        train_mask = mask_3d[train_idx]
        X_tr = torch.FloatTensor(X_3d[train_idx][train_mask])
        y_tr = torch.FloatTensor(y_3d[train_idx][train_mask])
        print(f"Train after denoising: {len(X_tr)} samples | Val: {len(val_idx)} | Test: {len(test_idx)}")

        X_val = torch.FloatTensor(X_3d[val_idx])
        y_val = torch.FloatTensor(y_3d[val_idx])

        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=params['batch_size'], shuffle=True)
        model = TimeSeriesTransformer3D(num_features=X_3d.shape[2], params=params).to(device)
        criterion = nn.L1Loss()
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        save_path = os.path.join(model_dir, 'transformer_best.pth')

        best_val_loss = float('inf')
        early_stop_patience = 15
        epochs_no_improve = 0
        for epoch in range(params['epochs']):
            model.train()
            train_loss = 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
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

        model.load_state_dict(torch.load(save_path))
        model.eval()
        with torch.no_grad():
            preds_scaled = model(X_te.to(device)).cpu().numpy()

        num_features = X_3d.shape[2]
        def inverse_transform_load_2d(scaled_data_2d):
            N, P = scaled_data_2d.shape
            flat_data = scaled_data_2d.flatten()
            dummy = np.zeros((len(flat_data), num_features))
            dummy[:, 0] = flat_data
            inv_flat = scaler_ts.inverse_transform(dummy)[:, 0]
            return inv_flat.reshape(N, P)

        all_preds_mw = inverse_transform_load_2d(preds_scaled)
        all_true_mw  = inverse_transform_load_2d(y_te_scaled)

        mape = mean_absolute_percentage_error(all_true_mw.flatten(), all_preds_mw.flatten())
        mae  = mean_absolute_error(all_true_mw.flatten(), all_preds_mw.flatten())
        rmse = np.sqrt(mean_squared_error(all_true_mw.flatten(), all_preds_mw.flatten()))

        self.transformer_preds = all_preds_mw
        self._evaluate_and_save_3d("Transformer", all_true_mw, all_preds_mw, test_timestamps, result_dir)
        print(f"\nMAPE: {mape*100:.2f}%")
        print(f"MAE : {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")

        return self.transformer_preds
    



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
        # x shape: [Batch, Seq_len, d_model]
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
        

        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(TRANSFORMER_FEATURE_CONFIG['lookback_hours'] * d_model, 128),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(128, params['out_dim'])
        )

    def forward(self, x):
        # x: [Batch, Seq, Features]
        x = self.input_projection(x) # -> [Batch, Seq, d_model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.fc_out(x)

