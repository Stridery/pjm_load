# src/model_trainer.py
import numpy as np
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

    def _plot_error_distributions(self, model_name, hourly_mape, dow_mape, dom_mape):
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
        
        save_dir = "results/" 
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name}_error_dashboard.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Error distribution plots saved to: {save_path}")

    def _evaluate_and_save(self, model_name, y_true_df, y_pred_np):
        print(f"\n====== {model_name} Error Analysis ======")
        

        mape = mean_absolute_percentage_error(y_true_df.values, y_pred_np)
        mae = mean_absolute_error(y_true_df.values, y_pred_np)
        rmse = np.sqrt(mean_squared_error(y_true_df.values, y_pred_np))
        print(f"Global -> MAPE: {mape*100:.2f}% | MAE: {mae:.2f} | RMSE: {rmse:.2f}")


        print(f"\n[1] Hourly MAPE (%):")
        hourly_mape = []
        for h in range(24):
            h_mape = mean_absolute_percentage_error(y_true_df.iloc[:, h], y_pred_np[:, h])
            hourly_mape.append(h_mape * 100)
            

        for h in range(0, 24, 8):
            formatted = ", ".join([f"h{i:02d}: {hourly_mape[i]:.2f}" for i in range(h, min(h+8, 24))])
            print(f"  {formatted}")


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


        print(f"\n[2] Day of Week MAPE (%)  *0=Mon, 6=Sun* :")
        dow_mape = detailed_df.groupby('day_of_week')['mape_pct'].mean()
        dow_formatted = ", ".join([f"Day {dow}: {mape:.2f}" for dow, mape in dow_mape.items()])
        print(f"  {dow_formatted}")


        print(f"\n[3] Day of Month MAPE (%)  *1st to 31st* :")
        dom_mape = detailed_df.groupby('day_of_month')['mape_pct'].mean()
        dom_items = list(dom_mape.items())
        for i in range(0, len(dom_items), 10):
            dom_formatted = ", ".join([f"{d:02d}th: {mape:.2f}" for d, mape in dom_items[i:i+10]])
            print(f"  {dom_formatted}")


        print(f"\n[4] Top 3 Worst Performing Days (Detailed Hourly Logs):")
        daily_mape = detailed_df.groupby('date')['mape_pct'].mean().sort_values(ascending=False)
        worst_days = daily_mape.head(3).index
        
        for i, w_date in enumerate(worst_days):
            w_mape = daily_mape.loc[w_date]
            print(f"  No.{i+1} Worst Day: {w_date} (Daily Avg MAPE: {w_mape:.2f}%)")
            print(f"    {'Time':<20} | {'True Load':<10} | {'Predicted':<10} | {'MAPE(%)':<8}")
            print(f"    {'-'*60}")
            
            day_data = detailed_df[detailed_df['date'] == w_date]
            for _, row in day_data.iterrows():
                dt_str = row['datetime'].strftime('%Y-%m-%d %H:%00')
                t_load = row['true_load']
                p_load = row[f'{model_name}_pred']
                err_pct = row['mape_pct']
                print(f"    {dt_str:<20} | {t_load:<10.2f} | {p_load:<10.2f} | {err_pct:<8.2f}")
            print("") 


        os.makedirs('results', exist_ok=True)
        save_path = f'results/{model_name}_detailed_errors.csv'
        detailed_df.drop(columns=['date']).to_csv(save_path, index=False) # 保存时去掉冗余的 date 列
        

        self._plot_error_distributions(model_name, hourly_mape, dow_mape, dom_mape)
        print(f"Detailed predictions and errors saved to: {save_path}\n")


    def _evaluate_and_save_3d(self, model_name, y_true_np, y_pred_np, test_timestamps):
        print(f"\n====== {model_name} Error Analysis ======")
        

        mape = mean_absolute_percentage_error(y_true_np.flatten(), y_pred_np.flatten())
        mae = mean_absolute_error(y_true_np.flatten(), y_pred_np.flatten())
        rmse = np.sqrt(mean_squared_error(y_true_np.flatten(), y_pred_np.flatten()))
        print(f"Global -> MAPE: {mape*100:.2f}% | MAE: {mae:.2f} | RMSE: {rmse:.2f}")


        print(f"\n[1] Hourly MAPE (%):")
        hourly_mape = []
        for h in range(24):
            h_mape = mean_absolute_percentage_error(y_true_np[:, h], y_pred_np[:, h])
            hourly_mape.append(h_mape * 100)
            
        for h in range(0, 24, 8):
            formatted = ", ".join([f"h{i:02d}: {hourly_mape[i]:.2f}" for i in range(h, min(h+8, 24))])
            print(f"  {formatted}")


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


        print(f"\n[2] Day of Week MAPE (%)  *0=Mon, 6=Sun* :")
        dow_mape = detailed_df.groupby('day_of_week')['mape_pct'].mean()
        dow_formatted = ", ".join([f"Day {dow}: {mape:.2f}" for dow, mape in dow_mape.items()])
        print(f"  {dow_formatted}")


        print(f"\n[3] Day of Month MAPE (%)  *1st to 31st* :")
        dom_mape = detailed_df.groupby('day_of_month')['mape_pct'].mean()
        dom_items = list(dom_mape.items())
        for i in range(0, len(dom_items), 10):
            dom_formatted = ", ".join([f"{d:02d}th: {mape:.2f}" for d, mape in dom_items[i:i+10]])
            print(f"  {dom_formatted}")


        print(f"\n[4] Top 3 Worst Performing Days (Detailed Hourly Logs):")
        daily_mape = detailed_df.groupby('date')['mape_pct'].mean().sort_values(ascending=False)
        worst_days = daily_mape.head(3).index
        
        for i, w_date in enumerate(worst_days):
            w_mape = daily_mape.loc[w_date]
            print(f"  No.{i+1} Worst Day: {w_date} (Daily Avg MAPE: {w_mape:.2f}%)")
            print(f"    {'Time':<20} | {'True Load':<10} | {'Predicted':<10} | {'MAPE(%)':<8}")
            print(f"    {'-'*60}")
            
            day_data = detailed_df[detailed_df['date'] == w_date]
            for _, row in day_data.iterrows():
                dt_str = row['datetime'].strftime('%Y-%m-%d %H:%00')
                t_load = row['true_load']
                p_load = row[f'{model_name}_pred']
                err_pct = row['mape_pct']
                print(f"    {dt_str:<20} | {t_load:<10.2f} | {p_load:<10.2f} | {err_pct:<8.2f}")
            print("")


        os.makedirs('results', exist_ok=True)
        save_path = f'results/{model_name}_detailed_errors.csv'
        detailed_df.drop(columns=['date']).to_csv(save_path, index=False)
        

        self._plot_error_distributions(model_name, hourly_mape, dow_mape, dom_mape)
        print(f"Detailed predictions and errors saved to: {save_path}\n")

    def train_xgboost(self, params):
        print("\n--- Training XGBoost Experts ---")
        self.xgb_models = []
        for h in tqdm(range(24), desc="XGBoost"):
            model = xgb.XGBRegressor(**params)
            model.fit(self.X_train, self.y_train[f'h{h}'])
            self.xgb_models.append(model)
        
        joblib.dump(self.xgb_models, 'models/xgboost_24_models.pkl')
        
        self.xgb_preds = np.array([m.predict(self.X_test) for m in self.xgb_models]).T
        mape = mean_absolute_percentage_error(self.y_test, self.xgb_preds)
        print(f"XGBoost MAPE: {mape*100:.2f}%")
        self._evaluate_and_save("XGBoost", self.y_test, self.xgb_preds)
        return self.xgb_preds

    def train_lightgbm(self, params):
        print("\n--- Training LightGBM Experts ---")
        self.lgb_models = []
        cat_features = ['tmrw_dayofweek', 'tmrw_is_weekend']
        
        for h in tqdm(range(24), desc="LightGBM"):
            model = lgb.LGBMRegressor(**params, n_estimators=1000)
            model.fit(self.X_train, self.y_train[f'h{h}'], categorical_feature=cat_features)
            self.lgb_models.append(model)

        joblib.dump(self.lgb_models, 'models/lightgbm_24_models.pkl')
            
        self.lgb_preds = np.array([m.predict(self.X_test) for m in self.lgb_models]).T
        mape = mean_absolute_percentage_error(self.y_test, self.lgb_preds)
        print(f"LightGBM MAPE: {mape*100:.2f}%")
        self._evaluate_and_save("LightGBM", self.y_test, self.lgb_preds)
        return self.lgb_preds
    

    def train_transformer_3d(self, X_3d, y_3d, mask_3d, timestamps_3d, scaler_ts, params):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\GPU Acceleration: {device}")


        train_size = int(len(X_3d) * 0.9)
        val_split = int(train_size * 0.9)

        test_timestamps = timestamps_3d[train_size:]
        

        X_tr_raw = X_3d[:val_split]

        y_tr_raw = y_3d[:val_split] 
        mask_tr = mask_3d[:val_split]
        

        X_tr_filtered = X_tr_raw[mask_tr]
        y_tr_filtered = y_tr_raw[mask_tr]
        print(f"Training Set Denoising: Reduced from {len(X_tr_raw)} to {len(X_tr_filtered)} samples")


        X_tr = torch.FloatTensor(X_tr_filtered)
        y_tr = torch.FloatTensor(y_tr_filtered)
        

        X_val = torch.FloatTensor(X_3d[val_split:train_size])
        y_val = torch.FloatTensor(y_3d[val_split:train_size])
        
        X_te = torch.FloatTensor(X_3d[train_size:])
        y_te_scaled = y_3d[train_size:] # shape: [N_test, 24]

        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=params['batch_size'], shuffle=True)
        

        model = TimeSeriesTransformer3D(num_features=X_3d.shape[2], params=params).to(device)
        criterion = nn.L1Loss()
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)


        best_val_loss = float('inf')
        early_stop_patience = 15  
        epochs_no_improve = 0
        for epoch in range(params['epochs']):
            model.train()
            train_loss = 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                

                out = model(bx) 
                loss = criterion(out, by) 
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            with torch.no_grad():
                v_out = model(X_val.to(device)) # 同样去掉 squeeze
                v_loss = criterion(v_out, y_val.to(device)).item()
            
            scheduler.step(v_loss)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                model_save_path = os.path.join('models', 'transformer_9am_full_features.pth')
                os.makedirs('models/', exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
                epochs_no_improve = 0  
            else:
                epochs_no_improve += 1 

            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1:03d} | Train: {train_loss/len(loader):.4f} | Val: {v_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            

            if epochs_no_improve >= early_stop_patience:
                print(f"\nEarly stopping triggered at Epoch {epoch+1}! Validation loss hasn't improved for {early_stop_patience} epochs.")
                break


        model.load_state_dict(torch.load('models/transformer_9am_full_features.pth'))
        model.eval()
        with torch.no_grad():
            # preds_scaled shape: [N_test, 24]
            preds_scaled = model(X_te.to(device)).cpu().numpy()
        

        num_features = X_3d.shape[2]
        def inverse_transform_load_2d(scaled_data_2d):
            N, P = scaled_data_2d.shape
            flat_data = scaled_data_2d.flatten() # 展平成 1D 数组 [N * 24]
            dummy = np.zeros((len(flat_data), num_features))
            dummy[:, 0] = flat_data # 放入 Load 列
            inv_flat = scaler_ts.inverse_transform(dummy)[:, 0]
            return inv_flat.reshape(N, P) # 重新变回 [N, 24]

        all_preds_mw = inverse_transform_load_2d(preds_scaled)
        all_true_mw = inverse_transform_load_2d(y_te_scaled)
        

        mape = mean_absolute_percentage_error(all_true_mw.flatten(), all_preds_mw.flatten())
        mae = mean_absolute_error(all_true_mw.flatten(), all_preds_mw.flatten())
        rmse = np.sqrt(mean_squared_error(all_true_mw.flatten(), all_preds_mw.flatten()))
        
        self.transformer_preds = all_preds_mw
        self._evaluate_and_save_3d("Transformer", all_true_mw, all_preds_mw, test_timestamps)
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
            nn.Linear(params['seq_len'] * d_model, 128),
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

