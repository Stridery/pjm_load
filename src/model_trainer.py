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
        """生成并保存三个维度的误差柱状图"""
        # 设置整个图表的尺寸
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
        plt.xticks(hours) # 强制显示所有的 0-23 刻度
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # ----------------------------------------
        # [2] Day of Week MAPE Plot
        # ----------------------------------------
        plt.subplot(3, 1, 2)
        days_labels = ['Mon (0)', 'Tue (1)', 'Wed (2)', 'Thu (3)', 'Fri (4)', 'Sat (5)', 'Sun (6)']
        # 确保按 0 到 6 的顺序获取数据，防止有些天缺失导致错位
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
        # 提取 1 到 31 号的数据并排序
        dom_idx = sorted(dom_mape.index)
        dom_values = [dom_mape[d] for d in dom_idx]
        plt.bar(dom_idx, dom_values, color='#C44E52', edgecolor='black', alpha=0.8)
        plt.title(f'{model_name} - Day of Month MAPE (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Day of Month (1-31)', fontsize=12)
        plt.ylabel('MAPE (%)', fontsize=12)
        plt.xticks(range(1, 32)) # 强制显示所有的 1-31 刻度
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 调整子图间距，防止文字重叠
        plt.tight_layout()
        
        # 保存图片 (这里填写你的 Placeholder 路径)
        save_dir = "results/" 
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name}_error_dashboard.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() # 释放内存，防止在服务器上后台跑的时候吃内存
        print(f"📊 Error distribution plots saved to: {save_path}")

    def _evaluate_and_save(self, model_name, y_true_df, y_pred_np):
        """就地计算误差，按小时、星期、日期分析，找出最差天数，并保存详细 CSV"""
        print(f"\n====== {model_name} Error Analysis ======")
        
        # 1. 计算全局指标
        mape = mean_absolute_percentage_error(y_true_df.values, y_pred_np)
        mae = mean_absolute_error(y_true_df.values, y_pred_np)
        rmse = np.sqrt(mean_squared_error(y_true_df.values, y_pred_np))
        print(f"Global -> MAPE: {mape*100:.2f}% | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

        # 2. 计算并打印 24 小时全部逐时 MAPE
        print(f"\n[1] Hourly MAPE (%):")
        hourly_mape = []
        for h in range(24):
            h_mape = mean_absolute_percentage_error(y_true_df.iloc[:, h], y_pred_np[:, h])
            hourly_mape.append(h_mape * 100)
            
        # 每 8 个小时打印一行
        for h in range(0, 24, 8):
            formatted = ", ".join([f"h{i:02d}: {hourly_mape[i]:.2f}" for i in range(h, min(h+8, 24))])
            print(f"  {formatted}")

        # 3. 构造带有精确 Datetime 的详情大表 (修复时间类型警告)
        dates = pd.to_datetime(y_true_df.index).repeat(24) 
        hours = np.tile(np.arange(24), len(y_true_df)) 
        datetimes = dates + pd.to_timedelta(hours, unit='h')

        detailed_df = pd.DataFrame({
            'datetime': datetimes,
            'true_load': y_true_df.values.flatten(),
            f'{model_name}_pred': y_pred_np.flatten()
        })
        
        # 强制洗成 datetime 类型
        detailed_df['datetime'] = pd.to_datetime(detailed_df['datetime'])
        
        detailed_df['abs_error'] = np.abs(detailed_df['true_load'] - detailed_df[f'{model_name}_pred'])
        detailed_df['mape_pct'] = (detailed_df['abs_error'] / detailed_df['true_load']) * 100

        # 提取时间特征用于 GroupBy 分析
        detailed_df['day_of_week'] = detailed_df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
        detailed_df['day_of_month'] = detailed_df['datetime'].dt.day       # 1-31
        detailed_df['date'] = detailed_df['datetime'].dt.date              # 提取纯日期用于查最差天数

        # 4. GroupBy: Day of Week 分析
        print(f"\n[2] Day of Week MAPE (%)  *0=Mon, 6=Sun* :")
        dow_mape = detailed_df.groupby('day_of_week')['mape_pct'].mean()
        dow_formatted = ", ".join([f"Day {dow}: {mape:.2f}" for dow, mape in dow_mape.items()])
        print(f"  {dow_formatted}")

        # 5. GroupBy: Day of Month 分析
        print(f"\n[3] Day of Month MAPE (%)  *1st to 31st* :")
        dom_mape = detailed_df.groupby('day_of_month')['mape_pct'].mean()
        dom_items = list(dom_mape.items())
        for i in range(0, len(dom_items), 10):
            dom_formatted = ", ".join([f"{d:02d}th: {mape:.2f}" for d, mape in dom_items[i:i+10]])
            print(f"  {dom_formatted}")

        # 6. 新增：找出表现最差的 Top 3 天并打印明细
        print(f"\n[4] Top 3 Worst Performing Days (Detailed Hourly Logs):")
        daily_mape = detailed_df.groupby('date')['mape_pct'].mean().sort_values(ascending=False)
        worst_days = daily_mape.head(3).index
        
        for i, w_date in enumerate(worst_days):
            w_mape = daily_mape.loc[w_date]
            print(f"  🔴 No.{i+1} Worst Day: {w_date} (Daily Avg MAPE: {w_mape:.2f}%)")
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

        # 7. 保存到本地与画图
        os.makedirs('results', exist_ok=True)
        save_path = f'results/{model_name}_detailed_errors.csv'
        detailed_df.drop(columns=['date']).to_csv(save_path, index=False) # 保存时去掉冗余的 date 列
        
        # 调用画图函数
        self._plot_error_distributions(model_name, hourly_mape, dow_mape, dom_mape)
        print(f"✅ Detailed predictions and errors saved to: {save_path}\n")


    def _evaluate_and_save_3d(self, model_name, y_true_np, y_pred_np, test_timestamps):
        """带有真实时间的 Transformer 评估逻辑 (支持分组统计与最差天数排查)"""
        print(f"\n====== {model_name} Error Analysis ======")
        
        # 1. 计算全局指标 
        mape = mean_absolute_percentage_error(y_true_np.flatten(), y_pred_np.flatten())
        mae = mean_absolute_error(y_true_np.flatten(), y_pred_np.flatten())
        rmse = np.sqrt(mean_squared_error(y_true_np.flatten(), y_pred_np.flatten()))
        print(f"Global -> MAPE: {mape*100:.2f}% | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

        # 2. 计算并打印 24 小时全部逐时 MAPE
        print(f"\n[1] Hourly MAPE (%):")
        hourly_mape = []
        for h in range(24):
            h_mape = mean_absolute_percentage_error(y_true_np[:, h], y_pred_np[:, h])
            hourly_mape.append(h_mape * 100)
            
        for h in range(0, 24, 8):
            formatted = ", ".join([f"h{i:02d}: {hourly_mape[i]:.2f}" for i in range(h, min(h+8, 24))])
            print(f"  {formatted}")

        # 3. 构造带有精确 Datetime 的详情大表 (修复时间类型警告)
        dates = pd.to_datetime(test_timestamps).repeat(24)
        hours = np.tile(np.arange(24), len(test_timestamps))
        datetimes = dates + pd.to_timedelta(hours, unit='h')

        detailed_df = pd.DataFrame({
            'datetime': datetimes,
            'true_load': y_true_np.flatten(),
            f'{model_name}_pred': y_pred_np.flatten()
        })
        
        # 强制洗成 datetime 类型
        detailed_df['datetime'] = pd.to_datetime(detailed_df['datetime'])
        
        detailed_df['abs_error'] = np.abs(detailed_df['true_load'] - detailed_df[f'{model_name}_pred'])
        detailed_df['mape_pct'] = (detailed_df['abs_error'] / detailed_df['true_load']) * 100

        # 提取时间特征用于 GroupBy 分析
        detailed_df['day_of_week'] = detailed_df['datetime'].dt.dayofweek  
        detailed_df['day_of_month'] = detailed_df['datetime'].dt.day       
        detailed_df['date'] = detailed_df['datetime'].dt.date              

        # 4. GroupBy: Day of Week 分析
        print(f"\n[2] Day of Week MAPE (%)  *0=Mon, 6=Sun* :")
        dow_mape = detailed_df.groupby('day_of_week')['mape_pct'].mean()
        dow_formatted = ", ".join([f"Day {dow}: {mape:.2f}" for dow, mape in dow_mape.items()])
        print(f"  {dow_formatted}")

        # 5. GroupBy: Day of Month 分析
        print(f"\n[3] Day of Month MAPE (%)  *1st to 31st* :")
        dom_mape = detailed_df.groupby('day_of_month')['mape_pct'].mean()
        dom_items = list(dom_mape.items())
        for i in range(0, len(dom_items), 10):
            dom_formatted = ", ".join([f"{d:02d}th: {mape:.2f}" for d, mape in dom_items[i:i+10]])
            print(f"  {dom_formatted}")

        # 6. 新增：找出表现最差的 Top 3 天并打印明细
        print(f"\n[4] Top 3 Worst Performing Days (Detailed Hourly Logs):")
        daily_mape = detailed_df.groupby('date')['mape_pct'].mean().sort_values(ascending=False)
        worst_days = daily_mape.head(3).index
        
        for i, w_date in enumerate(worst_days):
            w_mape = daily_mape.loc[w_date]
            print(f"  🔴 No.{i+1} Worst Day: {w_date} (Daily Avg MAPE: {w_mape:.2f}%)")
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

        # 7. 保存到本地与画图
        os.makedirs('results', exist_ok=True)
        save_path = f'results/{model_name}_detailed_errors.csv'
        detailed_df.drop(columns=['date']).to_csv(save_path, index=False)
        
        # 调用画图函数
        self._plot_error_distributions(model_name, hourly_mape, dow_mape, dom_mape)
        print(f"✅ Detailed predictions and errors saved to: {save_path}\n")

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
    
    # ==========================================
# 3. 在 PowerForecaster 类中添加 3D 训练方法
# ==========================================
# (请将此方法放入你的 PowerForecaster 类内)
    def train_transformer_3d(self, X_3d, y_3d, mask_3d, timestamps_3d, scaler_ts, params):
        # 1. 环境检查
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n🚀 GPU Acceleration: {device}")

        # --- 2. 数据划分与 Valid 过滤 ---
        train_size = int(len(X_3d) * 0.9)
        val_split = int(train_size * 0.9)

        test_timestamps = timestamps_3d[train_size:]
        
        # 获取原始训练集 
        X_tr_raw = X_3d[:val_split]
        # 🌟 修改 1：取消 [:, 0] 切片，保留完整的 24 小时 [N, 24]
        y_tr_raw = y_3d[:val_split] 
        mask_tr = mask_3d[:val_split]
        
        # 仅在训练集上剔除 is_valid == 0 的数据
        X_tr_filtered = X_tr_raw[mask_tr]
        y_tr_filtered = y_tr_raw[mask_tr]
        print(f"🧹 训练集去噪: 从 {len(X_tr_raw)} 缩减至 {len(X_tr_filtered)} 条高质量样本")

        # 转为 Tensor
        X_tr = torch.FloatTensor(X_tr_filtered)
        y_tr = torch.FloatTensor(y_tr_filtered)
        
        # 验证集和测试集（保留所有数据保证连续对齐，同样取消 [:, 0] 切片）
        X_val = torch.FloatTensor(X_3d[val_split:train_size])
        y_val = torch.FloatTensor(y_3d[val_split:train_size])
        
        X_te = torch.FloatTensor(X_3d[train_size:])
        y_te_scaled = y_3d[train_size:] # shape: [N_test, 24]

        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=params['batch_size'], shuffle=True)
        
        # 3. 模型定义
        model = TimeSeriesTransformer3D(num_features=X_3d.shape[2], params=params).to(device)
        criterion = nn.L1Loss()
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # 4. 训练循环
        best_val_loss = float('inf')
        early_stop_patience = 15  # 🌟 设定早停容忍度（比如连续 15 个 epoch 没提升就彻底停掉，通常要大于 scheduler 的 patience）
        epochs_no_improve = 0
        for epoch in range(params['epochs']):
            model.train()
            train_loss = 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                
                # 🌟 修改 2：去掉 squeeze(-1)，out 的 shape 变为 [batch_size, 24]
                out = model(bx) 
                loss = criterion(out, by) # 此时 out 和 by 都是 [batch_size, 24]
                
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
                torch.save(model.state_dict(), model_save_path)
                epochs_no_improve = 0  # 发现更好模型，容忍度计数器清零
            else:
                epochs_no_improve += 1 # 没发现更好模型，计数器 +1

            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1:03d} | Train: {train_loss/len(loader):.4f} | Val: {v_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 🌟 触发早停
            if epochs_no_improve >= early_stop_patience:
                print(f"\n🛑 Early stopping triggered at Epoch {epoch+1}! Validation loss hasn't improved for {early_stop_patience} epochs.")
                break

        # 5. 推理
        model.load_state_dict(torch.load('models/transformer_9am_full_features.pth'))
        model.eval()
        with torch.no_grad():
            # preds_scaled shape: [N_test, 24]
            preds_scaled = model(X_te.to(device)).cpu().numpy()
        
        # --- 6. 逆向还原 (🌟 修改 3：支持 2D 矩阵的逆变换) ---
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
        
        # 8. 计算 MAPE (展平后计算全局 MAPE)
        mape = mean_absolute_percentage_error(all_true_mw.flatten(), all_preds_mw.flatten())
        mae = mean_absolute_error(all_true_mw.flatten(), all_preds_mw.flatten())
        rmse = np.sqrt(mean_squared_error(all_true_mw.flatten(), all_preds_mw.flatten()))
        
        self.transformer_preds = all_preds_mw
        self._evaluate_and_save_3d("Transformer", all_true_mw, all_preds_mw, test_timestamps)
        print(f"\nMAPE: {mape*100:.2f}%")
        print(f"MAE : {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        
        return self.transformer_preds
    


# ==========================================
# 1. 正经时序 Transformer 必备：位置编码
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, Seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

# ==========================================
# 2. 3D Time Series Transformer 模型
# ==========================================
class TimeSeriesTransformer3D(nn.Module):
    def __init__(self, num_features, params):
        super().__init__()
        d_model = params['d_model']
        
        # 输入线性层：将每一时刻的 D 个特征投影到 d_model 维
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=params['nhead'], 
            dim_feedforward=d_model*4, dropout=params['dropout'], batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=params['num_layers'])
        
        # 输出头：打平所有时间步的输出，映射到 24 小时
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

