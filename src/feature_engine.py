# src/feature_engine.py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================================
# 1. 2D 矩阵构建 (专供 XGBoost / LightGBM)
# ==========================================
def build_or_load_matrix(cleaned_path, matrix_dir):
    """生成 2D CSV 矩阵，特征是打平的 (Flattened)"""
    x_path = os.path.join(matrix_dir, 'X_opt.csv')
    y_path = os.path.join(matrix_dir, 'y_opt.csv')

    if os.path.exists(x_path) and os.path.exists(y_path):
        print("=== Loading Pre-built 2D Matrix ===")
        return pd.read_csv(x_path, index_col=0, parse_dates=True), \
               pd.read_csv(y_path, index_col=0, parse_dates=True)

    print("=== Constructing 2D Matrix from Cleaned Data ===")
    df_final = pd.read_csv(cleaned_path, index_col=0, parse_dates=True)
    X_list, y_list = [], []
    unique_days = pd.Series(df_final.index.date).unique()

    # 这里的逻辑保持你最初的滑动窗口提取特征
    for i in tqdm(range(14, len(unique_days) - 1), desc="Building 2D Samples"):
        today, tomorrow, yesterday, last_week = unique_days[i], unique_days[i+1], unique_days[i-1], unique_days[i-6]
        y_tomorrow = df_final.loc[df_final.index.date == tomorrow, 'Load'].values
        if len(y_tomorrow) != 24: continue
        
        tmrw_valid = 1 if df_final.loc[df_final.index.date == tomorrow, 'is_valid'].all() else 0
        f = {'timestamp': today}
        
        # 特征提取 (2D打平)
        today_load = df_final.loc[df_final.index.date == today, 'Load'].values
        for h in range(9): f[f'load_today_h{h}'] = today_load[h]
        
        yest_load = df_final.loc[df_final.index.date == yesterday, 'Load'].values
        for h in range(24): f[f'load_yest_h{h}'] = yest_load[h]
        
        tmrw_meta = df_final.loc[df_final.index.date == tomorrow].iloc[0]
        f.update({'tmrw_month_sin': tmrw_meta['month_sin'], 'tmrw_month_cos': tmrw_meta['month_cos'],
                  'tmrw_dayofweek': tmrw_meta['dayofweek'], 'tmrw_is_weekend': tmrw_meta['is_weekend'],
                  'is_target_valid': tmrw_valid})

        y_dict = {'timestamp': today}
        for h in range(24): y_dict[f'h{h}'] = y_tomorrow[h]
        X_list.append(f); y_list.append(y_dict)

    X_opt = pd.DataFrame(X_list).set_index('timestamp')
    y_opt = pd.DataFrame(y_list).set_index('timestamp')
    os.makedirs(matrix_dir, exist_ok=True)
    X_opt.to_csv(x_path); y_opt.to_csv(y_path)
    return X_opt, y_opt

# ==========================================
# 2. 3D 矩阵构建 (专供 TimeSeries Transformer)
# ==========================================
def build_timeseries_matrix(cleaned_path, matrix_dir, seq_len=72, pred_len=24):
    """
    生成 3D NumPy 矩阵 [Batch, Seq, Feature]
    严格遵循 9 AM 截断逻辑：用今天 9 AM 之前的数据，预测明天全天 (24h)
    """
    x_path = os.path.join(matrix_dir, f'X_3d_9am_seq{seq_len}.npy')
    y_path = os.path.join(matrix_dir, f'y_3d_9am_seq{seq_len}.npy')
    mask_path = os.path.join(matrix_dir, f'mask_3d_9am_seq{seq_len}.npy')
    scaler_path = os.path.join(matrix_dir, 'scaler_ts.pkl')

    timestamp_path = os.path.join(matrix_dir, f'timestamps_3d_9am_seq{seq_len}.npy')

    if all(os.path.exists(p) for p in [x_path, y_path, mask_path, timestamp_path]):
        print(f"=== Loading 9 AM Cutoff Matrix (Seq={seq_len}) ===")
        return np.load(x_path), np.load(y_path), np.load(mask_path), np.load(timestamp_path, allow_pickle=True)

    print("=== Constructing 9 AM Cutoff Matrix (Day-Ahead Logic) ===")
    df = pd.read_csv(cleaned_path, index_col=0, parse_dates=True)
    
    # 确保索引有序
    df = df.sort_index()
    
    feature_cols = [
        'Load',                  # 负荷与周基准
        'Temp_F', 'Dewpoint_F', 'HeatIndex_F',  # 热力指标
        'SolarRadiation_Wm2', 'Windchill_F',    # 辐射与风寒
        'WindSpeed_mph', 'WindDirection_deg',   # 风速风向
        'CloudCover_pct', 'Precip_in',          # 云量与降水
        'RelativeHumidity_pct',                 # 湿度
        'hour_sin', 'hour_cos',                 # 日周期编码
        'month_sin', 'month_cos',               # 月周期编码
        'dayofweek',      # 周周期编码
        'is_weekend'         
    ]
    # 1. 严格防泄漏标准化 (Fit on first 80%)
    split_idx = int(len(df) * 0.9)
    scaler = StandardScaler()
    scaler.fit(df.iloc[:split_idx][feature_cols])
    
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    joblib.dump(scaler, scaler_path)

    X_list, y_list, valid_mask_list, timestamps_list = [], [], [], []
    data_array = df_scaled[feature_cols].values
    load_array = df_scaled['Load'].values 
    is_valid_array = df['is_valid'].values 
    timestamps = df.index

    # 2. 核心循环：寻找每天 9:00 AM
    # 我们需要留够 15h (Gap) + 24h (Target) = 39 小时的尾部空间
    for i in tqdm(range(seq_len, len(df) - 39), desc="9AM Anchoring"):
        # 🌟 只有在当天 9:00 AM 这一刻才生成样本
        if timestamps[i].hour != 9:
            continue
            
        # X: 截至今天 9 AM 的前 seq_len 小时 [i-72 : i]
        X_window = data_array[i - seq_len : i]
        
        # y: 明天 00:00 - 23:00 的 24 小时
        # i+1 到 i+15 是今天剩下的时间 (15h)，所以 y 从 i+15 开始
        y_window = load_array[i + 15 : i + 39]
        
        # Mask: 检查整段 111 小时 (72 + 15 + 24) 的有效性
        # 只要这段时间内有一个 Outlier，这整天就标记为 False (训练时剔除)
        is_seq_valid = is_valid_array[i - seq_len : i + 39].all()
        
        X_list.append(X_window)
        y_list.append(y_window)
        valid_mask_list.append(is_seq_valid)
        timestamps_list.append(timestamps[i + 15])

    X_3d = np.array(X_list, dtype='float32')
    y_3d = np.array(y_list, dtype='float32')
    mask_3d = np.array(valid_mask_list, dtype=bool)
    timestamps_3d = np.array(timestamps_list)
    
    # 存储
    np.save(x_path, X_3d)
    np.save(y_path, y_3d)
    np.save(mask_path, mask_3d)
    np.save(timestamp_path, timestamps_3d)
    
    print(f"Done. Generated {len(X_3d)} daily samples. Valid samples: {mask_3d.sum()}")
    return X_3d, y_3d, mask_3d, timestamps_3d

def get_train_test_split(X_opt, y_opt):
    split_idx = int(len(X_opt) * 0.9)
    X_train_raw, X_test_raw = X_opt.iloc[:split_idx], X_opt.iloc[split_idx:]
    y_train_raw, y_test = y_opt.iloc[:split_idx], y_opt.iloc[split_idx:]

    # Train exclusively on valid days
    train_mask = X_train_raw['is_target_valid'] == 1
    X_train = X_train_raw[train_mask].drop(columns=['is_target_valid'])
    y_train = y_train_raw[train_mask]
    
    # Test on all days
    X_test = X_test_raw.drop(columns=['is_target_valid'])
    return X_train, y_train, X_test, y_test