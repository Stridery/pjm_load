# src/feature_engine.py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib


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
        today, tomorrow = unique_days[i], unique_days[i+1]
        y_tomorrow = df_final.loc[df_final.index.date == tomorrow, 'Load'].values
        if len(y_tomorrow) != 24: continue
        
        tmrw_valid = 1 if df_final.loc[df_final.index.date == tomorrow, 'is_valid'].all() else 0
        f = {'timestamp': today}
        
        # 特征提取 (2D打平)
        weather_cols = ['Temp_F', 'Dewpoint_F', 'HeatIndex_F', 'SolarRadiation_Wm2', 'Windchill_F',
                        'WindSpeed_mph', 'WindDirection_deg', 'CloudCover_pct', 'Precip_in', 'RelativeHumidity_pct']

        for d in range(1, 8):  # d=1 昨天, d=7 七天前
            past_df = df_final.loc[df_final.index.date == unique_days[i - d]]
            for h in range(24): f[f'load_d{d}_h{h}'] = past_df['Load'].values[h]
            for col in weather_cols:
                vals = past_df[col].values
                for h in range(24): f[f'{col}_d{d}_h{h}'] = vals[h]

        today_meta = df_final.loc[df_final.index.date == today].iloc[0]
        tmrw_meta = df_final.loc[df_final.index.date == tomorrow].iloc[0]
        f.update({
            'today_month': today_meta['month'],
            'today_hour_sin': today_meta['hour_sin'], 'today_hour_cos': today_meta['hour_cos'],
            'today_month_sin': today_meta['month_sin'], 'today_month_cos': today_meta['month_cos'],
            'today_dayofweek': today_meta['dayofweek'],
            'tmrw_is_weekend': tmrw_meta['is_weekend'], 'tmrw_is_holiday': tmrw_meta['is_holiday'],
            'is_target_valid': tmrw_valid
        })

        y_dict = {'timestamp': today}
        for h in range(24): y_dict[f'h{h}'] = y_tomorrow[h]
        X_list.append(f); y_list.append(y_dict)

    X_opt = pd.DataFrame(X_list).set_index('timestamp')
    y_opt = pd.DataFrame(y_list).set_index('timestamp')
    os.makedirs(matrix_dir, exist_ok=True)
    X_opt.to_csv(x_path); y_opt.to_csv(y_path)
    return X_opt, y_opt


def build_timeseries_matrix(cleaned_path, matrix_dir, seq_len=72, pred_len=24):
    x_path = os.path.join(matrix_dir, f'X_3d_0am_seq{seq_len}.npy')
    y_path = os.path.join(matrix_dir, f'y_3d_0am_seq{seq_len}.npy')
    mask_path = os.path.join(matrix_dir, f'mask_3d_0am_seq{seq_len}.npy')
    scaler_path = os.path.join(matrix_dir, 'scaler_ts.pkl')

    timestamp_path = os.path.join(matrix_dir, f'timestamps_3d_0am_seq{seq_len}.npy')
    os.makedirs(matrix_dir, exist_ok=True)

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


    for i in tqdm(range(seq_len, len(df) - 48), desc="Midnight Anchoring"):

        if timestamps[i].hour != 0:
            continue

        X_window = data_array[i - seq_len : i]

        y_window = load_array[i + 24 : i + 48]

        is_seq_valid = is_valid_array[i + 24 : i + 48].all()

        X_list.append(X_window)
        y_list.append(y_window)
        valid_mask_list.append(is_seq_valid)
        timestamps_list.append(timestamps[i + 24])

    X_3d = np.array(X_list, dtype='float32')
    y_3d = np.array(y_list, dtype='float32')
    mask_3d = np.array(valid_mask_list, dtype=bool)
    timestamps_3d = np.array(timestamps_list)
    

    np.save(x_path, X_3d)
    np.save(y_path, y_3d)
    np.save(mask_path, mask_3d)
    np.save(timestamp_path, timestamps_3d)
    
    print(f"Done. Generated {len(X_3d)} daily samples. Valid samples: {mask_3d.sum()}")
    return X_3d, y_3d, mask_3d, timestamps_3d

def random_split_indices(n, test_frac=0.1, random_state=42):
    idx = np.arange(n)
    test_idx = np.random.default_rng(random_state).choice(idx, size=int(n * test_frac), replace=False)
    train_idx = np.setdiff1d(idx, test_idx)
    return train_idx, test_idx


def get_train_test_split(X_opt, y_opt, random_state=42):
    train_pos, test_pos = random_split_indices(len(X_opt), random_state=random_state)
    train_idx = X_opt.index[train_pos]
    test_idx = X_opt.index[test_pos]

    X_test_raw = X_opt.loc[test_idx]
    y_test = y_opt.loc[test_idx]
    X_train_raw = X_opt.loc[train_idx]
    y_train_raw = y_opt.loc[train_idx]

    train_mask = X_train_raw['is_target_valid'] == 1
    X_train = X_train_raw[train_mask].drop(columns=['is_target_valid'])
    y_train = y_train_raw[train_mask]

    X_test = X_test_raw.drop(columns=['is_target_valid'])
    return X_train, y_train, X_test, y_test