# src/config.py

# --- File Paths ---
RAW_LOAD_PATH = 'data/raw/dom_load.csv'
RAW_WEATHER_PATH = 'data/raw/pjm_dominionhub_hourly_2015_2025_openmeteo.csv'
MERGED_PATH = 'data/joined/merged_pjm_load_weather.csv'
CLEANED_PATH = 'data/cleaned/cleaned_pjm_load_weather.csv'
MATRIX_DIR = 'data/matrix/'

# --- Models to Train (1 = train, 0 = skip) ---
TRAIN_CONFIG = {
    'xgboost':     0,
    'lightgbm':    0,
    'transformer': 0,
    'lstm':        1,
}

# --- Tree Model Feature Generation ---
TREE_FEATURE_CONFIG = {
    'lookback_hours': 168,      # number of past hours used as features (e.g. 168 = 7 days)
    'latest_info_hour': 0,      # cutoff hour for available data when making the forecast:
                                #   <= 9 → today at that hour (e.g. 0 = midnight, 9 = 9am)
                                #   > 9  → previous day at that hour (e.g. 18 = yesterday 6pm)
    'split_strategy': 'tail',   # 'random' | 'head' | 'tail'
    'test_frac': 0.1,
    'random_state': 42,         # only used when split_strategy='random'
}

# --- Best Model Hyperparameters ---
XGB_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.012770027311556128,
    'max_depth': 3, 
    'subsample': 0.8739004809370342,
    'colsample_bytree': 0.7324710427863419,
    'reg_lambda': 9.33710650352912,
    'reg_alpha': 0.5117878514902956,
    'random_state': 42,
    'n_jobs': -1
}

LGBM_PARAMS = {
    'objective': 'regression', 
    'metric': 'mape', 
    'verbosity': -1,
    'boosting_type': 'gbdt', 
    'random_state': 42, 
    'learning_rate': 0.007546878079421479,
    'max_depth': 3, 
    'num_leaves': 49, 
    'feature_fraction': 0.7425994295050544, 
    'bagging_fraction': 0.6999177990613019,
    'bagging_freq': 6, 
    'lambda_l1': 0.0010071917640963458, 
    'lambda_l2': 1.4022452913170822, 
    'min_child_samples': 40, 
    'n_jobs': -1
}

# --- Transformer Feature Generation ---
TRANSFORMER_FEATURE_CONFIG = {
    'lookback_hours': 168,      # number of past hours used as input sequence (formerly seq_len)
    'latest_info_hour': 0,      # cutoff hour for available data when making the forecast:
                                #   <= 9 → today at that hour (e.g. 0 = midnight, 9 = 9am)
                                #   > 9  → previous day at that hour (e.g. 18 = yesterday 6pm)
    'split_strategy': 'tail',   # 'random' | 'head' | 'tail'
    'test_frac': 0.1,
    'val_strategy': 'random',     # 'random' | 'head' | 'tail' — how to split val from train pool
    'val_frac': 0.1,            # fraction of train pool used as validation
    'random_state': 42,         # only used when split_strategy or val_strategy='random'
}

TRANSFORMER_PARAMS = {
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'dropout': 0.3,
    'out_dim': 24,
    'epochs': 200,
    'batch_size': 32,
    'learning_rate': 3e-4,
    'weight_decay': 0.05
}

# --- LSTM Feature Generation (shares the same 3D matrix as Transformer) ---
LSTM_FEATURE_CONFIG = {
    'lookback_hours': 168,
    'latest_info_hour': 0,
    'split_strategy': 'tail',
    'test_frac': 0.1,
    'val_strategy': 'random',
    'val_frac': 0.1,
    'random_state': 42,
}

LSTM_PARAMS = {
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'out_dim': 24,
    'epochs': 200,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
}

# --- Evaluation Config ---
EVAL_CONFIG = {
    # Shared test split (all models use the same split)
    'split_strategy': 'tail',   # 'head' | 'tail' | 'random'
    'test_frac': 0.1,
    'random_state': 42,
    'result_dir': 'results/evaluation',

    # Which models to evaluate and where their saved files are
    'models': {
        'xgboost': {
            'enabled': 0,
            'model_path': 'models/xgboost/tail_test0.1/xgboost_24_models.pkl',
        },
        'lightgbm': {
            'enabled': 0,
            'model_path': 'models/lightgbm/tail_test0.1/lightgbm_24_models.pkl',
        },
        'transformer': {
            'enabled': 0,
            'model_path': 'models/transformer/tail_test0.1_random_val0.1/transformer_best.pth',
        },
        'lstm': {
            'enabled': 0,
            'model_path': 'models/lstm/tail_test0.1_random_val0.1/lstm_best.pth',
        },
    },

    # Single-day plot mode: load model, find date in matrix, show plot interactively
    'single_day': {
        'enabled': 1,
        'model': 'lstm',
        'model_path': 'models/lstm/tail_test0.1_random_val0.1/lstm_best.pth',
        'date': '2025-08-15',
    },
}