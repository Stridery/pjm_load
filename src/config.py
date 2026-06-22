# src/config.py
import os

# --- Dataset Selection (controls all data / model / result paths) ---
DATASET = os.environ.get('PJM_DATASET', 'dom')   # override: PJM_DATASET=dom python ...

# --- Weather Features (per dataset) ---
# dom columns match the output of data_crawler (Open-Meteo native variables).
# bge columns match the legacy manually-prepared weather file.
_WEATHER_COLS = {
    'dom': [
        'Temp_F', 'ApparentTemp_F', 'Dewpoint_F', 'RelativeHumidity_pct',
        'SolarRadiation_Wm2', 'WindSpeed_mph', 'WindDirection_deg',
        'WindGusts_mph', 'Precip_in', 'CloudCover_pct', 'SoilTemp0_7cm_F',
    ],
    'bge': [
        'Temp_F', 'ApparentTemp_F', 'Dewpoint_F', 'RelativeHumidity_pct',
        'SolarRadiation_Wm2', 'WindSpeed_mph', 'WindDirection_deg',
        'WindGusts_mph', 'Precip_in', 'CloudCover_pct', 'SoilTemp0_7cm_F',
    ],
}
WEATHER_COLS = _WEATHER_COLS.get(DATASET, [])  # empty for 'joint'; joint uses _WEATHER_COLS per zone

# --- File Paths ---
RAW_LOAD_PATH    = f'data/{DATASET}/raw/dom_load.csv'
RAW_WEATHER_PATH = f'data/{DATASET}/raw/pjm_dominionhub_hourly_2015_2025_openmeteo.csv'
MERGED_PATH      = f'data/{DATASET}/joined/merged_pjm_load_weather.csv'
CLEANED_PATH     = f'data/{DATASET}/cleaned/cleaned_pjm_load_weather.csv'
MATRIX_DIR       = f'data/{DATASET}/matrix/'

# ---------------------------------------------------------------------------
# Data Crawler Configuration
# ---------------------------------------------------------------------------
# Controls src/data_crawler.run_pipeline().
# Set PJM_API_KEY in your environment (or directly here for local runs).
CRAWLER_CONFIG = {
    # PJM zone abbreviation used in both metered-load and forecast endpoints.
    # Common values: 'DOM', 'BGE', 'PECO', 'PPL', 'PSEG', 'AEP', 'DAY', 'DUQ'
    'pjm_zone': os.environ.get('PJM_DATASET', 'BGE').upper(),

    # PJM Dataminer 2 API subscription key.
    # Obtain a free key at https://dataminer2.pjm.com/
    'pjm_api_key': os.environ.get('PJM_API_KEY', ''),

    # Location name passed to Open-Meteo Geocoding API.
    # Should be a city / region representative of the load area's geography.
    'location_name': os.environ.get('OPENMETEO_LOCATION', 'Baltimore'),

    # IANA timezone string for Open-Meteo requests and timezone alignment.
    # All output timestamps are normalised to this local time (naive).
    'timezone': 'America/New_York',

    # Inclusive year range for batch crawling.
    'start_year': int(os.environ.get('CRAWLER_START_YEAR', 2020)),
    'end_year':   int(os.environ.get('CRAWLER_END_YEAR',   2024)),
}

# --- Models to Train (1 = train, 0 = skip) ---
TRAIN_CONFIG = {
    'xgboost':     0,
    'lightgbm':    0,
    'transformer': 1,
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
    'n_jobs': -1,
    'use_lds': True,    # Label Distribution Smoothing: upweight rare peak-load samples
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
    'n_jobs': -1,
    'use_lds': True,    # Label Distribution Smoothing: upweight rare peak-load samples
}

# --- Transformer Feature Generation ---
TRANSFORMER_FEATURE_CONFIG = {
    'lookback_hours': 168,      # number of past hours used as input sequence (formerly seq_len)
    'latest_info_hour': 0,      # cutoff hour for available data when making the forecast:
                                #   <= 9 → today at that hour (e.g. 0 = midnight, 9 = 9am)
                                #   > 9  → previous day at that hour (e.g. 18 = yesterday 6pm)
    'split_strategy': 'tail',   # 'random' | 'head' | 'tail'
    'test_frac': 0.1,
    'val_strategy': 'tail',     # 'random' | 'head' | 'tail' — how to split val from train pool
    'val_frac': 0.1,            # fraction of train pool used as validation
    'random_state': 42,         # only used when split_strategy or val_strategy='random'
}

TRANSFORMER_PARAMS = {
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'dropout': 0.4,
    'out_dim': 24,
    'epochs': 200,
    'batch_size': 32,
    'learning_rate': 3e-4,
    'weight_decay': 0.05,
    'use_lds': True,    # Label Distribution Smoothing: upweight rare peak-load days
    'use_fds': True,    # Feature Distribution Smoothing: calibrate encoder features
    'fds_start_epoch': 5,   # warmup epochs before FDS calibration activates
    'fds_n_bins': 50,
    'fds_ks': 5,
    'fds_sigma': 2.0,
}

# --- LSTM Feature Generation (shares the same 3D matrix as Transformer) ---
LSTM_FEATURE_CONFIG = {
    'lookback_hours': 168,
    'latest_info_hour': 0,
    'split_strategy': 'tail',
    'test_frac': 0.1,
    'val_strategy': 'tail',
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
    'weight_decay': 1e-3,
    'use_lds': True,    # Label Distribution Smoothing: upweight rare peak-load days
    'use_fds': True,    # Feature Distribution Smoothing: calibrate encoder features
    'fds_start_epoch': 5,
    'fds_n_bins': 50,
    'fds_ks': 5,
    'fds_sigma': 2.0,
}

# --- Joint Model Configuration ---
# Add/remove zone abbreviations here; everything else is derived automatically.
JOINT_ZONES   = ['dom', 'bge']
JOINT_DATASET = 'joint_' + '_'.join(JOINT_ZONES)
JOINT_CLEANED_PATH = f'data/{JOINT_DATASET}/cleaned/joint_cleaned.csv'
JOINT_MATRIX_DIR   = f'data/{JOINT_DATASET}/matrix/'

JOINT_FEATURE_CONFIG = {
    'lookback_hours':  168,
    'latest_info_hour': 0,
    'split_strategy':  'tail',
    'test_frac':        0.1,
    'val_strategy':    'tail',
    'val_frac':         0.1,
    'random_state':     42,
}

JOINT_TRANSFORMER_PARAMS = {
    **TRANSFORMER_PARAMS,
    'out_dim':              24 * len(JOINT_ZONES),
    'fc_hidden':            256,   # wider output head for 48-dim joint prediction
    'epochs':               400,
    'early_stop_patience':  50,
}

JOINT_LSTM_PARAMS = {
    **LSTM_PARAMS,
    'out_dim':              24 * len(JOINT_ZONES),
    'fc_hidden':            256,
    'epochs':               400,
    'early_stop_patience':  50,
}

# --- Path suffixes (derived from param dicts, appended to model directory names) ---
_xgb_lds  = '_lds' if XGB_PARAMS.get('use_lds')         else ''
_lgbm_lds = '_lds' if LGBM_PARAMS.get('use_lds')        else ''
_tr_lds   = '_lds' if TRANSFORMER_PARAMS.get('use_lds') else ''
_lstm_lds = '_lds' if LSTM_PARAMS.get('use_lds')        else ''
_tr_fds   = '_fds' if TRANSFORMER_PARAMS.get('use_fds') else ''
_lstm_fds = '_fds' if LSTM_PARAMS.get('use_fds')        else ''

# --- Evaluation Config ---
EVAL_CONFIG = {
    # Shared test split (all models use the same split)
    'split_strategy': 'tail',   # 'head' | 'tail' | 'random'
    'test_frac': 0.1,
    'val_strategy': 'tail',   # sequence model result path only; 'head' | 'tail' | 'random'
    'val_frac': 0.1,
    'random_state': 42,
    'result_dir': f'results/{DATASET}/evaluation',

    # Which models to evaluate and where their saved files are
    'models': {
        'xgboost': {
            'enabled': 0,
            'model_path': f'models/{DATASET}/xgboost/tail_test0.1{_xgb_lds}/xgboost_24_models.pkl',
        },
        'lightgbm': {
            'enabled': 0,
            'model_path': f'models/{DATASET}/lightgbm/tail_test0.1{_lgbm_lds}/lightgbm_24_models.pkl',
        },
        'transformer': {
            'enabled': 0,
            'model_path': f'models/{DATASET}/transformer/tail_test0.1_tail_val0.1{_tr_lds}{_tr_fds}/transformer_best.pth',
        },
        'lstm': {
            'enabled': 0,
            'model_path': f'models/{DATASET}/lstm/tail_test0.1_tail_val0.1{_lstm_lds}{_lstm_fds}/lstm_best.pth',
        },
    },

    # Single-day plot mode: load model, find date in matrix, show plot interactively
    'single_day': {
        'enabled': 1,
        'model': 'transformer',
        'model_path': f'models/{DATASET}/transformer/tail_test0.1_tail_val0.1{_tr_lds}{_tr_fds}/transformer_best.pth',
        'date': '2026-01-24',
    },
}

# --- Joint Evaluation Config ---
JOINT_EVAL_CONFIG = {
    'split_strategy': 'tail',
    'test_frac':       0.1,
    'val_strategy':   'tail',
    'val_frac':        0.1,
    'random_state':    42,
    'result_dir':     f'results/{JOINT_DATASET}/evaluation',
    'models': {
        'transformer': {
            'enabled': 1,
            'model_path': f'models/{JOINT_DATASET}/transformer/tail_test0.1_tail_val0.1{_tr_lds}{_tr_fds}/transformer_best.pth',
        },
        'lstm': {
            'enabled': 1,
            'model_path': f'models/{JOINT_DATASET}/lstm/tail_test0.1_tail_val0.1{_lstm_lds}{_lstm_fds}/lstm_best.pth',
        },
    },
    'single_day': {
        'enabled':    0,
        'model_type': 'transformer',
        'model_path': f'models/{JOINT_DATASET}/transformer/tail_test0.1_random_val0.1{_tr_lds}{_tr_fds}/transformer_best.pth',
        'date':       '2024-08-15',
    },
}