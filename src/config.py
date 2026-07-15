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

from src.thermal_features import THERMAL_SEQ_COLS   # per-hour thermal feats (go into the lookback)
N_SEQ_FEATURES = 1 + len(WEATHER_COLS) + len(THERMAL_SEQ_COLS)   # Load + weather + thermal

# --- File Paths ---
RAW_LOAD_PATH    = f'data/{DATASET}/raw/dom_load.csv'
RAW_WEATHER_PATH = f'data/{DATASET}/raw/pjm_dominionhub_hourly_2015_2025_openmeteo.csv'
MERGED_PATH      = f'data/{DATASET}/joined/merged_pjm_load_weather.csv'
CLEANED_PATH     = f'data/{DATASET}/cleaned/cleaned_pjm_load_weather.csv'   # labelled rows -> training
PREDICT_PATH     = f'data/{DATASET}/cleaned/predict.csv'                    # all rows, no Load -> forecasting
MATRIX_DIR       = f'data/{DATASET}/matrix/'

# ---------------------------------------------------------------------------
# Data Crawler Configuration
# ---------------------------------------------------------------------------
# Controls src/data_crawler.run_pipeline().
# Set PJM_API_KEY in your environment (or directly here for local runs).

# Weather location PER ZONE — the city Open-Meteo is geocoded against. This MUST
# follow the dataset: a single hardcoded default would silently fetch the wrong
# region's weather on a re-crawl (e.g. Baltimore weather for the Virginia zone).
_ZONE_LOCATION = {
    'dom': 'Richmond',    # Dominion Energy Virginia
    'bge': 'Baltimore',   # Baltimore Gas & Electric
}

CRAWLER_CONFIG = {
    # PJM zone abbreviation used in both metered-load and forecast endpoints.
    # Common values: 'DOM', 'BGE', 'PECO', 'PPL', 'PSEG', 'AEP', 'DAY', 'DUQ'
    'pjm_zone': os.environ.get('PJM_DATASET', 'BGE').upper(),

    # PJM Dataminer 2 API subscription key.
    # Obtain a free key at https://dataminer2.pjm.com/
    'pjm_api_key': os.environ.get('PJM_API_KEY', ''),

    # Location name passed to Open-Meteo Geocoding API — resolved from the dataset,
    # so `PJM_DATASET=dom` fetches Richmond, not Baltimore. Override with the
    # OPENMETEO_LOCATION env var or run_crawler.py's --location flag.
    'location_name': os.environ.get('OPENMETEO_LOCATION', _ZONE_LOCATION.get(DATASET, 'Baltimore')),

    # IANA timezone string for Open-Meteo requests and timezone alignment.
    # All output timestamps are normalised to this local time (naive).
    'timezone': 'America/New_York',

    # Inclusive year range for batch crawling.
    'start_year': int(os.environ.get('CRAWLER_START_YEAR', 2020)),
    'end_year':   int(os.environ.get('CRAWLER_END_YEAR',   2026)),
}

# --- Models to Train (1 = train, 0 = skip) ---
TRAIN_CONFIG = {
    'xgboost':              0,
    'lightgbm':             0,
    'transformer':          0,
    'lstm':                 0,
    'moe_transformer':      1,
    'mstnn':                1,
    'xgboost_residual':     0,
    'transformer_residual': 0,
}

# ---------------------------------------------------------------------------
# Regime / Mixture-of-Experts definition (used by MoE Transformer)
# ---------------------------------------------------------------------------
# Two-axis hard routing:
#   season  -> chosen per sample (day) from its month  (sample-level route)
#   hour    -> fixed ownership of each output hour 0-23 (output-index route)
# Each expert = (season group, a fixed set of hour indices). Season groups may
# hold a different number of experts, and the hour partition is season-specific.
# To retune a boundary, edit REGIME_MAP only — the model reads it directly.
SEASON_ORDER  = ['summer', 'upper_shoulder', 'shoulder', 'winter']   # fixed index order
SEASON_MONTHS = {
    'summer':         [7, 8],
    'upper_shoulder': [6, 9],
    'shoulder':       [3, 4, 5, 10, 11],
    'winter':         [12, 1, 2],
}
MONTH_TO_SEASON = {m: s for s, months in SEASON_MONTHS.items() for m in months}

# season -> {expert_name: [owned hour indices]}.  Hour lists within a season
# must be disjoint and together tile 0..23 (validated at model build time).
# Hour ranges are left-closed/right-open and wrap past midnight (e.g. [21,1) = 21,22,23,0).
REGIME_MAP = {
    'summer': {           # ☀️ 7/8 月
        'low':  [1, 2, 3, 4, 5, 6],                                    # 6h  [1,7)   深夜基负荷
        'peak': [12, 13, 14, 15, 16, 17, 18, 19, 20],                 # 9h  [12,21) 制冷尖峰
        'high': [0, 7, 8, 9, 10, 11, 21, 22, 23],                     # 9h  [7,12)∪[21,1)
    },
    'upper_shoulder': {   # 🌤️ 6/9 月
        'low':  [1, 2, 3, 4, 5, 6],                                    # 6h  [1,7)
        'peak': [14, 15, 16, 17, 18, 19],                             # 6h  [14,20)
        'high': [0, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23],         # 12h [7,14)∪[20,1)
    },
    'shoulder': {         # 🍂 3/4/5/10/11 月
        'low':  [0, 1, 2, 3, 4],                                       # 5h  [0,5)
        'peak': [6, 7, 8, 9, 17, 18, 19, 20],                         # 8h  [6,10)∪[17,21)
        'high': [5, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23],          # 11h {5}∪[10,17)∪[21,0)
    },
    'winter': {           # ❄️ 12/1/2 月
        'low':  [0, 1, 2, 3, 4],                                       # 5h  [0,5)
        'peak': [6, 7, 8, 9, 17, 18, 19, 20, 21],                     # 9h  [6,10)∪[17,22)
        'high': [5, 10, 11, 12, 13, 14, 15, 16, 22, 23],              # 10h {5}∪[10,17)∪[22,0)
    },
}

# --- Split embargo (for the 3-week macro features) ---
# Drop this many samples (days) on the training side of each split boundary
# (train|val and val|test) so a sample's 3-week feature window never straddles a
# boundary. 14 = 2 weeks (the macro window reaches 2 weeks past the 168h lookback).
# The test set is left intact. Applied on top of the front-end history trim
# (samples lacking 504h of history are skipped at matrix build time).
EMBARGO_DAYS = 14

# --- Tree Model Feature Generation ---
TREE_FEATURE_CONFIG = {
    'lookback_hours': 168,      # number of past hours used as features (e.g. 168 = 7 days)
    'latest_info_hour': 0,      # cutoff hour for available data when making the forecast:
                                #   <= 9 → today at that hour (e.g. 0 = midnight, 9 = 9am)
                                #   > 9  → previous day at that hour (e.g. 18 = yesterday 6pm)
    'split_strategy': 'tail',   # 'random' | 'head' | 'tail'
    'test_frac': 0.16,
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
    'use_lds': False,
    'lds_bin_width': 200.0,
    'lds_ks': 5,
    'lds_sigma': 2.0,
    'lds_min_freq_ratio': 0.05,
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
    'use_lds': False,
    'lds_bin_width': 200.0,
    'lds_ks': 5,
    'lds_sigma': 2.0,
    'lds_min_freq_ratio': 0.05,
}

# --- Transformer Feature Generation ---
TRANSFORMER_FEATURE_CONFIG = {
    'lookback_hours': 168,      # number of past hours used as input sequence (formerly seq_len)
    'latest_info_hour': 0,      # cutoff hour for available data when making the forecast:
                                #   <= 9 → today at that hour (e.g. 0 = midnight, 9 = 9am)
                                #   > 9  → previous day at that hour (e.g. 18 = yesterday 6pm)
    'split_strategy': 'tail',   # 'random' | 'head' | 'tail'
    'test_frac': 0.16,
    'val_strategy': 'tail',     # 'random' | 'head' | 'tail' — how to split val from train pool
    'val_frac': 0.1,            # fraction of train pool used as validation
    'random_state': 42,         # only used when split_strategy or val_strategy='random'
}

TRANSFORMER_PARAMS = {
    'loss': 'huber',                # Stage-1 loss: 'huber' | 'mse' | 'l1'
    'huber_delta': 1.0,             # knee in units of target std (targets are standardized)
    'n_seq_features': N_SEQ_FEATURES,   # per-timestep feats (Load+weather+thermal);
                                # the rest (calendar+macro+thermal-static) bypass the encoder
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'dropout': 0.3,
    'out_dim': 24,
    'epochs': 200,
    'batch_size': 32,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'use_lds': False,
    'lds_bin_width': 200.0,
    'lds_ks': 5,
    'lds_sigma': 1.0,
    'lds_min_freq_ratio': 0.05,
    'use_fds': False,
    'fds_start_epoch': 30,
    'fds_bin_width': 200.0,
    'fds_ks': 5,
    'fds_sigma': 0.5,
    'fds_momentum': 0.1,   # EMA weight on current epoch's stats (lower = more stable history)
    'early_stop_patience': 50,
    'stage2_epochs': 0,        # Stage 2 calibration epochs (0 = disabled)
    'stage2_mode': 'pinball',       # 'mse' | 'bft' | 'pinball'
    'stage2_lr': 3e-4,
    'stage2_bft_n_bins': 10,    # bft: equal-width load bins for resampling
    'stage2_q_max': 0.6,        # pinball: max quantile target for peak-load hours
    'stage2_p': 4.0,            # pinball: curvature of q(load_pct) schedule
    'stage2_routing': 'pred',   # pinball: 'pred' (deployable) | 'true' (diagnostic upper bound)
}

# --- MoE Transformer (shares the same 3D matrix + feature config as Transformer) ---
# Shared encoder + 8 regime expert heads (see REGIME_MAP). Reuses the transformer
# feature config for splitting; only the head/routing differs.
MOE_TRANSFORMER_FEATURE_CONFIG = {
    **TRANSFORMER_FEATURE_CONFIG,
    'test_frac': 0.16,   # last ~1 year  = out-of-time test  (all seasons, dom: 2025-05→2026-05)
    'val_frac':  0.1,   # prior ~1 year = validation        (all seasons, dom: 2024-04→2025-05)
}                        # train = the earlier ~4 years (dom: 2020-01→2024-04)

MOE_TRANSFORMER_PARAMS = {
    'loss': 'huber',                # Stage-1 loss: 'huber' | 'mse' | 'l1'
    'huber_delta': 1.0,             # knee in units of target std (targets are standardized)
    'n_seq_features': N_SEQ_FEATURES,   # per-timestep feats (Load+weather+thermal);
                                # the rest (calendar+macro+thermal-static) bypass the encoder
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'dropout': 0.3,
    'out_dim': 24,             # total hours; must equal sum of REGIME_MAP hour lists per season
    'expert_fc_hidden': 64,    # hidden width of each expert head MLP
    'epochs': 200,
    'batch_size': 32,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'early_stop_patience': 50,
    'use_lds': False,          # per-day LDS sample weighting (same scheme as transformer)
    'lds_bin_width': 200.0,
    'lds_ks': 5,
    'lds_sigma': 1.0,
    'lds_min_freq_ratio': 0.05,
    # --- FDS: feature (encoder-representation) distribution smoothing ---
    'use_fds': False,
    'fds_start_epoch': 30,
    'fds_bin_width': 200.0,
    'fds_ks': 5,
    'fds_sigma': 0.5,
    'fds_momentum': 0.1,       # EMA weight on current epoch's stats (lower = more stable)
    # --- Stage 2: pluggable calibration of the expert heads (freezes encoder) ---
    'stage2_epochs': 0,        # 0 = disabled
    'stage2_mode': 'pinball',  # 'mse' | 'bft' | 'pinball'
    'stage2_lr': 3e-4,
    'stage2_bft_n_bins': 10,   # bft: equal-frequency load bins for resampling
    'stage2_q_max': 0.6,       # pinball: max quantile target for peak-load hours
    'stage2_p': 4.0,           # pinball: curvature of q(load_pct) schedule
    'stage2_routing': 'pred',  # pinball: 'pred' (deployable) | 'true' (diagnostic upper bound)
}

# --- MSTNN (simplified Multi-Scale Temporal NN; shares the 3D matrix + split) ---
MSTNN_FEATURE_CONFIG = dict(MOE_TRANSFORMER_FEATURE_CONFIG)   # same 0.16/0.19 split + embargo

MSTNN_PARAMS = {
    'loss': 'huber',                # Stage-1 loss: 'huber' | 'mse' | 'l1'
    'huber_delta': 1.0,             # knee in units of target std (targets are standardized)
    'lookback_hours': 168,          # reshaped to (7 days, 24 hours) grid; must be a multiple of 24
    'n_seq_features': N_SEQ_FEATURES,   # per-timestep feats (Load+weather+thermal) -> conv;
                                    #   the rest (calendar+macro+thermal-static) bypass the conv
    'mstnn_channels': 32,           # conv channels per multi-scale branch
    'mstnn_kernels': [[3, 3], [5, 5]],   # parallel branches with different receptive fields
    'mstnn_pool': 'avg',            # 'avg' = global avg pool (small head, less overfit) | 'flatten'
    'fc_hidden': 128,
    'dropout': 0.3,
    'out_dim': 24,
    'epochs': 200,
    'batch_size': 32,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'early_stop_patience': 50,
    'use_lds': False,
    'lds_bin_width': 200.0,
    'lds_ks': 5,
    'lds_sigma': 1.0,
    'lds_min_freq_ratio': 0.05,
    'use_fds': False,               # FDS needs a fixed encoder feat_dim; not wired for MSTNN
    'stage2_epochs': 0,             # 0 = disabled; 'mse' | 'bft' | 'pinball' supported
    'stage2_mode': 'pinball',
    'stage2_lr': 3e-4,
    'stage2_bft_n_bins': 10,
    'stage2_q_max': 0.6,
    'stage2_p': 4.0,
    'stage2_routing': 'pred',
}

# --- LSTM Feature Generation (shares the same 3D matrix as Transformer) ---
LSTM_FEATURE_CONFIG = {
    'lookback_hours': 168,
    'latest_info_hour': 0,
    'split_strategy': 'tail',
    'test_frac': 0.16,
    'val_strategy': 'tail',
    'val_frac': 0.1,
    'random_state': 42,
}

LSTM_PARAMS = {
    'loss': 'huber',                # Stage-1 loss: 'huber' | 'mse' | 'l1'
    'huber_delta': 1.0,             # knee in units of target std (targets are standardized)
    'n_seq_features': N_SEQ_FEATURES,   # per-timestep feats (Load+weather+thermal);
                                # the rest (calendar+macro+thermal-static) bypass the encoder
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'out_dim': 24,
    'epochs': 200,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'use_lds': False,
    'lds_bin_width': 200.0,
    'lds_ks': 5,
    'lds_sigma': 1,
    'lds_min_freq_ratio': 0.05,
    'use_fds': False,
    'fds_start_epoch': 30,
    'fds_bin_width': 200.0,
    'fds_ks': 5,
    'fds_sigma': 0.5,
    'fds_momentum': 0.1,
    'early_stop_patience': 50,
    'stage2_epochs': 0,
    'stage2_mode': 'pinball',       # 'mse' | 'bft' | 'pinball'
    'stage2_lr': 1e-4,
    'stage2_bft_n_bins': 10,    # bft: equal-width load bins for resampling
    'stage2_q_max': 0.6,
    'stage2_p': 4.0,
    'stage2_routing': 'pred',
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
    'n_seq_features':       None,  # joint feature engine is not static-skip aware; use all features
    'out_dim':              24 * len(JOINT_ZONES),
    'fc_hidden':            256,   # wider output head for 48-dim joint prediction
    'epochs':               400,
    'early_stop_patience':  50,
}

JOINT_LSTM_PARAMS = {
    **LSTM_PARAMS,
    'n_seq_features':       None,  # joint feature engine is not static-skip aware; use all features
    'out_dim':              24 * len(JOINT_ZONES),
    'fc_hidden':            256,
    'epochs':               400,
    'early_stop_patience':  50,
}

# ---------------------------------------------------------------------------
# Residual models  (src/models/xgboost_residual.py, transformer_residual.py)
# ---------------------------------------------------------------------------
# Same features, same estimator, same split as the plain models — only the TARGET moves:
# predict the deviation from a naive same-hour-last-week baseline, then add the baseline
# back at inference. The baseline is built per-sample from that sample's own lookback of
# PRELIMINARY load, so nothing is fit on the training set (it cannot leak) and it is
# available in real time — the residual models deploy on the same forecast path as the rest.
#
#   'baseline': 'hourly' -> baseline[h] = mean of the past 7 days at clock-hour h (24-dim)
#               'scalar' -> one number per day (mean of the 168 h window), broadcast to 24 h
_RESIDUAL_BASELINE = 'hourly'

XGB_RESIDUAL_PARAMS = {
    **XGB_PARAMS,
    'baseline': _RESIDUAL_BASELINE,
}

TRANSFORMER_RESIDUAL_PARAMS = {
    **TRANSFORMER_PARAMS,
    'baseline': _RESIDUAL_BASELINE,
    # NOTE: use_lds / use_fds bin on the TARGET, which for this model is the residual and
    # not the load. Inherited from TRANSFORMER_PARAMS on purpose ("same model, only the
    # target changes"), but it is a different thing than it is over there.
}

# --- Path suffixes (derived from param dicts, appended to model directory names) ---
_xgb_lds  = '_lds' if XGB_PARAMS.get('use_lds')         else ''
_lgbm_lds = '_lds' if LGBM_PARAMS.get('use_lds')        else ''
_tr_lds   = '_lds' if TRANSFORMER_PARAMS.get('use_lds') else ''
_lstm_lds = '_lds' if LSTM_PARAMS.get('use_lds')        else ''
_tr_fds   = '_fds' if TRANSFORMER_PARAMS.get('use_fds') else ''
_lstm_fds = '_fds' if LSTM_PARAMS.get('use_fds')        else ''
_moe_lds  = '_lds' if MOE_TRANSFORMER_PARAMS.get('use_lds') else ''
_mstnn_lds = '_lds' if MSTNN_PARAMS.get('use_lds') else ''
_xgbres_lds = '_lds' if XGB_RESIDUAL_PARAMS.get('use_lds')         else ''
_trres_lds  = '_lds' if TRANSFORMER_RESIDUAL_PARAMS.get('use_lds') else ''
_trres_fds  = '_fds' if TRANSFORMER_RESIDUAL_PARAMS.get('use_fds') else ''

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
            'enabled': 1,
            'model_path': f'models/{DATASET}/transformer/tail_test0.1_tail_val0.1{_tr_lds}{_tr_fds}/transformer_best.pth',
        },
        'lstm': {
            'enabled': 1,
            'model_path': f'models/{DATASET}/lstm/tail_test0.1_tail_val0.1{_lstm_lds}{_lstm_fds}/lstm_best.pth',
        },
        'moe_transformer': {
            'enabled': 1,
            'model_path': f'models/{DATASET}/moe_transformer/tail_test0.16_tail_val0.19{_moe_lds}/moe_transformer_best.pth',
        },
        'mstnn': {
            'enabled': 0,
            'model_path': f'models/{DATASET}/mstnn/tail_test0.16_tail_val0.19{_mstnn_lds}/mstnn_best.pth',
        },
        'xgboost_residual': {
            'enabled': 1,
            'model_path': f'models/{DATASET}/xgboost_residual/tail_test0.1{_xgbres_lds}/xgboost_residual_24_models.pkl',
        },
        'transformer_residual': {
            'enabled': 1,
            'model_path': f'models/{DATASET}/transformer_residual/tail_test0.1_tail_val0.1{_trres_lds}{_trres_fds}/transformer_residual_best.pth',
        },
    },

    # Single-day plot mode: load model, find date in matrix, show plot interactively
    'single_day': {
        'enabled': 0,
        'model': 'transformer',
        'model_path': f'models/{DATASET}/transformer/tail_test0.1_tail_val0.1{_tr_lds}{_tr_fds}/transformer_best.pth',
        'date': '2026-01-24',
    },
}

# ---------------------------------------------------------------------------
# Forecast Config  (src/model_predictor.py — run after evaluation, or standalone)
# ---------------------------------------------------------------------------
# Forecasts the days PJM has not verified yet: everything in cleaned/predict.csv that
# has Load_Estimated + weather but no metered Load. Metered lags ~7 days and is only
# ever the LABEL — every model input comes from Load_Estimated + weather — so those
# days are fully forecastable, they just have no verified truth yet.
PREDICT_CONFIG = {
    'predict_path': PREDICT_PATH,
    # No result_dir of its own: a model's forecast is written next to that model's
    # detailed_errors.csv, under EVAL_CONFIG['result_dir'] / {model} / {run_tag}.

    # Score the forecast against the preliminary load series?
    #   dom: preliminary is load_area=DOM — the zone itself (prelim/metered ratio 1.01).
    #        A real near-real-time truth. Scored.
    #   bge: PJM publishes NO zone-level preliminary for BC. What we have is load_area=
    #        MIDATL, the whole Mid-Atlantic region — ~8.8x BGE's load, and the ratio moves
    #        seasonally (CV 4.2%, vs the model's own 8.7% MAPE). Rescaling it by a constant
    #        would invent a truth noisier than the thing it is meant to judge. So bge emits
    #        forecasts only; its true curve comes from metered, once PJM verifies it.
    'compare_to_preliminary': DATASET == 'dom',

    # Same shape as EVAL_CONFIG['models'] — enable a model and point at its weights.
    'models': {
        'xgboost': {
            'enabled': 1,
            'model_path': f'models/{DATASET}/xgboost/tail_test0.16{_xgb_lds}/xgboost_24_models.pkl',
        },
        'lightgbm': {
            'enabled': 1,
            'model_path': f'models/{DATASET}/lightgbm/tail_test0.16{_lgbm_lds}/lightgbm_24_models.pkl',
        },
        'transformer': {
            'enabled': 1,
            'model_path': f'models/{DATASET}/transformer/tail_test0.16_tail_val0.1{_tr_lds}{_tr_fds}/transformer_best.pth',
        },
        'lstm': {
            'enabled': 1,
            'model_path': f'models/{DATASET}/lstm/tail_test0.16_tail_val0.1{_lstm_lds}{_lstm_fds}/lstm_best.pth',
        },
        'moe_transformer': {
            'enabled': 1,
            'model_path': f'models/{DATASET}/moe_transformer/tail_test0.16_tail_val0.1{_moe_lds}/moe_transformer_best.pth',
        },
        'mstnn': {
            'enabled': 1,
            'model_path': f'models/{DATASET}/mstnn/tail_test0.16_tail_val0.1{_mstnn_lds}/mstnn_best.pth',
        },
        'xgboost_residual': {
            'enabled': 1,
            'model_path': f'models/{DATASET}/xgboost_residual/tail_test0.16{_xgbres_lds}/xgboost_residual_24_models.pkl',
        },
        'transformer_residual': {
            'enabled': 1,
            'model_path': f'models/{DATASET}/transformer_residual/tail_test0.16_tail_val0.1{_trres_lds}{_trres_fds}/transformer_residual_best.pth',
        },
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