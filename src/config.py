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

# --- Forecast Weather ---
# Always on: the 48 h forecast issued the day before the cutoff (day d + the target day) is
# spliced into the STATIC feature block at matrix-build time (see src/forecast_features.py).
# The cleaned CSVs don't carry it — it lives on its own index in cleaned/forecast.csv and is
# joined by EPT date. Days without a complete forecast are dropped from the matrix.
MODEL_ROOT  = f'models/{DATASET}'
RESULT_ROOT = f'results/{DATASET}'

# --- File Paths ---
MERGED_PATH      = f'data/{DATASET}/joined/merged_pjm_load_weather.csv'
CLEANED_PATH     = f'data/{DATASET}/cleaned/cleaned_pjm_load_weather.csv'   # labelled rows -> training
FORECAST_PATH    = f'data/{DATASET}/cleaned/forecast.csv'                   # archived weather forecasts
MATRIX_DIR       = f'data/{DATASET}/matrix/'

# --- Load_Estimated scale correction (per zone) ---
# Most zones publish their OWN preliminary load, so Load_Estimated is already at zone scale
# (ratio ~1). BGE does NOT — PJM only publishes the MIDATL regional aggregate, ~8.8x BGE
# (median over history; the ratio is flat, CV 4.2%, 8.5–9.0 across all months). The crawler
# divides it to BGE scale at ingest so Load_Estimated is a same-scale recent-load proxy
# everywhere downstream. This matters for the RESIDUAL models: their baseline reconstructs
# raw MW from Load_Estimated, and a ~26,000 MW baseline made the residual target track
# -MIDATL instead of BGE's own deviation. Direct models are unaffected (a constant divisor
# vanishes under standardization / tree scale-invariance), so nothing else needs to change.
LOAD_ESTIMATED_DIVISOR = {'bge': 8.8}

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
    # PJM zone abbreviation used in both metered-load and forecast endpoints, and — via
    # run_pipeline — the data/{zone}/ folder everything is written to. Derived from
    # DATASET, like every other path here, so the zone and the geocoded location below
    # cannot disagree. (It used to re-read PJM_DATASET with its own 'BGE' fallback, which
    # made a plain run write into data/bge/ while fetching Richmond weather.)
    'pjm_zone': DATASET.upper(),

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

    # Inclusive year range for batch crawling. History starts 2021 (2020 dropped).
    'start_year': int(os.environ.get('CRAWLER_START_YEAR', 2021)),
    'end_year':   int(os.environ.get('CRAWLER_END_YEAR',   2026)),
}

# --- PJM load auto-fetch (Dataminer2 CSV-download endpoint) ---
# Hardcoded per zone. metered uses the zone's own load_area; preliminary uses whatever
# aggregate PJM actually publishes for it — DOM has its own, but BGE (zone 'BC') has no
# zone-level preliminary, only the MIDATL regional total (rescaled by LOAD_ESTIMATED_DIVISOR).
PJM_LOAD_FETCH = {
    'dom': {'metered_area': 'DOM', 'prelim_area': 'DOM'},
    'bge': {'metered_area': 'BC',  'prelim_area': 'MIDATL'},
}
# Public anonymous key the dataminer2.pjm.com web app ships to every browser — not a secret.
# Override with your own PJM_API_KEY if you have one.
PJM_SUBSCRIPTION_KEY = os.environ.get('PJM_API_KEY') or '6a75d9f6d933401dbb4f36f8e70b95b3'

# --- Which years each crawl RE-FETCHES (EDIT THESE per run) ---
# Only years listed here are fetched — overwriting/creating their per-year files. Every other
# year is loaded from its cached file (a listed year is force-refreshed even if cached; a
# missing unlisted year is still auto-fetched so the merge stays complete). `--no-skip` ignores
# these and re-fetches the whole range.
# All four sources share ONE list — the recent years whose data is still settling (metered
# verifies for ~weeks into the next year; weather/forecast revise for a few days).
UPDATE_YEARS = [2025, 2026]   # metered, preliminary, observed weather, weather-forecast archive

# --- Models to Train (1 = train, 0 = skip) ---
TRAIN_CONFIG = {
    'xgboost':              1,
    'lightgbm':             1,
    'transformer':          1,
    'lstm':                 1,
    'moe_transformer':      1,
    'mstnn':                1,
    'xgboost_residual':     1,
    'lightgbm_residual':    1,
    'transformer_residual': 1,
    'lstm_residual':        1,
    'moe_transformer_residual': 1,
    'mstnn_residual':      1,
    'moe_mstnn':           1,
    'moe_mstnn_residual':  1,
    'moe_lstm':            1,
    'moe_lstm_residual':   1,
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
    'early_stop_patience': 50,
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
    # --- FDS: feature (encoder-representation) distribution smoothing ---
    # --- Stage 2: pluggable calibration of the expert heads (freezes encoder) ---
}

# --- MoE Transformer RESIDUAL (same architecture, learns y_metered - baseline) ---
# Same MoE model, but the target is the deviation from a naive same-hour-last-week
# baseline (mean of the past 7 days' preliminary load per clock-hour); the baseline
# is added back at inference. Treated as a SEPARATE model: own switch, own save path.
MOE_TRANSFORMER_RESIDUAL_FEATURE_CONFIG = dict(MOE_TRANSFORMER_FEATURE_CONFIG)
MOE_TRANSFORMER_RESIDUAL_PARAMS = {
    **MOE_TRANSFORMER_PARAMS,
    'residual_baseline': 'hourly',   # 'hourly' (24h profile) | 'scalar' (one number/day)
}

# --- MSTNN (simplified Multi-Scale Temporal NN; shares the 3D matrix + split) ---
MSTNN_FEATURE_CONFIG = dict(MOE_TRANSFORMER_FEATURE_CONFIG)   # same 0.16/0.19 split + embargo

MSTNN_PARAMS = {
    'loss': 'huber',                # Stage-1 loss: 'huber' | 'mse' | 'l1'
    'huber_delta': 1.0,             # knee in units of target std (targets are standardized)
    'lookback_hours': 168,          # reshaped to (7 days, 24 hours) grid; must be a multiple of 24
    'n_seq_features': N_SEQ_FEATURES,   # per-timestep feats (Load+weather+thermal) -> conv;
                                    #   the rest (calendar+macro+thermal-static) bypass the conv
    'mstnn_channels': 40,           # conv channels per branch (was 32 — that was too thin)
    'mstnn_kernels': [[3, 3], [5, 5], [7, 3]],   # 3 scales: local / medium / full-week (7d x 3h)
    'mstnn_pool': 'avg',            # 'avg' = global avg pool (small head, less overfit) | 'flatten'
    'fc_hidden': 128,               # ~69k params total: ~1.9x the old, well under transformer's 115k
    'dropout': 0.3,
    'out_dim': 24,
    'epochs': 200,
    'batch_size': 32,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'early_stop_patience': 50,
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
    'early_stop_patience': 50,
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

LGBM_RESIDUAL_PARAMS = {
    **LGBM_PARAMS,
    'baseline': _RESIDUAL_BASELINE,
}

TRANSFORMER_RESIDUAL_PARAMS = {
    **TRANSFORMER_PARAMS,
    'baseline': _RESIDUAL_BASELINE,
}

LSTM_RESIDUAL_PARAMS = {
    **LSTM_PARAMS,
    'baseline': _RESIDUAL_BASELINE,
}

# --- MSTNN RESIDUAL (same MSTNN, learns y_metered - baseline) ---
MSTNN_RESIDUAL_PARAMS = {
    **MSTNN_PARAMS,
    'baseline': _RESIDUAL_BASELINE,   # same model, only the target changes
}

# --- MoE-MSTNN (MoE regime head on the MSTNN conv encoder) ---
MOE_MSTNN_FEATURE_CONFIG = dict(MSTNN_FEATURE_CONFIG)
MOE_MSTNN_PARAMS = {
    **MSTNN_PARAMS,
    'expert_fc_hidden': 64,   # width of each of the 12 regime expert heads
}
MOE_MSTNN_RESIDUAL_PARAMS = {
    **MOE_MSTNN_PARAMS,
    'baseline': _RESIDUAL_BASELINE,
}

# --- MoE-LSTM (MoE regime head on the LSTM encoder) ---
MOE_LSTM_FEATURE_CONFIG = dict(LSTM_FEATURE_CONFIG)
MOE_LSTM_PARAMS = {
    **LSTM_PARAMS,
    'expert_fc_hidden': 64,   # width of each of the 12 regime expert heads
}
MOE_LSTM_RESIDUAL_PARAMS = {
    **MOE_LSTM_PARAMS,
    'baseline': _RESIDUAL_BASELINE,
}

# --- Evaluation Config ---
EVAL_CONFIG = {
    # Shared test split (all models use the same split)
    'split_strategy': 'tail',   # 'head' | 'tail' | 'random'
    'test_frac': 0.1,
    'val_strategy': 'tail',   # sequence model result path only; 'head' | 'tail' | 'random'
    'val_frac': 0.1,
    'random_state': 42,
    'result_dir': f'{RESULT_ROOT}/evaluation',

    # Which models to evaluate and where their saved files are
    'models': {
        'xgboost': {
            'enabled': 0,
            'model_path': f'{MODEL_ROOT}/xgboost/tail_test0.1/xgboost_24_models.pkl',
        },
        'lightgbm': {
            'enabled': 0,
            'model_path': f'{MODEL_ROOT}/lightgbm/tail_test0.1/lightgbm_24_models.pkl',
        },
        'transformer': {
            'enabled': 0,
            'model_path': f'{MODEL_ROOT}/transformer/tail_test0.1_tail_val0.1/transformer_best.pth',
        },
        'lstm': {
            'enabled': 0,
            'model_path': f'{MODEL_ROOT}/lstm/tail_test0.1_tail_val0.1/lstm_best.pth',
        },
        'moe_transformer': {
            'enabled': 1,
            'model_path': f'{MODEL_ROOT}/moe_transformer/tail_test0.16_tail_val0.1/moe_transformer_best.pth',
        },
        'mstnn': {
            'enabled': 0,
            'model_path': f'{MODEL_ROOT}/mstnn/tail_test0.16_tail_val0.19/mstnn_best.pth',
        },
        'mstnn_residual': {
            'enabled': 0,
            'model_path': f'{MODEL_ROOT}/mstnn_residual/tail_test0.16_tail_val0.19/mstnn_residual_best.pth',
        },
        'moe_mstnn': {
            'enabled': 0,
            'model_path': f'{MODEL_ROOT}/moe_mstnn/tail_test0.16_tail_val0.19/moe_mstnn_best.pth',
        },
        'moe_mstnn_residual': {
            'enabled': 0,
            'model_path': f'{MODEL_ROOT}/moe_mstnn_residual/tail_test0.16_tail_val0.1/moe_mstnn_residual_best.pth',
        },
        'moe_transformer_residual': {
            'enabled': 0,
            'model_path': f'{MODEL_ROOT}/moe_transformer_residual/tail_test0.16_tail_val0.1/moe_transformer_residual_best.pth',
        },
        'xgboost_residual': {
            'enabled': 0,
            'model_path': f'{MODEL_ROOT}/xgboost_residual/tail_test0.1/xgboost_residual_24_models.pkl',
        },
        'transformer_residual': {
            'enabled': 0,
            'model_path': f'{MODEL_ROOT}/transformer_residual/tail_test0.1_tail_val0.1/transformer_residual_best.pth',
        },
    },

    # Single-day plot mode: load model, find date in matrix, show plot interactively
    'single_day': {
        'enabled': 0,
        'model': 'moe_transformer_residual',
        'model_path': f'{MODEL_ROOT}/moe_transformer_residual/tail_test0.16_tail_val0.1/moe_transformer_residual_best.pth',
        'date': '2026-01-24',
    },
}

# ---------------------------------------------------------------------------

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
            'model_path': f'models/{JOINT_DATASET}/transformer/tail_test0.1_tail_val0.1/transformer_best.pth',
        },
        'lstm': {
            'enabled': 1,
            'model_path': f'models/{JOINT_DATASET}/lstm/tail_test0.1_tail_val0.1/lstm_best.pth',
        },
    },
    'single_day': {
        'enabled':    0,
        'model_type': 'transformer',
        'model_path': f'models/{JOINT_DATASET}/transformer/tail_test0.1_random_val0.1/transformer_best.pth',
        'date':       '2024-08-15',
    },
}