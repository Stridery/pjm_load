# src/config.py
import os

# --- Dataset Selection (controls all data / model / result paths) ---
# ONE zone per process: every path below is baked from it at import, so a process cannot
# straddle two zones. Train_Model.sbatch accepts a space-separated LIST in the same variable
# and loops, handing one zone to each child — hence the guard, which turns "I passed the
# sbatch spelling to python" into a sentence instead of a mkdir called 'data/dom bge'.
_DATASET = os.environ.get('PJM_DATASET', 'dom').strip()
if len(_DATASET.split()) != 1:
    raise ValueError(
        f"PJM_DATASET={_DATASET!r} names {len(_DATASET.split())} zones. A single process runs "
        f"exactly one zone; pass a list only to Train_Model.sbatch, which loops over it.")
DATASET = _DATASET

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
CONGESTION_PATH  = f'data/{DATASET}/cleaned/congestion.csv'                 # hourly DA shadow prices
MATRIX_DIR       = f'data/{DATASET}/matrix/'

# --- Congestion (PJM Day-Ahead marginal value) ---
# RTO-wide transmission-constraint shadow prices for the BLIND-SPOT DAY (the day between the
# cutoff and the target day, which the 168 h lookback cannot see). Spliced into the STATIC
# feature block at matrix-build time, exactly like the forecast weather above; the cleaned
# load CSV doesn't carry it. See src/congestion_features.py for what the features are and
# src/data_crawler/pjm_marginal_value.py for why Day-Ahead and why RTO-wide rather than
# per-zone. Days with no Day-Ahead data at all are dropped from the matrix.
# Parsed before the dict so the error names the variable rather than surfacing as an opaque
# int() failure. Same shape as PJM_DATASET: one value per process, a list only to the sbatch.
_CONG = os.environ.get('PJM_CONGESTION', '1').strip()
if _CONG not in ('0', '1'):
    raise ValueError(
        f"PJM_CONGESTION={_CONG!r} is not 0 or 1. A single process runs one arm; pass a list "
        f"(e.g. \"1 0\") only to Train_Model.sbatch, which loops over it.")
_CONGESTION = int(_CONG)

CONGESTION_CONFIG = {
    # 0 reverts every artifact to its pre-congestion shape — matrices, scalers, model run
    # dirs and the serving lookup all key off MATRIX_TAG below, so the old feature set stays
    # fully reproducible rather than being replaced. Verified bit-identical: building with
    # PJM_CONGESTION=0 reproduces the pre-congestion matrix exactly.
    #
    # The flag is part of MATRIX_TAG rather than a silent switch because the static block's
    # WIDTH depends on it: flipping it while a matrix cache sat on disk would otherwise feed
    # a narrower sample to models trained on the wider one, with no error until the
    # numbers came out wrong.
    #
    # Overridable per run, like PJM_DATASET, so an A/B needs no file edit (and cannot be left
    # flipped by accident):
    #     PJM_CONGESTION=0 python Model_Training.py
    #     PJM_CONGESTION=1 python Model_Training.py
    'enabled': _CONGESTION,

    # The trailing-mean windows the departures are taken against live in
    # src/congestion_features (W_SHORT / W_PRIMARY / W_LONG) rather than here: changing them
    # changes what the features MEAN, so they belong with the definitions and the measurement
    # that justifies them, not in a knob that looks tunable per run.
}

# Matrix-cache / scaler / model-run-dir tag. Every artifact whose CONTENT depends on the
# static block's width carries it, so a cache or a set of weights built under one setting can
# never be silently reused under another. Read identically by src/feature_engine,
# src/models/_utils.run_tag and src/serving.
MATRIX_TAG = '_cong' if CONGESTION_CONFIG['enabled'] else ''

# The two tags this repo writes. Needed because '' is a suffix of everything: with the block
# off, a plain endswith('') test would happily match a '_cong' directory and load weights that
# expect the wider sample. Longest match wins.
#
# Feature-set changes are NOT versioned into the tag — obsolete artifacts are deleted instead,
# which is why there is no '_cong2'/'_s2'. That keeps the naming honest, but it does mean a
# change to the block's contents must be followed by wiping data/{zone}/matrix*: the builders
# cache on filename alone, so a stale matrix under the same name would be silently reused.
MATRIX_TAGS = ('', '_cong')


def matrix_tag_of(name):
    """The tag `name` (a matrix filename stem or a model run dir) was written under."""
    return max((t for t in MATRIX_TAGS if name.endswith(t)), key=len)

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

# Per-run override: train ONLY these models, comma-separated, everything else skipped.
#     PJM_TRAIN_ONLY=transformer_residual python Model_Training.py
#     PJM_TRAIN_ONLY=residual python Model_Training.py        # every *_residual model
# For probing one change on one model without editing (and having to restore) the dict above
# — the A/B of the congestion block being the case in point. Unknown names are rejected here
# rather than silently training nothing, which is what a typo would otherwise look like.
_TRAIN_ONLY = os.environ.get('PJM_TRAIN_ONLY', '').strip()
if _TRAIN_ONLY:
    if _TRAIN_ONLY == 'residual':
        # Shorthand, resolved HERE rather than by whoever sets the variable. A caller that
        # tried to expand it first would have to import this module to read TRAIN_CONFIG —
        # and this module validates PJM_TRAIN_ONLY at import, so the expansion would die on
        # the very word it was expanding. Deriving it from the registry also picks up models
        # added later.
        _wanted = {m for m in TRAIN_CONFIG if m.endswith('_residual')}
    else:
        _wanted = {s.strip() for s in _TRAIN_ONLY.split(',') if s.strip()}
        _unknown = _wanted - set(TRAIN_CONFIG)
        if _unknown:
            raise ValueError(f"PJM_TRAIN_ONLY names unknown model(s): {sorted(_unknown)}. "
                             f"Valid: {sorted(TRAIN_CONFIG)}, or the shorthand 'residual'.")
    # Applies to BOTH branches — keeping this inside the else was what made
    # PJM_TRAIN_ONLY=residual compute the right set and then silently train all sixteen.
    TRAIN_CONFIG = {k: int(k in _wanted) for k in TRAIN_CONFIG}

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

# ---------------------------------------------------------------------------
# Probabilistic baselines — Xie & Hong (2018), IEEE TSG 9(3):1680-1687
# ---------------------------------------------------------------------------
# Temperature scenarios fed to an MLR point model, scored with the pinball loss. Driven by
# train_quantile.py; shares NOTHING with the point-model stack above except the cleaned CSV
# (no matrix cache, no scalers, no src/models registry — see src/quantile/__init__.py).
QUANTILE_CONFIG = {
    # Which of the paper's three MLR specifications to run (1 = run, 0 = skip).
    #
    # Vanilla is the shipped baseline, and it is the only one of the three that can be
    # claimed as a reproduction: it is the GEFCom2012 benchmark model, with a frozen
    # published specification the paper notes was "later reproduced by other scholars".
    #   hong2014 - the paper states its variable list "may change from year to year
    #              depending on the validation dataset that is used for the model selection
    #              process", so there is no specification to reproduce; ours is an invented
    #              instantiation. It scores within 0.5% of Vanilla either way.
    #   tcube    - is literally GSP + poly3(T), and no PJM zone has a state-product series,
    #              so substituting the linear Trend replaces HALF the model (in Vanilla the
    #              same substitution is one term among 285 columns).
    # Both were run before being switched off and reproduce the same qualitative findings
    # (optimum at the same n, same shifted-date gain, bootstrap no better than fixed-date);
    # their results are kept under results/{zone}/quantile/ as evidence. Flip to 1 to re-run.
    'models': {'tcube': 0, 'vanilla': 1, 'hong2014': 0},

    # Tail split on HOURLY rows, snapped back to an EPT day boundary so the per-day plots
    # have no partial day at either end. 0.16 matches the point models, so the probabilistic
    # baselines are scored on the same test period.
    'test_frac': 0.16,

    # Length of the estimation window ending at the split, in years (None = everything
    # before the split). 3 is the paper's fixed rolling-window length, and on DOM it also
    # matters a lot: the models carry growth in ONE linear Trend term (we have no GSP index
    # to stand in for the paper's macroeconomic driver), so a longer window averages the
    # zone's accelerating data-centre growth and under-forecasts the recent peaks. Measured
    # on dom: perfect-weather MAPE 3.28% at 3 y vs 3.68% on the full 4.66 y, and mean error
    # at the top decile +836 MW vs +1095 MW.
    'train_years': 3,

    # Scenario pool: complete calendar years of observed temperature that end strictly
    # BEFORE the test period starts (the paper keeps 1976-2005 clear of its 2006-2011 test
    # years). None = use every year that qualifies.
    #
    # Temperature is read from data/{zone}/raw/weather_long/ when that exists (run
    # fetch_scenario_weather.py; ERA5 reaches back to 1940), otherwise it falls back to the
    # cleaned CSV, which caps the pool at k=4. The long store is what makes the paper's
    # central results reachable at all: its findings are functions of k over 1..30, and four
    # points cannot show a curve flattening at fifteen. Verified to be the same series as the
    # cleaned CSV — 35,060 overlapping hours agree to within 1e-6. Note the pool spans a
    # warming climate (+0.85 F/decade at Richmond, +0.76 at Baltimore), so its oldest profiles
    # run ~2.5 F cool; the paper assigns past temperatures date-by-date with no such
    # adjustment and this follows it.
    'scenario_years': None,

    # Which k the grid is computed on. The n=0 column (fixed-date) is swept over EVERY
    # available k, which is cheap and reproduces the paper's Fig. 2/3; the full k x n grid is
    # restricted to the paper's own Table I columns, because a 30 x 41 grid spends most of its
    # time sorting ensembles of thousands of scenarios for cells nobody reports.
    'k_grid_shifted': [1, 2, 3, 4, 5, 10, 20, 30],

    # Shifted-date sweep. The paper's rule of thumb is n = ceil(30/k - 1) — 7 for k=4, 29
    # for k=1 — but the point of the grid is not to hardcode it: sweep 0..n_max and read the
    # optimum off the result (its Table III). 40 covers the full range of its Table I,
    # including the k=1 rule-of-thumb cell, and the sweep is cheap (the cost is the scenario
    # cube, which is built once).
    'n_max': 40,

    # Block bootstrap: m and l in the paper's notation.
    'bootstrap_block_days': 9,
    'bootstrap_scenarios': 100,

    'random_state': 42,
    'result_dir': f'{RESULT_ROOT}/quantile',
}


# ---------------------------------------------------------------------------
# Probabilistic baseline 2 — Liu, Nowotarski, Hong & Weron (2017), IEEE TSG 8(2):730-737
# ---------------------------------------------------------------------------
# Sister point forecasts -> quantile regression averaging. Driven by train_qra.py. Parallel
# to QUANTILE_CONFIG in every structural respect: same one input (the cleaned CSV), no matrix
# cache, no src/models registration, invisible to serving.
QRA_CONFIG = {
    'test_frac': 0.16,      # same tail split, so both baselines score the same test period
    'val_days': 365,        # validation year: selects the recency orders, m, and L

    # Sisters: `sisters_per_window` recency orders per estimation window (2 y and 3 y) = 8,
    # chosen off a (lag, dma) grid by validation MAE. See src/quantile/sisters.py for why the
    # orders are selected directly rather than via [26]'s four calibration schemes.
    'sisters_per_window': 4,
    'max_lag': 3,
    'max_dma': 3,


    'm_grid': [2, 3, 4, 5, 6, 7, 8],       # how many sisters QRA averages over

    # QRA calibration windows, in days. The paper sweeps 91/122/183/365; 365 CANNOT fit this
    # record and is dropped rather than faked. A 365-day window in front of a 365-day
    # validation year pushes the earliest sister forecast to 2023-08, whose own 3-year
    # estimation window would start 2020-08 — four months before the data begins. Keeping the
    # full validation year and capping L at 183 is the better of the two trims: a half-year
    # validation would select m and L on a seasonally lopsided sample, then apply them across
    # a full test year.
    'L_grid': [91, 122, 183],

    # Refit cadence, in days. The paper re-estimates everything daily; 1 reproduces that.
    # The validation sweep runs 21 (m, L) combinations, so it gets a coarser cadence — the
    # weights barely move when a 183-day window slides by one day, and this is model
    # SELECTION, not the reported result.
    'sister_refit_every': 1,
    'qra_refit_every_val': 7,
    'qra_refit_every_test': 1,

    'result_dir': f'{RESULT_ROOT}/qra',
}


# ---------------------------------------------------------------------------
# Probabilistic baseline 3 — Li, Wang & Zhang (2020), IEEE TSG 11(2):1679-1690
# ---------------------------------------------------------------------------
# Fifteen base density models -> a Gaussian mixture whose weights minimise the CRPS, solved
# as a small quadratic program. Driven by train_density.py. Same isolation as the other two
# baselines: one input (the cleaned CSV), no matrix cache, no src/models registration.
DENSITY_CONFIG = {
    'test_frac': 0.16,        # same tail split as the other two, so T4 is the shared test set

    # T2 tunes/reports the base models, T3 optimises the fifteen weights. The paper's split is
    # 10:1:1:1 on two years; here T4 is pinned to the shared test period instead and these two
    # take a fixed number of days before it. T3 only has to pin down fifteen weights — the
    # paper itself allows five weeks — so lengthening it would only starve T1.
    't2_days': 56,
    't3_days': 56,

    # Which of base_models.SPECS to run; None = all fifteen, as published.
    'models': None,

    # The paper reports Q_new positive definite in every case but does not prove it, so the
    # solver multi-starts and reports the certificate rather than assuming it.
    'n_starts': 8,
    'random_state': 0,

    'n_workers': None,        # None = the SLURM allocation, capped at 8
    'result_dir': f'{RESULT_ROOT}/density',
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