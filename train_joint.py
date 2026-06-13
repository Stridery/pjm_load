"""
Joint model training: Transformer + LSTM on combined multi-zone data.

Usage
-----
    python train_joint.py

Zone list is configured via JOINT_ZONES in src/config.py.
"""

import logging
import os

import src.config as cfg
from src.joint_feature_engine import build_joint_cleaned, build_joint_timeseries_matrix
from src.joint_model_evaluator import JointModelEvaluator
from src.models import transformer as transformer_mod
from src.models import lstm as lstm_mod

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)

ZONES    = cfg.JOINT_ZONES
DATASET  = cfg.JOINT_DATASET
FEAT_CFG = cfg.JOINT_FEATURE_CONFIG
T_PARAMS = cfg.JOINT_TRANSFORMER_PARAMS
L_PARAMS = cfg.JOINT_LSTM_PARAMS

TEST_STRATEGIES = ['tail', 'random']
VAL_STRATEGIES  = ['tail', 'random']

# ---------------------------------------------------------------------------
# Step 1: Build joint cleaned CSV (skip if already exists)
# ---------------------------------------------------------------------------
if not os.path.exists(cfg.JOINT_CLEANED_PATH):
    print('=== Building joint cleaned CSV ===')
    build_joint_cleaned(ZONES)
else:
    print(f'Joint cleaned CSV exists: {cfg.JOINT_CLEANED_PATH}')

# ---------------------------------------------------------------------------
# Step 2: Build joint 3D matrix (cached after first build)
# ---------------------------------------------------------------------------
print('=== Building / loading joint matrix ===')
X_3d, y_3d, mask_3d, timestamps, y_scalers = build_joint_timeseries_matrix(
    zones=ZONES,
    weather_cols=cfg._WEATHER_COLS,
    joint_cleaned_path=cfg.JOINT_CLEANED_PATH,
    matrix_dir=cfg.JOINT_MATRIX_DIR,
    lookback_hours=FEAT_CFG['lookback_hours'],
    latest_info_hour=FEAT_CFG['latest_info_hour'],
    test_frac=FEAT_CFG['test_frac'],
)
print(f'X shape: {X_3d.shape} | y shape: {y_3d.shape} | valid: {mask_3d.sum()}/{len(mask_3d)}')

# ---------------------------------------------------------------------------
# Step 3: Train and evaluate
# ---------------------------------------------------------------------------
for test_strategy in TEST_STRATEGIES:
    for val_strategy in VAL_STRATEGIES:
        FEAT_CFG['split_strategy'] = test_strategy
        FEAT_CFG['val_strategy']   = val_strategy

        run_tag = f'{test_strategy}_test{FEAT_CFG["test_frac"]}_{val_strategy}_val{FEAT_CFG["val_frac"]}'
        eval_cfg = {
            'split_strategy': test_strategy,
            'test_frac':      FEAT_CFG['test_frac'],
            'val_strategy':   val_strategy,
            'val_frac':       FEAT_CFG['val_frac'],
            'random_state':   FEAT_CFG['random_state'],
            'result_dir':     f'results/{DATASET}/evaluation',
            'models': {
                'transformer': {'enabled': 0, 'model_path': f'models/{DATASET}/transformer/{run_tag}/transformer_best.pth'},
                'lstm':        {'enabled': 0, 'model_path': f'models/{DATASET}/lstm/{run_tag}/lstm_best.pth'},
            },
        }

        # --- Transformer ---
        print(f"\n{'='*60}")
        print(f'Joint Transformer | test: {test_strategy} | val: {val_strategy}')
        print('='*60)
        transformer_mod.train(X_3d, y_3d, mask_3d, T_PARAMS, FEAT_CFG, DATASET)
        t_path = f'models/{DATASET}/transformer/{run_tag}/transformer_best.pth'
        eval_cfg['models']['transformer'] = {'enabled': 1, 'model_path': t_path}
        evaluator = JointModelEvaluator(eval_cfg, ZONES)
        evaluator.X_3d       = X_3d
        evaluator.y_3d       = y_3d
        evaluator.mask_3d    = mask_3d
        evaluator.timestamps = timestamps
        evaluator.y_scalers  = y_scalers
        evaluator.evaluate_all()
        eval_cfg['models']['transformer']['enabled'] = 0

        # --- LSTM ---
        print(f"\n{'='*60}")
        print(f'Joint LSTM | test: {test_strategy} | val: {val_strategy}')
        print('='*60)
        lstm_mod.train(X_3d, y_3d, mask_3d, L_PARAMS, FEAT_CFG, DATASET)
        l_path = f'models/{DATASET}/lstm/{run_tag}/lstm_best.pth'
        eval_cfg['models']['lstm'] = {'enabled': 1, 'model_path': l_path}
        evaluator = JointModelEvaluator(eval_cfg, ZONES)
        evaluator.X_3d       = X_3d
        evaluator.y_3d       = y_3d
        evaluator.mask_3d    = mask_3d
        evaluator.timestamps = timestamps
        evaluator.y_scalers  = y_scalers
        evaluator.evaluate_all()
        eval_cfg['models']['lstm']['enabled'] = 0

print('\nAll joint training complete.')
