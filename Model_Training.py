import src.config as cfg
from src.feature_engine import build_or_load_matrix, get_train_test_split, build_timeseries_matrix
from src.model_evaluator import ModelEvaluator, TREE_MODELS, SEQ_MODELS
from src.models import xgboost as xgb_mod
from src.models import lightgbm as lgb_mod
from src.models import transformer as transformer_mod
from src.models import lstm as lstm_mod
from src.models import moe_transformer as moe_mod
from src.models import mstnn as mstnn_mod
from src.models import xgboost_residual as xgb_res_mod
from src.models import lightgbm_residual as lgbm_res_mod
from src.models import transformer_residual as tr_res_mod
from src.models import lstm_residual as lstm_res_mod
from src.models import moe_transformer_residual as moe_res_mod
from src.models import mstnn_residual as mstnn_res_mod
from src.models import moe_mstnn as moe_mstnn_mod
from src.models import moe_mstnn_residual as moe_mstnn_res_mod
from src.models import moe_lstm as moe_lstm_mod
from src.models import moe_lstm_residual as moe_lstm_res_mod

TEST_STRATEGIES = ['tail']
VAL_STRATEGIES  = ['tail']


def _only(model_name, model_path):
    m = {n: {'enabled': 0, 'model_path': ''} for n in TREE_MODELS + SEQ_MODELS}
    m[model_name] = {'enabled': 1, 'model_path': model_path}
    return m


def _run_eval(model_name, model_path, split_strategy, feature_cfg):
    """训练完成后立即评估。预测已从训练流水线里剥离，是每天独立跑的事；
    训练只负责 train + evaluate。"""
    eval_cfg = {
        'split_strategy': split_strategy,
        'test_frac':      feature_cfg['test_frac'],
        'random_state':   feature_cfg['random_state'],
        'result_dir':     f'{cfg.RESULT_ROOT}/evaluation',
        'models':         _only(model_name, model_path),
        'single_day':     {'enabled': 0, 'model': model_name, 'model_path': model_path, 'date': ''},
    }
    if 'val_strategy' in feature_cfg:
        eval_cfg['val_strategy'] = feature_cfg['val_strategy']
        eval_cfg['val_frac']     = feature_cfg['val_frac']
    evaluator = ModelEvaluator(eval_cfg)
    evaluator.load_data()
    evaluator.evaluate_all()


# --- Tree Models (XGBoost / LightGBM) ---
if cfg.TRAIN_CONFIG['xgboost'] or cfg.TRAIN_CONFIG['lightgbm']:
    X_opt, y_opt = build_or_load_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)

    for test_strategy in TEST_STRATEGIES:
        print(f"\n{'='*60}")
        print(f"Tree Models | test split: {test_strategy}")
        print(f"{'='*60}")

        cfg.TREE_FEATURE_CONFIG['split_strategy'] = test_strategy
        X_train, y_train, X_test, y_test = get_train_test_split(X_opt, y_opt)

        if cfg.TRAIN_CONFIG['xgboost']:
            xgb_mod.train(X_train, y_train, cfg.XGB_PARAMS, cfg.TREE_FEATURE_CONFIG)
            tf = cfg.TREE_FEATURE_CONFIG
            _run_eval('xgboost',
                      f'{cfg.MODEL_ROOT}/xgboost/{test_strategy}_test{tf["test_frac"]}/xgboost_24_models.pkl',
                      test_strategy, tf)

        if cfg.TRAIN_CONFIG['lightgbm']:
            lgb_mod.train(X_train, y_train, cfg.LGBM_PARAMS, cfg.TREE_FEATURE_CONFIG)
            tf = cfg.TREE_FEATURE_CONFIG
            _run_eval('lightgbm',
                      f'{cfg.MODEL_ROOT}/lightgbm/{test_strategy}_test{tf["test_frac"]}/lightgbm_24_models.pkl',
                      test_strategy, tf)

        if cfg.TRAIN_CONFIG['xgboost_residual']:
            xgb_res_mod.train(X_train, y_train, cfg.XGB_RESIDUAL_PARAMS, cfg.TREE_FEATURE_CONFIG)
            tf = cfg.TREE_FEATURE_CONFIG
            _run_eval('xgboost_residual',
                      f'{cfg.MODEL_ROOT}/xgboost_residual/{test_strategy}_test{tf["test_frac"]}/xgboost_residual_24_models.pkl',
                      test_strategy, tf)

        if cfg.TRAIN_CONFIG['lightgbm_residual']:
            lgbm_res_mod.train(X_train, y_train, cfg.LGBM_RESIDUAL_PARAMS, cfg.TREE_FEATURE_CONFIG)
            tf = cfg.TREE_FEATURE_CONFIG
            _run_eval('lightgbm_residual',
                      f'{cfg.MODEL_ROOT}/lightgbm_residual/{test_strategy}_test{tf["test_frac"]}/lightgbm_residual_24_models.pkl',
                      test_strategy, tf)

# --- Transformer ---
if cfg.TRAIN_CONFIG['transformer']:
    X_3d, y_3d, mask_3d, timestamps_3d = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)

    for test_strategy in TEST_STRATEGIES:
        for val_strategy in VAL_STRATEGIES:
            print(f"\n{'='*60}")
            print(f"Transformer | test split: {test_strategy} | val split: {val_strategy}")
            print(f"{'='*60}")

            cfg.TRANSFORMER_FEATURE_CONFIG['split_strategy'] = test_strategy
            cfg.TRANSFORMER_FEATURE_CONFIG['val_strategy']   = val_strategy

            transformer_mod.train(X_3d, y_3d, mask_3d, cfg.TRANSFORMER_PARAMS, cfg.TRANSFORMER_FEATURE_CONFIG)
            tf = cfg.TRANSFORMER_FEATURE_CONFIG
            _run_eval('transformer',
                      f'{cfg.MODEL_ROOT}/transformer/{test_strategy}_test{tf["test_frac"]}_{val_strategy}_val{tf["val_frac"]}/transformer_best.pth',
                      test_strategy, tf)

# --- Transformer (residual target) ---
# Same 3D matrix, same network, same split as above — it regresses the deviation from a
# naive same-hour-last-week baseline instead of the load itself. See src/models/_residual.py.
if cfg.TRAIN_CONFIG['transformer_residual']:
    X_3d, y_3d, mask_3d, timestamps_3d = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)

    for test_strategy in TEST_STRATEGIES:
        for val_strategy in VAL_STRATEGIES:
            print(f"\n{'='*60}")
            print(f"Transformer (residual) | test split: {test_strategy} | val split: {val_strategy}")
            print(f"{'='*60}")

            cfg.TRANSFORMER_FEATURE_CONFIG['split_strategy'] = test_strategy
            cfg.TRANSFORMER_FEATURE_CONFIG['val_strategy']   = val_strategy

            tr_res_mod.train(X_3d, y_3d, mask_3d,
                             cfg.TRANSFORMER_RESIDUAL_PARAMS, cfg.TRANSFORMER_FEATURE_CONFIG)
            tf = cfg.TRANSFORMER_FEATURE_CONFIG
            _run_eval('transformer_residual',
                      f'{cfg.MODEL_ROOT}/transformer_residual/{test_strategy}_test{tf["test_frac"]}_{val_strategy}_val{tf["val_frac"]}/transformer_residual_best.pth',
                      test_strategy, tf)

# --- MoE Transformer ---
if cfg.TRAIN_CONFIG['moe_transformer']:
    X_3d, y_3d, mask_3d, timestamps_3d = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)

    for test_strategy in TEST_STRATEGIES:
        for val_strategy in VAL_STRATEGIES:
            print(f"\n{'='*60}")
            print(f"MoE Transformer | test split: {test_strategy} | val split: {val_strategy}")
            print(f"{'='*60}")

            cfg.MOE_TRANSFORMER_FEATURE_CONFIG['split_strategy'] = test_strategy
            cfg.MOE_TRANSFORMER_FEATURE_CONFIG['val_strategy']   = val_strategy

            moe_mod.train(X_3d, y_3d, mask_3d, timestamps_3d,
                          cfg.MOE_TRANSFORMER_PARAMS, cfg.MOE_TRANSFORMER_FEATURE_CONFIG)
            mf = cfg.MOE_TRANSFORMER_FEATURE_CONFIG
            _run_eval('moe_transformer',
                      f'{cfg.MODEL_ROOT}/moe_transformer/{test_strategy}_test{mf["test_frac"]}_{val_strategy}_val{mf["val_frac"]}/moe_transformer_best.pth',
                      test_strategy, mf)

# --- MSTNN ---
if cfg.TRAIN_CONFIG['mstnn']:
    X_3d, y_3d, mask_3d, timestamps_3d = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)

    for test_strategy in TEST_STRATEGIES:
        for val_strategy in VAL_STRATEGIES:
            print(f"\n{'='*60}")
            print(f"MSTNN | test split: {test_strategy} | val split: {val_strategy}")
            print(f"{'='*60}")

            cfg.MSTNN_FEATURE_CONFIG['split_strategy'] = test_strategy
            cfg.MSTNN_FEATURE_CONFIG['val_strategy']   = val_strategy

            mstnn_mod.train(X_3d, y_3d, mask_3d, cfg.MSTNN_PARAMS, cfg.MSTNN_FEATURE_CONFIG)
            mf = cfg.MSTNN_FEATURE_CONFIG
            _run_eval('mstnn',
                      f'{cfg.MODEL_ROOT}/mstnn/{test_strategy}_test{mf["test_frac"]}_{val_strategy}_val{mf["val_frac"]}/mstnn_best.pth',
                      test_strategy, mf)

# --- LSTM ---
if cfg.TRAIN_CONFIG['lstm']:
    X_3d, y_3d, mask_3d, timestamps_3d = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)

    for test_strategy in TEST_STRATEGIES:
        for val_strategy in VAL_STRATEGIES:
            print(f"\n{'='*60}")
            print(f"LSTM | test split: {test_strategy} | val split: {val_strategy}")
            print(f"{'='*60}")

            cfg.LSTM_FEATURE_CONFIG['split_strategy'] = test_strategy
            cfg.LSTM_FEATURE_CONFIG['val_strategy']   = val_strategy

            lstm_mod.train(X_3d, y_3d, mask_3d, cfg.LSTM_PARAMS, cfg.LSTM_FEATURE_CONFIG)
            lf = cfg.LSTM_FEATURE_CONFIG
            _run_eval('lstm',
                      f'{cfg.MODEL_ROOT}/lstm/{test_strategy}_test{lf["test_frac"]}_{val_strategy}_val{lf["val_frac"]}/lstm_best.pth',
                      test_strategy, lf)

# --- LSTM (residual target) ---
if cfg.TRAIN_CONFIG['lstm_residual']:
    X_3d, y_3d, mask_3d, timestamps_3d = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)

    for test_strategy in TEST_STRATEGIES:
        for val_strategy in VAL_STRATEGIES:
            print(f"\n{'='*60}")
            print(f"LSTM (residual) | test split: {test_strategy} | val split: {val_strategy}")
            print(f"{'='*60}")

            cfg.LSTM_FEATURE_CONFIG['split_strategy'] = test_strategy
            cfg.LSTM_FEATURE_CONFIG['val_strategy']   = val_strategy

            lstm_res_mod.train(X_3d, y_3d, mask_3d, cfg.LSTM_RESIDUAL_PARAMS, cfg.LSTM_FEATURE_CONFIG)
            lf = cfg.LSTM_FEATURE_CONFIG
            _run_eval('lstm_residual',
                      f'{cfg.MODEL_ROOT}/lstm_residual/{test_strategy}_test{lf["test_frac"]}_{val_strategy}_val{lf["val_frac"]}/lstm_residual_best.pth',
                      test_strategy, lf)

# --- MoE Transformer (residual target) ---
if cfg.TRAIN_CONFIG['moe_transformer_residual']:
    X_3d, y_3d, mask_3d, timestamps_3d = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)

    for test_strategy in TEST_STRATEGIES:
        for val_strategy in VAL_STRATEGIES:
            print(f"\n{'='*60}")
            print(f"MoE Transformer (residual) | test split: {test_strategy} | val split: {val_strategy}")
            print(f"{'='*60}")

            cfg.MOE_TRANSFORMER_RESIDUAL_FEATURE_CONFIG['split_strategy'] = test_strategy
            cfg.MOE_TRANSFORMER_RESIDUAL_FEATURE_CONFIG['val_strategy']   = val_strategy

            moe_res_mod.train(X_3d, y_3d, mask_3d, timestamps_3d,
                              cfg.MOE_TRANSFORMER_RESIDUAL_PARAMS, cfg.MOE_TRANSFORMER_RESIDUAL_FEATURE_CONFIG)
            mf = cfg.MOE_TRANSFORMER_RESIDUAL_FEATURE_CONFIG
            _run_eval('moe_transformer_residual',
                      f'{cfg.MODEL_ROOT}/moe_transformer_residual/{test_strategy}_test{mf["test_frac"]}_{val_strategy}_val{mf["val_frac"]}/moe_transformer_residual_best.pth',
                      test_strategy, mf)

# --- MSTNN (residual target) ---
if cfg.TRAIN_CONFIG['mstnn_residual']:
    X_3d, y_3d, mask_3d, timestamps_3d = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)

    for test_strategy in TEST_STRATEGIES:
        for val_strategy in VAL_STRATEGIES:
            print(f"\n{'='*60}")
            print(f"MSTNN (residual) | test split: {test_strategy} | val split: {val_strategy}")
            print(f"{'='*60}")

            cfg.MSTNN_FEATURE_CONFIG['split_strategy'] = test_strategy
            cfg.MSTNN_FEATURE_CONFIG['val_strategy']   = val_strategy

            mstnn_res_mod.train(X_3d, y_3d, mask_3d, cfg.MSTNN_RESIDUAL_PARAMS, cfg.MSTNN_FEATURE_CONFIG)
            mf = cfg.MSTNN_FEATURE_CONFIG
            _run_eval('mstnn_residual',
                      f'{cfg.MODEL_ROOT}/mstnn_residual/{test_strategy}_test{mf["test_frac"]}_{val_strategy}_val{mf["val_frac"]}/mstnn_residual_best.pth',
                      test_strategy, mf)

# --- MoE-MSTNN + MoE-LSTM (+ residuals) ---
for _key, _mod, _P, _FC, _mt in [
    ('moe_mstnn', moe_mstnn_mod, cfg.MOE_MSTNN_PARAMS, cfg.MOE_MSTNN_FEATURE_CONFIG, 'moe_mstnn'),
    ('moe_mstnn_residual', moe_mstnn_res_mod, cfg.MOE_MSTNN_RESIDUAL_PARAMS, cfg.MOE_MSTNN_FEATURE_CONFIG, 'moe_mstnn_residual'),
    ('moe_lstm', moe_lstm_mod, cfg.MOE_LSTM_PARAMS, cfg.MOE_LSTM_FEATURE_CONFIG, 'moe_lstm'),
    ('moe_lstm_residual', moe_lstm_res_mod, cfg.MOE_LSTM_RESIDUAL_PARAMS, cfg.MOE_LSTM_FEATURE_CONFIG, 'moe_lstm_residual'),
]:
    if not cfg.TRAIN_CONFIG[_key]:
        continue
    X_3d, y_3d, mask_3d, timestamps_3d = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)
    for test_strategy in TEST_STRATEGIES:
        for val_strategy in VAL_STRATEGIES:
            print(f"\n{'='*60}")
            print(f"{_key} | test split: {test_strategy} | val split: {val_strategy}")
            print(f"{'='*60}")
            _FC['split_strategy'] = test_strategy
            _FC['val_strategy']   = val_strategy
            _mod.train(X_3d, y_3d, mask_3d, timestamps_3d, _P, _FC)
            _run_eval(_key,
                      f'{cfg.MODEL_ROOT}/{_mt}/{test_strategy}_test{_FC["test_frac"]}_{val_strategy}_val{_FC["val_frac"]}/{_mt}_best.pth',
                      test_strategy, _FC)
