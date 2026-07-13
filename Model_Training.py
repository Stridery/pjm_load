import src.config as cfg
from src.feature_engine import build_or_load_matrix, get_train_test_split, build_timeseries_matrix
from src.model_evaluator import ModelEvaluator
from src.models import xgboost as xgb_mod
from src.models import lightgbm as lgb_mod
from src.models import transformer as transformer_mod
from src.models import lstm as lstm_mod
from src.models import moe_transformer as moe_mod
from src.models import mstnn as mstnn_mod

TEST_STRATEGIES = ['tail']
VAL_STRATEGIES  = ['tail']


def _run_eval(model_name, model_path, split_strategy, feature_cfg):
    """训练完成后立即评估，动态构建 eval config 传入 ModelEvaluator。"""
    models = {m: {'enabled': 0, 'model_path': ''} for m in ['xgboost', 'lightgbm', 'transformer', 'lstm', 'moe_transformer', 'mstnn']}
    models[model_name] = {'enabled': 1, 'model_path': model_path}
    eval_cfg = {
        'split_strategy': split_strategy,
        'test_frac':      feature_cfg['test_frac'],
        'random_state':   feature_cfg['random_state'],
        'result_dir':     f'results/{cfg.DATASET}/evaluation',
        'models':         models,
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
                      f'models/{cfg.DATASET}/xgboost/{test_strategy}_test{tf["test_frac"]}{cfg._xgb_lds}/xgboost_24_models.pkl',
                      test_strategy, tf)

        if cfg.TRAIN_CONFIG['lightgbm']:
            lgb_mod.train(X_train, y_train, cfg.LGBM_PARAMS, cfg.TREE_FEATURE_CONFIG)
            tf = cfg.TREE_FEATURE_CONFIG
            _run_eval('lightgbm',
                      f'models/{cfg.DATASET}/lightgbm/{test_strategy}_test{tf["test_frac"]}{cfg._lgbm_lds}/lightgbm_24_models.pkl',
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
                      f'models/{cfg.DATASET}/transformer/{test_strategy}_test{tf["test_frac"]}_{val_strategy}_val{tf["val_frac"]}{cfg._tr_lds}{cfg._tr_fds}/transformer_best.pth',
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
                      f'models/{cfg.DATASET}/moe_transformer/{test_strategy}_test{mf["test_frac"]}_{val_strategy}_val{mf["val_frac"]}{cfg._moe_lds}/moe_transformer_best.pth',
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
                      f'models/{cfg.DATASET}/mstnn/{test_strategy}_test{mf["test_frac"]}_{val_strategy}_val{mf["val_frac"]}{cfg._mstnn_lds}/mstnn_best.pth',
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
                      f'models/{cfg.DATASET}/lstm/{test_strategy}_test{lf["test_frac"]}_{val_strategy}_val{lf["val_frac"]}{cfg._lstm_lds}{cfg._lstm_fds}/lstm_best.pth',
                      test_strategy, lf)
