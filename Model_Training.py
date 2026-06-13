import src.config as cfg
from src.feature_engine import build_or_load_matrix, get_train_test_split, build_timeseries_matrix
from src.model_trainer import PowerForecaster
from src.model_evaluator import ModelEvaluator

TEST_STRATEGIES = ['tail', 'random']
VAL_STRATEGIES  = ['tail', 'random']


def _run_eval(model_name, model_path, split_strategy, feature_cfg):
    """训练完成后立即评估，动态构建 eval config 传入 ModelEvaluator。"""
    models = {m: {'enabled': 0, 'model_path': ''} for m in ['xgboost', 'lightgbm', 'transformer', 'lstm']}
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
        forecaster = PowerForecaster(X_train, y_train, X_test, y_test)

        if cfg.TRAIN_CONFIG['xgboost']:
            forecaster.train_xgboost(cfg.XGB_PARAMS)
            tf = cfg.TREE_FEATURE_CONFIG
            _run_eval('xgboost',
                      f'models/{cfg.DATASET}/xgboost/{test_strategy}_test{tf["test_frac"]}/xgboost_24_models.pkl',
                      test_strategy, tf)

        if cfg.TRAIN_CONFIG['lightgbm']:
            forecaster.train_lightgbm(cfg.LGBM_PARAMS)
            tf = cfg.TREE_FEATURE_CONFIG
            _run_eval('lightgbm',
                      f'models/{cfg.DATASET}/lightgbm/{test_strategy}_test{tf["test_frac"]}/lightgbm_24_models.pkl',
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

            PowerForecaster(None, None, None, None).train_transformer_3d(
                X_3d=X_3d, y_3d=y_3d, mask_3d=mask_3d, params=cfg.TRANSFORMER_PARAMS
            )
            tf = cfg.TRANSFORMER_FEATURE_CONFIG
            _run_eval('transformer',
                      f'models/{cfg.DATASET}/transformer/{test_strategy}_test{tf["test_frac"]}_{val_strategy}_val{tf["val_frac"]}/transformer_best.pth',
                      test_strategy, tf)

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

            PowerForecaster(None, None, None, None).train_lstm_3d(
                X_3d=X_3d, y_3d=y_3d, mask_3d=mask_3d, params=cfg.LSTM_PARAMS
            )
            lf = cfg.LSTM_FEATURE_CONFIG
            _run_eval('lstm',
                      f'models/{cfg.DATASET}/lstm/{test_strategy}_test{lf["test_frac"]}_{val_strategy}_val{lf["val_frac"]}/lstm_best.pth',
                      test_strategy, lf)
