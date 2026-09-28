# Entry points live in scripts/ but import the package from the repo root, and running
# `python scripts/foo.py` puts scripts/ on sys.path rather than the root. Prepend the root so
# `import src...` resolves from any working directory.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import os

import src.config as cfg
from src.feature_engine import build_or_load_matrix, get_train_test_split, build_timeseries_matrix
from src.models._utils import run_tag
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
from src.models import archformer as archformer_mod
from src.models import archformer_residual as archformer_res_mod
from src.models import archformer_hp as archformer_hp_mod
from src.models import archformer_hp_residual as archformer_hp_res_mod
from src.models import archformer_skip as archformer_skip_mod
from src.models import archformer_skip_residual as archformer_skip_res_mod

TEST_STRATEGIES = ['tail']
VAL_STRATEGIES  = ['tail']

# Residual models subtract a baseline taken from the target's own lookback channel, and not
# every (target, arm) has one: rt_price's baseline is the price history that rides in the LMP
# block, so a PJM_LMP=0 price arm cannot build it. Rather than let such a run abort halfway
# through on the guard in src/models/_residual.py, drop them up front and say why once.
_residual_reason = cfg.residual_baseline_reason()
if _residual_reason:
    _skipped = [m for m, on in cfg.TRAIN_CONFIG.items() if on and m.endswith('_residual')]
    if _skipped:
        print(f"Skipping {len(_skipped)} residual model(s): {_residual_reason}")
        print(f"  ({', '.join(_skipped)})")
    cfg.TRAIN_CONFIG = {m: (0 if m.endswith('_residual') else on)
                        for m, on in cfg.TRAIN_CONFIG.items()}
else:
    print(f"Residual baseline: per-clock-hour mean of {cfg.BASELINE_COL} over the lookback "
          f"window, in {cfg.TARGET_UNIT}"
          + (f" (decoding {cfg.BASELINE_TRANSFORM})" if cfg.BASELINE_TRANSFORM != 'none' else ""))


def _model_path(model_type, feature_cfg, filename):
    """Where the trainer just saved this model, so the evaluator reads the same directory.

    The run tag comes from the SAME helper the trainers call (src/models/_utils.run_tag), not
    from a second copy of the format string. It used to be spelled out inline at each of the
    thirteen call sites, which meant any change to the tag — the feature-set suffix being the
    first — had to be made thirteen times or the evaluator would look in a directory the
    trainer had not written.
    """
    return os.path.join(cfg.MODEL_ROOT, model_type, run_tag(feature_cfg), filename)


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
# Guard on the whole TREE_MODELS registry, not on the two plain variants: the residual
# ones are trained INSIDE this block, so naming only xgboost/lightgbm here meant asking
# for lightgbm_residual alone built no matrix, entered no loop, and exited silently with
# nothing trained. Deriving it from the registry also means a new tree model cannot
# reintroduce the omission.
if any(cfg.TRAIN_CONFIG[_m] for _m in TREE_MODELS):
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
                      _model_path('xgboost', tf, 'xgboost_24_models.pkl'),
                      test_strategy, tf)

        if cfg.TRAIN_CONFIG['lightgbm']:
            lgb_mod.train(X_train, y_train, cfg.LGBM_PARAMS, cfg.TREE_FEATURE_CONFIG)
            tf = cfg.TREE_FEATURE_CONFIG
            _run_eval('lightgbm',
                      _model_path('lightgbm', tf, 'lightgbm_24_models.pkl'),
                      test_strategy, tf)

        if cfg.TRAIN_CONFIG['xgboost_residual']:
            xgb_res_mod.train(X_train, y_train, cfg.XGB_RESIDUAL_PARAMS, cfg.TREE_FEATURE_CONFIG)
            tf = cfg.TREE_FEATURE_CONFIG
            _run_eval('xgboost_residual',
                      _model_path('xgboost_residual', tf, 'xgboost_residual_24_models.pkl'),
                      test_strategy, tf)

        if cfg.TRAIN_CONFIG['lightgbm_residual']:
            lgbm_res_mod.train(X_train, y_train, cfg.LGBM_RESIDUAL_PARAMS, cfg.TREE_FEATURE_CONFIG)
            tf = cfg.TREE_FEATURE_CONFIG
            _run_eval('lightgbm_residual',
                      _model_path('lightgbm_residual', tf, 'lightgbm_residual_24_models.pkl'),
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
                      _model_path('transformer', tf, 'transformer_best.pth'),
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
                      _model_path('transformer_residual', tf, 'transformer_residual_best.pth'),
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
                      _model_path('moe_transformer', mf, 'moe_transformer_best.pth'),
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
                      _model_path('mstnn', mf, 'mstnn_best.pth'),
                      test_strategy, mf)

# --- GARCHFormer / ARCHFormer3 (+ residual) ---
# Same shape as the MSTNN block. Both variants are driven from one matrix build,
# because the two runs differ only in their target.
# Three versions side by side: the author's as published, a pure-hyperparameter shrink
# (patch_len 6), and the static-skip rewiring. Separate run dirs, so they are three
# scoreboard rows on one test set rather than a claim about which wins.
for _name, _mod, _params in [
    ('archformer',               archformer_mod,          cfg.ARCHFORMER_PARAMS),
    ('archformer_residual',      archformer_res_mod,      cfg.ARCHFORMER_RESIDUAL_PARAMS),
    ('archformer_hp',            archformer_hp_mod,       cfg.ARCHFORMER_HP_PARAMS),
    ('archformer_hp_residual',   archformer_hp_res_mod,   cfg.ARCHFORMER_HP_RESIDUAL_PARAMS),
    ('archformer_skip',          archformer_skip_mod,     cfg.ARCHFORMER_SKIP_PARAMS),
    ('archformer_skip_residual', archformer_skip_res_mod, cfg.ARCHFORMER_SKIP_RESIDUAL_PARAMS),
]:
    if not cfg.TRAIN_CONFIG[_name]:
        continue
    X_3d, y_3d, mask_3d, timestamps_3d = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)

    for test_strategy in TEST_STRATEGIES:
        for val_strategy in VAL_STRATEGIES:
            print(f"\n{'='*60}")
            print(f"{_name.upper()} | test split: {test_strategy} | val split: {val_strategy}")
            print(f"{'='*60}")

            cfg.ARCHFORMER_FEATURE_CONFIG['split_strategy'] = test_strategy
            cfg.ARCHFORMER_FEATURE_CONFIG['val_strategy']   = val_strategy

            _mod.train(X_3d, y_3d, mask_3d, _params, cfg.ARCHFORMER_FEATURE_CONFIG)
            mf = cfg.ARCHFORMER_FEATURE_CONFIG
            _run_eval(_name,
                      _model_path(_name, mf, f'{_name}_best.pth'),
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
                      _model_path('lstm', lf, 'lstm_best.pth'),
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
                      _model_path('lstm_residual', lf, 'lstm_residual_best.pth'),
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
                      _model_path('moe_transformer_residual', mf, 'moe_transformer_residual_best.pth'),
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
                      _model_path('mstnn_residual', mf, 'mstnn_residual_best.pth'),
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
                      _model_path(_mt, _FC, f'{_mt}_best.pth'),
                      test_strategy, _FC)
