# src/models/_residual.py
"""
Shared machinery for the residual-learning models.

Instead of predicting metered load directly, predict its DEVIATION from a naive
same-hour-last-week baseline, and add the baseline back at inference:

    baseline[h] = mean of the past 7 days' PRELIMINARY load at clock-hour h   -> (24,)
    train target : y_residual = y_metered - baseline
    inference    : pred_metered = model.predict(X) + baseline

Features are unchanged and the estimator is unchanged — only the target moves. The
baseline is derived per-sample from that sample's OWN lookback window, so nothing is fit
on the training set and it cannot leak; and it reads only the preliminary load, which is
published in real time. That makes the residual models deployable on exactly the same
forecast path as everything else.

The baseline is recovered from the feature matrix itself rather than re-read from the
cleaned CSV. That is what lets predict() keep the same signature as the plain models — and
it means the forecast path (which runs off predict.csv, not the cleaned CSV) needs no
special plumbing at all.
"""

import os

import joblib
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error)

BASELINE_MODES = ('hourly', 'scalar')


def print_metrics(tag, y_true, y_pred):
    """One-line score. Used to report the naive baseline ALONE next to the model's — the
    residual models are only worth their complexity if they beat the thing they lean on."""
    t, p = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    print(f"  {tag:28s} MAPE {mean_absolute_percentage_error(t, p) * 100:6.2f}%  "
          f"MAE {mean_absolute_error(t, p):8.1f}  "
          f"RMSE {np.sqrt(mean_squared_error(t, p)):8.1f}  "
          f"ME {np.mean(p - t):+8.1f}")


def baseline_from_windows(win, mode='hourly'):
    """(N, lookback) of raw-MW preliminary load -> (N, 24) baseline.

    'hourly' : baseline[h] = mean of the past 7 days at clock-hour h. The window ends at
               the midnight cutoff, so window position k sits at clock-hour k % 24 and
               win[:, h::24] is exactly the 7 values for hour h. (DST shifts this by an
               hour a couple of times a year.)
    'scalar' : one number per day (mean of the whole window), broadcast across all 24 h.
    """
    if mode not in BASELINE_MODES:
        raise ValueError(f"baseline mode must be one of {BASELINE_MODES}, got {mode!r}")
    if mode == 'scalar':
        return np.repeat(win.mean(axis=1, keepdims=True), 24, axis=1)
    return np.stack([win[:, h::24].mean(axis=1) for h in range(24)], axis=1)


def tree_baseline(X_opt, mode='hourly', lookback=168):
    """Baseline for the 2D matrix — the lookback window is sitting right there in columns."""
    cols = [f'load_estimated_h{k}' for k in range(lookback)]
    missing = [c for c in cols if c not in X_opt.columns]
    if missing:
        raise KeyError(
            f"2D matrix is missing {len(missing)} load_estimated_h* column(s) "
            f"(first: {missing[0]}). The residual model reads the preliminary-load window "
            f"straight out of the features; without it there is no baseline."
        )
    return baseline_from_windows(X_opt[cols].values, mode)


def sequence_baseline(X_3d, seq_scaler, mode='hourly'):
    """Baseline for the 3D matrix.

    Channel 0 of the window is Load_Estimated, standardized by the sequence scaler that
    build_timeseries_matrix fitted. Undo that one channel to get raw MW — the same 168 h
    window the 2D path reads, so both models see an identical baseline.
    """
    win = X_3d[:, :, 0] * seq_scaler.scale_[0] + seq_scaler.mean_[0]
    return baseline_from_windows(win, mode)


# ---------------------------------------------------------------------------
# Shared residual orchestration for the SEQUENCE (3D-matrix) models
# ---------------------------------------------------------------------------
# The four sequence residual models (transformer / mstnn / moe_transformer / moe_mstnn) do
# the identical dance: recover the baseline, fit a residual scaler on the exact training
# samples the net will see, standardize the residual target, train the plain estimator on
# it, then invert at predict time (net -> res_scaler -> +baseline -> y_scaler). This factory
# is that dance in one place; each model file becomes a ~10-line registration. The config
# TRAIN switch still picks the model, and it automatically takes the residual route here.
# (The 2D tree residual, xgboost_residual.py, is left alone — different matrix and format.)
#
# `is_moe` is the only real branch: MoE nets thread the per-sample season route through
# train/predict, and their evaluate adds the regime / per-expert breakdown. The returned
# train/predict keep the SAME signatures as the plain vs MoE base models, so callers
# (ModelEvaluator / ModelPredictor / Model_Training / single_day) need no changes.

_RES_SCALER = 'residual_scaler.pkl'      # saved beside the weights — part of the model


def _glob_scaler(matrix_dir, pattern):
    import glob
    hits = glob.glob(os.path.join(matrix_dir, pattern))
    if not hits:
        raise FileNotFoundError(f"No scaler matching '{pattern}' in {matrix_dir}")
    return joblib.load(sorted(hits)[0])


def make_residual_model(*, name, model_type, filename, feature_cfg, params_default,
                        model_cls, is_moe, base_predict=None, expert_prefix='MOE'):
    """Return (train, predict, evaluate) for a residual sequence model."""
    from sklearn.preprocessing import StandardScaler

    from src.config import MATRIX_DIR, EMBARGO_DAYS
    from src.feature_engine import _split_indices, apply_embargo
    from src.models._eval_utils import EvalUtils
    from src.models._utils import _make_run_dir
    from src.models._seq_trainer import train_sequence
    from src.models import moe_transformer as moe

    lb = feature_cfg['lookback_hours']
    hh = feature_cfg['latest_info_hour']

    def _matrix_scalers():
        seq = _glob_scaler(MATRIX_DIR, f'scaler_ts_lb{lb}_h{hh}.pkl')
        ysc = _glob_scaler(MATRIX_DIR, f'y_scaler_lb{lb}_h{hh}.pkl')
        return seq, ysc

    def _scalers(model_path):
        seq, ysc = _matrix_scalers()
        res_path = os.path.join(os.path.dirname(model_path), _RES_SCALER)
        if not os.path.exists(res_path):
            raise FileNotFoundError(
                f"{res_path} is missing. The residual scaler is saved at train time and is "
                f"required to turn the network's output back into MW.")
        return seq, joblib.load(res_path), ysc

    def _save_dir(params, dataset):
        # match how each trainer names its dir: moe.train uses use_lds only; train_sequence
        # uses use_lds AND use_fds (plain models get an _fds suffix when FDS is on).
        if is_moe:
            return _make_run_dir('models', model_type, feature_cfg, dataset,
                                 use_lds=params.get('use_lds', False))
        return _make_run_dir('models', model_type, feature_cfg, dataset,
                             use_lds=params.get('use_lds', False),
                             use_fds=params.get('use_fds', False))

    def _train(X_3d, y_3d, mask_3d, timestamps_3d, params, dataset):
        params = {**(params or params_default)}
        mode = params.pop('baseline', 'hourly')
        N, H = y_3d.shape
        seq_scaler, y_scaler = _matrix_scalers()

        baseline = sequence_baseline(X_3d, seq_scaler, mode)                     # (N, 24) MW
        y_mw = y_scaler.inverse_transform(y_3d.reshape(-1, 1)).reshape(N, H)     # (N, 24) MW
        residual_mw = y_mw - baseline

        # The same split (+embargo, +mask denoise) the plain model trains on, so the residual
        # scaler sees exactly the samples the net will fit on and not one more.
        train_pool, _t = _split_indices(N, feature_cfg['split_strategy'],
                                        feature_cfg['test_frac'], feature_cfg['random_state'])
        rtr, rval = _split_indices(len(train_pool), feature_cfg['val_strategy'],
                                   feature_cfg['val_frac'], feature_cfg['random_state'])
        train_idx, _v = apply_embargo(train_pool[rtr], train_pool[rval], EMBARGO_DAYS)
        fit_idx = train_idx[mask_3d[train_idx]]

        res_scaler = StandardScaler().fit(residual_mw[fit_idx].reshape(-1, 1))
        y_res = res_scaler.transform(residual_mw.reshape(-1, 1)).reshape(N, H).astype('float32')

        print(f"Baseline mode:    {mode}")
        print(f"Residual (MW):    mean {residual_mw[fit_idx].mean():+.1f}  "
              f"std {residual_mw[fit_idx].std():.1f}  -> standardized by its own scaler "
              f"(fit on {len(fit_idx)} train samples only)")
        if params.get('use_lds') or params.get('use_fds'):
            print(f"NOTE: use_lds={params.get('use_lds')} use_fds={params.get('use_fds')} — these "
                  f"bin on the TARGET, which here is the residual, not the load.")
        print_metrics('naive baseline alone (train)', y_mw[fit_idx], baseline[fit_idx])

        if is_moe:
            moe.train(X_3d, y_res, mask_3d, timestamps_3d, params, feature_cfg, dataset,
                      model_type_name=model_type, save_name=filename, model_cls=model_cls)
        else:
            train_sequence(model_cls, model_type, filename, X_3d, y_res, mask_3d,
                           params, feature_cfg, dataset)

        model_dir = _save_dir(params, dataset)
        joblib.dump({'scaler': res_scaler, 'baseline': mode},
                    os.path.join(model_dir, _RES_SCALER))
        print(f"Residual scaler saved to: {os.path.join(model_dir, _RES_SCALER)}")

    def _predict(model_path, X_np, timestamps, params):
        params = params or params_default
        seq_scaler, res_bundle, y_scaler = _scalers(model_path)
        res_scaler, mode = res_bundle['scaler'], res_bundle['baseline']

        if is_moe:
            res_scaled = moe.predict(model_path, X_np, timestamps, params, model_cls=model_cls)
        else:
            res_scaled = base_predict(model_path, X_np, params)
        n, h = res_scaled.shape
        res_mw = res_scaler.inverse_transform(res_scaled.reshape(-1, 1)).reshape(n, h)
        load_mw = res_mw + sequence_baseline(X_np, seq_scaler, mode)
        # back to y-standardized space — the caller inverse-transforms with y_scaler, so the
        # residual plumbing is exactly cancelled and stays invisible.
        return y_scaler.transform(load_mw.reshape(-1, 1)).reshape(n, h)

    def evaluate(model_path, X_test, y_true_mw, y_scaler, timestamps, result_dir,
                 params=None, X_train=None, y_true_train_mw=None, timestamps_train=None):
        params = params or params_default
        seq_scaler, res_bundle, _ = _scalers(model_path)

        y_pred_scaled = _predict(model_path, X_test, timestamps, params)
        n, h = y_pred_scaled.shape
        y_pred_mw = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(n, h)

        base_test = sequence_baseline(X_test, seq_scaler, res_bundle['baseline'])
        print(f"\n=== {name}: model vs the baseline it leans on (test) ===")
        print_metrics('naive baseline alone', y_true_mw, base_test)
        print_metrics('residual + baseline', y_true_mw, y_pred_mw)

        y_pred_train_mw = None
        train_df = None
        if X_train is not None and y_true_train_mw is not None:
            p = _predict(model_path, X_train, timestamps_train, params)
            n2, h2 = p.shape
            y_pred_train_mw = y_scaler.inverse_transform(p.reshape(-1, 1)).reshape(n2, h2)
            train_df = EvalUtils.build_detailed_df(name, y_true_train_mw, y_pred_train_mw, timestamps_train)
        EvalUtils.evaluate_one(name, y_true_mw, y_pred_mw, timestamps, result_dir, train_df)

        if is_moe:   # MoE regime + per-expert views, same as the plain MoE
            moe._regime_breakdown(y_true_mw, y_pred_mw, timestamps, result_dir, name=name)
            moe._evaluate_experts(y_true_mw, y_pred_mw, timestamps, result_dir,
                                  y_true_train_mw, y_pred_train_mw, timestamps_train,
                                  name_prefix=expert_prefix)

    # Return train/predict with the SAME arity as the base model (MoE threads timestamps).
    if is_moe:
        def train(X_3d, y_3d, mask_3d, timestamps_3d, params=None, feature_cfg=None, dataset=None):
            _train(X_3d, y_3d, mask_3d, timestamps_3d, params, dataset)

        def predict(model_path, X_np, timestamps, params=None):
            return _predict(model_path, X_np, timestamps, params)
    else:
        def train(X_3d, y_3d, mask_3d, params=None, feature_cfg=None, dataset=None):
            _train(X_3d, y_3d, mask_3d, None, params, dataset)

        def predict(model_path, X_np, params=None):
            return _predict(model_path, X_np, None, params)

    return train, predict, evaluate
