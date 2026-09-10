# src/models/_residual.py
"""
Shared machinery for the residual-learning models.

Instead of predicting the target directly, predict its DEVIATION from a naive
same-hour-last-week baseline, and add the baseline back at inference:

    baseline[h] = mean of the lookback window's days at clock-hour h   -> (24,)
    train target : y_residual = y - baseline
    inference    : pred = model.predict(X) + baseline

Features are unchanged and the estimator is unchanged — only the target moves. The
baseline is derived per-sample from that sample's OWN lookback window, so nothing is fit
on the training set and it cannot leak; and it reads only series published in real time.
That makes the residual models deployable on exactly the same forecast path as everything
else.

WHICH CHANNEL THE BASELINE COMES FROM IS PER TARGET (config.BASELINE_COL), because the one
thing it must be is in the target's own units:

    load       Load_Estimated   MW      preliminary load, the proxy published in real time
    load_met   Load             MW      metered load, the series being forecast
    rt_price   RT_Total_slog    $/MWh   the price's own history, signed-log encoded

For rt_price the encoding is undone BEFORE the mean is taken (config.BASELINE_TRANSFORM),
so the baseline is a mean of dollars rather than a mean of logs. With a 120x price tail
those are very different numbers and only the first can be added back to a $/MWh target.
The price channel rides in the LMP lookback block, so a residual price model needs
PJM_LMP=1; config.residual_baseline_reason() is the check, and it names the fix.

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


def _decode(x):
    """Undo the lookback channel's encoding so the mean is taken in the target's real units.

    'signed_log' is src.lmp_features.signed_log, sign(x)*log1p(|x|); its inverse is
    sign(z)*expm1(|z|). Applied per-hour BEFORE averaging on purpose — averaging logs and
    then exponentiating gives a geometric mean, which on a series whose median is 36 and
    maximum 4392 sits far below the arithmetic level the target actually needs added back.
    """
    from src.config import BASELINE_TRANSFORM
    if BASELINE_TRANSFORM == 'none':
        return x
    if BASELINE_TRANSFORM == 'signed_log':
        return np.sign(x) * np.expm1(np.abs(x))
    raise ValueError(f"unknown baseline_transform {BASELINE_TRANSFORM!r}")


def assert_baseline_available(what):
    """Raise with the arm-specific fix if this run cannot build a baseline at all."""
    from src.config import residual_baseline_reason
    reason = residual_baseline_reason()
    if reason:
        raise ValueError(f"{what} is a residual model and cannot run here. {reason}")


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
    """Baseline for the 2D matrix — the lookback window is sitting right there in columns.

    The column stem is PJM_TARGET's baseline channel (`load_estimated_h*`, `load_h*`,
    `rt_total_slog_h*`), lower-cased the way build_or_load_matrix flattens it. Naming it from
    config rather than hard-coding one stem is what keeps the three classes in step: a wrong
    stem raises here instead of averaging whichever channel happens to be present.
    """
    from src.config import BASELINE_COL, residual_baseline_reason
    cols = [f'{BASELINE_COL.lower()}_h{k}' for k in range(lookback)]
    missing = [c for c in cols if c not in X_opt.columns]
    if missing:
        raise KeyError(
            f"2D matrix is missing {len(missing)} {BASELINE_COL.lower()}_h* column(s) "
            f"(first: {missing[0]}). The residual model reads its baseline window straight "
            f"out of the features; without it there is no baseline. "
            f"{residual_baseline_reason() or ''}".strip())
    return baseline_from_windows(_decode(X_opt[cols].values), mode)


def sequence_baseline(X_3d, seq_scaler, mode='hourly'):
    """Baseline for the 3D matrix.

    The baseline channel is standardized by the sequence scaler build_timeseries_matrix
    fitted, so undo that one channel, then undo any encoding it carries — the same window the
    2D path reads, so both models see an identical baseline.

    The channel INDEX comes from config.sequence_feature_cols(), the same list the builder
    laid the channels out with. Hard-coding 0 was correct only while every target's baseline
    was the sequence's load channel; rt_price's is channel 22.
    """
    from src.config import BASELINE_COL, sequence_feature_cols, residual_baseline_reason
    cols = sequence_feature_cols()
    if BASELINE_COL not in cols:
        raise ValueError(residual_baseline_reason()
                         or f"{BASELINE_COL!r} is not among the lookback channels {cols}.")
    c = cols.index(BASELINE_COL)
    win = X_3d[:, :, c] * seq_scaler.scale_[c] + seq_scaler.mean_[c]
    return baseline_from_windows(_decode(win), mode)


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

    from src.config import MATRIX_DIR, EMBARGO_DAYS, MATRIX_TAG, TARGET_TAG
    from src.feature_engine import _split_indices, apply_embargo
    from src.models._eval_utils import EvalUtils
    from src.models._utils import _make_run_dir
    from src.models._seq_trainer import train_sequence
    from src.models import moe_transformer as moe

    lb = feature_cfg['lookback_hours']
    hh = feature_cfg['latest_info_hour']

    def _matrix_scalers():
        seq = _glob_scaler(MATRIX_DIR, f'scaler_ts_lb{lb}_h{hh}{MATRIX_TAG}{TARGET_TAG}.pkl')
        ysc = _glob_scaler(MATRIX_DIR, f'y_scaler_lb{lb}_h{hh}{MATRIX_TAG}{TARGET_TAG}.pkl')
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
        # Same directory the base trainer (moe.train / train_sequence) wrote the weights to,
        # so the residual scaler lands beside them.
        return _make_run_dir('models', model_type, feature_cfg, dataset)

    def _train(X_3d, y_3d, mask_3d, timestamps_3d, params, dataset):
        # Refuse rather than produce plausible noise. Subtracting a baseline in the wrong
        # units still cancels arithmetically at inference -- nothing raises -- but the
        # network spends its capacity rebuilding a baseline it is about to add back.
        # residual_baseline_reason() names both the cause and the fix.
        assert_baseline_available(name)
        params = {**(params or params_default)}
        mode = params.pop('baseline', 'hourly')
        N, H = y_3d.shape
        seq_scaler, y_scaler = _matrix_scalers()

        # All three arrays are in the TARGET's own units — MW for the load classes, $/MWh
        # for rt_price — which is the whole reason baseline_col is per target.
        baseline = sequence_baseline(X_3d, seq_scaler, mode)                     # (N, 24)
        y_real   = y_scaler.inverse_transform(y_3d.reshape(-1, 1)).reshape(N, H)  # (N, 24)
        residual = y_real - baseline

        # The same split (+embargo, +mask denoise) the plain model trains on, so the residual
        # scaler sees exactly the samples the net will fit on and not one more.
        train_pool, _t = _split_indices(N, feature_cfg['split_strategy'],
                                        feature_cfg['test_frac'], feature_cfg['random_state'])
        rtr, rval = _split_indices(len(train_pool), feature_cfg['val_strategy'],
                                   feature_cfg['val_frac'], feature_cfg['random_state'])
        train_idx, _v = apply_embargo(train_pool[rtr], train_pool[rval], EMBARGO_DAYS)
        fit_idx = train_idx[mask_3d[train_idx]]

        res_scaler = StandardScaler().fit(residual[fit_idx].reshape(-1, 1))
        y_res = res_scaler.transform(residual.reshape(-1, 1)).reshape(N, H).astype('float32')

        from src.config import BASELINE_COL, TARGET_UNIT
        print(f"Baseline mode:    {mode}  (from {BASELINE_COL}, in {TARGET_UNIT})")
        print(f"Residual ({TARGET_UNIT}): mean {residual[fit_idx].mean():+.2f}  "
              f"std {residual[fit_idx].std():.2f}  -> standardized by its own scaler "
              f"(fit on {len(fit_idx)} train samples only)")
        print_metrics('naive baseline alone (train)', y_real[fit_idx], baseline[fit_idx])

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
        res_real  = res_scaler.inverse_transform(res_scaled.reshape(-1, 1)).reshape(n, h)
        pred_real = res_real + sequence_baseline(X_np, seq_scaler, mode)
        # back to y-standardized space — the caller inverse-transforms with y_scaler, so the
        # residual plumbing is exactly cancelled and stays invisible.
        return y_scaler.transform(pred_real.reshape(-1, 1)).reshape(n, h)

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
