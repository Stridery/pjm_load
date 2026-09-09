"""One benchmark run: (adapter, mode, context length) -> scores and figures on disk.

The run is pinned to the same test days as the 16 point models by taking the day list and the
truth from the y matrix rather than re-splitting the cleaned CSV. That matters more than it
looks: the matrix builder drops days with no complete 48 h weather forecast and (when the block
is on) days with no Day-Ahead congestion data, so a fresh split of the CSV would score a
different -- larger -- set of days and quietly report a better or worse number for reasons that
have nothing to do with the model.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error)
from tqdm import tqdm

import src.config as cfg
from src.feature_engine import _split_indices
from src.models._eval_utils import EvalUtils
from src.foundation import _calibrate, _context
from src.foundation.registry import get as get_adapter

MODES = ('direct', 'residual')

# Which series the model is handed as context.
#   'estimated' — preliminary load. Current as of the real cutoff, but a proxy for the target:
#                 it carries a scale offset (dom) or a shape mismatch (bge, where it is
#                 MIDATL/8.8). See _calibrate.py.
#   'metered'   — the target series itself, so no proxy gap at all. But metered trails
#                 preliminary by ~1 day (measured: unverified metered max is exactly 24 h behind
#                 preliminary max on both zones), so the cutoff moves back a day and the target
#                 lands at forecast steps 49-72 instead of 25-48. Trading horizon for fidelity.
#
# The backtest reads the cleaned CSV's Load column, which is VERIFIED metered — at forecast time
# only the UNVERIFIED value would have existed, and PJM revises it. That revision is not
# recoverable from this data (no archived snapshots), so metered-context scores carry an
# unmeasured optimism. It is expected to be small, but it is not zero and it is one-directional.
#   'rt_price'  — the Real-Time zonal price. Realised hour by hour during the operating day,
#                 so at the cutoff it is available through exactly the same last hour as
#                 preliminary load: extra_lag is 0 and the geometry is the 48-step one, NOT
#                 metered's 72-step one. Only reachable with PJM_TARGET=rt_price, where the
#                 truth matrix is price too; pairing a price context with a load target would
#                 be a category error, so the two are checked against each other below.
SERIES = {'estimated': 'Load_Estimated', 'metered': 'Load', 'rt_price': 'LMP_RT_Total'}

# Which context series belongs to which PJM_TARGET. A run is refused rather than allowed to
# forecast one series and score it against another.
SERIES_TARGET = {'estimated': 'load', 'metered': 'load', 'rt_price': 'rt_price'}


def _score(y_true, y_pred):
    t, p = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    return {'MAPE': mean_absolute_percentage_error(t, p) * 100,
            'MAE': mean_absolute_error(t, p),
            'RMSE': float(np.sqrt(mean_squared_error(t, p))),
            'ME': float(np.mean(p - t))}


def _fmt(tag, s):
    return f"  {tag:34s} MAPE {s['MAPE']:6.2f}%   MAE {s['MAE']:8.1f}   RMSE {s['RMSE']:8.1f}   ME {s['ME']:+8.1f}"


def eval_name(adapter, mode, context_hours, series='estimated', issue='day_of'):
    """EvalUtils prefixes every artifact with this, so it carries the run's identity."""
    tag = '' if series != 'metered' else ('_MET' if issue == 'day_of' else '_METDA')
    stem = adapter.upper() + tag + ('_RES' if mode == 'residual' else '')
    return f'{stem}_{context_hours}'


def model_dir(adapter, mode):
    """Directory name under results/{zone}/evaluation/ — direct and residual are separate
    models here exactly as xgboost and xgboost_residual are."""
    return adapter if mode == 'direct' else f'{adapter}_residual'


def run_tag(feature_cfg, context_hours, series='estimated', issue='day_of'):
    suffix = '' if series != 'metered' else ('_met' if issue == 'day_of' else '_met_da')
    return (f"{feature_cfg['split_strategy']}_test{feature_cfg['test_frac']}"
            f"{cfg.MATRIX_TAG}_ctx{context_hours}{suffix}")


def _load_truth(feature_cfg):
    lb, hh = feature_cfg['lookback_hours'], feature_cfg['latest_info_hour']
    path = os.path.join(cfg.MATRIX_DIR, f'y_opt_lb{lb}_h{hh}{cfg.MATRIX_TAG}{cfg.TARGET_TAG}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No target matrix at {path}. It carries the day list this benchmark must score to "
            f"stay comparable with the 16 point models. Build it with:\n"
            f"    PJM_DATASET={cfg.DATASET} python -c "
            f"'import src.config as c; from src.feature_engine import build_or_load_matrix; "
            f"build_or_load_matrix(c.CLEANED_PATH, c.MATRIX_DIR)'")
    y = pd.read_csv(path, index_col=0, parse_dates=True)
    return y, np.array([d.date() for d in y.index]), y[[f'h{i}' for i in range(24)]].values


def _batched_forecast(adapter, contexts, horizon, params, batch_size, label,
                      past_only=None, past_future=None):
    """Slices the covariate lists with the same window as the contexts, so a batch can never
    pair one day's context with another day's covariates."""
    out = []
    for lo in tqdm(range(0, len(contexts), batch_size), desc=label, unit='batch'):
        hi = lo + batch_size
        out.append(adapter.forecast(
            contexts[lo:hi], horizon, params,
            past_only=None if past_only is None else past_only[lo:hi],
            past_future=None if past_future is None else past_future[lo:hi]))
    return np.concatenate(out, axis=0)


def run_one(adapter_name, mode, context_hours, fcfg=None, feature_cfg=None, series=None):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    fcfg = fcfg or cfg.FOUNDATION_CONFIG
    feature_cfg = feature_cfg or cfg.TREE_FEATURE_CONFIG
    series = series or fcfg.get('series', 'estimated')
    if series not in SERIES:
        raise ValueError(f"series must be one of {sorted(SERIES)}, got {series!r}")
    if SERIES_TARGET[series] != cfg.TARGET:
        raise ValueError(
            f"series={series!r} forecasts {SERIES[series]}, but PJM_TARGET={cfg.TARGET!r} scores "
            f"{cfg.TARGET_COL}. The truth comes from the y matrix, so this would compare a "
            f"forecast of one series against another. Use "
            f"PJM_TARGET={SERIES_TARGET[series]} for series={series!r}.")
    # Metered trails preliminary by ~1 day, and 'metered_issue' says which way that is paid for:
    # 'day_of' keeps the horizon at 48 and gives up the day-ahead issue time (the forecast can
    # first be made at the START of the target day); 'day_ahead' keeps the issue time and pays
    # with a 72-step horizon. See FOUNDATION_CONFIG for why this is not a tuning knob.
    issue = fcfg.get('metered_issue', 'day_of')
    if issue not in ('day_of', 'day_ahead'):
        raise ValueError(f"metered_issue must be 'day_of' or 'day_ahead', got {issue!r}")
    extra_lag = (int(fcfg.get('metered_lag_days', 1))
                 if series == 'metered' and issue == 'day_ahead' else 0)
    adapter = get_adapter(adapter_name)
    # series_is_positive tells an adapter whether the series it is about to see is a load
    # (strictly positive, worth clamping) or a signed residual (clamping would delete every
    # downward correction). It is a property of the MODE, not of the adapter's config.
    params = {**fcfg['adapters'].get(adapter_name, {}),
              'device': fcfg.get('device', 'auto'),
              'series_is_positive': mode == 'direct',
              # The RT price channels ride the SAME switch the point models use, so the run tag
              # already tells the two arms apart and an A/B is two runs.
              'include_prices': bool(cfg.LMP_CONFIG['enabled'])}

    name = eval_name(adapter_name, mode, context_hours, series, issue)
    result_dir = os.path.join(fcfg['result_root'], model_dir(adapter_name, mode),
                              run_tag(feature_cfg, context_hours, series, issue))
    print(f"\n{'=' * 72}\n{name} | zone {cfg.DATASET} | mode {mode} | context {context_hours} h"
          f" | series {series}" + (f" ({issue}, lag {extra_lag} d)" if series == 'metered' else "") +
          f"\n{'=' * 72}")

    # --- data -------------------------------------------------------------------------
    df, ept_dates, ept_hours = _context.load_frame(cfg.CLEANED_PATH)
    raw = df[SERIES[series]].values          # the context series
    est = df['Load_Estimated'].values
    load = df['Load'].values
    y_idx, target_days, y_all = _load_truth(feature_cfg)

    cutoff = _context.cutoff_positions(ept_dates, ept_hours, target_days,
                                       feature_cfg['latest_info_hour'], extra_lag)
    spans = _context.day_row_span(ept_dates)
    warmup = _context.baseline_warmup(extra_lag) if mode == 'residual' else 0

    # The horizon is DERIVED, not configured: how many steps the cutoff actually sits from each
    # target day's final hour (48-49 for the preliminary series, 72-73 once metered's day of lag
    # is added). A hand-set number that came up short would silently drop days rather than fail.
    need_all = _context.steps_needed(cutoff, target_days, spans)
    cfg_h = fcfg.get('horizon_steps', 'auto')
    horizon = (int(need_all[need_all > 0].max()) + 2 if cfg_h in ('auto', None, 0)
               else int(cfg_h))
    print(f"horizon {horizon} steps (target day occupies steps "
          f"{int(np.median(need_all[need_all > 0])) - 23}-{int(np.median(need_all[need_all > 0]))})")
    ok = _context.usable(cutoff, context_hours, horizon, len(df), spans, target_days, warmup)
    if not ok.any():
        raise SystemExit(f"No usable day at context {context_hours} h — the record is too short.")

    # --- the split, taken on the FULL matrix so it matches the point models exactly ------
    train_pos, test_pos = _split_indices(len(y_idx), feature_cfg['split_strategy'],
                                         feature_cfg['test_frac'], feature_cfg['random_state'])
    kept = np.where(ok)[0]
    is_test = np.isin(kept, test_pos)
    dropped_test = len(test_pos) - int(is_test.sum())
    print(f"days: {len(kept)}/{len(target_days)} usable   "
          f"test {int(is_test.sum())}/{len(test_pos)}   train {int((~is_test).sum())}/{len(train_pos)}")
    if dropped_test:
        print(f"  !! {dropped_test} TEST day(s) dropped for lack of history at this context "
              f"length — scores are NOT on the same day set as the point models.")

    # Forecast only what will actually be scored. Without this a default run puts all 1942 days
    # through the model to report on 310 of them — 6x the inference for nothing, which on a
    # 200M-parameter model is the whole runtime.
    if not fcfg.get('eval_train'):
        kept, is_test = kept[is_test], np.ones(int(is_test.sum()), dtype=bool)
        print(f"  (train split not evaluated — forecasting the {len(kept)} test days only)")

    # --- the series the model sees ------------------------------------------------------
    ctx_series = raw if mode == 'direct' else _context.residual_series(raw, extra_lag)
    contexts = _context.gather_contexts(ctx_series, cutoff[kept], context_hours)

    # Covariates, for the adapters that take them. Days whose 48 h weather forecast is missing
    # come back as None and are dropped here rather than filled — the same rule the matrix
    # builders apply, and the reason the point models score 1957 days rather than 2072.
    # The horizon a sample needs is the number of rows from its cutoff to the end of its target
    # day: 72 normally, 73 when a fall-back day falls inside D-2..D, 71 on a spring-forward.
    # A single fixed horizon cannot serve all three — too short drops the DST days, too long
    # asks the covariate builder for rows past the target day, which it can only fill by
    # inventing them. So samples are GROUPED by required horizon and each group is run at its
    # own, which keeps every day and fabricates nothing. steps_to_day24 selects the target
    # day's rows by EPT date, so a group whose target starts at step 50 rather than 49 is
    # extracted correctly without special-casing.
    need_kept = _context.steps_needed(cutoff, target_days, spans)[kept]

    past_only = past_future = None
    if params.get('covariates'):
        from src.foundation import _covariates as COV
        fc_hourly = COV.load_forecast_hourly(cfg.FORECAST_PATH)
        po, pf = [None] * len(kept), [None] * len(kept)
        for h in sorted(set(need_kept.tolist())):
            g = np.where(need_kept == h)[0]
            gpo, gpf = COV.build(df, ept_dates, ept_hours, cutoff[kept][g], target_days[kept][g],
                                 fc_hourly, include_prices=bool(params.get('include_prices', True)),
                                 context_hours=context_hours, horizon=int(h),
                                 exclude=(SERIES[series],))
            for j, i in enumerate(g):
                po[i], pf[i] = gpo[j], gpf[j]
        have = np.array([x is not None for x in pf])
        if not have.all():
            lost_test = int((~have & is_test).sum())
            print(f"  covariates: {int((~have).sum())} day(s) dropped for want of a 48 h "
                  f"forecast ({lost_test} of them in the test split)")
            kept, is_test, contexts = kept[have], is_test[have], contexts[have]
            need_kept = need_kept[have]
            po = [x for x, k in zip(po, have) if k]
            pf = [x for x, k in zip(pf, have) if k]
        past_only, past_future = po, pf
        names = COV.channel_names(bool(params.get('include_prices', True)),
                                  exclude=(SERIES[series],))
        print(f"  covariates: past_only {len(names['past_only'])}x{context_hours}, "
              f"past_future {len(names['past_future'])}x{context_hours + horizon}"
              f"  (prices {'on' if params.get('include_prices', True) else 'off'})")
    if not np.isfinite(contexts).all():
        raise SystemExit("Non-finite values in the context windows — the cleaned CSV has holes.")

    if past_future is None:
        steps = _batched_forecast(adapter, contexts, horizon, params,
                                  int(fcfg.get('batch_size', 64)), f'{adapter_name}:{mode}')
    else:
        # One call per required horizon; results are written back into the sample's own slot so
        # the row order still matches `kept`.
        steps = np.full((len(contexts), int(need_kept.max())), np.nan)
        for h in sorted(set(need_kept.tolist())):
            g = np.where(need_kept == h)[0]
            out = _batched_forecast(adapter, contexts[g], int(h), params,
                                    int(fcfg.get('batch_size', 64)),
                                    f'{adapter_name}:{mode}:h{h}',
                                    past_only=[past_only[i] for i in g],
                                    past_future=[past_future[i] for i in g])
            steps[np.ix_(g, np.arange(int(h)))] = out
        horizon = int(need_kept.max())
    pred = _context.steps_to_day24(steps, cutoff[kept], target_days[kept], ept_dates, ept_hours)

    # --- back to metered MW -------------------------------------------------------------
    y_true = y_all[kept]
    extra, ratio = [], None
    # The baseline goes through the identical step -> day24 path as the forecast, from one
    # pass over the series (it used to be rebuilt per day inside a comprehension, which was
    # 1942 full-series scans and an unpadded slice that broke at the end of the record).
    baseline_24 = _context.steps_to_day24(
        _context.gather_steps(_context.baseline_series(raw, extra_lag), cutoff[kept], horizon),
        cutoff[kept], target_days[kept], ept_dates, ept_hours)

    if mode == 'residual':
        raw = pred.copy()
        pred = pred + baseline_24
        extra.append(('naive baseline alone', baseline_24))
        print("\n=== model vs the baseline it leans on (test) ===")
        print(_fmt('naive baseline alone', _score(y_true[is_test], baseline_24[is_test])))
        print(_fmt('residual + baseline', _score(y_true[is_test], pred[is_test])))
        print(f"  mean |predicted residual| = {np.abs(raw[is_test]).mean():.1f} {cfg.TARGET_UNIT}  "
              f"(0 would reproduce the baseline exactly)")
    else:
        uncorrected = pred.copy()
        # A metered context is already in the target's own space; there is no offset to correct
        # and applying one would introduce the very error it exists to remove.
        mode_cal = fcfg.get('est_to_metered', 'none') if series == 'estimated' else 'none'
        if mode_cal not in _calibrate.MODES:
            raise ValueError(f"est_to_metered must be one of {_calibrate.MODES}")
        if mode_cal == 'hourly_ratio':
            first_test_day = target_days[test_pos.min()]
            train_rows = ept_dates < first_test_day
            ratio = _calibrate.fit_hourly_ratio(load, est, ept_hours, train_rows)
            pred = _calibrate.apply_hourly_ratio(uncorrected, ratio)
            print(f"\n=== Est -> metered correction ({int(train_rows.sum())} train hours) ===")
            print(f"  {_calibrate.describe(ratio)}")
            print(_fmt('raw (Load_Estimated space)', _score(y_true[is_test], uncorrected[is_test])))
            print(_fmt('scale-corrected', _score(y_true[is_test], pred[is_test])))
            extra.append(('uncorrected (Est space)', uncorrected))
        extra.append(('naive baseline (reference)', baseline_24))

    if not np.isfinite(pred).all():
        bad = int((~np.isfinite(pred)).any(axis=1).sum())
        raise SystemExit(f"{bad} day(s) came back with non-finite predictions.")

    # --- evaluation: the same suite the 16 point models get -----------------------------
    ts_test = pd.to_datetime(y_idx.index[kept][is_test])
    train_df = None
    if fcfg.get('eval_train'):
        train_df = EvalUtils.build_detailed_df(name, y_true[~is_test], pred[~is_test],
                                               pd.to_datetime(y_idx.index[kept][~is_test]))
    EvalUtils.evaluate_one(name, y_true[is_test], pred[is_test], ts_test, result_dir, train_df)

    # --- what EvalUtils has no slot for, appended to its own metrics file ---------------
    scores = _score(y_true[is_test], pred[is_test])
    with open(os.path.join(result_dir, f'{name}_metrics.txt'), 'a') as f:
        f.write(f"\n--- foundation run ---\n")
        f.write(f"adapter:        {adapter_name}\n")
        f.write(f"context series: {SERIES[series]} ({series}"
                + (f", {issue})" if series == 'metered' else ")") + "\n")
        if series == 'metered':
            f.write(f"issue time:     {'target day 00:00 (day-of)' if issue == 'day_of' else 'target day - 1 at 00:00 (same as the 16 point models)'}\n")
        f.write(f"mode:           {mode}\n")
        f.write(f"context_hours:  {context_hours}\n")
        f.write(f"horizon_steps:  {horizon}\n")
        f.write(f"checkpoint:     {params.get('checkpoint', '-')}\n")
        f.write(f"est_to_metered: {fcfg.get('est_to_metered') if mode == 'direct' else 'n/a (cancels)'}\n")
        f.write(f"test days:      {int(is_test.sum())} of {len(test_pos)} in the matrix"
                f"{'  (INCOMPLETE)' if dropped_test else ''}\n")
        if extra:
            f.write("\nreference points on the same test days:\n")
            for tag, arr in extra:
                s = _score(y_true[is_test], arr[is_test])
                f.write(f"  {tag:28s} MAPE {s['MAPE']:6.3f} %   MAE {s['MAE']:8.2f} {cfg.TARGET_UNIT}\n")

    manifest = {'zone': cfg.DATASET, 'adapter': adapter_name, 'mode': mode,
                'series': series, 'metered_issue': issue if series == 'metered' else None,
                'extra_lag_days': extra_lag,
                'context_hours': context_hours, 'horizon_steps': horizon,
                'matrix_tag': cfg.MATRIX_TAG, 'params': {k: v for k, v in params.items()},
                'est_to_metered': fcfg.get('est_to_metered') if mode == 'direct' else None,
                'hourly_ratio': None if ratio is None else ratio.tolist(),
                'test_days': int(is_test.sum()), 'test_days_in_matrix': len(test_pos),
                'scores': scores}
    with open(os.path.join(result_dir, 'run_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n-> {result_dir}")
    return {'model': model_dir(adapter_name, mode),
            'run': run_tag(feature_cfg, context_hours, series, issue),
            'name': name, 'context_hours': context_hours, 'mode': mode, 'series': series, **scores,
            'test_days': int(is_test.sum())}
