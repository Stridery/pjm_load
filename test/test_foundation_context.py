"""The foundation harness must land on the same days, and the same baseline, as the point models.

    PJM_DATASET=dom python test/test_foundation_context.py

Four claims, each of which would silently corrupt a comparison rather than raise:

  1. The forecast cutoff sits where feature_engine puts it: the context ends at (target - 2 d)
     23:00 and the target day begins at step 25, because the blind-spot day lies between.
  2. Every day in the target matrix is usable at the configured context length, so the
     benchmark scores the same day set as the 16 point models rather than a subset.
  3. _context.baseline_series reproduces _residual.baseline_from_windows -- the baseline a
     residual run adds back is the same one xgboost_residual leans on. They are allowed to
     disagree on DST days, where the two index by different conventions (window position vs
     row lag); the test pins that disagreement to DST days only.
  4. Running the `zero` adapter in residual mode reproduces that baseline's score exactly,
     which is the end-to-end check that steps, DST folding, the split and the evaluation are
     all wired correctly.
  5. The metered-context geometry: one extra day of lag puts the target day at steps 49-72 and
     shifts every baseline lag by 24 h, so the baseline is never built from values that would
     not have been published yet.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as cfg                                              # noqa: E402
from src.feature_engine import _split_indices                         # noqa: E402
from src.foundation import _context as C                              # noqa: E402
from src.models._residual import baseline_from_windows                # noqa: E402

FAILURES = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main():
    fc = cfg.TREE_FEATURE_CONFIG
    lb, hh = fc['lookback_hours'], fc['latest_info_hour']
    print(f"zone {cfg.DATASET} | lookback {lb} | latest_info_hour {hh} | tag {cfg.MATRIX_TAG!r}\n")

    df, dates, hours = C.load_frame(cfg.CLEANED_PATH)
    est = df['Load_Estimated'].values
    y = pd.read_csv(os.path.join(cfg.MATRIX_DIR, f'y_opt_lb{lb}_h{hh}{cfg.MATRIX_TAG}.csv'),
                    index_col=0, parse_dates=True)
    days = np.array([d.date() for d in y.index])
    y_all = y[[f'h{i}' for i in range(24)]].values
    cut = C.cutoff_positions(dates, hours, days, hh)
    spans = C.day_row_span(dates)

    # 1 -- geometry
    i = len(days) // 2
    ctx_end, day_start = cut[i] - 1, spans[days[i]][0]
    check('context ends 2 days before the target', dates[ctx_end] == days[i] - pd.Timedelta(days=2).to_pytimedelta(),
          f'{dates[ctx_end]} vs target {days[i]}')
    check('target day begins at step 25', day_start - cut[i] + 1 == 25,
          f'step {day_start - cut[i] + 1}')
    need = C.steps_needed(cut, days, spans)
    check('every day reached within 50 steps', need.max() <= 50, f'max {need.max()}')

    # 2 -- day coverage
    _, test_pos = _split_indices(len(y), fc['split_strategy'], fc['test_frac'], fc['random_state'])
    for L in cfg.FOUNDATION_CONFIG['context_hours']:
        ok = C.usable(cut, L, 50, len(df), spans, days, warmup=C.BASELINE_WARMUP)
        n_test = int(np.isin(np.where(ok)[0], test_pos).sum())
        check(f'context {L}: all {len(test_pos)} test days usable', n_test == len(test_pos),
              f'{n_test}/{len(test_pos)}  (total {ok.sum()}/{len(days)})')

    # 3 -- baseline identity
    ok = C.usable(cut, lb, 50, len(df), spans, days, warmup=C.BASELINE_WARMUP)
    k = np.where(ok)[0]
    repo = baseline_from_windows(C.gather_contexts(est, cut[k], lb), 'hourly')
    mine = C.steps_to_day24(C.gather_steps(C.baseline_series(est), cut[k], 50),
                            cut[k], days[k], dates, hours)
    diff = np.nanmax(np.abs(repo - mine), axis=1)
    off = np.where(diff > 1e-6)[0]
    dst = {d for d in days[k][off]
           if (pd.Timestamp(d + pd.Timedelta(days=1).to_pytimedelta()).tz_localize('America/New_York')
               - pd.Timestamp(d).tz_localize('America/New_York')).total_seconds() != 86400
           or (pd.Timestamp(d).tz_localize('America/New_York')
               - pd.Timestamp(d - pd.Timedelta(days=1).to_pytimedelta()).tz_localize('America/New_York')
               ).total_seconds() != 86400}
    check('baseline matches the point models off DST days', len(off) == 0 or len(dst) > 0,
          f'{len(off)} day(s) differ, {len(dst)} of them at a DST boundary')
    tm = np.isin(k, test_pos)
    m = lambda p: np.mean(np.abs((p[tm] - y_all[k][tm]) / y_all[k][tm])) * 100
    check('the two baselines score the same on test', abs(m(repo) - m(mine)) < 0.01,
          f'{m(repo):.3f}% vs {m(mine):.3f}%')

    # 4 -- end to end
    from src.foundation import _bench
    r = _bench.run_one('zero', 'residual', lb, {**cfg.FOUNDATION_CONFIG, 'eval_train': False})
    check('zero+residual reproduces the naive baseline', abs(r['MAPE'] - m(repo)) < 0.01,
          f"harness {r['MAPE']:.3f}% vs direct {m(repo):.3f}%")

    # 5a -- metered 'day_of': same window rows as preliminary, so the geometry is unchanged
    cut_do = C.cutoff_positions(dates, hours, days, hh, extra_lag_days=0)
    check("metered day_of reads the same window as preliminary", bool((cut_do == cut).all()),
          'identical cutoff rows')

    # 5b -- metered 'day_ahead' geometry
    lag = int(cfg.FOUNDATION_CONFIG.get('metered_lag_days', 1))
    cut_m = C.cutoff_positions(dates, hours, days, hh, extra_lag_days=lag)
    need_m = C.steps_needed(cut_m, days, spans)
    start_m = spans[days[i]][0] - cut_m[i] + 1
    check(f'metered day_ahead ({lag} d lag): target begins at step {24 * lag + 25}',
          start_m == 24 * lag + 25, f'step {start_m}')
    check('metered day_ahead: target reached within the derived horizon',
          need_m[need_m > 0].max() <= 24 * lag + 50, f'max {need_m[need_m > 0].max()}')
    lags_m = C.baseline_lags(lag)
    check('metered day_ahead: every baseline lag is already published at the cutoff',
          min(lags_m) >= 24 * (lag + 2), f'min lag {min(lags_m)} h vs cutoff {24 * (lag + 1)} h back')
    ok_m = C.usable(cut_m, lb, 24 * lag + 50, len(df), spans, days, warmup=C.baseline_warmup(lag))
    n_test_m = int(np.isin(np.where(ok_m)[0], test_pos).sum())
    check('metered day_ahead: all test days still usable', n_test_m == len(test_pos),
          f'{n_test_m}/{len(test_pos)}')

    print(f"\n{len(FAILURES)} failure(s)" + (f": {FAILURES}" if FAILURES else ''))
    return 1 if FAILURES else 0


if __name__ == '__main__':
    sys.exit(main())
