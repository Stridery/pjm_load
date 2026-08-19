"""The process pool must not change a single number.

    PJM_DATASET=dom python test/test_qra_pool.py

Parallelism is the only optimisation applied to the QRA baseline, and it was chosen over the
faster alternatives precisely because it is defensible without a numerical argument. That
claim is worth checking rather than asserting: this runs both stages serially and pooled on a
short window and requires the results to be bit-identical.

Kept small on purpose (a few weeks, two sisters) — it verifies equivalence, not throughput.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantile import qra as Q                                   # noqa: E402
from src.quantile import sisters as S                               # noqa: E402
from train_quantile import load_frame, split_at_day_boundary        # noqa: E402


def main():
    df, ept, load, temp, hol, valid = load_frame()
    split = split_at_day_boundary(ept, 0.16)
    test_start = ept[split].normalize()

    # A short window well inside the record, so every rolling estimation window is available.
    days = [d.date() for d in pd.date_range(test_start - pd.Timedelta(days=120),
                                            test_start - pd.Timedelta(days=61), freq='D')]
    specs = [(2, (1, 1)), (3, (0, 1))]

    print(f"=== sister forecasts | {len(specs)} sisters x {len(days)} days ===")
    t0 = time.time()
    ser, rows, dl = S.rolling_forecasts(ept, load, temp, valid, specs, days,
                                        workers=1, verbose=False)
    t_ser = time.time() - t0
    t0 = time.time()
    par, rows2, dl2 = S.rolling_forecasts(ept, load, temp, valid, specs, days,
                                          workers=4, verbose=False)
    t_par = time.time() - t0
    assert np.array_equal(rows, rows2) and dl == dl2, "row/day alignment differs"
    same = np.array_equal(ser, par)
    print(f"  serial {t_ser:5.1f}s   pooled {t_par:5.1f}s   "
          f"max|diff| = {np.abs(ser - par).max():.3e}   bit-identical: {same}")
    assert same, "pooled sister forecasts differ from serial"

    # --- QRA: the pooled worker against a direct call --------------------------
    y = load[rows]
    lens = pd.Series(ept[rows].normalize()).value_counts().sort_index()
    edges = np.r_[0, np.cumsum([int(lens[pd.Timestamp(d).normalize()]) for d in dl])]
    slices = [(int(edges[i]), int(edges[i + 1])) for i in range(len(dl))]
    L, eval_from = 30, 30
    sl = slice(slices[eval_from][0], None)

    print(f"=== QRA | L={L}, {len(dl) - eval_from} evaluated days ===")
    direct = Q.qra(ser, y, slices, eval_from, m=2, L=L, refit_every=1)[:, sl]
    scored = Q.score_config((ser, y, slices, eval_from, None, sl, 2, L, 1))
    from src.quantile.qeval import pinball
    ref = float(pinball(y[sl], direct).mean())
    print(f"  direct pinball {ref:.10f}   worker pinball {scored['pinball']:.10f}   "
          f"diff {abs(ref - scored['pinball']):.3e}")
    assert ref == scored['pinball'], "pooled worker disagrees with a direct qra() call"

    print("\nPASS — the pool changes timing only.")


if __name__ == '__main__':
    main()
