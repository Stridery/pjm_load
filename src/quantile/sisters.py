"""Sister point forecasts — stage 1 of Liu et al. (2017), IEEE TSG 8(2):730-737.

Sister models are one model FAMILY whose members differ only in their variable-selection
outcome, so their forecasts are highly correlated ("the differences between them are hardly
visible", their Fig. 2) and their DISAGREEMENT is what carries information about
uncertainty. QRA (src/quantile/qra.py) is what turns that disagreement into quantiles.

The family here is Tao's Vanilla benchmark plus the recency effect (their eq. 4):

    y_t = [Vanilla eq. (1)] + sum_{d=1..D} f(T~_{t,d}) + sum_{j=1..J} f(T_{t-j})

where f(.) is the same Month/Hour-interacted cubic as the base temperature term and
T~_{t,d} is the mean temperature over the d-th preceding 24 h block (their eq. 3). Note the
base model carries NO trend and no macroeconomic term — eq. (1) has neither, and this
follows it. Note that Xie & Hong's Vanilla, the same benchmark model, DOES carry one (GSP);
Liu et al. drop it because the model is re-estimated daily on a rolling 2-3 year window,
which tracks growth by moving rather than by extrapolating.

That works when load is level-stationary, as GEFCom2014's is. DOM's is not (+33% over
2021-2026 on data-centre demand), and with no growth term ~99% of a sister's error there is
a constant level offset (measured: MAE 1493, ME +1475, de-biased MAE 471 — the shape fit is
fine, the level is a year stale). QRA, Ind and BI are INVARIANT to that: a common log-space
offset is absorbed exactly by the QRA intercept and by the residual quantiles. `Direct` is
not — it reads percentiles straight off the raw sister forecasts with no calibration — so
expect it to score badly on dom for this reason and not because combining fails.


What is reproduced, and the one thing that is not
------------------------------------------------
The paper builds its eight sisters as (2 or 3 years of training data) x (four calibration
schemes from [26], Wang/Liu/Hong "Electrical Load Forecasting With Recency Effect"), and the
four schemes are what produce four different (D, J) pairs. [26] is not in hand and the paper
describes only two of its four schemes, so the schemes themselves cannot be reproduced.

What the schemes ARE is reproducible from the paper's own description of them: they are
different "partition[s] of the training and validation datasets for model selection", one
of which treats all hourly values as one series and another as 24 independent series. So
four partitions are defined here (SELECTION_SCHEMES) and each picks its own best (J, D).
Same family, same mechanism, same count (2 x 4 = 8); only the specific partitions differ.

Selecting this way rather than by "keep the four best" is not cosmetic. Measured on dom,
the four lowest-MAE orders have a mean pairwise ERROR correlation of 0.963, while four
orders chosen by different partitions sit near 0.81 (the family's floor is 0.709). Sister
forecasts exist to disagree; ranking them by accuracy and keeping the top of the list
selects against exactly the property the method needs.


Fitting is in LOG space
-----------------------
The paper log-transforms load and forecasts throughout the QRA and residual benchmarks.
Doing the same here keeps stage 1 and stage 2 on one scale, and it is also what makes the
prediction intervals widen at peak in MW terms (a constant log-space width is a proportional
MW width), which is the behaviour the paper points out in its Section IV-C.
"""

import numpy as np
import pandas as pd

from src.quantile._pool import n_workers, run, single_threaded
from src.quantile.design import Design

TRAIN_YEARS = (2, 3)          # the paper's two estimation-window lengths

# Four "partition[s] of the training and validation datasets for model selection". The first
# two are the paper's own examples (all hourly values as one series; as 24 independent
# series); the other two split the validation year in time, which is the other obvious axis
# a partition can vary along. Each returns its own best recency order.
SELECTION_SCHEMES = ('pooled', 'per_hour', 'first_half', 'second_half')


def candidate_recency(max_lag=3, max_dma=3):
    """The (J, D) grid searched on the validation year. (0, 0) is plain Vanilla."""
    return [(j, d) for j in range(max_lag + 1) for d in range(max_dma + 1)]


def _day_bounds(ept, days):
    """{day: (first_row, last_row_exclusive)} — position slices, so no per-day masking."""
    dates = ept.normalize().values
    out = {}
    for d in days:
        k = np.datetime64(pd.Timestamp(d).normalize())
        lo, hi = np.searchsorted(dates, k, 'left'), np.searchsorted(dates, k, 'right')
        if hi > lo:
            out[d] = (int(lo), int(hi))
    return out


def build_design(ept, temp, recency, train_mask):
    """(Design, X) for one (J, D) pair, built ONCE over the whole record.

    The rolling refits below are then pure row slices of X. Rebuilding the matrix per day
    would cost more than the fit it feeds (0.45 s vs 1.19 s at the widest recency), and the
    lagged columns are identical either way because they are lagged off the same series.
    """
    des = Design.fit('vanilla', ept, temp, train_mask, trend=False, recency=recency)
    X = np.hstack([des.calendar_block(ept), des.temp_block(ept, temp)])
    return des, X


def _fit_log(X, y, rows):
    beta, *_ = np.linalg.lstsq(X[rows], np.log(y[rows]), rcond=None)
    return beta


def _scheme_scores(err, hours, half):
    """One MAE per selection scheme, in SELECTION_SCHEMES order."""
    return (float(np.abs(err).mean()),
            float(np.mean([np.abs(err[hours == h]).mean() for h in range(24)])),
            float(np.abs(err[half == 0]).mean()),
            float(np.abs(err[half == 1]).mean()))


def select_recency(ept, load, temp, valid, val_start, test_start, n_keep=4,
                   max_lag=3, max_dma=3, verbose=True):
    """{train_years: [(J, D), ...]} — one recency order per selection scheme.

    Each scheme scores the same validation-year forecasts a different way and keeps its own
    argmin; a scheme whose winner is already taken falls through to its next-best, so the
    n_keep sisters of a training window are always distinct. Coefficients are fit once,
    ending at the validation start — the paper rolls coefficients daily but selects (D, J)
    once, and rolling the sweep too would multiply its cost by 365 for a frozen choice.
    """
    va = (ept >= val_start) & (ept < test_start)
    hours = ept[va].hour.values
    half = (np.arange(int(va.sum())) >= int(va.sum()) // 2).astype(int)
    grid = candidate_recency(max_lag, max_dma)

    out = {}
    for ty in TRAIN_YEARS:
        tr = (ept >= val_start - pd.DateOffset(years=ty)) & (ept < val_start) & (valid == 1)
        scores = []
        for rec in grid:
            _, X = build_design(ept, temp, rec, tr)
            pred = np.exp(X[va] @ _fit_log(X, load, np.where(tr)[0]))
            scores.append(_scheme_scores(load[va] - pred, hours, half))
        scores = np.array(scores)                                   # (n_grid, n_schemes)

        chosen = []
        for si in range(min(n_keep, len(SELECTION_SCHEMES))):
            for gi in np.argsort(scores[:, si]):
                if grid[gi] not in chosen:
                    chosen.append(grid[gi])
                    if verbose:
                        print(f"  train {ty}y  {SELECTION_SCHEMES[si]:<12} -> "
                              f"(J={grid[gi][0]},D={grid[gi][1]})  MAE {scores[gi, si]:.1f}")
                    break
        out[ty] = chosen
    return out


def sister_specs(selected):
    """[(train_years, (J, D)), ...] in a stable order — the paper's Ind1..Ind8."""
    return [(ty, rec) for ty in TRAIN_YEARS for rec in selected[ty]]


def _one_sister(task):
    """One sister's whole rolling forecast. Module-level so a process pool can pickle it."""
    si, ty, rec, ept, load, temp, valid, day_list, bounds, refit_every = task
    with single_threaded():
        first = pd.Timestamp(day_list[0]) - pd.DateOffset(years=ty)
        seed = (ept >= first) & (ept < pd.Timestamp(day_list[0])) & (valid == 1)
        _, X = build_design(ept, temp, rec, seed)
        dates = ept.values

        n_rows = sum(hi - lo for lo, hi in (bounds[d] for d in day_list))
        out = np.empty(n_rows)
        beta, off, n_fit = None, 0, 0
        for di, d in enumerate(day_list):
            lo, hi = bounds[d]
            if di % refit_every == 0:
                w0 = np.searchsorted(
                    dates, np.datetime64(pd.Timestamp(d) - pd.DateOffset(years=ty)), 'left')
                rows = np.arange(w0, lo)
                rows = rows[valid[rows] == 1]    # same 3-sigma mask as the point models
                beta = _fit_log(X, load, rows)
                n_fit += 1
            out[off:off + hi - lo] = np.exp(X[lo:hi] @ beta)
            off += hi - lo
    return si, out, X.shape[1], n_fit


def rolling_forecasts(ept, load, temp, valid, specs, day_list, refit_every=1,
                      workers=None, verbose=True):
    """(n_sisters, n_hours) point forecasts in MW over the hours of `day_list`.

    Coefficients are re-estimated on a rolling window ending the instant before each forecast
    day — the paper's "re-estimated each day". `refit_every > 1` reuses the previous fit for
    that many days; it is a speed knob and a DEVIATION, so the caller reports it.

    The eight sisters are independent, so they run in a process pool. That changes nothing
    about the numbers — each sister's arithmetic is untouched — only how long the stage takes.
    """
    bounds = _day_bounds(ept, day_list)
    day_list = [d for d in day_list if d in bounds]
    rows_out = np.concatenate([np.arange(*bounds[d]) for d in day_list])

    w = n_workers(workers)
    tasks = [(si, ty, rec, ept, load, temp, valid, day_list, bounds, refit_every)
             for si, (ty, rec) in enumerate(specs)]
    if verbose:
        print(f"  {len(specs)} sisters over {len(day_list)} days on {min(w, len(tasks))} "
              f"process(es)")

    out = np.empty((len(specs), len(rows_out)))
    for si, row, ncol, n_fit in run(_one_sister, tasks, w,
                                    label='sisters' if verbose else None):
        out[si] = row
        if verbose:
            ty, rec = specs[si]
            print(f"    sister {si + 1}: train {ty}y, recency (J={rec[0]},D={rec[1]}), "
                  f"{ncol} cols, {n_fit} refits")
    return out, rows_out, day_list
