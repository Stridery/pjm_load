"""Quantile Regression Averaging — stage 2 of Liu et al. (2017).

Consumes a matrix of sister POINT forecasts and emits 99 quantiles per hour: quantile
regression of the actual load on the m best sisters, refit per quantile on a rolling window.

The paper's three benchmarks (Direct, Ind, BI) are deliberately NOT implemented — only the
method itself is shipped. They would have cost ~0.3% of the runtime, so this is a choice
about what the deliverable claims to be, not about compute.

Works in LOG space and exponentiates at the end, as the paper does. One consequence worth
stating: a constant log-space width is a PROPORTIONAL width in MW, so every interval here is
automatically wider at peak than at trough.


Why QRA can be calibrated when scenario generation cannot
---------------------------------------------------------
The sister forecasts disagree more on days that are hard to forecast. QRA fits, per quantile
level, a different weighting of them, on the most recent L days of REALISED error — so the
interval width is estimated from how wrong this family of models has actually been lately,
and it breathes with the sisters' disagreement. A scenario ensemble has neither property:
its width comes only from temperature spread and carries no model error at all.
"""

import numpy as np

from src.quantile.qeval import QUANTILES


# ---------------------------------------------------------------------------
# Quantile regression
# ---------------------------------------------------------------------------

def quantreg_irls(X, y, q, max_iter=1000, tol=1e-6, beta0=None):
    """Koenker quantile regression by IRLS. Minimises the pinball loss of eq. (6).

    This is the algorithm statsmodels.QuantReg uses, transcribed rather than imported:
    statsmodels is not a dependency of this repo, and the daily serving workflow installs
    requirement.txt on every run, so adding one for a module the workflow never touches
    would be pure cost. Verified against sklearn's linear-programming solver — identical
    pinball loss to six decimals, in-sample coverage exactly q — and ~100x faster (the LP
    carries 2L slack variables; this is a handful of 9x9 weighted least-squares solves).

    beta0 seeds the iteration from a neighbouring day's solution. NOTE it must not be the
    OLS fit: iteration 1 reproduces OLS exactly, so an OLS seed satisfies `tol` immediately
    and returns the mean regression for every q.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if beta0 is None:
        beta = np.ones(X.shape[1])
        xstar = X.copy()
    else:
        beta = np.asarray(beta0, float).copy()
        xstar = X / _irls_weight(y - X @ beta, q)[:, None]

    for _ in range(max_iter):
        prev = beta
        beta = np.linalg.pinv(xstar.T @ X) @ (xstar.T @ y)
        xstar = X / _irls_weight(y - X @ beta, q)[:, None]
        if np.max(np.abs(beta - prev)) < tol:
            break
    return beta


def _irls_weight(r, q):
    """|asymmetrically weighted residual|, floored away from zero."""
    small = np.abs(r) < 1e-6
    r = r.copy()
    r[small] = np.where(r[small] >= 0, 1e-6, -1e-6)
    return np.abs(np.where(r < 0, q * r, (1 - q) * r))


# ---------------------------------------------------------------------------
# Rolling window plumbing
# ---------------------------------------------------------------------------

def rolling_windows(day_slices, eval_from, L, eval_to=None):
    """(i, calibration_rows, target_rows) for each evaluated day in [eval_from, eval_to).

    The calibration window is the L whole days ENDING THE DAY BEFORE the target — never
    the target day itself, which at forecast time has no actual load yet.

    `eval_to` matters for cost, not just tidiness: the validation sweep runs 21 (m, L)
    combinations, and without a stop it would forecast the test period 21 more times for
    numbers nobody reads.
    """
    for i in range(eval_from, len(day_slices) if eval_to is None else eval_to):
        j = i - L
        if j < 0:
            raise ValueError(f"calibration window L={L} reaches before the available history")
        yield (i,
               np.arange(day_slices[j][0], day_slices[i - 1][1]),
               np.arange(*day_slices[i]))


def _monotone(q):
    """Sort each hour's 99 values. Independent quantile regressions can cross; rearranging
    is the standard remedy and weakly improves the pinball loss, so it can only help.
    Neither paper mentions it — with crossing left in, a "90% interval" is not always an
    interval at all."""
    return np.sort(q, axis=0)


# ---------------------------------------------------------------------------
# The method
# ---------------------------------------------------------------------------


def qra(sister_fc, actual, day_slices, eval_from, m, L, refit_every=1, warm_start=True, eval_to=None):
    """(99, N) quantile regression averaging over the m best sisters.  Their eq. (5)-(6).

    The m sisters are the lowest-MAE ones in the calibration window, as the paper selects
    them. Weights are per-quantile, which is the whole point: the mixing rule for an interval
    forecast is not the mixing rule for a point forecast, so a fixed average cannot deliver
    nominal coverage.
    """
    sister_fc = np.asarray(sister_fc, float)
    ln_fc, ln_y = np.log(sister_fc), np.log(np.asarray(actual, float))
    out = np.full((len(QUANTILES), sister_fc.shape[1]), np.nan)

    pick, betas = None, None
    for k, (i, cal, tgt) in enumerate(rolling_windows(day_slices, eval_from, L, eval_to)):
        if k % refit_every == 0:
            mae = np.mean(np.abs(sister_fc[:, cal] - actual[cal]), axis=1)
            pick = np.argsort(mae)[:m]
            Xc = np.column_stack([np.ones(len(cal)), ln_fc[np.ix_(pick, cal)].T])
            seed = betas if (warm_start and betas is not None) else [None] * len(QUANTILES)
            betas = [quantreg_irls(Xc, ln_y[cal], q / 100.0, beta0=s)
                     for q, s in zip(QUANTILES, seed)]
        Xt = np.column_stack([np.ones(len(tgt)), ln_fc[np.ix_(pick, tgt)].T])
        out[:, tgt] = np.exp(np.stack([Xt @ b for b in betas]))
    return _monotone(out)


def score_config(task):
    """Fit + score one (m, L) configuration. Module-level so a process pool can pickle it.

    Scoring happens INSIDE the worker on purpose: returning the (99, N) quantile array would
    ship ~16 MB per task back through the pipe, while the three scores are a few bytes.
    """
    from src.quantile._pool import single_threaded
    from src.quantile.qeval import pinball, winkler
    fc, y, slices, eval_from, eval_to, sl, m, L, refit = task
    with single_threaded():
        q = qra(fc, y, slices, eval_from, m=m, L=L, eval_to=eval_to, refit_every=refit)[:, sl]
        yv = y[sl]
        return {'model': f'QRA{m}', 'm': m, 'L': L,
                'pinball': float(pinball(yv, q).mean()),
                'winkler50': winkler(yv, q, 50), 'winkler90': winkler(yv, q, 90)}
