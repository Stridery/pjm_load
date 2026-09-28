"""The chapter's metric set, for an arbitrary quantile grid.

src/quantile/qeval.py indexes into a fixed 1..99 percentile grid, which cannot score the
three levels {0.05, 0.50, 0.95} this chapter reports. Everything here takes the levels as an
argument instead, so the same functions score a 3-level forecast and the 999-level grid used
for a reference CRPS.

    Pinball   mean over levels and steps of the asymmetric absolute error
    PICP      share of steps inside the central interval — should sit near 90%
    MPIW      mean width of that interval
    Winkler   width, plus a 2/alpha penalty on the steps that fall outside it
    MAE RMSE MSE   on the median as the point forecast
    MAPE SMAPE     scale-free, so computed in PHYSICAL units (see below)

Units. Pinball, MPIW, Winkler, MAE, RMSE and MSE are reported on the z-scored target, which
is what puts the three datasets on one axis. A percentage error in z units is not a
percentage error of anything — the z series crosses zero — so MAPE and SMAPE are computed
after mapping the median and the actuals back through (mean, sd).
"""

import numpy as np


def pinball(y, q, levels):
    """(K, N) pinball loss. `q` is (K, N) in the same order as `levels`."""
    y = np.asarray(y, float).reshape(1, -1)
    lv = np.asarray(levels, float).reshape(-1, 1)
    d = np.asarray(q, float) - y
    return np.where(d > 0, (1.0 - lv) * d, lv * (-d))


def _band(q, levels, coverage):
    """Rows of `q` holding the lower and upper end of a central `coverage` interval."""
    lv = np.asarray(levels, float)
    lo, hi = (1.0 - coverage) / 2.0, 1.0 - (1.0 - coverage) / 2.0
    i, j = int(np.argmin(np.abs(lv - lo))), int(np.argmin(np.abs(lv - hi)))
    if max(abs(lv[i] - lo), abs(lv[j] - hi)) > 1e-6:
        raise ValueError(f"levels {lv} do not contain the {coverage:.0%} interval")
    return np.asarray(q, float)[i], np.asarray(q, float)[j]


def picp_mpiw(y, q, levels, coverage=0.90):
    L, U = _band(q, levels, coverage)
    y = np.asarray(y, float)
    return float(100.0 * ((y >= L) & (y <= U)).mean()), float(np.mean(U - L))


def winkler(y, q, levels, coverage=0.90):
    """Mean interval score: width everywhere, plus 2/alpha times the miss distance."""
    L, U = _band(q, levels, coverage)
    y = np.asarray(y, float)
    a = 1.0 - coverage
    w = U - L
    s = np.where(y < L, w + 2 * (L - y) / a, np.where(y > U, w + 2 * (y - U) / a, w))
    return float(np.mean(s))


def median_of(q, levels):
    lv = np.asarray(levels, float)
    i = int(np.argmin(np.abs(lv - 0.5)))
    if abs(lv[i] - 0.5) > 1e-6:
        raise ValueError("levels must contain the median for the point metrics")
    return np.asarray(q, float)[i]


def score(y, q, levels, scale=None, coverage=0.90):
    """Every metric for one forecast. `scale` is the (mean, sd) used to z-score the target;
    supplying it adds MAPE and SMAPE, computed in physical units."""
    y = np.asarray(y, float)
    med = median_of(q, levels)
    picp, mpiw = picp_mpiw(y, q, levels, coverage)
    e = y - med
    out = {'Pinball': float(pinball(y, q, levels).mean()),
           'PICP': picp, 'MPIW': mpiw,
           'Winkler': winkler(y, q, levels, coverage),
           'MAE': float(np.mean(np.abs(e))),
           'RMSE': float(np.sqrt(np.mean(e ** 2))),
           'MSE': float(np.mean(e ** 2))}
    if scale is not None:
        m, s = scale
        yp, mp = y * s + m, med * s + m
        out['MAPE'] = float(100 * np.mean(np.abs((yp - mp) / yp)))
        out['SMAPE'] = float(100 * np.mean(2 * np.abs(yp - mp) / (np.abs(yp) + np.abs(mp))))
    return out


def crossing_rate(q):
    """Share of (level, step) pairs where the quantile function decreases. Reported, not
    fixed: every model here sorts its output, so a non-zero value means a bug upstream."""
    q = np.asarray(q, float)
    return float(100.0 * (np.diff(q, axis=0) < 0).mean()) if q.shape[0] > 1 else 0.0


DDOF = 1        # sample standard deviation; see aggregate()


def aggregate(runs, ddof=DDOF):
    """{'metric': (mean, sd)} over a list of per-seed metric dicts.

    `ddof=1` is the SAMPLE standard deviation. The seeds are a sample of the run-to-run
    variation, not the population of it, and at three seeds the choice is not cosmetic: the
    population form (ddof=0) is smaller by a factor sqrt(2/3) = 0.82, so reporting it would
    understate the spread by 18%. With one seed the sd is undefined and comes back as nan
    rather than as a fabricated zero.
    """
    keys = list(runs[0])
    return {k: (float(np.mean([r[k] for r in runs])),
                float(np.std([r[k] for r in runs], ddof=ddof)) if len(runs) > ddof
                else float('nan'))
            for k in keys}
