"""Probabilistic forecast evaluation, shared by every baseline in this repo.

One scorer for all of them: src.density and src.qrnn import from here
rather than each computing its own pinball loss, because a table in which two rows were
scored by two implementations is not a comparison. Lower is better for every metric defined
below.

    pinball / quantile score   the metric of GEFCom2014; the mean over quantile levels and
                               hours of the asymmetrically weighted absolute error
    winkler                    interval score; charges for width at every hour and adds a
                               1/alpha-scaled penalty only on a miss, which is what pinball
                               alone cannot separate
    crps_from_quantiles        distribution-level score, read off a quantile function so
                               that density and quantile forecasts land on one scale
    coverage_report            nominal vs empirical coverage — a calibration diagnostic,
                               not a proper scoring rule
"""

import numpy as np

QUANTILES = np.arange(1, 100)          # the 1st..99th percentiles, as in the paper


def pinball(y_true, y_quantiles):
    """(99, N) pinball loss per quantile per hour.

    Xie & Hong (2018) eq. (2); identical to the max-of-two-branches form in
    src.qrnn.model.pinball_loss, which is the objective the neural baseline trains on.

        y_t <  y_q :  (1 - p) (y_q - y_t)        over-forecast at that quantile
        y_t >= y_q :       p  (y_t - y_q)
    """
    y_true = np.asarray(y_true, dtype=float)
    p = (QUANTILES / 100.0).reshape(-1, 1)
    d = np.asarray(y_quantiles, dtype=float) - y_true.reshape(1, -1)
    return np.where(d > 0, (1.0 - p) * d, p * (-d))


def winkler(y_true, y_quantiles, coverage):
    """Mean Winkler (interval) score for a central `coverage`% PI.  Liu et al. eq. (8).

        d,                      L <= y <= U
        d + 2(L - y)/alpha,     y < L
        d + 2(y - U)/alpha,     y > U          with d = U - L,  alpha = 1 - coverage/100

    Pinball alone cannot separate "covers everything because it is enormously wide" from
    "covers everything and is tight". Winkler charges for width on every hour and adds a
    1/alpha-scaled penalty only when the interval misses, so the two attributes are scored
    together. Lower is better, and it is reported alongside the pinball loss because the two
    disagree exactly when width and coverage trade off against each other.
    """
    y_true = np.asarray(y_true, dtype=float)
    lo_q, hi_q = (100 - coverage) // 2, 100 - (100 - coverage) // 2
    L = np.asarray(y_quantiles)[np.where(QUANTILES == lo_q)[0][0]]
    U = np.asarray(y_quantiles)[np.where(QUANTILES == hi_q)[0][0]]
    alpha = 1.0 - coverage / 100.0
    width = U - L
    score = np.where(y_true < L, width + 2 * (L - y_true) / alpha,
                     np.where(y_true > U, width + 2 * (y_true - U) / alpha, width))
    return float(np.mean(score))


def coverage_report(y_true, y_quantiles, levels=(50, 80, 90, 95, 98)):
    """Nominal vs empirical coverage of central intervals — the calibration diagnostic.

    Not from either paper: both score with pinball/Winkler only. Kept because a probabilistic
    baseline that is never checked for coverage is the one thing a reader will ask about
    first.
    """
    y_true = np.asarray(y_true, dtype=float)
    q = np.asarray(y_quantiles)
    rows = []
    for c in levels:
        lo, hi = (100 - c) // 2, 100 - (100 - c) // 2
        L, U = q[np.where(QUANTILES == lo)[0][0]], q[np.where(QUANTILES == hi)[0][0]]
        below, above = float((y_true < L).mean()), float((y_true > U).mean())
        rows.append({'nominal_pct': c, 'empirical_pct': 100 * (1 - below - above),
                     'below_pct': 100 * below, 'above_pct': 100 * above,
                     'mean_width': float(np.mean(U - L))})
    return rows


def crps_from_quantiles(levels, values, y_true):
    """(N,) CRPS per hour from a quantile function.  CRPS = 2 * integral_0^1 pinball dtau.

    Added for baseline 2 (Li et al. 2020), whose headline metric is the CRPS — but written
    against a quantile function rather than a Gaussian mixture so that the quantile-emitting
    baselines can be scored on it too. That is the whole point: it is the one metric every
    baseline here shares.

    `levels` are probabilities in (0, 1); the integral is a trapezoid over whatever grid is
    supplied, so a denser grid is a more accurate CRPS.
    """
    levels = np.asarray(levels, dtype=float)
    d = np.asarray(values, dtype=float) - np.asarray(y_true, dtype=float)[None, :]
    pin = np.where(d > 0, (1.0 - levels)[:, None] * d, -levels[:, None] * d)
    return 2.0 * np.trapezoid(pin, levels, axis=0)


def point_metrics(y_true, y_pred):
    """MAPE / MAE / RMSE for a single point forecast — the deterministic sanity check."""
    t, p = np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel()
    return {'MAPE': float(np.mean(np.abs((t - p) / t)) * 100),
            'MAE': float(np.mean(np.abs(t - p))),
            'RMSE': float(np.sqrt(np.mean((t - p) ** 2)))}
