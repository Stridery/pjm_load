"""Probabilistic forecast evaluation: Xie & Hong (2018) eq. (1) and eq. (2).

Scenario point forecasts -> percentiles -> pinball loss -> quantile score. A LOWER quantile
score is better.

Why eq. (1) is transcribed instead of calling np.percentile
-----------------------------------------------------------
The paper uses the empirical distribution function WITH AVERAGING, and says explicitly that
it chose it because it still produces percentiles when j is small. That is our regime: with
four scenario years and no shifting, j = 4. numpy's default linear interpolation between
order statistics gives different numbers there, and the resulting quantile scores would not
be comparable with the published ones. At large j the two converge, so nothing is lost by
always using the paper's rule.

    N non-missing forecasts, sorted ascending as x_1..x_N;  p = q/100;  i = floor(Np);
    g = Np - i:
        g == 0  ->  y_q = 1/2 (x_i + x_{i+1})
        g >  0  ->  y_q = x_{i+1}
"""

import numpy as np

QUANTILES = np.arange(1, 100)          # the 1st..99th percentiles, as in the paper


def empirical_percentiles(preds):
    """(J, N) scenario point forecasts -> (99, N) percentile forecasts.  Paper eq. (1).

    Indices below are converted from the paper's 1-based x_i to 0-based numpy: x_i is
    s[i-1] and x_{i+1} is s[i]. Both are clamped, which only ever binds for degenerate J.
    """
    preds = np.asarray(preds, dtype=float)
    if preds.ndim != 2:
        raise ValueError(f"preds must be (J, N), got shape {preds.shape}")
    s = np.sort(preds, axis=0)
    J = s.shape[0]

    out = np.empty((len(QUANTILES), s.shape[1]))
    for qi, q in enumerate(QUANTILES):
        Np = J * (q / 100.0)
        i = int(np.floor(Np))
        if np.isclose(Np - i, 0.0):
            out[qi] = 0.5 * (s[min(max(i - 1, 0), J - 1)] + s[min(i, J - 1)])
        else:
            out[qi] = s[min(i, J - 1)]
    return out


def pinball(y_true, y_quantiles):
    """(99, N) pinball loss per quantile per hour.  Paper eq. (2).

        y_t <  y_q :  (1 - p) (y_q - y_t)        over-forecast at that quantile
        y_t >= y_q :       p  (y_t - y_q)
    """
    y_true = np.asarray(y_true, dtype=float)
    p = (QUANTILES / 100.0).reshape(-1, 1)
    d = np.asarray(y_quantiles, dtype=float) - y_true.reshape(1, -1)
    return np.where(d > 0, (1.0 - p) * d, p * (-d))


def quantile_score(y_true, preds):
    """Mean pinball loss over all 99 quantiles and all forecast hours. Lower is better."""
    return float(pinball(y_true, empirical_percentiles(preds)).mean())


def winkler(y_true, y_quantiles, coverage):
    """Mean Winkler (interval) score for a central `coverage`% PI.  Liu et al. eq. (8).

        d,                      L <= y <= U
        d + 2(L - y)/alpha,     y < L
        d + 2(y - U)/alpha,     y > U          with d = U - L,  alpha = 1 - coverage/100

    Pinball alone cannot separate "covers everything because it is enormously wide" from
    "covers everything and is tight". Winkler charges for width on every hour and adds a
    1/alpha-scaled penalty only when the interval misses, so the two attributes are scored
    together. Lower is better. This is where the QRA paper's method separates itself most
    clearly from its benchmarks, which is why it is reported alongside the pinball loss.
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
    first (see the scenario baseline, whose nominal 98% covers 77%).
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

    Added for baseline 3 (Li et al. 2020), whose headline metric is the CRPS — but written
    against a quantile function rather than a Gaussian mixture so that the scenario and QRA
    baselines can be scored on it too. That is the whole point: it is the first metric all
    three papers share.

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
