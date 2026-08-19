"""Quantiles -> Gaussian, by regression on probits.  Li et al. Section IV-D.

Two of the three base-model families emit quantiles, not densities, and the combination
step needs densities. The paper's conversion is deliberately simple: if the quantiles really
came from N(mu, sigma), then

    y_q = mu + p_q * sigma ,      p_q = Phi^-1(q)

so the points (p_q, y_q) lie on a line whose intercept is mu and whose slope is sigma. Fit
that line by OLS and read the two parameters off it (their Fig. 3).

It is an approximation and the paper says so — its justification is that regional loads are
close to Gaussian, and that the mixture built downstream is what absorbs the residual
non-normality. Worth remembering when reading the result: any skew a base model captured is
discarded here, before the combination ever sees it.

The probit grid is the same for every hour, so the whole conversion is one vectorised
least-squares over the quantile axis rather than T separate fits.
"""

import numpy as np
from scipy.special import ndtri

MIN_SIGMA = 1e-6


def quantiles_to_gaussian(levels, values):
    """(mu, sigma), each (T,), from `values` (Q, T) at probability `levels` (Q,).

    `levels` are probabilities in (0, 1) — deciles in the paper. A degenerate or decreasing
    fit is floored rather than allowed through: a non-positive sigma would make the CRPS
    terms downstream undefined, and it means the base model produced crossing quantiles.
    """
    levels = np.asarray(levels, dtype=float)
    values = np.asarray(values, dtype=float)
    if levels.ndim != 1 or values.shape[0] != len(levels):
        raise ValueError(f"values must be (Q, T) with Q={len(levels)}, got {values.shape}")
    if np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("levels must lie strictly inside (0, 1) — probits are infinite at 0/1")

    p = ndtri(levels)                       # probit of each level
    pc = p - p.mean()
    denom = float(pc @ pc)
    vbar = values.mean(axis=0)
    sigma = (pc[:, None] * (values - vbar)).sum(axis=0) / denom
    mu = vbar - sigma * p.mean()
    return mu, np.maximum(sigma, MIN_SIGMA)


def mixture_quantiles(weights, mu, sigma, levels, grid=4096, span=6.0):
    """(len(levels), T) quantiles of the Gaussian mixture, by inverting its CDF on a grid.

    The mixture CDF has no closed-form inverse, so the pinball / Winkler / coverage numbers
    that put this baseline on the same footing as the other two are read off a fine grid
    spanning `span` standard deviations either side of the widest component.
    """
    from scipy.special import ndtr
    weights = np.asarray(weights, dtype=float)
    lo = (mu - span * sigma).min(axis=0)
    hi = (mu + span * sigma).max(axis=0)
    T = mu.shape[1]
    out = np.empty((len(levels), T))
    for t in range(T):
        z = np.linspace(lo[t], hi[t], grid)
        cdf = (weights[:, None] * ndtr((z[None, :] - mu[:, t:t + 1]) / sigma[:, t:t + 1])).sum(0)
        out[:, t] = np.interp(levels, cdf, z)
    return out


def mixture_mean(weights, mu):
    """The mixture's expectation — the paper's point forecast, scored by MAPE and MAE."""
    return np.asarray(weights, dtype=float) @ mu
