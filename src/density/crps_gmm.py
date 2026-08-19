"""Closed-form CRPS of a Gaussian mixture.  Li et al. eq. (3), (5)-(9).

This is what makes the paper's combination a quadratic program instead of a simulation. The
CRPS has no readily computable integral form, but Baringhaus & Franz's identity

    CRPS(F, y) = E|Y - y| - 1/2 E|Y - Y'|                                   (their eq. 3)

turns it into expectations of absolute values, and for a finite Gaussian mixture those are
available in closed form because a difference of Gaussian mixtures is again one (their
Lemma 1) and the expectation of an absolute value is additive over components (Lemma 2). The
result is quadratic in the weights:

    CRPS(F, y) = sum_i sum_k alpha_{i,k} w_i w_k + sum_i beta_i w_i        (their eq. 7)

with alpha_{i,k} = -1/2 E|Y_i - Y'_k| and beta_i = E|Y_i - y|, both instances of

    E|X| = sigma sqrt(2/pi) exp(-m^2 / 2 sigma^2) + m (2 Phi(m/sigma) - 1),  X ~ N(m, sigma^2)

which is their eq. (6).

Every alpha is negative (alpha_{i,i} = -sigma_i/sqrt(pi)), so Q is indefinite and the
combination problem is not convex as stated — see src/density/combine.py for the change of
variables that fixes it.


Checkability
------------
This module is why this paper was picked over the alternatives: the closed form can be
checked against two independent computations of the same quantity — Monte-Carlo sampling of
eq. (3), and numerical integration of the quantile form CRPS = 2 * integral of the pinball
loss over tau. `verify()` runs all three and is exercised by test/test_density_crps.py.
"""

import numpy as np
from scipy.special import ndtr

from src.quantile.qeval import crps_from_quantiles  # noqa: F401  (re-exported)


def _abs_normal(m, s):
    """E|X| for X ~ N(m, s^2).  Their eq. (6), vectorised."""
    s = np.maximum(s, 1e-12)
    return s * np.sqrt(2.0 / np.pi) * np.exp(-m ** 2 / (2.0 * s ** 2)) + m * (2.0 * ndtr(m / s) - 1.0)


def crps_terms(mu, sigma, y):
    """(alpha, beta) with shapes (N, N, T) and (N, T) — their eq. (8) and (9).

    Summing these over t gives the Q and c of the combination QP; `alpha` is the expensive
    object at N^2 T floats, which is why the caller reduces it immediately.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    y = np.asarray(y, dtype=float)

    beta = _abs_normal(mu - y[None, :], sigma)
    d = mu[:, None, :] - mu[None, :, :]
    s = np.sqrt(sigma[:, None, :] ** 2 + sigma[None, :, :] ** 2)
    return -0.5 * _abs_normal(d, s), beta


def crps(weights, mu, sigma, y):
    """(T,) CRPS per hour for the mixture with these weights.  Their eq. (7)."""
    w = np.asarray(weights, dtype=float)
    alpha, beta = crps_terms(mu, sigma, y)
    return np.einsum('i,j,ijt->t', w, w, alpha) + np.einsum('i,it->t', w, beta)


def qc_matrices(mu, sigma, y):
    """(Q, c) of the combination objective, summed over the calibration period.  Eq. (11)."""
    alpha, beta = crps_terms(mu, sigma, y)
    return alpha.sum(axis=2), beta.sum(axis=1)


# ---------------------------------------------------------------------------
# Independent computations of the same number, for verification
# ---------------------------------------------------------------------------

def crps_mc(weights, mu, sigma, y, n_samples=200_000, random_state=0):
    """CRPS by sampling eq. (3) directly. Slow, unbiased, and free of the closed form."""
    w = np.asarray(weights, dtype=float)
    rng = np.random.default_rng(random_state)
    out = np.empty(len(y))
    for t in range(len(y)):
        c1 = rng.choice(len(w), size=n_samples, p=w / w.sum())
        c2 = rng.choice(len(w), size=n_samples, p=w / w.sum())
        a = rng.normal(mu[c1, t], sigma[c1, t])
        b = rng.normal(mu[c2, t], sigma[c2, t])
        out[t] = np.abs(a - y[t]).mean() - 0.5 * np.abs(a - b).mean()
    return out



def verify(seed=0, n_models=4, n_hours=3, tol_mc=0.02, tol_q=0.005, verbose=True):
    """Agreement of the three routes on a random mixture. Returns the worst relative gaps."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(15000, 800, (n_models, n_hours))
    sigma = rng.uniform(200, 900, (n_models, n_hours))
    w = rng.dirichlet(np.ones(n_models))
    y = mu.mean(axis=0) + rng.normal(0, 500, n_hours)

    closed = crps(w, mu, sigma, y)
    mc = crps_mc(w, mu, sigma, y, random_state=seed)
    lv = np.concatenate([[1e-4], np.arange(1, 1000) / 1000.0, [1 - 1e-4]])
    from src.density.gaussfit import mixture_quantiles
    quant = crps_from_quantiles(lv, mixture_quantiles(w, mu, sigma, lv, grid=8192), y)

    g_mc = float(np.max(np.abs(closed - mc) / closed))
    g_q = float(np.max(np.abs(closed - quant) / closed))
    if verbose:
        print(f"  closed form   {np.round(closed, 2)}")
        print(f"  Monte Carlo   {np.round(mc, 2)}   max rel gap {g_mc:.2e} (tol {tol_mc})")
        print(f"  quantile int. {np.round(quant, 2)}   max rel gap {g_q:.2e} (tol {tol_q})")
    return g_mc, g_q
