"""Finding the weights.  Li et al. Section III-C, plus their two comparison schemes.

The CRPS of the mixture is quadratic in the weights, so the combination is

    min_w  w' Q w + c' w      s.t.  1' w = 1,  w >= 0                      (their eq. 12)

with Q and c summed over the weight-optimisation period. A weight that comes back at zero
drops its model, so selection and weighting happen in one solve — which is the paper's stated
reason for posing it this way.

Q is INDEFINITE: every alpha is negative (alpha_ii = -sigma_i/sqrt(pi)), so every diagonal
entry is too. The paper handles that by eliminating the equality constraint (their eq. 21-25)
and reports that the resulting Q_new came out positive definite in all of their cases, while
noting it is not proven. That claim is worth carrying over as a CHECK rather than an
assumption, so `solve` computes Q_new's smallest eigenvalue and reports whether the solution
it returns is certified global or merely a local optimum of a non-convex program.

The solver is scipy's SLSQP rather than the paper's CVXOPT (not a dependency here). For
fifteen variables the difference is milliseconds, and SLSQP does not require convexity — it
just cannot certify it, which is what the eigenvalue check is for.
"""

import numpy as np
from scipy.optimize import minimize


def _convexity_certificate(Q):
    """Smallest eigenvalue of the paper's Q_new (their eq. 24), after eliminating 1'w = 1.

    Positive => the reduced problem is a convex QP and any local optimum is global.
    """
    Qp, t, Qnn = Q[:-1, :-1], Q[:-1, -1], Q[-1, -1]
    one = np.ones(len(Qp))
    Q_new = Qp - np.outer(one, t) - np.outer(t, one) + Qnn * np.outer(one, one)
    return float(np.linalg.eigvalsh(0.5 * (Q_new + Q_new.T))[0])


def solve_crps(Q, c, n_starts=8, random_state=0):
    """(weights, info) minimising w'Qw + c'w on the simplex.  Their eq. (12).

    Multi-start because the program is not convex as stated; if the certificate below is
    positive the extra starts are redundant, and if it is not they are the only defence.
    """
    n = len(c)
    Q = 0.5 * (np.asarray(Q, float) + np.asarray(Q, float).T)
    c = np.asarray(c, float)

    def f(w):
        return float(w @ Q @ w + c @ w)

    def g(w):
        return 2.0 * Q @ w + c

    rng = np.random.default_rng(random_state)
    starts = [np.full(n, 1.0 / n)] + [rng.dirichlet(np.ones(n)) for _ in range(n_starts - 1)]
    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0, 'jac': lambda w: np.ones(n)}]
    best = None
    for w0 in starts:
        r = minimize(f, w0, jac=g, bounds=[(0.0, 1.0)] * n, constraints=cons,
                     method='SLSQP', options={'maxiter': 500, 'ftol': 1e-12})
        if r.success and (best is None or r.fun < best.fun):
            best = r
    if best is None:
        raise RuntimeError("CRPS weight optimisation failed from every start")

    w = np.clip(best.x, 0.0, None)
    w /= w.sum()
    lam = _convexity_certificate(Q)
    return w, {'objective': float(best.fun), 'min_eig_Qnew': lam,
               'certified_global': bool(lam > 0), 'n_selected': int((w > 1e-3).sum())}


def solve_mape(mu, y, n_starts=4, random_state=0):
    """Weights minimising the MAPE of the mixture MEAN — the paper's competing scheme.

    Their Section VI-B solves it as an LP; SLSQP on the same objective is equivalent at this
    size. The point of carrying it is the paper's own observation: a MAPE-guided combination
    knows nothing about the shape of the density, so it can win on MAPE while losing on CRPS.
    """
    mu = np.asarray(mu, float)
    y = np.asarray(y, float)
    n = mu.shape[0]

    def f(w):
        return float(np.mean(np.abs(y - w @ mu) / y))

    rng = np.random.default_rng(random_state)
    starts = [np.full(n, 1.0 / n)] + [rng.dirichlet(np.ones(n)) for _ in range(n_starts - 1)]
    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
    best = None
    for w0 in starts:
        r = minimize(f, w0, bounds=[(0.0, 1.0)] * n, constraints=cons,
                     method='SLSQP', options={'maxiter': 500, 'ftol': 1e-10})
        if r.success and (best is None or r.fun < best.fun):
            best = r
    w = np.clip(best.x, 0.0, None)
    return w / w.sum()


def simple_average(n):
    """Equal weights — the paper's third scheme, and the one it shows can be dragged down by
    a single poor member."""
    return np.full(n, 1.0 / n)
