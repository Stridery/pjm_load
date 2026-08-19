"""Ordinary least squares for the MLR benchmark models, with the scenario sweep in mind.

Fitting is one lstsq call (285 columns x ~41k rows solves in milliseconds). The only thing
worth designing here is PREDICTION, because the scenario sweep runs it hundreds of times
over the same rows:

    pred = X_cal @ beta_cal  +  X_temp(z) @ beta_temp
           ^^^^^^^^^^^^^^^^
           identical for every scenario — computed once, reused

`calendar_contrib` returns that first term so the caller can hoist it out of the loop; see
train_quantile.py. Coefficients are stored as one vector with `n_cal` marking the split, so
the full-matrix path and the block path use literally the same numbers.

lstsq (SVD, minimum-norm) rather than a normal-equation solve: the design carries 168-level
class dummies whose rank can drop if a (weekday, hour) cell is empty in a short training
window, and the minimum-norm solution keeps predictions well-defined when that happens.
"""

import joblib
import numpy as np

from src.quantile.design import Design


class MLR:
    """A fitted benchmark model: the frozen design spec plus its coefficients."""

    def __init__(self, design, beta, n_cal):
        self.design, self.beta, self.n_cal = design, np.asarray(beta, float), int(n_cal)

    # -- fitting -----------------------------------------------------------------

    @classmethod
    def fit(cls, model, ept, temp, y, train_mask, is_holiday=None):
        """Fit on `train_mask` rows; the design constants are learned from the same rows."""
        design = Design.fit(model, ept, temp, train_mask)
        X_cal = design.calendar_block(ept, is_holiday)
        X = np.hstack([X_cal, design.temp_block(ept, temp)])
        beta, *_ = np.linalg.lstsq(X[train_mask], np.asarray(y, float)[train_mask], rcond=None)
        return cls(design, beta, X_cal.shape[1])

    # -- prediction --------------------------------------------------------------

    def calendar_contrib(self, ept, is_holiday=None):
        """(N,) scenario-independent part of the prediction. Hoist this out of the sweep."""
        return self.design.calendar_block(ept, is_holiday) @ self.beta[:self.n_cal]

    def predict_with_temp(self, ept, temp, cal_contrib=None, is_holiday=None):
        """(N,) MW forecast under the temperature series `temp`."""
        base = self.calendar_contrib(ept, is_holiday) if cal_contrib is None else cal_contrib
        return base + self.design.temp_block(ept, temp) @ self.beta[self.n_cal:]

    # -- persistence -------------------------------------------------------------

    def save(self, path):
        joblib.dump({'model': self.design.model, 'mu': self.design.mu,
                     'sigma': self.design.sigma, 't0': self.design.t0,
                     'beta': self.beta, 'n_cal': self.n_cal}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        return cls(Design(d['model'], d['mu'], d['sigma'], d['t0']), d['beta'], d['n_cal'])
