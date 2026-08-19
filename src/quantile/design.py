"""Design matrices for the three MLR benchmark models of Xie & Hong (2018), Section II-B.

Three models with deliberately different predictive power, so that a conclusion holding
across all three is not an artefact of one specification:

    tcube     b0 + Trend + poly3(T)
    vanilla   b0 + Trend + Day x Hour + Month + Month x poly3(T) + Hour x poly3(T)
    hong2014  vanilla + Holiday x Hour + recency (T_{t-1..t-3}, mean of the previous 24 h)

Row granularity is ONE HOUR — not the per-day samples the rest of this repo builds. That is
why nothing here touches src/feature_engine: the cleaned CSV already IS the design frame,
and the matrix layer's per-day flattening would have to be undone to use it.


Three deliberate departures from the paper
------------------------------------------
GSP.        The paper's T-cube and Vanilla models carry a Gross State Product index as the
            macroeconomic trend. No such series exists for DOM/BGE, so the linear `Trend`
            term (years since the start of the record) carries the growth instead. It
            matters for DOM, whose load is growing fast on data-centre demand.

T scaling.  Polynomials are taken of the STANDARDIZED temperature z = (T - mu)/sigma, with
            mu/sigma fit on the training rows only. {1, z, z^2, z^3} spans exactly the same
            column space as {1, T, T^2, T^3}, so the fitted function is identical — but T^3
            reaches ~1e6 in Fahrenheit and the raw version is badly conditioned.

Hong-2014.  The paper states its variable list "may change from year to year depending on
            the validation dataset that is used for the model selection process", so there
            is no canonical specification to reproduce. What is implemented here is one
            fixed, reasonable instantiation of the three effects the paper names (recency,
            weekend, holiday) — not a claim of exact reproduction. The weekend effect is
            already carried by the Day x Hour block, so only recency and holiday are added.


Why the block split
-------------------
A scenario sweep re-predicts the SAME rows under hundreds of temperature series. Only the
temperature-interaction columns change; the calendar columns do not. So the design is
exposed as two blocks — `calendar_block` is built once per run, `temp_block` once per
scenario — and MLR.predict_with_temp reuses the pre-multiplied calendar contribution.
Rebuilding the full 285-column matrix per scenario would be ~3x the work for nothing.
"""

import numpy as np
import pandas as pd

MODELS = ('tcube', 'vanilla', 'hong2014')
POLY_ORDERS = (1, 2, 3)


def _indicator(codes, n_levels, drop_first=True):
    """(N,) integer codes in [0, n_levels) -> (N, n_levels - drop_first) indicator matrix."""
    codes = np.asarray(codes, dtype=int)
    out = np.zeros((len(codes), n_levels))
    out[np.arange(len(codes)), codes] = 1.0
    return out[:, 1:] if drop_first else out


def _f_block(month_d, hour_d, z):
    """f(T) of Xie & Hong eq. (2) / Liu et al. eq. (2): poly3 of z, interacted with Month
    and with Hour. 3 x (12 + 23) = 105 columns.

    Month gets FULL dummies and Hour drops one: both full blocks would each sum to z^p
    across their columns, making the two exactly collinear. With Month full, the z^p main
    effect the paper writes separately is already inside this span.
    """
    z = np.asarray(z, dtype=float)
    out = []
    for p in POLY_ORDERS:
        zp = (z ** p).reshape(-1, 1)
        out.append(month_d * zp)
        out.append(hour_d * zp)
    return out


class Design:
    """Column spec for one MLR benchmark model.

    `mu`/`sigma` standardize temperature and `t0` anchors the trend; all three are fit on
    the training rows and then frozen, so a scenario (or a test row) is transformed with
    exactly the constants the coefficients were fit against.

    Two switches serve the QRA sister models of Liu et al. (2017):
      trend   - their eq. (1) has NO trend or macroeconomic term at all; it is a short-term
                model re-estimated daily on a rolling 2-3 year window, which tracks growth
                by moving rather than by extrapolating. Pass trend=False to match it.
      recency - (n_lag, n_dma): add f(T_{t-1..t-n_lag}) and f(T~_{t,1..n_dma}) to the
                temperature block, where f is the SAME interacted form as the base
                temperature (their eq. 4), and T~_{t,d} is the mean of the d-th preceding
                24 h block (their eq. 3). (0, 0) is the plain Vanilla model.
    """

    def __init__(self, model, mu, sigma, t0, trend=True, recency=(0, 0)):
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {model!r}")
        if model == 'hong2014' and recency != (0, 0):
            raise ValueError("hong2014 already carries its own fixed recency block")
        self.model, self.mu, self.sigma, self.t0 = model, mu, sigma, t0
        self.trend, self.recency = bool(trend), tuple(int(v) for v in recency)
        self._ck = None

    @classmethod
    def fit(cls, model, ept, temp, train_mask, trend=True, recency=(0, 0)):
        """Learn the temperature standardization and the trend origin from training rows."""
        t = np.asarray(temp, dtype=float)[train_mask]
        return cls(model, float(t.mean()), float(t.std()), ept[0].normalize(),
                   trend=trend, recency=recency)

    # -- blocks ------------------------------------------------------------------

    def _dummy_cache(self, ept):
        """Month/hour indicator matrices — scenario-independent, so built once per index."""
        key = (len(ept), ept[0], ept[-1])
        if self._ck != key:
            self._ck = key
            self._month_d = _indicator(ept.month.values - 1, 12, drop_first=False)
            self._hour_d = _indicator(ept.hour.values, 24, drop_first=True)
        return self._month_d, self._hour_d

    def calendar_block(self, ept, is_holiday=None):
        """Everything that does not depend on temperature. Built once per run."""
        n = len(ept)
        cols = [np.ones((n, 1))]
        if self.trend:
            cols.append(((ept.normalize() - self.t0).days.values / 365.25).reshape(-1, 1))
        if self.model == 'tcube':
            return np.hstack(cols)

        # Day x Hour (168 levels) and Month (12) as class variables. The weekend effect is
        # already inside Day x Hour — it is a partition of the same 168 cells.
        cols.append(_indicator(ept.dayofweek.values * 24 + ept.hour.values, 168))
        cols.append(_indicator(ept.month.values - 1, 12))

        if self.model == 'hong2014':
            if is_holiday is None:
                raise ValueError("hong2014 needs is_holiday (the cleaned CSV's own column)")
            hol = np.asarray(is_holiday, dtype=float).reshape(-1, 1)
            # Holiday x Hour, not a bare holiday flag: a holiday flattens the morning ramp
            # rather than shifting the whole day by a constant.
            cols.append(_indicator(ept.hour.values, 24, drop_first=False) * hol)
        return np.hstack(cols)

    def temp_block(self, ept, temp):
        """Everything that depends on temperature. Rebuilt once per scenario.

        For hong2014 the recency terms are lagged off THIS series, so a scenario's lags come
        from the scenario itself and never leak observed weather into the test period. That
        is also why the caller builds on the full record and slices afterwards — the lags at
        the start of the test period then reach into real preceding scenario hours.
        """
        z = (np.asarray(temp, dtype=float) - self.mu) / self.sigma
        if self.model == 'tcube':
            return np.stack([z ** p for p in POLY_ORDERS], axis=1)

        month_d, hour_d = self._dummy_cache(ept)
        cols = list(_f_block(month_d, hour_d, z))

        # Recency (Liu et al. eq. 4): the same interacted form f(.) applied to lagged
        # temperature and to preceding daily means. Lags are taken off THIS series, so a
        # sister forecast never mixes in a different temperature path.
        n_lag, n_dma = self.recency
        if n_lag or n_dma:
            s = pd.Series(z)
            for j in range(1, n_lag + 1):
                cols += _f_block(month_d, hour_d, s.shift(j).bfill().values)
            for d in range(1, n_dma + 1):
                # T~_{t,d} = mean of T over the d-th preceding 24 h block (their eq. 3)
                dma = s.shift(24 * (d - 1) + 1).rolling(24, min_periods=1).mean()
                cols += _f_block(month_d, hour_d, dma.bfill().values)

        if self.model == 'hong2014':
            s = pd.Series(z)
            for lag in (1, 2, 3):
                zl = s.shift(lag).bfill().values.reshape(-1, 1)
                cols.append(np.hstack([zl ** p for p in POLY_ORDERS]))
            zbar = s.shift(1).rolling(24, min_periods=1).mean().bfill().values.reshape(-1, 1)
            cols.append(np.hstack([zbar ** p for p in POLY_ORDERS]))
            cols.append(hour_d * zbar)
        return np.hstack(cols)

    def build(self, ept, temp, is_holiday=None):
        """Full design matrix — the two blocks concatenated, calendar first."""
        return np.hstack([self.calendar_block(ept, is_holiday), self.temp_block(ept, temp)])
