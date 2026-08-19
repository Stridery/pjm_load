"""Temperature scenario generation — the three practical methods of Xie & Hong (2018), III.

    fixed-date    take year y's temperature profile, assign it date-by-date to the target
                  year.  k scenarios from k years of history.
    shifted-date  the same, but also slid +/- 1..n days.  (2n+1)k scenarios.  Preserves the
                  autocorrelation of the series (it IS a real series, just relabelled),
                  which is the paper's stated reason to prefer it over resampling.
    bootstrap     cut every year into m-day blocks, then draw each block with replacement
                  from any of the k years.  l scenarios.  The paper finds it does not beat
                  fixed-date — adjacent blocks from different years leave an unphysical
                  jump at the seam (its Fig. 9) — so it is here for the comparison, not
                  because it is expected to win.

The scenario pool must predate the test period, exactly as the paper keeps its 1976-2005
temperature history strictly before the 2006-2011 test years. Using test-period weather to
build scenarios would be leakage; run_quantile picks the years accordingly.


Pool layout
-----------
`arr[year]` is a dense (13, 32, 24) array indexed [month, day, hour]. Dense rather than a
MultiIndex because the lookup runs over ~7.8k rows x a few hundred scenarios, where numpy
fancy indexing beats a pandas reindex by two orders of magnitude. The wasted cells (day 0,
day 31 of short months) cost 10 kB per year.

Three holes are structural, and all three are filled the same way:
    - a target 29 Feb with a non-leap source year  -> arr[2, 29] is all NaN
    - the fall-back day has two 01:00 rows         -> averaged when the pool is built
    - the spring-forward day has no 02:00 row      -> arr[m, d, 2] is NaN
The lookup returns values in TARGET-INDEX time order, so a plain interpolate over that
vector is interpolating between adjacent clock hours — physically the right thing, and it
also covers any incidental gap in the archive.
"""

import numpy as np
import pandas as pd


def build_pool(ept, temp, years):
    """{year: (13, 32, 24) array indexed [month, day, hour]} of observed temperature."""
    df = pd.DataFrame({'y': ept.year, 'm': ept.month, 'd': ept.day,
                       'h': ept.hour, 't': np.asarray(temp, dtype=float)})
    pool = {}
    for y in years:
        arr = np.full((13, 32, 24), np.nan)
        # mean(): the fall-back day publishes 01:00 twice, and they are two real hours.
        g = df[df['y'] == y].groupby(['m', 'd', 'h'])['t'].mean()
        if g.empty:
            raise ValueError(f"No temperature rows for scenario year {y}")
        idx = np.array(g.index.tolist())
        arr[idx[:, 0], idx[:, 1], idx[:, 2]] = g.values
        pool[y] = arr
    return pool


def series_for(ept, arr, shift=0):
    """(N,) temperature for the target index, read off one historical year at `shift` days.

    shift=0 is the fixed-date method; shift=+/-1..n adds the shifted-date scenarios. The
    hour is always the target row's own hour — only the DATE is remapped.
    """
    src = ept.normalize() + pd.Timedelta(days=int(shift))
    t = arr[src.month.values, src.day.values, ept.hour.values]
    s = pd.Series(t)
    if s.isna().any():
        s = s.interpolate(limit_direction='both')
    return s.values


def fixed_and_shifted(ept, pool, years, n_max):
    """(len(years), 2*n_max+1, N) temperature scenarios, indexed [year, shift, row].

    Every (year, shift) pair is generated once. A (k, n) configuration is then a SLICE of
    this cube — years[-k:] and shifts within +/-n — which is what makes the full k x n grid
    affordable: the expensive part (point-forecasting each scenario) never repeats.
    """
    shifts = np.arange(-n_max, n_max + 1)
    return np.array([[series_for(ept, pool[y], s) for s in shifts] for y in years]), shifts


def bootstrap(ept, pool, years, block_days=9, n_scenarios=100, random_state=42):
    """(n_scenarios, N) block-bootstrapped temperature series.

    Each m-day block of the target calendar is drawn, with replacement, from one of the k
    historical years. Blocks are cut on day-of-year so a scenario is assembled the same way
    the paper's Fig. 7 shows: block 1 from one year, block 2 from another, and so on.
    """
    base = np.array([series_for(ept, pool[y], 0) for y in years])       # (K, N)
    blk = (ept.dayofyear.values - 1) // int(block_days)
    n_blocks = int(blk.max()) + 1
    rng = np.random.default_rng(random_state)
    rows = np.arange(len(ept))

    out = np.empty((n_scenarios, len(ept)))
    for i in range(n_scenarios):
        pick = rng.integers(0, len(years), size=n_blocks)   # which year each block comes from
        out[i] = base[pick[blk], rows]
    return out
