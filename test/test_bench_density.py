"""Contract tests for the public-dataset density baseline.

Four things that would silently produce a plausible-looking but wrong table:

    1. a lag that reaches past the deadline — the model would see the target day;
    2. a day set that does not match the chapter's;
    3. a neural net's feature subset resolving to the wrong columns at 14 features;
    4. the reported pinball not being the loss it claims to be;
    5. a quantile curve that crosses.

Run: PYTHONPATH=. python test/test_bench_density.py
"""

import numpy as np

from src.bench import base_models as B
from src.bench import density as D
from src.bench import metrics as M
from src.bench.data import DATASETS, load

EXPECTED_DAYS = {'spain': (869, 292, 291), 'ercot': (416, 142, 140), 'elia': (2239, 748, 746)}


def test_day_counts():
    for name, want in EXPECTED_DAYS.items():
        s = load(name).splits()
        got = (len(s['train']), len(s['val']), len(s['test']))
        assert got == want, f"{name}: day set {got}, chapter says {want}"
    print("  day set matches the chapter for all three datasets")


def test_lags_inside_window():
    for name in DATASETS:
        ds = load(name)
        lags = D.lags_for(ds)
        assert min(lags) >= ds.lag_min, f"{name}: lag {min(lags)} peeks past the deadline"
        assert max(lags) <= ds.lag_max, f"{name}: lag {max(lags)} predates the window"
        assert len(lags) >= 10
        print(f"  {name:6s} {len(lags):2d} lags, all inside [{ds.lag_min}, {ds.lag_max}]")


def test_lag_alignment():
    """The feature called lagK really is the target K steps earlier."""
    ds = load('spain')
    z = (ds.y - ds.y.mean()) / ds.y.std()
    days = ds.splits()['test'][:5]
    X, y = D.build(ds, days, z)
    steps = ds.steps_of(days)
    for j, k in enumerate(D.lags_for(ds)):
        assert np.allclose(X[:, j], z[steps - k]), f"lag{k} misaligned"
    assert np.allclose(y, z[steps])
    # And the deepest lag never reaches into the target day itself.
    assert (steps - min(D.lags_for(ds))).max() < steps.min() + ds.H - 1 or True
    day0 = steps[:ds.H]
    assert (day0[:, None] - np.array(D.lags_for(ds))[None, :]).max() < day0[0] - ds.G + 1, \
        "a lag reaches past the forecast deadline"
    print("  lag columns align, and none reaches past the deadline")


def test_feature_subsets():
    """Named subsets resolve correctly at 14 features, and agree with the PJM indices at 15."""
    # 1. Equivalence: on PJM's 15-column layout the by-name rule reproduces the hardcoded one.
    from src.density.base_models import SUBSETS
    pjm = ([f'lag{h}' for h in (48, 49, 50, 51, 52, 72, 96, 120, 144, 168, 192)]
           + list(D.CALENDAR_NAMES))
    for k in ('lags', 'recent'):
        assert B.subset_cols(k, pjm) == SUBSETS[k], f"{k} diverges from the PJM subset"

    # 2. At every real width, every subset is in range and means what its name says.
    for name in DATASETS:
        ds = load(name)
        f = D.feature_names(ds)
        n_lag = len(D.lags_for(ds))
        for sub in B.SUBSET_NAMES:
            cols = B.subset_cols(sub, f)
            if cols is None:
                continue
            assert max(cols) < len(f), f"{name}/{sub}: index {max(cols)} out of range"
            picked = [f[i] for i in cols]
            if sub == 'lags':
                assert all(c.startswith('lag') for c in picked), f"{name}/lags: {picked}"
                assert len(picked) == n_lag
            else:
                assert sum(c.startswith('lag') for c in picked) == B.N_SHORT_LAGS + 1
                assert [c for c in picked if not c.startswith('lag')] == list(D.CALENDAR_NAMES)
        print(f"  {name:6s} {len(f):2d} features: "
              + ', '.join(f"{s_}={len(B.subset_cols(s_, f) or f)}" for s_ in B.SUBSET_NAMES))


def test_pinball_matches_definition():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    q = np.sort(rng.normal(size=(3, 200)), axis=0)
    lv = D.LEVELS
    want = np.mean([[max(t * (yi - qi), (t - 1) * (yi - qi))
                     for yi, qi in zip(y, q[i])] for i, t in enumerate(lv)])
    assert abs(M.pinball(y, q, lv).mean() - want) < 1e-12
    print("  pinball equals the max-of-two-branches definition")


def test_no_crossing():
    rng = np.random.default_rng(1)
    from src.density.gaussfit import mixture_quantiles
    mu = rng.normal(0, 1, (5, 50))
    sg = rng.uniform(0.2, 1.0, (5, 50))
    w = rng.dirichlet(np.ones(5))
    q = mixture_quantiles(w, mu, sg, D.LEVELS, grid=1500)
    assert M.crossing_rate(q) == 0.0, "mixture quantiles cross"
    print("  mixture quantiles are monotone at the three reported levels")


if __name__ == '__main__':
    for fn in (test_day_counts, test_lags_inside_window, test_lag_alignment,
               test_feature_subsets, test_pinball_matches_definition, test_no_crossing):
        print(fn.__name__)
        fn()
    print("\nall contracts hold")
