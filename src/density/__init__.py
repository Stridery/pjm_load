"""Probabilistic baseline 3 — Li, Wang & Zhang (2020), IEEE TSG 11(2):1679-1690.

Combining probability DENSITY forecasts: several base models each emit a Gaussian per hour,
and the combination is a Gaussian mixture whose weights minimise the CRPS. Because the CRPS
of a Gaussian mixture is available in closed form, that search is a small quadratic program
rather than a simulation.

Where this sits next to the other two baselines. Hong & Fan's tutorial (cited by all three
papers) splits probabilistic load forecasting into three sources: simulating predictors,
probabilistic models, and converting/combining point forecasts.

    src/quantile/scenarios.py   Xie & Hong 2018   simulating predictors
    src/quantile/qra.py         Liu et al. 2017   combining point forecasts into quantiles
    src/density/                Li et al. 2020    combining DENSITIES

Isolated from the point-model stack on the same terms as the other two: reads only the
cleaned hourly CSV, writes only under results/{zone}/density/, and is invisible to
src/models, the matrix cache and the serving path. Shared evaluation helpers are imported
from src.quantile (pinball/Winkler/coverage, the process pool, the figure style) rather than
duplicated — the package name is about which paper it implements, not which metric it uses.
"""
