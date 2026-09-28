"""Evaluation and data plumbing shared by the probabilistic baselines.

Everything here is INFRASTRUCTURE. The package no longer holds a baseline of its own:

    qeval.py    pinball, Winkler, coverage, CRPS from a quantile function
    frame.py    the cleaned-CSV reader and the day-boundary split
    plots.py    palette, style helpers and the generic probabilistic figures
    _pool.py    process-pool helpers, with BLAS pinned inside each worker

src/density and src/qrnn import from these rather than duplicating them, so that every
baseline reads the same input, is cut at the same hour, is scored by the same functions, and
is drawn by the same rules. A table whose rows came from three implementations of the pinball
loss is not a comparison.

The package keeps the name `quantile` because that is what it scores.

Deliberately isolated from src/models/. That package has a fixed train/predict/evaluate
contract bound to the per-DAY tree/sequence matrices, and its registry is what
src/serving/predict.available_models() scans; a baseline dropped in there would be picked up
by the daily CI forecast job and blow it up.


Historical note
---------------
Two baselines were removed from this package.

The temperature-scenario baseline of Xie & Hong (2018), driven by train_quantile.py with a
thirty-year ERA5 archive fetched by fetch_scenario_weather.py, went first: the scenario pool
made it the one baseline that could not be run against an arbitrary load dataset.

Quantile Regression Averaging on sister forecasts — Liu, Nowotarski, Hong & Weron (2017),
IEEE TSG 8(2):730-737 — followed, with qra.py, sisters.py, design.py, qra_plots.py and
train_qra.py. Its sister family is built from a TEMPERATURE recency effect, and the public
datasets this framework is now pointed at carry no temperature at all; driving the family
from lagged load instead makes all eight sisters share one signal, which collapses the
disagreement the method converts into quantiles. That is a change to the mechanism, not to
the features, so the result would not have been the published method.

What remains is two baselines, each of which survives that substitution intact:

    src/density/    Li, Wang & Zhang (2020)      combining DENSITIES, weights by CRPS
    src/qrnn/       Zhang, Quan & Srinivasan (2019)   one network -> every quantile at once
"""
