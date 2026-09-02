"""zero — the exact self-check adapter. Forecasts nothing at all.

In RESIDUAL mode this is not a joke model: the prediction becomes 0 + baseline, so the run must
reproduce the same-hour-last-week baseline's score to the last digit (dom 7.707%, bge 10.084%
on the tail-0.16 test set, computed independently from the matrices). If it does, then cutoff
location, step-to-EPT-hour mapping, DST normalisation, the baseline add-back, the train/test
split and the evaluation wiring are all correct, and any later disagreement belongs to the
model under test rather than to the harness.

In DIRECT mode it predicts zero load, which is only useful as a smoke test that nothing
crashes.
"""

import numpy as np

NAME = 'zero'


def forecast(contexts, horizon, params, **_kw):
    return np.zeros((contexts.shape[0], horizon), dtype=float)
