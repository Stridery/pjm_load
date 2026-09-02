"""Adapter registry. An adapter is any module exposing:

    NAME     : str
    forecast(contexts: np.ndarray (N, L), horizon: int, params: dict) -> np.ndarray (N, horizon)

Deliberately narrow. The adapter never sees the cleaned frame, the split, the baseline or the
evaluation -- it turns context windows into forecast steps and nothing else, so a new model is
one file and one line here.
"""

from src.foundation import _naive, _zero, timesfm

ADAPTERS = {
    _naive.NAME:   _naive,
    _zero.NAME:    _zero,
    timesfm.NAME:  timesfm,
}


def get(name):
    if name not in ADAPTERS:
        raise ValueError(f"Unknown foundation model {name!r}. Available: {sorted(ADAPTERS)}")
    return ADAPTERS[name]
