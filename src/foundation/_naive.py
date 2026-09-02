"""seasonal_naive — the harness's self-check adapter.

Returns the value one week earlier. In DIRECT mode that makes the whole pipeline reproduce the
same-hour-last-week baseline the residual point models lean on, whose score is already known
(dom 7.707%, bge 10.084% on the tail-0.16 test set). If this adapter does not land on those
numbers, the fault is in the context/alignment/evaluation wiring rather than in whatever
foundation model is being tested -- which is the entire reason it exists.

No dependencies, so it also lets the harness be exercised before any checkpoint is downloaded.
"""

import numpy as np

NAME = 'seasonal_naive'


def forecast(contexts, horizon, params, **_kw):
    """(N, L) -> (N, horizon): step k is the observation `period` hours before it."""
    period = int(params.get('period', 168))
    n, L = contexts.shape
    if L < period:
        raise ValueError(
            f"seasonal_naive needs context_hours >= period ({period}), got {L}. Step 1 looks "
            f"back {period} h from the cutoff, which a shorter context does not contain.")
    # Step k (1-based) is cleaned-frame row cutoff+k-1; the value `period` hours earlier sits at
    # context position L - period + (k-1).
    idx = L - period + np.arange(horizon)
    if idx.max() >= L:
        raise ValueError(f"horizon {horizon} exceeds one period ({period}) past the context end.")
    return contexts[:, idx]
