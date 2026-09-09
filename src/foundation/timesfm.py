"""TimesFM 3.0 adapter — the only file that knows TimesFM exists.

Replaces the 2.5 adapter rather than sitting beside it. 2.5 is univariate: its only route to
exogenous information was XReg, a ridge fitted on the model's residuals, which is the same
two-stage shape as the TimesFM+LightGBM experiment that measured 4.10% against a 3.43%
baseline. 3.0 takes covariates natively through variate attention, so the weather that
explains ~80% of the daily residual finally reaches the model instead of being bolted on
afterwards. Keeping both would mean maintaining a path we have already measured and rejected.

What this model is handed
-------------------------
    context               metered load, 144 h, days D-8..D-3
    horizon               72 steps; the target day is steps 49-72
    past_future (216)     forecast temperature + 8 calendar channels
    past_only   (144)     preliminary load, the RT price channels (PJM_LMP), and the ten
                          weather variables the forecast archive cannot carry forward

The six-day window and the 72-step horizon both come from metered being published a day later
than everything else; see src/foundation/_covariates.py for the derivation and the leak
argument.

Licence
-------
The 3.0 CHECKPOINT is under the TimesFM Non-Commercial License v1.0 — unlike 2.5, which is
Apache 2.0. The pip package is Apache either way; it is the weights that are restricted. Fine
for research, not for anything shipped.
"""

import numpy as np

NAME = 'timesfm'
_CACHE = {}


def _resolve_device(want):
    import torch
    if want in (None, 'auto'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return want


def _load(params):
    """Load the checkpoint once. Unlike 2.5 there is no compile step and no shape to pin, so
    one instance serves every context length and horizon this pipeline asks for."""
    try:
        import torch
        from timesfm3 import TimesFM3Forecaster
    except ImportError as e:
        raise ImportError(
            "TimesFM 3.0 is not installed. Install it into this venv:\n"
            "    .venv/bin/pip install 'timesfm[torch]'\n"
            "Use --model zero or --model seasonal_naive to exercise the harness without it."
        ) from e

    ckpt = params['checkpoint']
    device = _resolve_device(params.get('device'))
    key = (ckpt, device)
    if key not in _CACHE:
        print(f"  loading {ckpt}  (device {device})")
        torch.set_float32_matmul_precision('high')
        _CACHE.clear()
        _CACHE[key] = TimesFM3Forecaster.from_pretrained(ckpt, device=device)
    return _CACHE[key]


def _as_array(out, n, horizon):
    """predict_batch returns ForecastOutput objects; normalise to (N, horizon) floats."""
    if hasattr(out, 'shape'):
        arr = np.asarray(out, dtype=float)
    else:
        arr = np.stack([np.asarray(getattr(o, 'forecast', o), dtype=float).ravel() for o in out])
    if arr.shape != (n, horizon):
        raise RuntimeError(
            f"TimesFM 3.0 returned {arr.shape}, expected {(n, horizon)}. The installed "
            f"package's predict_batch contract differs from the one this adapter targets; "
            f"only this file needs updating.")
    return arr


def forecast(contexts, horizon, params, past_only=None, past_future=None):
    """(N, L) -> (N, horizon) point forecast, in the same units as the context.

    past_only / past_future are lists of (channels, L) and (channels, L+horizon) arrays, or
    None for a univariate run. They are passed through untouched: the alignment they carry is
    the whole point, and this layer is not the place to second-guess it.
    """
    n = len(contexts)
    model = _load(params)
    out = model.predict_batch(
        contexts=[np.asarray(row, dtype=np.float32) for row in contexts],
        horizon=horizon,
        past_only_covariates=past_only,
        past_future_covariates=past_future,
        return_quantiles=False,
        # Load is strictly positive, so clamping helps in DIRECT mode. In RESIDUAL mode the
        # series is a signed deviation that is negative about half the time, and clamping it
        # would delete every downward correction the model makes — the same trap 2.5's
        # infer_is_positive had. _bench sets series_is_positive per mode; the adapter's own
        # config default only applies when nothing said otherwise.
        make_positive=bool(params.get('series_is_positive',
                                      params.get('make_positive', True))),
    )
    return _as_array(out, n, horizon)
