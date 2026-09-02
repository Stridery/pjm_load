"""Load_Estimated -> metered scale correction, for DIRECT mode only.

A zero-shot model is handed Load_Estimated (the preliminary series, the only one available in
real time) and its output is therefore in Load_Estimated space, while the target is metered
Load. Scoring one against the other charges the model for an offset it has no way to know
about. Measured over the full record:

    dom   Est-as-Load  MAPE 0.82%   ratio 1.0079, CV 0.67%, flat across years and hours
                                    -> a near-constant LEVEL offset; one ratio removes it
    bge   Est-as-Load  MAPE 3.30%   ratio 0.9995 but CV 4.20%, 2.18% spread across hours
                                    -> a SHAPE mismatch (bge's proxy is MIDATL/8.8, a regional
                                       aggregate whose diurnal shape is not bge's). A perfect
                                       per-day level oracle still leaves 1.99%, so this cannot
                                       be rescaled away -- only reduced.

The ratio is fitted on the TRAIN split alone and frozen over the test period. That is the same
arrangement the 16 point models get (they see metered load in training and none at inference),
and it avoids the alternative's problem: a trailing window would have to read metered load from
days inside the test period, which is both leakage and unavailable in reality, since metered
verifies weeks late.

Residual mode does not use any of this. The baseline it adds back is itself built from
Load_Estimated, so the offset sits on both sides of the subtraction and cancels.
"""

import numpy as np

MODES = ('hourly_ratio', 'none')


def fit_hourly_ratio(load, est, ept_hours, train_mask):
    """24 medians of Est/Load, one per EPT hour, from the train rows only.

    Median rather than mean: the ratio has a long tail on low-load hours, and one bad metered
    revision should not move the correction for a whole hour of the year.
    """
    ratio = np.ones(24)
    load = np.asarray(load, dtype=float)
    est = np.asarray(est, dtype=float)
    good = train_mask & np.isfinite(load) & np.isfinite(est) & (load > 0)
    for h in range(24):
        sel = good & (ept_hours == h)
        if sel.any():
            ratio[h] = float(np.median(est[sel] / load[sel]))
    return ratio


def apply_hourly_ratio(pred_24, ratio):
    """(N, 24) in Load_Estimated space -> metered space. Column h is EPT hour h."""
    return np.asarray(pred_24, dtype=float) / ratio[None, :]


def describe(ratio):
    return (f"hourly Est/Load ratio: mean {ratio.mean():.4f}  "
            f"min {ratio.min():.4f}@h{int(ratio.argmin())}  "
            f"max {ratio.max():.4f}@h{int(ratio.argmax())}")
