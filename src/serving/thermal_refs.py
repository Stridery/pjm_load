"""
Persist (and load) the train-fitted thermal references so serving never needs the 6-year
history to reproduce them.

build_thermal_references returns five things; only two are TRAIN-FITTED CONSTANTS:
    threshold  : the heat-wave temperature (P75 of training-summer daily means), one float.
    climatology: day-of-year mean temperature over the training rows, array[1..366].
The other three (heat_streak, day_index, doy) are frame-local — a function of whatever data
is in hand — so serving recomputes them from its 40-day window using the persisted threshold.
That is safe because the longest heat streak ever observed (~21 days) fits inside the window.

`temp_anomaly_vs_climatology` is the single most important static feature for dom, so getting
this climatology right at serve time is not optional: recomputing it from 40 days instead of
6 years would quietly shift it and every model would drift.
"""

import os

import joblib
import numpy as np
import pandas as pd

from src.config import CLEANED_PATH, MATRIX_DIR, TRANSFORMER_FEATURE_CONFIG
from src.thermal_features import build_thermal_references

# One file per zone, next to the scalers. All models (tree + sequence) share it: every
# feature config uses the same test_frac, so build_thermal_references sees the same split_idx
# and fits one climatology.
FILENAME = 'thermal_refs.pkl'


def _refs_path(matrix_dir):
    return os.path.join(matrix_dir, FILENAME)


def extract_and_save(cleaned_path=None, matrix_dir=None):
    """Compute the thermal references from the TRAINING cleaned CSV and save them.

    Run once per zone after training (no retrain needed — it only reads the cleaned data that
    training already produced, and reproduces the same split_idx the matrix builders used).
    """
    cleaned_path = cleaned_path or CLEANED_PATH
    matrix_dir   = matrix_dir or MATRIX_DIR
    os.makedirs(matrix_dir, exist_ok=True)

    df = pd.read_csv(cleaned_path, index_col=0, parse_dates=True).sort_index()
    ept_dates = pd.to_datetime(df['Datetime_EPT']).dt.date.values
    unique_days = np.unique(ept_dates)
    # Same split_idx the builders use (all feature configs share test_frac), so the fitted
    # threshold + climatology are byte-identical to what training baked into the matrices.
    split_idx = int(len(df) * (1 - TRANSFORMER_FEATURE_CONFIG['test_frac']))

    thr, _streak, clim, _day_index, _doy = build_thermal_references(
        df, ept_dates, unique_days, split_idx)

    path = _refs_path(matrix_dir)
    joblib.dump({'threshold': float(thr), 'climatology': np.asarray(clim, dtype='float64')}, path)
    print(f"Thermal refs saved → {path}  (threshold={thr:.2f} F, climatology[{len(clim)}])")
    return path


def load(matrix_dir=None):
    """Return {'threshold': float, 'climatology': array[1..366]} — raises if not extracted."""
    path = _refs_path(matrix_dir or MATRIX_DIR)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing. Run src.serving.thermal_refs.extract_and_save() once per zone "
            f"after training — serving cannot rebuild the 6-year climatology from a short window."
        )
    return joblib.load(path)
