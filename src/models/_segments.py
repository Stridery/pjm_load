# src/models/_segments.py
"""Segmented point-forecast metrics: by load band, and on the most volatile hours.

A single MAPE over 7512 hours says which model is better on average and nothing about where.
Two cuts answer "where", and both are defined on the TRUE target so that every model is scored
on the identical set of points. A split derived from predictions would hand each model its own
bands and the comparison would not survive the first question about it.

BRS is deliberately absent. It is the slope of binned error against load across the whole
range; inside one band the range is compressed by construction, so the slope stops meaning
anything. It belongs to the unsegmented report.
"""

import os

import numpy as np
import pandas as pd

from src.config import (
    CLEANED_PATH, DATASET, LOAD_BANDS, LOAD_BAND_LABELS, TARGET_UNIT,
    VOL_SERIES_COL, VOL_TOP_FRACTION, VOL_WINDOW_DAYS,
)

_VOL_CACHE = {}


def band_edges(zone=None):
    """(lo, hi) in target units for this zone, or None when the zone has no bands configured."""
    return LOAD_BANDS.get(zone or DATASET)


def load_band(truth, zone=None):
    """Label every point low / mid / high. Returns None when the zone is not configured.

    Fixed MW thresholds rather than quantiles of this test set: a tercile split moves with the
    period and the zone, so the same label would not denote the same physical regime twice.
    """
    e = band_edges(zone)
    if e is None:
        return None
    lo, hi = e
    return pd.cut(pd.Series(np.asarray(truth).ravel()),
                  [-np.inf, lo, hi, np.inf], labels=list(LOAD_BAND_LABELS))


def relative_volatility(timestamps, zone=None, cleaned_path=None):
    """r = this hour's day-over-day move / its own trailing mean move, same clock hour.

    Read from the cleaned CSV rather than from a run's own file, because the trailing window
    reaches back before the test period starts and a detailed_errors.csv only holds test rows.

    Returns a float array aligned with `timestamps`; NaN where the window is incomplete.
    """
    zone = zone or DATASET
    key = (zone, cleaned_path or CLEANED_PATH, VOL_SERIES_COL, VOL_WINDOW_DAYS)
    if key not in _VOL_CACHE:
        df = pd.read_csv(cleaned_path or CLEANED_PATH, index_col=0, parse_dates=True)
        ept = pd.to_datetime(df['Datetime_EPT'])
        s = pd.Series(df[VOL_SERIES_COL].values, index=ept).sort_index()
        s = s[~s.index.duplicated(keep='first')]
        # Pivot to (date x clock hour) so "the same clock hour yesterday" is one row shift and
        # the trailing mean is a rolling window down a column. DST days that are short or long
        # simply leave a NaN in the affected cell rather than shifting every later lag by one.
        g = pd.DataFrame({'d': s.index.normalize(), 'h': s.index.hour, 'y': s.values})
        wide = g.pivot_table(index='d', columns='h', values='y', aggfunc='first').sort_index()
        move = wide.diff().abs()                                   # |y[d,h] - y[d-1,h]|
        # closed='left' so the baseline uses the previous VOL_WINDOW_DAYS days and NOT the day
        # being scored -- otherwise a turbulent day deflates its own ratio.
        base = move.rolling(VOL_WINDOW_DAYS, min_periods=VOL_WINDOW_DAYS, closed='left').mean()
        ratio = move / (base + 1e-6)
        _VOL_CACHE[key] = ratio.stack(future_stack=True)
    r = _VOL_CACHE[key]
    ts = pd.to_datetime(pd.Series(np.asarray(timestamps).ravel()))
    idx = pd.MultiIndex.from_arrays([ts.dt.normalize(), ts.dt.hour])
    return r.reindex(idx).to_numpy(dtype=float)


def _metrics(truth, pred):
    """MAPE / MAE / RMSE / ME on one subset. Empty subset -> NaNs, never an exception."""
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if truth.size == 0:
        return dict(n=0, MAPE=np.nan, MAE=np.nan, RMSE=np.nan, ME=np.nan)
    e = pred - truth
    return dict(n=int(truth.size),
                MAPE=float(np.mean(np.abs(e) / np.abs(truth)) * 100),
                MAE=float(np.mean(np.abs(e))),
                RMSE=float(np.sqrt(np.mean(e ** 2))),
                ME=float(np.mean(e)))


def compute(timestamps, truth, pred, zone=None, cleaned_path=None):
    """Both cuts for one run.

    Returns (rows, extra_cols):
        rows       list of dicts, one per segment, ready for a DataFrame
        extra_cols {'load_band': ..., 'vol_ratio': ..., 'vol_subset': ...} aligned with the
                   inputs, for appending to the detailed-errors file
    """
    truth = np.asarray(truth, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    zone = zone or DATASET
    rows, extra = [], {}

    rows.append(dict(cut='all', segment='all', **_metrics(truth, pred)))

    bands = load_band(truth, zone)
    if bands is not None:
        lo, hi = band_edges(zone)
        extra['load_band'] = bands.to_numpy()
        spans = {'low': f'<{lo:,.0f}', 'mid': f'{lo:,.0f}-{hi:,.0f}', 'high': f'>={hi:,.0f}'}
        for lab in LOAD_BAND_LABELS:
            m = (bands == lab).to_numpy()
            rows.append(dict(cut='load_band', segment=lab, span=spans[lab],
                             **_metrics(truth[m], pred[m])))

    r = relative_volatility(timestamps, zone, cleaned_path)
    extra['vol_ratio'] = r
    ok = np.isfinite(r)
    if ok.any():
        # Threshold on the points that HAVE a ratio, so an incomplete leading window shrinks
        # the eligible pool rather than silently biasing the cut.
        thr = float(np.quantile(r[ok], 1.0 - VOL_TOP_FRACTION))
        top = ok & (r >= thr)
        extra['vol_subset'] = np.where(~ok, 'na', np.where(top, 'volatile', 'calm'))
        rows.append(dict(cut='volatility', segment=f'top{int(VOL_TOP_FRACTION * 100)}',
                         span=f'r>={thr:.3f}', **_metrics(truth[top], pred[top])))
        rest = ok & ~top
        rows.append(dict(cut='volatility', segment='rest',
                         span=f'r<{thr:.3f}', **_metrics(truth[rest], pred[rest])))
    else:
        extra['vol_subset'] = np.full(len(r), 'na')
    return rows, extra


def to_text(rows, unit=None):
    """The block appended to a run's metrics.txt."""
    U = unit or TARGET_UNIT
    df = pd.DataFrame(rows)
    out = []
    for cut, title in (('load_band', 'by load band (fixed thresholds)'),
                       ('volatility', f'by relative volatility '
                                      f'(vs trailing {VOL_WINDOW_DAYS} d, same clock hour)')):
        d = df[df.cut == cut]
        if d.empty:
            continue
        out.append(f"\n--- {title} ---")
        out.append(f"{'segment':<10}{'range':>16}{'n':>7}{'MAPE %':>9}"
                   f"{'MAE ' + U:>12}{'RMSE ' + U:>12}{'ME ' + U:>12}")
        for _, r in d.iterrows():
            out.append(f"{r.segment:<10}{str(r.get('span', '')):>16}{int(r.n):>7}"
                       f"{r.MAPE:>9.3f}{r.MAE:>12.2f}{r.RMSE:>12.2f}{r.ME:>+12.2f}")
    return "\n".join(out) + "\n"


def write(rows, result_dir, model_name):
    """Machine-readable copy beside the run's other artefacts, for the scoreboard to collect."""
    path = os.path.join(result_dir, f'{model_name}_segments.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
