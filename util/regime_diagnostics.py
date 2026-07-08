"""Regime-design diagnostics for the load data.

Each task is a standalone function. Enable the ones you want in main() — nothing
runs automatically except what you leave uncommented there.

Stats are computed on the REAL (MW) target-day load, restricted to the training
split (same tail 0.16/0.19 split as the MoE, so test days are never touched).

Run:
  python util/regime_diagnostics.py
  PJM_DATASET=dom python util/regime_diagnostics.py
"""

import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as cfg
from src.feature_engine import build_timeseries_matrix, _split_indices
from src.models.moe_transformer import season_indices

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)

_OUT_DIR = os.path.join('results', cfg.DATASET, 'diagnostics')


def _save(df, name):
    os.makedirs(_OUT_DIR, exist_ok=True)
    path = os.path.join(_OUT_DIR, name)
    df.to_csv(path)
    print(f"  saved -> {path}")


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def load_hourly(which='train', valid_only=True):
    """Hourly cleaned load (real MW) restricted to a split's target days.

    which:      'train' (default) | 'val' | 'trainpool' (train+val) | 'test' | 'all'
    valid_only: keep only metered (is_valid==1) rows, dropping imputed hours.

    Returns a DataFrame with columns ['Load', 'month', 'hour'].
    """
    _, _, _, ts = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)
    n = len(ts)
    fc = cfg.MOE_TRANSFORMER_FEATURE_CONFIG
    train_pool, test_idx = _split_indices(n, fc['split_strategy'], fc['test_frac'], fc['random_state'])
    rtr, rval = _split_indices(len(train_pool), fc['val_strategy'], fc['val_frac'], fc['random_state'])
    sel = {
        'train':     train_pool[rtr],
        'val':       train_pool[rval],
        'trainpool': train_pool,
        'test':      test_idx,
        'all':       np.arange(n),
    }[which]
    days = set(pd.to_datetime(ts[sel]).date)

    df = pd.read_csv(cfg.CLEANED_PATH, index_col=0, parse_dates=True)
    ept = pd.to_datetime(df['Datetime_EPT'])
    df = df.assign(_date=ept.dt.date, month=ept.dt.month, hour=ept.dt.hour)
    df = df[df['_date'].isin(days)]
    if valid_only and 'is_valid' in df.columns:
        df = df[df['is_valid'] == 1]
    print(f"[load_hourly] which={which} | days={len(days)} | hourly rows={len(df)} "
          f"| valid_only={valid_only}")
    return df[['Load', 'month', 'hour']].copy()


# ---------------------------------------------------------------------------
# Task 1 — monthly load mean / variance
# ---------------------------------------------------------------------------

def month_load_stats(df=None, save=True):
    """Per-month load mean and variance (and std, count) over the training set."""
    if df is None:
        df = load_hourly('train')
    out = df.groupby('month')['Load'].agg(
        mean='mean', var='var', std='std', count='count').round(1)
    print("\n=== Monthly load stats (MW) ===")
    print(out.to_string())
    if save:
        _save(out, 'month_load_stats.csv')
    return out


# ---------------------------------------------------------------------------
# Task 2 — month x hour-of-day load mean / variance
# ---------------------------------------------------------------------------

def month_hour_load_stats(df=None, save=True):
    """Per (month, hour-of-day) load mean and variance, pivoted to month x hour."""
    if df is None:
        df = load_hourly('train')
    stats = df.groupby(['month', 'hour'])['Load'].agg(mean='mean', var='var')
    mean_p = stats['mean'].unstack('hour').round(0)   # rows=month(1-12), cols=hour(0-23)
    var_p = stats['var'].unstack('hour').round(0)
    print("\n=== Load MEAN by month (rows) x hour (cols), MW ===")
    print(mean_p.to_string())
    print("\n=== Load VARIANCE by month (rows) x hour (cols), MW^2 ===")
    print(var_p.to_string())
    if save:
        _save(mean_p, 'month_hour_load_mean.csv')
        _save(var_p, 'month_hour_load_var.csv')
    return mean_p, var_p


# ---------------------------------------------------------------------------
# Task 3 — y vs lookback-window feature means (one big scatter grid)
# ---------------------------------------------------------------------------

def _feature_names(num_features):
    """Names for the 3D-matrix feature axis, in build_timeseries_matrix order."""
    from src.macro_features import MACRO_FEATURE_NAMES
    names = (['Load_Estimated'] + list(cfg.WEATHER_COLS) +
             ['tmrw_month_sin', 'tmrw_month_cos', 'tmrw_dow_sin', 'tmrw_dow_cos',
              'tmrw_is_weekend', 'tmrw_is_holiday'] + list(MACRO_FEATURE_NAMES))
    return names if len(names) == num_features else [f'feat_{i}' for i in range(num_features)]


def y_vs_lookback_feature_scatter(save=True):
    """One big figure: for EVERY feature, scatter (mean of that feature over the
    sample's 168h lookback window) vs (that sample's mean daily load, MW), across
    ALL matrix samples. One subplot per feature, points colored by season group,
    Pearson r in each title. Reveals which lookback averages track next-day load.
    """
    X, y, _, ts = build_timeseries_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)
    names = _feature_names(X.shape[2])
    is_forecast = np.array([nm.startswith('tmrw_') for nm in names])   # forecast-day standalone feats

    # Lookback features: mean over the 168h window. Forecast-day features are
    # broadcast constants (not a window) -> plot their value directly.
    feat_x = X.mean(axis=1)                                      # (N, F)
    feat_x[:, is_forecast] = X[:, 0, :][:, is_forecast]

    y_scaler = joblib.load(glob.glob(os.path.join(cfg.MATRIX_DIR, 'y_scaler_*.pkl'))[0])
    y_rep = y_scaler.inverse_transform(y.mean(axis=1).reshape(-1, 1)).ravel()   # (N,) MW

    season = season_indices(ts)
    palette = np.array(['#C44E52', '#DD8452', '#55A868', '#4C72B0'])            # per SEASON_ORDER
    pt_colors = palette[season % len(palette)]

    F = len(names)
    ncols = 5
    nrows = int(np.ceil(F / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, nm in enumerate(names):
        ax = axes[i]
        xf = feat_x[:, i]
        ax.scatter(xf, y_rep, c=pt_colors, s=5, alpha=0.35, linewidths=0, rasterized=True)
        std = xf.std()
        r = np.corrcoef(xf, y_rep)[0, 1] if std > 1e-9 else np.nan
        ax.set_title(f'{nm}  (r={r:+.2f})', fontsize=9)
        ax.set_xlabel('forecast-day value' if is_forecast[i] else 'lookback mean (scaled)', fontsize=7)
        ax.set_ylabel('mean load (MW)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)
    for j in range(F, len(axes)):
        axes[j].axis('off')

    handles = [plt.Line2D([], [], marker='o', ls='', color=palette[k % len(palette)], label=s)
               for k, s in enumerate(cfg.SEASON_ORDER)]
    fig.legend(handles=handles, loc='upper right', fontsize=9, title='season')
    fig.suptitle(f'{cfg.DATASET}: mean daily load vs features  '
                 f'(lookback mean for continuous / forecast-day value for calendar)  '
                 f'[N={len(y_rep)} samples]', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if save:
        os.makedirs(_OUT_DIR, exist_ok=True)
        path = os.path.join(_OUT_DIR, 'y_vs_lookback_feature_scatter.png')
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved -> {path}")
    return fig


# ---------------------------------------------------------------------------
# main — enable the tasks you want to run
# ---------------------------------------------------------------------------

def main():
    # df = load_hourly('train')          # training-set hourly load (real MW)
    # month_load_stats(df)               # Task 1
    # month_hour_load_stats(df)          # Task 2

    y_vs_lookback_feature_scatter()      # Task 3


if __name__ == '__main__':
    main()
