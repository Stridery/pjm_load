"""
Standalone bottleneck-diagnostics for the load-forecasting pipeline.

All tasks live inside `run_diagnostics()` so a single run produces every
diagnostic together and results stay easy to compare across runs.

Usage:
    python util/diagnostics.py                 # uses PJM_DATASET (default 'bge')
    PJM_DATASET=bge python util/diagnostics.py
"""

import math
import os

import numpy as np
import pandas as pd

# --- Paths (mirror src/config.py without importing the training stack) ---
DATASET = os.environ.get("PJM_DATASET", "bge")
MATRIX_DIR = f"data/{DATASET}/matrix/"
LOOKBACK = 168
HORIZON = 0
X_OPT_PATH = os.path.join(MATRIX_DIR, f"X_opt_lb{LOOKBACK}_h{HORIZON}.csv")
Y_OPT_PATH = os.path.join(MATRIX_DIR, f"y_opt_lb{LOOKBACK}_h{HORIZON}.csv")


def _load_opt():
    """Load X_opt and y_opt, datetime-indexed and aligned on common timestamps."""
    X = pd.read_csv(X_OPT_PATH, parse_dates=["timestamp"]).set_index("timestamp")
    y = pd.read_csv(Y_OPT_PATH, parse_dates=["timestamp"]).set_index("timestamp")
    idx = X.index.intersection(y.index)
    return X.loc[idx].sort_index(), y.loc[idx].sort_index()


def _base_feature_groups(X):
    """Map each base feature -> its list of lookback columns ('<feat>_h<k>').

    Only keeps features that have the full set of LOOKBACK columns (the
    windowed weather/load features); scalar calendar columns are excluded.
    """
    groups = {}
    for col in X.columns:
        base = col.rsplit("_h", 1)[0] if "_h" in col else col
        groups.setdefault(base, []).append(col)
    return {b: cols for b, cols in groups.items() if len(cols) == LOOKBACK}


# ---------------------------------------------------------------------------
# Task 1: one combined figure relating every feature to daily-mean load.
#   - windowed features (168h mean)  -> scatter + linear fit
#   - discrete calendar features     -> boxplots
#   - month sin/cos cyclical encoding-> circle scatter colored by load
#   - hour sin/cos                   -> dead-feature check (no variance)
# ---------------------------------------------------------------------------
def _plot_feature_overview(X, y_target, target_label, out_path):
    """One combined figure relating every feature to a given target load series.

    `y_target` is a per-day load series (daily mean, or a single hour of the day);
    `target_label` is used in titles / axis labels.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_mean = y_target.rename("load")  # target load series (one value per day)
    groups = _base_feature_groups(X)

    # --- panel layout: scatters first, then calendar panels ---
    n_scatter = len(groups)
    n_extra = 6  # dayofweek, month, weekend, holiday boxplots + month-circle + hour-note
    total = n_scatter + n_extra
    ncols = 4
    nrows = math.ceil(total / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    ai = 0

    # --- windowed features: scatter of 168h mean vs daily-mean load ---
    for base, cols in groups.items():
        ax = axes[ai]; ai += 1
        feat_mean = X[cols].mean(axis=1)
        df = pd.concat([feat_mean, y_mean], axis=1).dropna()
        xv, yv = df.iloc[:, 0].values, df.iloc[:, 1].values
        r = np.corrcoef(xv, yv)[0, 1]
        ax.scatter(xv, yv, s=6, alpha=0.35, edgecolors="none")
        m, b = np.polyfit(xv, yv, 1)
        xs = np.array([xv.min(), xv.max()])
        ax.plot(xs, m * xs + b, color="crimson", lw=1.2)
        ax.set_title(f"{base}  (r={r:+.3f})", fontsize=9)
        ax.set_xlabel("168h mean", fontsize=8)
        ax.set_ylabel(target_label, fontsize=8)
        ax.tick_params(labelsize=7)

    # --- discrete calendar features: boxplots ---
    def _box_by(ax, key, title, labels=None):
        g = pd.concat([X[key].rename("k"), y_mean.rename("load")], axis=1).dropna()
        cats = sorted(g["k"].unique())
        data = [g.loc[g["k"] == c, "load"].values for c in cats]
        ax.boxplot(data, showfliers=False,
                   tick_labels=[labels[int(c)] if labels else f"{c:g}" for c in cats])
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(target_label, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.3)

    dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    _box_by(axes[ai], "today_dayofweek", "by day-of-week", labels=dow); ai += 1
    _box_by(axes[ai], "today_month", "by month"); ai += 1
    _box_by(axes[ai], "tmrw_is_weekend", "by tmrw_is_weekend", labels=["No", "Yes"]); ai += 1
    _box_by(axes[ai], "tmrw_is_holiday", "by tmrw_is_holiday", labels=["No", "Yes"]); ai += 1

    # --- month sin/cos: cyclic scatter on the unit circle, colored by load ---
    ax = axes[ai]; ai += 1
    if {"today_month_cos", "today_month_sin"}.issubset(X.columns):
        g = pd.concat([X["today_month_cos"], X["today_month_sin"], y_mean.rename("load")],
                      axis=1).dropna()
        sc = ax.scatter(g["today_month_cos"], g["today_month_sin"], c=g["load"],
                        cmap="viridis", s=10, alpha=0.7)
        fig.colorbar(sc, ax=ax, label="load")
        ax.set_title("month_sin/cos (color=load)", fontsize=9)
        ax.set_xlabel("month_cos", fontsize=8)
        ax.set_ylabel("month_sin", fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    # --- hour sin/cos: dead-feature note ---
    ax = axes[ai]; ai += 1
    ax.axis("off")
    hs, hc = X.get("today_hour_sin"), X.get("today_hour_cos")
    if hs is not None and hc is not None:
        ax.text(0.05, 0.95,
                "hour_sin / hour_cos\n\n"
                f"sin: const={hs.iloc[0]:.3f} std={hs.std():.4f}\n"
                f"cos: const={hc.iloc[0]:.3f} std={hc.std():.4f}\n\n"
                "DEAD FEATURES — every sample is a\n"
                "day stamped at hour 0, so these\n"
                "carry zero variance / no signal.",
                va="top", ha="left", fontsize=9, family="monospace",
                bbox=dict(boxstyle="round", facecolor="mistyrose", edgecolor="crimson"))

    for ax in axes[ai:]:
        ax.set_visible(False)

    fig.suptitle(f"Features vs {target_label}  (dataset={DATASET}, n={len(y_mean.dropna())})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved -> {os.path.abspath(out_path)}\n")


# ---------------------------------------------------------------------------
# Task 1: feature overview against daily-mean load.
# ---------------------------------------------------------------------------
def task1_feature_overview(X, y, out_path="feature_vs_load_overview.png"):
    print("=" * 70)
    print("Task 1: feature vs daily-mean load (single combined figure)")
    print(f"  dataset={DATASET}  X={X.shape}  y={y.shape}  n_samples={len(X)}")
    print("=" * 70)
    dead = [c for c in X.columns if c.startswith(("today_", "tmrw_")) and X[c].std() == 0]
    if dead:
        print(f"  DEAD calendar columns (no variance): {dead}")
    _plot_feature_overview(X, y.mean(axis=1), "daily-mean load", out_path)


# ---------------------------------------------------------------------------
# Task 2: same overview, but per target hour (not averaged) -> one figure each.
# ---------------------------------------------------------------------------
def task2_feature_overview_by_hour(X, y, hours=(0, 3, 9, 12, 15, 20)):
    print("=" * 70)
    print(f"Task 2: feature overview per target hour {list(hours)}")
    print("=" * 70)
    for h in hours:
        col = f"h{h}"
        if col not in y.columns:
            print(f"  skip {col}: not in y columns")
            continue
        _plot_feature_overview(X, y[col], f"load @ {h:02d}:00",
                               f"feature_vs_load_h{h:02d}.png")


def run_diagnostics():
    X, y = _load_opt()
    task1_feature_overview(X, y)
    task2_feature_overview_by_hour(X, y)


if __name__ == "__main__":
    run_diagnostics()
