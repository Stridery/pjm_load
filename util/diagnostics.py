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


def _feature_values(X):
    """One representative series per numeric variable: 168h mean for windowed
    features, raw value for non-constant scalar calendar columns."""
    variables = {}
    for base, cols in _base_feature_groups(X).items():
        variables[base] = X[cols].mean(axis=1)
    for c in X.columns:
        if c.startswith(("today_", "tmrw_")) and c != "is_target_valid" and X[c].std() > 0:
            variables[c] = X[c]
    return variables


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

    # --- panel layout: one scatter per windowed feature, + month sin/cos circle ---
    has_month_circle = {"today_month_cos", "today_month_sin"}.issubset(X.columns)
    total = len(groups) + (1 if has_month_circle else 0)
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

    # --- month sin/cos: cyclic scatter on the unit circle, colored by load ---
    if has_month_circle:
        ax = axes[ai]; ai += 1
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


# ---------------------------------------------------------------------------
# Task 3: which hours dominate the worst over- and under-predictions.
#   Reads a model's *_detailed_errors.csv and reports the hour distribution of
#   the top-10% most positive and most negative signed errors. Terminal only.
#   signed_error = pred - true  (positive = over-prediction, negative = under).
# ---------------------------------------------------------------------------
ERR_CSV = (f"results/{DATASET}/evaluation/transformer/"
           f"tail_test0.1_tail_val0.1_fds/TRANSFORMER_detailed_errors.csv")


def task3_error_hours(csv_path=ERR_CSV, frac=0.10):
    print("=" * 70)
    print(f"Task 3: hour distribution of worst {frac:.0%} over/under-predictions")
    print(f"  file: {csv_path}")
    print("=" * 70)

    if not os.path.exists(csv_path):
        print(f"  NOT FOUND: {csv_path}\n")
        return

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df["hour"] = df["datetime"].dt.hour
    n = len(df)
    k = max(1, int(round(n * frac)))
    base = df["hour"].value_counts(normalize=True)  # baseline hour share in test set

    pos = df.nlargest(k, "signed_error")   # most over-predicted (pred >> true)
    neg = df.nsmallest(k, "signed_error")  # most under-predicted (pred << true)

    def _report(name, sub):
        cnt = sub["hour"].value_counts().sort_values(ascending=False)
        share = cnt / len(sub)
        print(f"\n  {name}: n={len(sub)} of {n}  "
              f"(signed_error range [{sub['signed_error'].min():+.0f}, "
              f"{sub['signed_error'].max():+.0f}], mean {sub['signed_error'].mean():+.0f} MW)")
        print(f"    {'hour':>4} {'count':>6} {'grp%':>7} {'base%':>7} {'lift':>6}")
        for h in cnt.index:
            b = base.get(h, 0.0)
            lift = (share[h] / b) if b > 0 else float("nan")
            print(f"    {h:>4d} {cnt[h]:>6d} {share[h]:>6.1%} {b:>6.1%} {lift:>5.1f}x")

    _report("MOST OVER-PREDICTED  (signed_error > 0)", pos)
    _report("MOST UNDER-PREDICTED (signed_error < 0)", neg)
    print()


# ---------------------------------------------------------------------------
# Task 4: scatter of each variable SQUARED vs daily-mean load (combined figure).
#   Checks for quadratic relationships. For windowed features the variable is the
#   168h mean; scalar numeric calendar columns are used as-is. Each is squared.
# ---------------------------------------------------------------------------
def _plot_squared_scatter(X, y_target, target_label, out_path):
    """Combined figure: scatter of each variable^2 vs a given target load series."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_t = y_target.rename("load")
    variables = _feature_values(X)
    n = len(variables)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, ser) in zip(axes, variables.items()):
        df = pd.concat([ser.rename("v"), y_t], axis=1).dropna()
        xv = (df["v"].values) ** 2
        yv = df["load"].values
        r = np.corrcoef(xv, yv)[0, 1]
        ax.scatter(xv, yv, s=6, alpha=0.35, edgecolors="none")
        m, b = np.polyfit(xv, yv, 1)
        xs = np.array([xv.min(), xv.max()])
        ax.plot(xs, m * xs + b, color="crimson", lw=1.2)
        ax.set_title(f"{name}^2  (r={r:+.3f})", fontsize=9)
        ax.set_xlabel("variable^2", fontsize=8)
        ax.set_ylabel(target_label, fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Variable^2 vs {target_label}  (dataset={DATASET}, n={len(y_t.dropna())})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved -> {os.path.abspath(out_path)}\n")


def task4_squared_feature_scatter(X, y, out_path="feature_squared_vs_load.png"):
    print("=" * 70)
    print("Task 4: scatter(variable^2  vs  daily-mean load)")
    print("=" * 70)
    _plot_squared_scatter(X, y.mean(axis=1), "daily-mean load", out_path)


# ---------------------------------------------------------------------------
# Task 5: same variable^2 scatter, but per target hour (not averaged).
# ---------------------------------------------------------------------------
def task5_squared_by_hour(X, y, hours=(0, 3, 9, 12, 15, 20)):
    print("=" * 70)
    print(f"Task 5: variable^2 scatter per target hour {list(hours)}")
    print("=" * 70)
    for h in hours:
        col = f"h{h}"
        if col not in y.columns:
            print(f"  skip {col}: not in y columns")
            continue
        _plot_squared_scatter(X, y[col], f"load @ {h:02d}:00",
                              f"feature_squared_vs_load_h{h:02d}.png")


# ---------------------------------------------------------------------------
# Task 6: correlation heatmap of the feature variables (+ daily-mean load).
#   Variable = 168h mean for windowed features, raw value for scalar columns.
#   Correlation (not covariance) so color depth is comparable across scales.
# ---------------------------------------------------------------------------
def task6_feature_corr_heatmap(X, y, out_path="feature_correlation_heatmap.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("Task 6: correlation heatmap of feature variables (+ daily-mean load)")
    print("=" * 70)

    df = pd.DataFrame(_feature_values(X))
    df["load"] = y.mean(axis=1)
    corr = df.corr()
    labels = list(corr.columns)
    m = len(labels)

    fig, ax = plt.subplots(figsize=(0.9 * m + 2, 0.9 * m + 1))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")

    ax.set_xticks(range(m)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(m)); ax.set_yticklabels(labels, fontsize=8)

    for i in range(m):
        for j in range(m):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.55 else "black")

    ax.set_title(f"Feature correlation  (dataset={DATASET}, n={len(df)})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved -> {os.path.abspath(out_path)}\n")
    return corr


def run_diagnostics():
    X, y = _load_opt()
    task1_feature_overview(X, y)
    task2_feature_overview_by_hour(X, y)
    task3_error_hours()
    task4_squared_feature_scatter(X, y)
    task5_squared_by_hour(X, y)
    task6_feature_corr_heatmap(X, y)


if __name__ == "__main__":
    run_diagnostics()
