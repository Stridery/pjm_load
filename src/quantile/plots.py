"""Figures for the probabilistic baselines — the paper's Fig. 2/5/8/10 on our data.

    PJM_DATASET=dom python -m src.quantile.plots

Reads only what train_quantile.py already wrote under results/{zone}/quantile/, so it can
be re-run and restyled without re-fitting anything.

Colour
------
Slots come from the repo's validated categorical order (generate_web.PALETTE_LIGHT) and are
assigned in that fixed order, never cycled. The first four were re-validated for this use:
worst adjacent CVD dE 9.1 (protan), worst normal-vision adjacent dE 22.9 — both above the
6.0 / 15.0 floors. Two of the four sit below 3:1 contrast on a light surface, which obliges
a secondary encoding rather than a dismissal: every line is DIRECT-LABELLED at its right
end, and grid_shifted.csv is the table view of the same numbers.

Label text stays in ink; a colour is carried by the mark beside it, never by the type.
These are static report figures, so they are rendered for a light surface only — the dark
theme belongs to the web artifact (docs/index.html), which does not consume them.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.quantile.qeval import QUANTILES, empirical_percentiles

# Fixed categorical order (see module docstring). Slot i is always the same entity.
PALETTE = ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7', '#e34948']
INK, INK2, MUTED, GRID = '#1a1a19', '#5c5c58', '#8a8a85', '#e3e3e0'
SURFACE = '#fcfcfb'

MODEL_LABEL = {'tcube': 'T-cube', 'vanilla': 'Vanilla', 'hong2014': 'Hong-2014'}


def _style(ax, title, xlabel, ylabel, title_pad=10):
    """Recessive grid and axes; the data is the only thing with weight."""
    ax.set_title(title, color=INK, fontsize=11, loc='left', pad=title_pad)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color(INK2)


def _stack_labels(ax, entries, min_gap_frac=0.05):
    """Right-end direct labels, pushed apart so converging lines stay readable.

    The colour-carrying dot stays on the true value; only the TEXT is nudged, so nothing is
    misreported. Call after the data is drawn — it needs the settled y-limits. Text never
    wears the series hue: the dot beside it carries identity.
    """
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    order = sorted(range(len(entries)), key=lambda i: entries[i][1])
    ys = [entries[i][1] for i in order]
    for j in range(1, len(ys)):
        ys[j] = max(ys[j], ys[j - 1] + gap)
    for j, i in enumerate(order):
        x, y, text, color = entries[i]
        ax.plot([x], [y], 'o', color=color, markersize=5, zorder=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f'  {text}', (x, ys[j]), color=INK, fontsize=8.5,
                    va='center', ha='left', annotation_clip=False)


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    print(f"  -> {path}")


def plot_qs_vs_n(grid, floor, model, zone, path):
    """Quantile score against the number of shifted days — the paper's Fig. 5 / Fig. 10.

    The shape is the whole finding: shifting a short weather history helps a lot, then stops
    helping, then hurts. One line per k so the "more history needs less shifting" reading is
    on the same axes.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(floor, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate('perfect-weather floor (temperature known exactly)',
                (grid.index[-1], floor), color=INK2, fontsize=8,
                va='bottom', ha='right', xytext=(0, 4), textcoords='offset points')

    # At most six k on one chart. The repo's palette validates cleanly to six slots; slots
    # 7-8 (#e87ba4 against its neighbours) sit at normal-vision dE 12.9-13.2, under the 15.0
    # floor, and that floor is not waivable by direct labels. The full k grid stays in
    # grid_shifted.csv, which is this figure's table view.
    cols = list(grid.columns)
    if len(cols) > len(PALETTE):
        idx = sorted({0, *(round(j * (len(cols) - 1) / (len(PALETTE) - 1))
                           for j in range(len(PALETTE)))})
        cols = [cols[i] for i in idx][:len(PALETTE)]

    labels = []
    for i, k in enumerate(cols):
        v = grid[k].values
        ax.plot(grid.index, v, color=PALETTE[i], linewidth=2, zorder=3)
        nb = int(np.argmin(v))                       # the optimum is the point of the figure
        ax.plot([grid.index[nb]], [v[nb]], 'o', color=PALETTE[i], markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        labels.append((grid.index[-1], v[-1], f'k={k}', PALETTE[i]))

    ax.set_xlim(grid.index[0] - 0.5, grid.index[-1] + max(3.5, 0.09 * len(grid)))
    _style(ax, f'{MODEL_LABEL.get(model, model)} — quantile score vs days shifted   ({zone.upper()})',
           'n  (temperature series shifted ±n days)', 'Quantile score  (MW, lower is better)')
    _stack_labels(ax, labels)
    ax.legend(handles=[plt.Line2D([], [], color=PALETTE[i], lw=2, label=f'{k} year(s) of history')
                       for i, k in enumerate(cols)],
              frameon=False, fontsize=8.5, labelcolor=INK, loc='upper right')
    _save(fig, path)


def plot_fixed_date_by_k(fixed, floor, model, zone, path):
    """Quantile score against years of temperature history — the paper's Fig. 2 / Fig. 3.

    This is the curve the paper's headline finding lives on ("the improvement ... is
    negligible when the history is beyond 15 years or so"), and it is the one figure that a
    four-year pool cannot draw. The 95%-of-total-improvement mark is annotated because that
    is the quantity the finding is actually about.
    """
    ks = list(fixed.index)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(floor, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate('perfect-weather floor (temperature known exactly)', (ks[-1], floor),
                color=INK2, fontsize=8, va='bottom', ha='right',
                xytext=(0, 4), textcoords='offset points')

    thr = fixed.iloc[0] - 0.95 * (fixed.iloc[0] - fixed.min())
    k95 = min(k for k in ks if fixed[k] <= thr)
    ax.axvline(k95, color=MUTED, linewidth=1.2, linestyle=(0, (2, 3)), zorder=1)
    ax.annotate(f'95% of the total improvement\nreached at k={k95}', (k95, fixed.iloc[0]),
                color=INK2, fontsize=8, va='top', ha='left',
                xytext=(6, -2), textcoords='offset points')

    ax.plot(ks, fixed.values, color=PALETTE[0], linewidth=2, zorder=3)
    ax.plot(ks, fixed.values, 'o', color=PALETTE[0], markersize=4.5, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=1.2)
    _style(ax, f'{MODEL_LABEL.get(model, model)} — fixed-date: quantile score vs years of '
               f'history   ({zone.upper()})',
           'k  (years of temperature history)', 'Quantile score  (MW, lower is better)')
    _save(fig, path)


def plot_qs_vs_k(grid, boot, floor, model, zone, path):
    """Quantile score against the length of weather history, one line per method (Fig. 2/3).

    Three methods on one axis because the comparison IS the question: does more history beat
    shifting, and does bootstrapping beat either.
    """
    ks = list(grid.columns)
    series = [('Fixed-date', grid.loc[0].values, PALETTE[0]),
              ('Shifted-date (best n)', grid.min(axis=0).values, PALETTE[1])]
    if boot is not None:
        series.append(('Bootstrap', boot.reindex(ks).values, PALETTE[2]))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(floor, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate('perfect-weather floor', (ks[-1], floor), color=INK2, fontsize=8,
                va='bottom', ha='right', xytext=(0, 4), textcoords='offset points')

    for label, vals, color in series:
        ax.plot(ks, vals, color=color, linewidth=2, marker='o', markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)

    ax.set_xticks(ks)
    ax.set_xlim(ks[0] - 0.15, ks[-1] + 2.4)
    _style(ax, f'{MODEL_LABEL.get(model, model)} — quantile score vs years of history   ({zone.upper()})',
           'k  (years of temperature history)', 'Quantile score  (MW, lower is better)')
    _stack_labels(ax, [(ks[-1], v[-1], l, c) for l, v, c in series])
    ax.legend(handles=[plt.Line2D([], [], color=c, lw=2, marker='o', label=l)
                       for l, _, c in series],
              frameon=False, fontsize=8.5, labelcolor=INK, loc='upper right')
    _save(fig, path)


def plot_bootstrap_scatter(per_model, zone, path):
    """Fixed-date vs bootstrap, the paper's Fig. 8.

    Its claim is geometric: every point lands on or above y = x, i.e. bootstrapping never
    improves the quantile score. A scatter against the diagonal is the form that shows it.
    """
    # Pair on the k both series actually have — `fixed` is swept over every k while the
    # bootstrap is only run on the grid's columns.
    pts = []
    for m, d in per_model.items():
        b = d.get('bootstrap')
        if b is None:
            continue
        common = d['fixed'].index.intersection(b.dropna().index)
        if len(common):
            pts.append((m, d['fixed'].reindex(common), b.reindex(common)))
    if not pts:
        return
    allv = np.concatenate([np.r_[f.values, b.values] for _, f, b in pts])
    lo, hi = allv.min(), allv.max()
    pad = 0.06 * (hi - lo)

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, linewidth=1.2,
            linestyle=(0, (4, 3)), zorder=1)
    ax.annotate('y = x  (no improvement)', (hi, hi), color=INK2, fontsize=8,
                va='top', ha='right', xytext=(-6, -6), textcoords='offset points')

    for i, (m, f, b) in enumerate(pts):
        ax.plot(f.values, b.values, 'o', color=PALETTE[i], markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
                label=MODEL_LABEL.get(m, m))
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    _style(ax, f'Bootstrap does not beat fixed-date   ({zone.upper()})',
           'Quantile score — fixed-date', 'Quantile score — bootstrap')
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc='upper left')
    _save(fig, path)


def plot_peak_week_fan(preds, actual, when, shifts, k, n, model, zone, path):
    """The predictive distribution over the peak week, against what happened (Fig. 6).

    Nested central intervals in ONE hue, light to dark, rather than 99 separate dashed lines:
    the job here is magnitude of spread, which is a sequential encoding. The median is the
    only line, and the actual load is the only marker, so neither can be confused for a band.
    """
    mid = int(np.where(shifts == 0)[0][0])
    sub = preds[-k:, mid - n:mid + n + 1, :].reshape(-1, len(actual))

    peak = int(np.argmax(actual))                      # centre the window on the peak hour
    lo = max(0, min(peak - 84, len(actual) - 168))
    sl = slice(lo, lo + 168)
    q = empirical_percentiles(sub[:, sl])
    t, a = when[sl], actual[sl]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # One hue, light -> dark as the interval narrows. Never a rainbow across intervals.
    for (a_q, b_q), shade in [((1, 99), '#d6e5f7'), ((5, 95), '#a8c8ee'),
                              ((25, 75), '#79ace4')]:
        ax.fill_between(t, q[np.where(QUANTILES == a_q)[0][0]],
                        q[np.where(QUANTILES == b_q)[0][0]],
                        color=shade, linewidth=0, zorder=2,
                        label=f'{b_q - a_q}% interval')
    ax.plot(t, q[49], color='#1a5aa8', linewidth=2, zorder=3, label='Median forecast')
    ax.plot(t, a, 'o', color=INK, markersize=3.2, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=0.5, label='Actual load')

    _style(ax, f'{MODEL_LABEL.get(model, model)} — peak week, {len(sub)} temperature scenarios '
               f'(k={k}, n={n})   ({zone.upper()})', '', 'Load  (MW)', title_pad=32)
    # Above the axes, not floating over the bands — at this density any in-plot legend lands
    # on the peaks, which are exactly what the reader came for.
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, ncol=5,
              loc='lower left', bbox_to_anchor=(0, 1.005), borderaxespad=0)
    fig.autofmt_xdate(rotation=0, ha='center')
    _save(fig, path)


# ---------------------------------------------------------------------------
# Generic probabilistic-forecast figures — shared by BOTH baselines
# ---------------------------------------------------------------------------
# These answer questions that belong to any set of predictive quantiles, not to one paper,
# so both train_quantile.py's and train_qra.py's results are drawn by the same code and can
# be laid side by side without a caption explaining why they look different.

def plot_reliability(curves, actual, zone, path, colors=None, title=None):
    """Nominal vs empirical coverage; the diagonal is a perfectly calibrated forecast.

    Neither paper reports coverage — both score with pinball/Winkler only — but it is the
    first thing a reader checks, and it is where the scenario method's weakness is visible
    at a glance rather than buried in a score.
    """
    from src.quantile.qeval import coverage_report
    levels = (50, 80, 90, 95, 98)
    colors = colors or PALETTE[:len(curves)]

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([45, 100], [45, 100], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate('perfectly calibrated', (68, 68), color=INK2, fontsize=8, rotation=38,
                rotation_mode='anchor', va='bottom', ha='center',
                xytext=(0, 4), textcoords='offset points')

    labels = []
    for (name, q), c in zip(curves.items(), colors):
        emp = [r['empirical_pct'] for r in coverage_report(actual, q, levels)]
        ax.plot(levels, emp, color=c, linewidth=2, marker='o', markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        labels.append((levels[-1], emp[-1], name, c))

    ax.set_xticks(levels)
    ax.set_xlim(46, 116)
    ax.set_ylim(min(15, ax.get_ylim()[0]), 108)
    _style(ax, title or f'Interval calibration   ({zone.upper()})',
           'Nominal coverage  (%)', 'Empirical coverage  (%)')
    ax.legend(handles=[plt.Line2D([], [], color=c, lw=2, marker='o', label=n)
                       for (n, _), c in zip(curves.items(), colors)],
              frameon=False, fontsize=8.5, labelcolor=INK, loc='upper left')
    _stack_labels(ax, labels, min_gap_frac=0.055)
    _save(fig, path)


def plot_pinball_by_quantile(csv_path, zone, path, colors=None):
    """Where in the distribution each configuration loses. Two forecasts with the same mean
    pinball can bleed in completely different places — a narrow one in the tails, a biased
    one on a single side."""
    d = pd.read_csv(csv_path)
    cols = [c for c in d.columns if c != 'quantile']
    colors = colors or PALETTE[:len(cols)]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    labels = []
    for c, col in zip(cols, colors):
        ax.plot(d['quantile'], d[c], color=col, linewidth=2, zorder=3)
        labels.append((d['quantile'].iloc[-1], d[c].iloc[-1], c, col))
    ax.set_xlim(0, 118)
    _style(ax, f'Pinball loss by quantile level   ({zone.upper()})',
           'Quantile  (%)', 'Mean pinball loss  (MW)')
    ax.legend(handles=[plt.Line2D([], [], color=col, lw=2, label=c)
                       for c, col in zip(cols, colors)],
              frameon=False, fontsize=8.5, labelcolor=INK, loc='upper center')
    _stack_labels(ax, labels)
    _save(fig, path)


def plot_monthly(csv_path, zone, path, colors=None):
    """Pinball and 90% coverage month by month. A year average hides a forecast that is
    calibrated in the shoulder seasons and broken at the summer peak."""
    d = pd.read_csv(csv_path)
    methods = [c.rsplit('_', 1)[0] for c in d.columns if c.endswith('_pinball')]
    colors = colors or PALETTE[:len(methods)]
    x = np.arange(len(d))

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, suffix, ylab, ref in [(axes[0], '_pinball', 'Pinball loss  (MW)', None),
                                  (axes[1], '_cover90', '90% interval coverage  (%)', 90)]:
        ax.set_facecolor(SURFACE)
        if ref is not None:
            ax.axhline(ref, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        for m, c in zip(methods, colors):
            ax.plot(x, d[m + suffix], color=c, linewidth=2, marker='o', markersize=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        _style(ax, '', '', ylab)
    axes[1].annotate('nominal 90%', (x[-1], 90), color=INK2, fontsize=8, va='bottom',
                     ha='right', xytext=(0, 4), textcoords='offset points')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(d['month'], rotation=45, ha='right')
    axes[0].set_title(f'Month by month   ({zone.upper()})', color=INK, fontsize=11,
                      loc='left', pad=10)
    axes[0].legend(handles=[plt.Line2D([], [], color=c, lw=2, marker='o', label=m)
                            for m, c in zip(methods, colors)],
                   frameon=False, fontsize=8.5, labelcolor=INK, loc='upper left')
    fig.tight_layout()
    _save(fig, path)


def main():
    import src.config as cfg
    root = cfg.QUANTILE_CONFIG['result_dir']
    zone = cfg.DATASET
    per_model = {}

    for model, on in cfg.QUANTILE_CONFIG['models'].items():
        if not on:
            continue
        d = os.path.join(root, model)
        gp = os.path.join(d, 'grid_shifted.csv')
        if not os.path.exists(gp):
            continue
        print(f"\n{model}:")
        grid = pd.read_csv(gp, index_col=0)
        grid.columns = [int(c) for c in grid.columns]
        ref = pd.read_csv(os.path.join(d, 'point_reference.csv'), header=None, index_col=0)[1]
        floor = float(ref['perfect_weather_QS_degenerate'])
        bp = os.path.join(d, 'bootstrap_vs_fixed.csv')
        boot = pd.read_csv(bp, index_col=0)['bootstrap'] if os.path.exists(bp) else None
        if boot is not None:
            boot.index = [int(i) for i in boot.index]

        fp = os.path.join(d, 'fixed_date_by_k.csv')
        if os.path.exists(fp):
            fx = pd.read_csv(fp, index_col=0).iloc[:, 0]
            fx.index = [int(i) for i in fx.index]
            plot_fixed_date_by_k(fx, floor, model, zone, os.path.join(d, 'fixed_date_by_k.png'))
        plot_qs_vs_n(grid, floor, model, zone, os.path.join(d, 'qs_vs_n.png'))
        plot_qs_vs_k(grid, boot, floor, model, zone, os.path.join(d, 'qs_vs_k.png'))

        preds = np.load(os.path.join(d, 'scenario_preds.npy'))
        actual = np.load(os.path.join(d, 'test_actual.npy'))
        when = pd.to_datetime(np.load(os.path.join(d, 'test_datetime.npy')))
        shifts = np.load(os.path.join(d, 'scenario_shifts.npy'))
        nbest, kbest = grid.stack().idxmin()
        plot_peak_week_fan(preds, actual, when, shifts, int(kbest), int(nbest), model, zone,
                           os.path.join(d, 'fan_peak_week.png'))

        zp = os.path.join(d, 'test_quantiles.npz')
        if os.path.exists(zp):
            z = np.load(zp, allow_pickle=True)
            names = list(z['labels'])
            cur = {n: z[f'cfg{i}'] for i, n in enumerate(names)}
            plot_reliability(cur, z['actual'], zone, os.path.join(d, 'reliability.png'))
        for csv, fn, out in [('pinball_by_quantile.csv', plot_pinball_by_quantile,
                              'pinball_by_quantile.png'),
                             ('monthly.csv', plot_monthly, 'monthly.png')]:
            if os.path.exists(os.path.join(d, csv)):
                fn(os.path.join(d, csv), zone, os.path.join(d, out))

        per_model[model] = {'fixed': grid.loc[0], 'bootstrap': boot}

    if per_model:
        print()
        plot_bootstrap_scatter(per_model, zone, os.path.join(root, 'bootstrap_vs_fixed.png'))


if __name__ == '__main__':
    main()
