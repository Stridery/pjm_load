"""Shared figure style and the generic probabilistic figures.

Style helpers, palette and the three method-agnostic figures (reliability, pinball by
quantile, monthly breakdown) used by every probabilistic baseline in this repo:
src.density.plots and src.qrnn.plots both import from here, so the
baselines are drawn by one set of rules rather than one per package.

Colour
------
Slots come from the repo's validated categorical order (generate_web.PALETTE_LIGHT) and are
assigned in that fixed order, never cycled. The first four were re-validated for this use:
worst adjacent CVD dE 9.1 (protan), worst normal-vision adjacent dE 22.9 — both above the
6.0 / 15.0 floors. Two of the four sit below 3:1 contrast on a light surface, which obliges
a secondary encoding rather than a dismissal: every line is DIRECT-LABELLED at its right
end, and each figure has a CSV counterpart written beside it as the table view.

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

from src.quantile.qeval import QUANTILES

# Fixed categorical order (see module docstring). Slot i is always the same entity.
PALETTE = ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7', '#e34948']
INK, INK2, MUTED, GRID = '#1a1a19', '#5c5c58', '#8a8a85', '#e3e3e0'
SURFACE = '#fcfcfb'



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


def plot_reliability(curves, actual, zone, path, colors=None, title=None):
    """Nominal vs empirical coverage; the diagonal is a perfectly calibrated forecast.

    Rarely reported in the source papers, which score with pinball/Winkler only, but it is
    the first thing a reader checks, and a miscalibrated interval shows up here at a glance
    rather than buried in a score.
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
