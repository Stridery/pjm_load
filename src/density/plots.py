"""Figures for the density-combination baseline — Li et al. Table I/II/IV and calibration.

    PJM_DATASET=dom python -m src.density.plots

Reads only what scripts/train_density.py wrote under results/{zone}/density/, so it restyles without
refitting. Style helpers, palette and the generic probabilistic figures come from
src.quantile.plots: the three baselines are read side by side, so they are drawn by one set
of rules rather than three.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.quantile.plots import (INK, INK2, MUTED, PALETTE, SURFACE, _save, _stack_labels,
                                _style, plot_monthly, plot_pinball_by_quantile,
                                plot_reliability)

# Fixed scheme -> hue, shared by every figure here. The subject of the paper leads; the two
# competing weightings and the single-model benchmark follow in a stable order, so no scheme
# ever changes colour between charts.
SCHEME_HUE = {'CRPS-guided': 0, 'MAPE-guided': 1, 'Simple average': 2, 'Best individual': 3}
FAMILY_HUE = {'GPR': 0, 'NN': 1, 'XGB': 2}
BENCH = '#b9b9b4'
GRIDLINE = '#e3e3e0'


def _hue(name, table=SCHEME_HUE):
    for k, i in table.items():
        if name.startswith(k):
            return PALETTE[i]
    return MUTED


def plot_scoreboard(tbl, zone, path):
    """CRPS, MAPE and Winkler(90%) as small multiples — three units never share one axis.

    CRPS is the paper's headline and the only one of the three that scores the whole density;
    MAPE is the point metric it is contrasted against, and Winkler adds the width/coverage
    view the other two baselines are also reported on.
    """
    metrics = [('CRPS', 'CRPS  (MW)'), ('MAPE', 'MAPE  (%)'),
               ('winkler90', 'Winkler score, 90% PI')]
    t = tbl.iloc[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 0.55 * len(t) + 2.2))
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, title) in zip(axes, metrics):
        ax.set_facecolor(SURFACE)
        colors = [PALETTE[0] if m.startswith('CRPS-guided') else BENCH for m in t.model]
        ax.barh(np.arange(len(t)), t[key], height=0.62, color=colors, zorder=3)
        for i, v in enumerate(t[key]):
            ax.annotate(f' {v:,.2f}' if key == 'MAPE' else f' {v:,.1f}', (v, i),
                        va='center', ha='left', color=INK, fontsize=8.5, annotation_clip=False)
        ax.set_yticks(np.arange(len(t)))
        ax.set_yticklabels([m.split(' (')[0] for m in t.model])
        ax.set_xlim(0, t[key].max() * 1.3)
        _style(ax, title, '', '')
        ax.grid(axis='y', visible=False)
    fig.suptitle(f'Test period — weighting schemes   ({zone.upper()})',
                 color=INK, fontsize=11, x=0.005, ha='left', y=1.0)
    fig.tight_layout()
    _save(fig, path)


def plot_weights(w, zone, path):
    """What each scheme actually selected — their Table III vs Table IV.

    A dot plot, not grouped bars: fifteen models times two schemes as bars is a thicket, and
    the question being asked is "did the two schemes pick the same members", which is a
    comparison of positions on one shared 0-1 scale.
    """
    w = w.iloc[::-1].reset_index(drop=True)
    ypos = np.arange(len(w))
    fig, ax = plt.subplots(figsize=(6.6, 0.34 * len(w) + 1.9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i in ypos:                                   # connector, so the pair reads as one row
        ax.plot([w.crps_weight[i], w.mape_weight[i]], [i, i], color=GRIDLINE, linewidth=1.4,
                zorder=2, solid_capstyle='round')
    ax.plot(w.mape_weight, ypos, 'o', color=PALETTE[1], markersize=7, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.plot(w.crps_weight, ypos, 'o', color=PALETTE[0], markersize=7, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=1.5)

    ax.set_yticks(ypos)
    ax.set_yticklabels(w.model)
    ax.set_xlim(-0.02, max(0.02 + float(max(w.crps_weight.max(), w.mape_weight.max())), 0.3))
    _style(ax, f'Combination weights   ({zone.upper()})', 'Weight in the mixture', '',
           title_pad=30)
    ax.grid(axis='y', visible=False)
    # Above the axes: with fifteen rows any in-plot corner is occupied by a dot.
    ax.legend(handles=[plt.Line2D([], [], color=PALETTE[0], lw=0, marker='o', label='CRPS-guided'),
                       plt.Line2D([], [], color=PALETTE[1], lw=0, marker='o', label='MAPE-guided')],
              frameon=False, fontsize=8.5, labelcolor=INK, ncol=2,
              loc='lower left', bbox_to_anchor=(0, 1.005), borderaxespad=0)
    _save(fig, path)


def plot_base_models(t2, w, zone, path):
    """The fifteen base models on T2, and which of them the combination kept.

    Their Table I makes the point that a low individual CRPS does not earn a weight — a model
    is chosen for what it adds to the others, not for its own score — so accuracy and
    selection are shown on the same axes.
    """
    m = t2.merge(w[['model', 'crps_weight']], on='model')
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for fam in FAMILY_HUE:
        s = m[m.model.str.startswith(fam)]
        if s.empty:
            continue
        picked = s.crps_weight > 0.01
        ax.plot(s.CRPS[~picked], s.MAPE[~picked], 'o', color=_hue(fam, FAMILY_HUE),
                markersize=7, alpha=0.35, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        ax.plot(s.CRPS[picked], s.MAPE[picked], 'o', color=_hue(fam, FAMILY_HUE),
                markersize=12, markeredgecolor=INK, markeredgewidth=1.6, zorder=4)
    for r in m.itertuples():
        # Offset in POINTS, not a leading space: a space is scaled with the data range, so on
        # a narrow axis the marker swallows the first character of the label.
        ax.annotate(r.model, (r.CRPS, r.MAPE), color=INK2, fontsize=7.5, va='center',
                    ha='left', xytext=(11, 0), textcoords='offset points')
    ax.margins(x=0.14, y=0.08)
    _style(ax, f'Base models on T2 — large marker = kept by the combination   ({zone.upper()})',
           'CRPS  (MW)', 'MAPE  (%)')
    ax.legend(handles=[plt.Line2D([], [], color=_hue(f, FAMILY_HUE), lw=0, marker='o', label=f)
                       for f in FAMILY_HUE],
              frameon=False, fontsize=8.5, labelcolor=INK, loc='upper left')
    _save(fig, path)


def plot_peak_week(q, actual, when, zone, path, label='CRPS-guided combination'):
    """The mixture's intervals over the peak week — same construction as the other two
    baselines' figures, so the three can be laid side by side."""
    from src.quantile.qeval import QUANTILES
    peak = int(np.argmax(actual))
    lo = max(0, min(peak - 84, len(actual) - 168))
    sl = slice(lo, lo + 168)
    qs, a, t = q[:, sl], actual[sl], when[sl]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for (a_q, b_q), shade in [((1, 99), '#d6e5f7'), ((5, 95), '#a8c8ee'), ((25, 75), '#79ace4')]:
        ax.fill_between(t, qs[np.where(QUANTILES == a_q)[0][0]],
                        qs[np.where(QUANTILES == b_q)[0][0]],
                        color=shade, linewidth=0, zorder=2, label=f'{b_q - a_q}% interval')
    ax.plot(t, qs[49], color='#1a5aa8', linewidth=2, zorder=3, label='Median forecast')
    ax.plot(t, a, 'o', color=INK, markersize=3.2, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=0.5, label='Actual load')
    _style(ax, f'{label} — peak week   ({zone.upper()})', '', 'Load  (MW)', title_pad=32)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, ncol=5,
              loc='lower left', bbox_to_anchor=(0, 1.005), borderaxespad=0)
    fig.autofmt_xdate(rotation=0, ha='center')
    _save(fig, path)



def main():
    import src.config as cfg
    d, zone = cfg.DENSITY_CONFIG['result_dir'], cfg.DATASET
    if not os.path.exists(os.path.join(d, 'test_scores.csv')):
        raise SystemExit(f"No results in {d} — run scripts/train_density.py first.")

    tbl = pd.read_csv(os.path.join(d, 'test_scores.csv'))
    plot_scoreboard(tbl, zone, os.path.join(d, 'scoreboard.png'))

    w = pd.read_csv(os.path.join(d, 'weights.csv'))
    plot_weights(w, zone, os.path.join(d, 'weights.png'))
    plot_base_models(pd.read_csv(os.path.join(d, 'base_models_T2.csv')), w, zone,
                     os.path.join(d, 'base_models.png'))

    z = np.load(os.path.join(d, 'test_quantiles.npz'), allow_pickle=True)
    actual, when = z['actual'], pd.to_datetime(z['datetime'])
    curves = {n: z[f'cfg{i}'] for i, n in enumerate(list(z['labels']))}
    plot_reliability(curves, actual, zone, os.path.join(d, 'reliability.png'),
                     colors=[_hue(n) for n in curves])
    plot_peak_week(curves['CRPS-guided'], actual, when, zone,
                   os.path.join(d, 'fan_peak_week.png'))

    # These two come from a later version of train_density; skip rather than crash so an
    # older results directory still re-plots everything it does have.
    p_pbq = os.path.join(d, 'pinball_by_quantile.csv')
    if os.path.exists(p_pbq):
        cols = [c for c in pd.read_csv(p_pbq, nrows=0).columns if c != 'quantile']
        plot_pinball_by_quantile(p_pbq, zone, os.path.join(d, 'pinball_by_quantile.png'),
                                 colors=[_hue(c) for c in cols])
    p_mth = os.path.join(d, 'monthly.csv')
    if os.path.exists(p_mth):
        cols = [c.rsplit('_', 1)[0] for c in pd.read_csv(p_mth, nrows=0).columns
                if c.endswith('_pinball')]
        plot_monthly(p_mth, zone, os.path.join(d, 'monthly.png'),
                     colors=[_hue(c) for c in cols])
    else:
        print("  (pinball_by_quantile / monthly not present — re-run scripts/train_density.py)")


if __name__ == '__main__':
    main()
