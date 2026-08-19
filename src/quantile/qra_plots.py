"""Figures for the QRA baseline — Liu et al. (2017) Table I/II and a calibration check.

    PJM_DATASET=dom python -m src.quantile.qra_plots

Reads only what train_qra.py wrote under results/{zone}/qra/, so it restyles without
refitting. Style helpers, palette and the fixed hue order come from src.quantile.plots —
the two baselines should look like one family, because they are read side by side.
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
from src.quantile.qeval import QUANTILES, coverage_report

# The subject of the study gets the lead hue; its benchmarks share a recessive grey, because
# the comparison the reader makes is length-against-length, not identity-against-identity.
SUBJECT, BENCH = PALETTE[0], '#b9b9b4'

# Fixed method -> hue slot, shared by EVERY figure here. Assigning by sort order or by rank
# would repaint a method from chart to chart (QRA blue in one, green in the next), which is
# the one thing colour must never do: the hue identifies the entity, not its position.
METHOD_HUE = {'QRA': 0, 'BI': 1, 'Ind': 2, 'Direct': 3, 'Vanilla': 5}


def _hue(name):
    for k, i in METHOD_HUE.items():
        if name.startswith(k):
            return PALETTE[i]
    return PALETTE[len(METHOD_HUE) % len(PALETTE)]


def plot_validation_sweep(sw, zone, path):
    """Pinball against the number of sisters averaged, one line per calibration window.

    The paper's Table I as a figure. Two things are meant to be readable at a glance: QRA
    improves with m, and it sits below every individual model at every window length.
    """
    q = sw[sw.model.str.startswith('QRA')].copy()
    q['m'] = q['m'].astype(int)
    ind = sw[sw.model.str.startswith('Ind')]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    best_ind = ind.pinball.min()
    ax.axhline(best_ind, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate('best individual sister', (q.m.max(), best_ind), color=INK2, fontsize=8,
                va='bottom', ha='right', xytext=(0, 4), textcoords='offset points')

    labels = []
    for i, (L, g) in enumerate(q.groupby('L')):
        g = g.sort_values('m')
        ax.plot(g.m, g.pinball, color=PALETTE[i], linewidth=2, marker='o', markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        labels.append((g.m.iloc[-1], g.pinball.iloc[-1], f'L={L}d', PALETTE[i]))

    ax.set_xticks(sorted(q.m.unique()))
    ax.set_xlim(q.m.min() - 0.2, q.m.max() + 1.4)
    _style(ax, f'QRA on the validation year — pinball vs sisters averaged   ({zone.upper()})',
           'm  (number of sister forecasts combined)', 'Pinball loss  (MW, lower is better)')
    ax.legend(handles=[plt.Line2D([], [], color=PALETTE[i], lw=2, marker='o',
                                  label=f'{L}-day calibration window')
                       for i, L in enumerate(sorted(q.L.unique()))],
              frameon=False, fontsize=8.5, labelcolor=INK, loc='upper right')
    _stack_labels(ax, labels)
    _save(fig, path)


def plot_scoreboard(tbl, zone, path):
    """The three scores as small multiples — never one axis with three different units.

    Pinball and the two Winkler scores answer different questions (average quantile error;
    coverage-and-width at 50%; at 90%), so they get three panels sharing one category axis
    rather than three series sharing one value axis.
    """
    metrics = [('pinball', 'Pinball loss'), ('winkler50', 'Winkler score, 50% PI'),
               ('winkler90', 'Winkler score, 90% PI')]
    t = tbl.iloc[::-1]                                  # barh draws bottom-up
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 0.55 * len(t) + 2.1))
    fig.patch.set_facecolor(SURFACE)

    for ax, (key, title) in zip(axes, metrics):
        ax.set_facecolor(SURFACE)
        colors = [SUBJECT if m == 'QRA' else BENCH for m in t.model]
        ax.barh(np.arange(len(t)), t[key], height=0.62, color=colors, zorder=3)
        for i, v in enumerate(t[key]):
            ax.annotate(f' {v:,.1f}', (v, i), va='center', ha='left',
                        color=INK, fontsize=8.5, annotation_clip=False)
        ax.set_yticks(np.arange(len(t)))
        ax.set_yticklabels(t.model)
        ax.set_xlim(0, t[key].max() * 1.28)
        _style(ax, title, '', '')
        ax.grid(axis='y', visible=False)

    fig.suptitle(f'Test period — QRA against its benchmarks   ({zone.upper()})',
                 color=INK, fontsize=11, x=0.005, ha='left', y=1.0)
    fig.tight_layout()
    _save(fig, path)


def plot_peak_week(q, actual, when, zone, path, label='QRA'):
    """The predictive intervals over the peak week against what happened.

    Same construction as the scenario baseline's figure so the two can be laid side by side:
    nested central intervals in one hue, one median line, actual load as the only marker.
    """
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
    d, zone = cfg.QRA_CONFIG['result_dir'], cfg.DATASET
    if not os.path.exists(os.path.join(d, 'test_scores.csv')):
        raise SystemExit(f"No results in {d} — run train_qra.py first.")

    sw = pd.read_csv(os.path.join(d, 'validation_sweep.csv'))
    plot_validation_sweep(sw, zone, os.path.join(d, 'validation_sweep.png'))

    tbl = pd.read_csv(os.path.join(d, 'test_scores.csv'))
    plot_scoreboard(tbl, zone, os.path.join(d, 'scoreboard.png'))

    z = np.load(os.path.join(d, 'test_quantiles.npz'), allow_pickle=True)
    actual, when = z['actual'], pd.to_datetime(z['datetime'])
    curves = {k: z[k] for k in z.files if k not in ('actual', 'datetime')}
    plot_reliability(curves, actual, zone, os.path.join(d, 'reliability.png'),
                     colors=[_hue(n) for n in curves])
    plot_peak_week(curves['QRA'], actual, when, zone, os.path.join(d, 'fan_peak_week.png'))
    pbq = pd.read_csv(os.path.join(d, 'pinball_by_quantile.csv'))
    plot_pinball_by_quantile(os.path.join(d, 'pinball_by_quantile.csv'), zone,
                             os.path.join(d, 'pinball_by_quantile.png'),
                             colors=[_hue(c) for c in pbq.columns if c != 'quantile'])
    mth = pd.read_csv(os.path.join(d, 'monthly.csv'))
    plot_monthly(os.path.join(d, 'monthly.csv'), zone, os.path.join(d, 'monthly.png'),
                 colors=[_hue(c.rsplit('_', 1)[0]) for c in mth.columns
                         if c.endswith('_pinball')])


if __name__ == '__main__':
    main()
