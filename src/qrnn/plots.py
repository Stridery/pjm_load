"""Figures for the iQRNN baseline — Zhang et al. Table IV, Fig. 3/5/8/9, and calibration.

    PJM_DATASET=dom python -m src.qrnn.plots

Reads only what scripts/train_qrnn.py wrote under results/{zone}/qrnn/, so it restyles without
refitting. Style helpers, palette and the generic probabilistic figures come from
src.quantile.plots: the baselines are read side by side, so they are drawn by one set of
rules rather than one per package.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.quantile.plots import (INK, INK2, MUTED, PALETTE, SURFACE, _save, _style,
                                plot_monthly, plot_pinball_by_quantile, plot_reliability)

# Fixed name -> hue, shared by every figure here, so no model changes colour between charts.
# The proposed network leads; its control is red because the whole point of that row is that
# it is the thing being beaten; the folded-in baseline is a muted green.
HUE = {'iQRNN': PALETTE[0], 'bQRNN': PALETTE[5], 'CRPS-guided': PALETTE[2]}
BENCH = '#b9b9b4'
GRIDLINE = '#e3e3e0'


def _hue(name):
    for k, c in HUE.items():
        if str(name).startswith(k):
            return c
    return MUTED


def plot_scoreboard(tbl, zone, path):
    """Pinball, Winkler(90%) and CRPS as small multiples — three units never share an axis.

    Pinball is the paper's headline (its quantile score); Winkler adds the width-and-coverage
    view that pinball alone cannot separate; CRPS is the one metric every baseline in this
    repo shares. Rows whose horizon differs from the run's are hatched, because a bar chart is
    exactly where a reader stops reading the horizon column.
    """
    metrics = [('pinball', 'Quantile score  (MW)'), ('winkler90', 'Winkler score, 90% PI'),
               ('CRPS', 'CRPS  (MW)')]
    t = tbl.sort_values('pinball', ascending=False).reset_index(drop=True)
    own = t.horizon.value_counts().idxmax() if 'horizon' in t else None

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 0.55 * len(t) + 2.4))
    fig.subplots_adjust(wspace=0.28)
    fig.patch.set_facecolor(SURFACE)
    for panel, (ax, (key, title)) in enumerate(zip(axes, metrics)):
        ax.set_facecolor(SURFACE)
        for i, r in t.iterrows():
            odd = 'horizon' in t and r.horizon != own
            ax.barh(i, r[key], height=0.62, color=_hue(r.model), zorder=3,
                    hatch='///' if odd else None, edgecolor=SURFACE if odd else 'none')
            ax.annotate(f'{r[key]:,.1f}', (r[key], i), textcoords='offset points',
                        xytext=(5, 0), va='center', ha='left', color=INK, fontsize=8.5,
                        annotation_clip=False)
        if key == 'pinball' and 'pinball_sd' in t:
            ax.errorbar(t[key], np.arange(len(t)), xerr=t.pinball_sd, fmt='none',
                        ecolor=INK, elinewidth=1.1, capsize=3, zorder=4)
        ax.set_yticks(np.arange(len(t)))
        # Only the leftmost panel is labelled: the three share a row order, and repeating
        # long model names in every panel is what pushed them onto the bars.
        ax.set_yticklabels(t.model if panel == 0 else [])
        ax.set_xlim(0, t[key].max() * 1.34)
        _style(ax, title, '', '')
        ax.grid(axis='x', color=GRIDLINE, zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle(f'Probabilistic baselines on the shared test period   ({zone.upper()})',
                 color=INK, fontsize=11, y=1.0)
    if own:
        fig.text(0.5, -0.02, f'hatched = a different horizon from this run ({own}); '
                 f'error bars are +/-1 SD over independent trainings',
                 ha='center', color=INK2, fontsize=8.5)
    _save(fig, path)


def plot_runs(runs, zone, path):
    """Their Fig. 5: the spread over independent trainings, not just the mean.

    This is the figure the paper's practicality argument actually rests on — bQRNN's quantile
    score has an SD two orders of magnitude larger than iQRNN's, which means a single bQRNN
    run tells you very little about the next one. A table of means hides that completely.
    """
    order = sorted(runs.model.unique(), key=lambda m: runs[runs.model == m].pinball.median())
    data = [runs[runs.model == m].pinball.values for m in order]

    fig, ax = plt.subplots(figsize=(1.5 * len(order) + 2.6, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    bp = ax.boxplot(data, tick_labels=order, widths=0.5, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.4),
                    whiskerprops=dict(color=INK2), capprops=dict(color=INK2),
                    flierprops=dict(marker='+', markeredgecolor=MUTED, markersize=5))
    for patch, m in zip(bp['boxes'], order):
        patch.set(facecolor=_hue(m), alpha=0.55, edgecolor=_hue(m))
    for i, (m, d) in enumerate(zip(order, data), start=1):
        ax.scatter(np.full(len(d), i) + np.linspace(-0.1, 0.1, len(d)), d, s=13,
                   color=INK, zorder=4, alpha=0.75)
        ax.annotate(f'sd {d.std(ddof=0):.2f}', (i, d.max()), textcoords='offset points',
                    xytext=(0, 8), ha='center', color=INK2, fontsize=8.5)
    _style(ax, f'Quantile score over {len(data[0])} independent trainings   ({zone.upper()})',
           '', 'Pinball loss  (MW)')
    ax.grid(axis='y', color=GRIDLINE, zorder=0)
    ax.set_axisbelow(True)
    _save(fig, path)


def plot_peak_week(q, actual, when, zone, path, label='iQRNN'):
    """The intervals over the peak week — same construction as the other three baselines'
    figures, so all four can be laid side by side."""
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


def plot_size_sweep(sw, zone, path):
    """Their Fig. 3 / Fig. 8. The claim is not that iQRNN is flat — it is that iQRNN is flat
    where the control is not, so both curves belong on one axis with a shared scale."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, title) in zip(axes, [('pinball', 'Quantile score  (MW)'),
                                       ('winkler50', 'Winkler score, 50% PI'),
                                       ('winkler90', 'Winkler score, 90% PI')]):
        ax.set_facecolor(SURFACE)
        for fam, grp in sw.groupby('family'):
            g = grp.groupby('hidden')[key].agg(['mean', 'std']).reset_index()
            ax.plot(g.hidden, g['mean'], marker='o', markersize=4.5, linewidth=1.8,
                    color=_hue(fam), label=fam, zorder=3)
            ax.fill_between(g.hidden, g['mean'] - g['std'].fillna(0),
                            g['mean'] + g['std'].fillna(0), color=_hue(fam), alpha=0.16,
                            linewidth=0, zorder=2)
        _style(ax, title, 'Units per hidden layer', '')
        ax.grid(color=GRIDLINE, zorder=0)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(f'Robustness to network size   ({zone.upper()})', color=INK, fontsize=11,
                 y=1.02)
    _save(fig, path)


def plot_noise_sweep(sw, zone, path):
    """Their Fig. 9: the Gaussian-noise SD sweep, the paper's largest single ablation."""
    g = sw.groupby('noise_sd')['pinball'].agg(['mean', 'std', 'min', 'max']).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between(g.noise_sd, g['mean'] - g['std'].fillna(0), g['mean'] + g['std'].fillna(0),
                    color=PALETTE[0], alpha=0.16, linewidth=0, zorder=2)
    ax.plot(g.noise_sd, g['mean'], marker='o', markersize=5, linewidth=1.9, color=PALETTE[0],
            zorder=3)
    best = g.loc[g['mean'].idxmin()]
    ax.annotate(f"best SD {best.noise_sd:g}", (best.noise_sd, best['mean']),
                textcoords='offset points', xytext=(0, -16), ha='center', color=INK2,
                fontsize=9)
    _style(ax, f'Gaussian noise SD vs quantile score   ({zone.upper()})',
           'Noise standard deviation', 'Pinball loss  (MW)')
    ax.grid(color=GRIDLINE, zorder=0)
    ax.set_axisbelow(True)
    _save(fig, path)


def main():
    import argparse
    import src.config as cfg
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dir', default=None,
                    help='results directory (default QRNN_CONFIG; use for a _lag24 arm)')
    a = ap.parse_args()
    d, zone = a.dir or cfg.QRNN_CONFIG['result_dir'], cfg.DATASET
    if not os.path.exists(os.path.join(d, 'test_scores.csv')):
        raise SystemExit(f"No results in {d} — run scripts/train_qrnn.py first.")

    plot_scoreboard(pd.read_csv(os.path.join(d, 'test_scores.csv')), zone,
                    os.path.join(d, 'scoreboard.png'))
    plot_runs(pd.read_csv(os.path.join(d, 'runs.csv')), zone, os.path.join(d, 'runs.png'))

    z = np.load(os.path.join(d, 'test_quantiles.npz'), allow_pickle=True)
    actual, when = z['actual'], pd.to_datetime(z['datetime'])
    curves = {n: z[f'cfg{i}'] for i, n in enumerate(list(z['labels']))}
    plot_reliability(curves, actual, zone, os.path.join(d, 'reliability.png'),
                     colors=[_hue(n) for n in curves])
    lead = next((k for k in curves if k.startswith('iQRNN')), list(curves)[0])
    plot_peak_week(curves[lead], actual, when, zone, os.path.join(d, 'fan_peak_week.png'),
                   label=lead)

    p = os.path.join(d, 'pinball_by_quantile.csv')
    cols = [c for c in pd.read_csv(p, nrows=0).columns if c != 'quantile']
    plot_pinball_by_quantile(p, zone, os.path.join(d, 'pinball_by_quantile.png'),
                             colors=[_hue(c) for c in cols])

    p = os.path.join(d, 'monthly.csv')
    cols = [c.rsplit('_', 1)[0] for c in pd.read_csv(p, nrows=0).columns
            if c.endswith('_pinball')]
    plot_monthly(p, zone, os.path.join(d, 'monthly.png'), colors=[_hue(c) for c in cols])

    # The two sweeps are optional runs, so their figures are drawn only if the run made them.
    for name, fn in [('size_sweep', plot_size_sweep), ('noise_sweep', plot_noise_sweep)]:
        p = os.path.join(d, f'{name}.csv')
        if os.path.exists(p):
            fn(pd.read_csv(p), zone, os.path.join(d, f'{name}.png'))
        else:
            print(f"  ({name} not present — re-run with --{name.replace('_', '-')})")


if __name__ == '__main__':
    main()
