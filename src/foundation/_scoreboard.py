"""One table putting the foundation runs next to the 16 point models.

The point models' scores are already durable files -- results/{zone}/evaluation/**/*_metrics.txt,
which the README calls the record of a run -- so the comparison is a matter of reading them
rather than re-running anything. Foundation runs write the same file in the same tree, so they
are picked up by the same scan with no special case.
"""

import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

_PATTERNS = {
    'MAPE':      re.compile(r'^MAPE:\s+([-\d.]+)\s*%', re.M),
    # Units are NOT pinned to MW: PJM_TARGET decides them, and a price run writes $/MWh.
    # Matching the number and letting the unit be any non-space token keeps one scoreboard
    # parser correct for every target instead of silently returning NaN on a price run.
    'MAE':       re.compile(r'^MAE:\s+([-\d.]+)\s*\S+', re.M),
    'RMSE':      re.compile(r'^RMSE:\s+([-\d.]+)\s*\S+', re.M),
    'ME':        re.compile(r'^ME \(bias\):\s+([-+\d.]+)\s*\S+', re.M),
    'BRS':       re.compile(r'^BRS:\s+([-+\d.]+)\s*\S+', re.M),
    'peak_MAPE': re.compile(r'^Peak MAPE \(top 10% \w+ hours\):\s+([-\d.]+)\s*%', re.M),
    'wMAPE':     re.compile(r'^wMAPE:\s+([-\d.]+)\s*%', re.M),
    'days':      re.compile(r'^test samples:\s+\d+\s+\((\d+) days\)', re.M),
}


# The run tag encodes the split AND the feature arm, and several models differ ONLY by arm --
# four `transformer_residual` rows whose `model` and `name` columns are identical. Pulling the
# arm out into its own column is what makes the table and the chart readable.
_CTX_SUFFIX = re.compile(r'_ctx\d+.*$')          # foundation: _ctx168, _ctx144_met_da, ...
_TEST_PREFIX = re.compile(r'^\w+_test[\d.]+')    # tail_test0.16
_VAL_PREFIX = re.compile(r'^_\w+_val[\d.]+')     # _tail_val0.1

ARM_LABEL = {'': 'base', '_cong': 'cong', '_lmp': 'lmp', '_cong_lmp': 'cong+lmp'}


def arm_of(run):
    """The feature-set tag inside a run directory name: '' | '_cong' | '_lmp' | '_cong_lmp'.

    Strips the split description and any foundation context suffix, so the same function works
    on `tail_test0.16_tail_val0.1_cong_lmp` (a sequence model), `tail_test0.16_cong` (a tree
    model) and `tail_test0.16_cong_lmp_ctx168` (a foundation run).
    """
    s = _CTX_SUFFIX.sub('', run)
    s = _TEST_PREFIX.sub('', s)
    s = _VAL_PREFIX.sub('', s)
    return s if s in ARM_LABEL else s          # unknown tags pass through rather than vanish


def arm_label(run):
    return ARM_LABEL.get(arm_of(run), arm_of(run) or 'base')


def collect(evaluation_root):
    """Every *_metrics.txt under results/{zone}/evaluation/ as one DataFrame."""
    rows = []
    for path in sorted(glob.glob(os.path.join(evaluation_root, '*', '*', '*_metrics.txt'))):
        parts = path.split(os.sep)
        text = open(path).read()
        row = {'model': parts[-3], 'arm': arm_label(parts[-2]), 'run': parts[-2],
               'name': os.path.basename(path).replace('_metrics.txt', '')}
        for key, pat in _PATTERNS.items():
            m = pat.search(text)
            row[key] = float(m.group(1)) if m else None
        # Foundation runs append a block naming their adapter; point models do not.
        row['kind'] = 'foundation' if '--- foundation run ---' in text else 'point'
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values('MAPE', na_position='last').reset_index(drop=True)


def plot(df, zone, path):
    if df.empty:
        return
    d = df.dropna(subset=['MAPE']).iloc[::-1]          # best at the top of a barh
    colors = ['#DD8452' if k == 'foundation' else '#4C72B0' for k in d['kind']]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(d))))
    ax.barh(range(len(d)), d['MAPE'], color=colors, edgecolor='black', alpha=0.85)
    ax.set_yticks(range(len(d)))
    # The arm is part of the identity, not a detail: without it four rows read
    # 'transformer_residual' and the chart cannot be used to pick one.
    ax.set_yticklabels([f"{m} [{a}]" + (f"  ({n})" if k == 'foundation' else '')
                        for m, a, n, k in zip(d['model'], d['arm'], d['name'], d['kind'])],
                       fontsize=8)
    for i, (v, dd) in enumerate(zip(d['MAPE'], d['days'])):
        ax.text(v, i, f"  {v:.2f}%" + (f"  ({int(dd)}d)" if pd.notna(dd) else ''),
                va='center', fontsize=7)
    ax.set_xlabel('Test MAPE (%)')
    ax.set_title(f'{zone} — point models vs foundation models, same tail-0.16 test period',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ('#4C72B0', '#DD8452')]
    ax.legend(handles, ['trained point model', 'zero-shot foundation'], loc='lower right', fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


# The MAPE bar chart answers one question. These three answer the question the point models are
# actually being compared on, and they disagree: on dom the best mean MAPE and the best peak
# MAPE are different models, and the flattest BRS is a third. A single-metric ranking hides that.
_PANELS = [
    ('MAPE',      'Mean MAPE  (%)',                  False),
    ('peak_MAPE', 'Peak MAPE  (%)   top-10% hours',  False),
    ('BRS',       'BRS  (|slope|, 0 = unbiased)',    True),
]


def best_arm_per_model(df):
    """One row per model -- its best arm by MAPE.

    Four rows of `transformer_residual` differing only by feature arm crowd out the other
    twenty models, and the question a scoreboard answers is which model to use, not which arm
    of one model. The arm that survives is named on the label so nothing is hidden.
    """
    d = df.dropna(subset=['MAPE'])
    return d.loc[d.groupby('model')['MAPE'].idxmin()].reset_index(drop=True)


def plot_by_metric(df, zone, path, per_model=True):
    """MAPE, peak MAPE and BRS as small multiples, each panel sorted on its own metric.

    BRS is ranked on |BRS| because it is a bias slope: zero is the target and a large negative
    slope is as wrong as a large positive one. Its bars are drawn from zero with the sign kept,
    so the direction stays visible even though the ordering ignores it.
    """
    d = best_arm_per_model(df) if per_model else df.dropna(subset=['MAPE'])
    if d.empty:
        return
    n = len(d)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 0.34 * n + 2.0))
    for ax, (key, title, by_abs) in zip(axes, _PANELS):
        if key not in d.columns or d[key].isna().all():
            ax.set_visible(False)
            continue
        s = d.dropna(subset=[key])
        order = s[key].abs() if by_abs else s[key]
        s = s.loc[order.sort_values(ascending=False).index]      # best ends up at the top
        colors = ['#DD8452' if k == 'foundation' else '#4C72B0' for k in s['kind']]
        ax.barh(range(len(s)), s[key], color=colors, edgecolor='black', alpha=0.85, height=0.7)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels([f"{m} [{a}]" for m, a in zip(s['model'], s['arm'])], fontsize=7.5)
        for i, v in enumerate(s[key]):
            ax.text(v, i, f'  {v:.3f}' if by_abs else f'  {v:.2f}',
                    va='center', ha='left' if v >= 0 else 'right', fontsize=6.8)
        ax.set_xlabel(title, fontsize=9)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        if by_abs:
            ax.axvline(0, color='black', linewidth=0.8)
            lim = s[key].abs().max() * 1.35
            ax.set_xlim(-lim, lim)
        else:
            ax.set_xlim(0, s[key].max() * 1.22)
    scope = 'best arm per model' if per_model else 'every run'
    fig.suptitle(f'{zone} — point models vs foundation models, each metric ranked '
                 f'independently ({scope})', fontsize=12, fontweight='bold')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ('#4C72B0', '#DD8452')]
    axes[0].legend(handles, ['trained point model', 'zero-shot foundation'],
                   loc='lower right', fontsize=8)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


def write(evaluation_root, zone):
    df = collect(evaluation_root)
    if df.empty:
        print(f"No *_metrics.txt found under {evaluation_root} — nothing to compare yet.")
        return df
    csv_path = os.path.join(evaluation_root, '_scoreboard.csv')
    df.to_csv(csv_path, index=False)
    plot(df, zone, os.path.join(evaluation_root, '_scoreboard.png'))
    plot_by_metric(df, zone, os.path.join(evaluation_root, '_scoreboard_by_metric.png'))
    plot_by_metric(df, zone, os.path.join(evaluation_root, '_scoreboard_by_metric_all.png'),
                   per_model=False)
    cols = ['model', 'arm', 'name', 'MAPE', 'peak_MAPE', 'MAE', 'RMSE', 'ME', 'days', 'kind']
    if 'wMAPE' in df.columns and df['wMAPE'].notna().any():
        cols.insert(5, 'wMAPE')          # price runs only; absent on load runs
    print(f"\n{'=' * 96}\nSCOREBOARD | zone {zone} | {evaluation_root}\n{'=' * 96}")
    print(df[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f"\n-> {csv_path}")
    print(f"-> {os.path.join(evaluation_root, '_scoreboard_by_metric.png')}  "
          f"(best arm per model, 3 metrics)")
    return df
