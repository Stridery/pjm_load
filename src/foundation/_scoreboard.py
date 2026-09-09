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


def write(evaluation_root, zone):
    df = collect(evaluation_root)
    if df.empty:
        print(f"No *_metrics.txt found under {evaluation_root} — nothing to compare yet.")
        return df
    csv_path = os.path.join(evaluation_root, '_scoreboard.csv')
    df.to_csv(csv_path, index=False)
    plot(df, zone, os.path.join(evaluation_root, '_scoreboard.png'))
    cols = ['model', 'arm', 'name', 'MAPE', 'peak_MAPE', 'MAE', 'RMSE', 'ME', 'days', 'kind']
    if 'wMAPE' in df.columns and df['wMAPE'].notna().any():
        cols.insert(5, 'wMAPE')          # price runs only; absent on load runs
    print(f"\n{'=' * 96}\nSCOREBOARD | zone {zone} | {evaluation_root}\n{'=' * 96}")
    print(df[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f"\n-> {csv_path}")
    return df
