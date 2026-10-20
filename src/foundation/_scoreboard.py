"""One table and one figure putting every scored model in a result root side by side.

The point models' scores are already durable files -- results/{zone}{tag}/evaluation/**/
*_metrics.txt, which the README calls the record of a run -- so the comparison is a matter of
reading them rather than re-running anything. Foundation runs write the same file in the same
tree, so they are picked up by the same scan with no special case.

EVERY metric in those files is carried through to the table, the CSV and the figure. They are
not interchangeable and a ranking by one of them hides the others:

    MAPE / peak_MAPE        per-observation error. On a price target a handful of near-zero
                            hours move MAPE more than the rest of the year.
    wMAPE / peak_wMAPE      sum|e| / sum|y| -- what the price literature reports, and the
                            reason it exists is precisely that MAPE misleads there. Written
                            only for targets with report_wmape, so absent on load runs.
    MAE / RMSE              level errors in the target's own unit. RMSE punishes the spike
                            days MAE averages away; where the two disagree, that is the
                            finding.
    ME / BRS                BIAS, not size. ME is the mean signed error; BRS is the slope of
                            that error against the truth (negative = under-forecasting harder
                            as the target rises). Both are best at ZERO, in either direction,
                            which is why they are drawn from a zero line and never ranked
                            "smaller is better".
    worst / best day        the tails of the daily MAPE. A model can win on the mean and be
                            the one that blows up on the day that mattered.
"""

import glob
import os
import re
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Single-number metrics. Units are NOT pinned to MW: PJM_TARGET decides them and a price run
# writes $/MWh, so every pattern matches the number and lets the unit be any non-space token.
# Pinning "MW" here is what would silently return NaN on a price run.
_PATTERNS = {
    'MAPE':       re.compile(r'^MAPE:\s+([-\d.]+)\s*%', re.M),
    'MAE':        re.compile(r'^MAE:\s+([-\d.]+)\s*\S+', re.M),
    'RMSE':       re.compile(r'^RMSE:\s+([-\d.]+)\s*\S+', re.M),
    'ME':         re.compile(r'^ME \(bias\):\s+([-+\d.]+)\s*\S+', re.M),
    'BRS':        re.compile(r'^BRS:\s+([-+\d.]+)\s*\S+', re.M),
    'peak_MAPE':  re.compile(r'^Peak MAPE \(top 10% \w+ hours\):\s+([-\d.]+)\s*%', re.M),
    'wMAPE':      re.compile(r'^wMAPE:\s+([-\d.]+)\s*%', re.M),
    'peak_wMAPE': re.compile(r'^Peak wMAPE \(top 10% \w+ hours\):\s+([-\d.]+)\s*%', re.M),
    'samples':    re.compile(r'^test samples:\s+(\d+)', re.M),
    'days':       re.compile(r'^test samples:\s+\d+\s+\((\d+) days\)', re.M),
}

# Date + percentage in one line, so they need two capture groups and their own pass.
_PAIRS = {
    'worst': re.compile(r'^Worst day:\s+(\S+)\s+\(([-\d.]+)%\)', re.M),
    'best':  re.compile(r'^Best day:\s+(\S+)\s+\(([-\d.]+)%\)', re.M),
}

# `target:  LMP_RT_Total ($/MWh)` -> the unit the MAE/RMSE/ME axes are labelled with.
_TARGET = re.compile(r'^target:\s+(\S+)\s+\(([^)]+)\)', re.M)

# key, axis label, polarity. 'bias' metrics are best at zero IN EITHER DIRECTION, so they get
# a zero line and a symmetric axis instead of being drawn as if less were always better.
_METRICS = [
    ('MAPE',       'MAPE (%)',           'lower'),
    ('peak_MAPE',  'Peak MAPE (%)',      'lower'),
    ('wMAPE',      'wMAPE (%)',          'lower'),
    ('peak_wMAPE', 'Peak wMAPE (%)',     'lower'),
    ('MAE',        'MAE ({unit})',       'lower'),
    ('RMSE',       'RMSE ({unit})',      'lower'),
    ('ME',         'ME bias ({unit})',   'bias'),
    ('BRS',        'BRS ({unit}/{unit})', 'bias'),
    ('worst_MAPE', 'Worst day (%)',      'lower'),
    ('best_MAPE',  'Best day (%)',       'lower'),
]

# Two categories, one colour job: WHICH KIND of model a row is. Polarity in the bias panels is
# carried by the bar's direction from the zero line, not by a second colour meaning -- two
# colour scales in one figure is the thing that makes small multiples unreadable.
# The pair is the repo's existing seaborn-deep blue/orange, checked rather than assumed:
# OKLab dE 26.9 normal vision, 19.9 protan / 26.8 deutan / 39.7 tritan, all above the 8 target,
# both inside the 0.43-0.77 lightness band and over the 0.10 chroma floor. The orange sits at
# 2.80:1 on white, under the 3:1 mark bar, which is why the legend is always drawn and the
# same numbers are printed as a table and written to CSV.
_C_POINT, _C_FOUNDATION = '#4C72B0', '#DD8452'
# Two more for the GARCHFormer grid-search ensembles (scripts/grid_archformer_collect.py
# --emit-eval): the hp wiring (every channel tokenised) and the skip wiring (statics fused at
# the decoder). Green and purple from the same seaborn-deep family, so the four kinds stay
# apart from each other at a glance; the legend names them.
_C_GRID_HP, _C_GRID_SKIP = '#55A868', '#8172B2'
# A fifth, for the GARCHFormer variants that are NOT grid-search ensembles -- the shipped
# archformer / _hp / _skip and their residual and 5-seed-ensemble forms. Fourteen of the
# sixteen GARCHFormer rows were landing in the generic point blue, so the one architecture the
# project is actually investigating was the hardest to find on its own chart. Seaborn-deep red,
# the remaining member of the family the other four come from.
_C_ARCHFORMER = '#C44E52'
_KIND_COLOR = {'point': _C_POINT, 'foundation': _C_FOUNDATION,
               'archformer': _C_ARCHFORMER,
               'grid_hp': _C_GRID_HP, 'grid_skip': _C_GRID_SKIP}
_KIND_LABEL = {'point': 'trained point model', 'foundation': 'zero-shot foundation',
               'archformer': 'GARCHFormer',
               'grid_hp': 'GARCHFormer grid, hp wiring (5-seed ens)',
               'grid_skip': 'GARCHFormer grid, skip wiring (5-seed ens)'}
_INK, _MUTED = '#222222', '#888888'


# The run tag encodes the split AND the feature arm, and several models differ ONLY by arm --
# four `transformer_residual` rows whose `model` and `name` columns are identical. Pulling the
# arm out into its own column is what makes the table and the figure readable.
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
    """Every *_metrics.txt under results/{zone}{tag}/evaluation/ as one DataFrame.

    The glob is deliberately exactly three levels deep -- {model}/{run}/{NAME}_metrics.txt.
    The MoE models also write per-expert metrics one level further down, under
    {run}/experts/{season__expert}/, and those are sub-populations of a row already in this
    table rather than rows of their own. A `**` here would list a MoE model fourteen times and
    rank its easiest expert first.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(evaluation_root, '*', '*', '*_metrics.txt'))):
        parts = path.split(os.sep)
        text = open(path).read()
        row = {'model': parts[-3], 'arm': arm_label(parts[-2]), 'run': parts[-2],
               'name': os.path.basename(path).replace('_metrics.txt', '')}
        for key, pat in _PATTERNS.items():
            m = pat.search(text)
            row[key] = float(m.group(1)) if m else None
        for key, pat in _PAIRS.items():
            m = pat.search(text)
            row[f'{key}_day'] = m.group(1) if m else None
            row[f'{key}_MAPE'] = float(m.group(2)) if m else None
        m = _TARGET.search(text)
        row['target'], row['unit'] = (m.group(1), m.group(2)) if m else (None, None)
        # Foundation runs append a block naming their adapter; point models do not.
        if '--- foundation run ---' in text:
            row['kind'] = 'foundation'
        elif '--- grid search ---' in text:
            # Grid-search ensembles say which wiring they are; colour them apart.
            w = re.search(r'^wiring:\s*(\w+)', text, re.M)
            row['kind'] = f"grid_{w.group(1)}" if w else 'point'
        elif row['model'].startswith('archformer'):
            # By directory name, not by a marker in the file: the shipped GARCHFormer runs
            # write an ordinary metrics.txt with nothing to distinguish them, and adding one
            # would mean re-running every past evaluation to get the colour.
            row['kind'] = 'archformer'
        else:
            row['kind'] = 'point'
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values('MAPE', na_position='last').reset_index(drop=True)


def _unit_of(df):
    u = [x for x in df.get('unit', []) if isinstance(x, str)]
    return Counter(u).most_common(1)[0][0] if u else ''


def _panels(df):
    """The metrics actually present in this root, in _METRICS order.

    wMAPE and peak_wMAPE are written only for targets with report_wmape, so a load root has
    eight panels and a price root ten. Dropping the empty ones beats drawing blank axes.
    """
    return [(k, lab, pol) for k, lab, pol in _METRICS
            if k in df.columns and df[k].notna().any()]


def _ranked(d, key, polarity):
    """Rows ordered best-first for THIS metric, and the label to put on each bar.

    'lower' metrics rank ascending. 'bias' metrics (ME, BRS) are best at ZERO in either
    direction, so they rank on |value| — ranking them ascending would put the worst
    under-forecaster first and call it the winner.
    """
    d = d.dropna(subset=[key]).copy()
    d['_rank'] = d[key].abs() if polarity == 'bias' else d[key]
    return d.sort_values('_rank').drop(columns='_rank')


def plot_metric(df, zone, key, label, polarity, path, unit='', target=''):
    """One metric, one figure, ranked by that metric with the best at the top.

    One chart per metric rather than one shared-axis grid: each metric gets its OWN order, so
    the figure answers "who wins on RMSE" directly instead of asking the reader to find a row
    in a MAPE ordering. Where two metrics disagree, that is visible as a model moving between
    the two files — which is the comparison worth making on a price target, where MAPE and
    wMAPE routinely rank differently.
    """
    d = _ranked(df, key, polarity)
    if d.empty:
        return None
    d = d.iloc[::-1]                       # matplotlib draws y upward; best ends up on top
    rows = len(d)
    colors = [_KIND_COLOR.get(k, _C_POINT) for k in d['kind']]
    v = d[key].astype(float)

    fig, ax = plt.subplots(figsize=(11, max(4.0, 0.30 * rows + 1.4)))
    # height < 1 leaves a surface gap between adjacent bars, so 70+ rows read as 70+ marks
    # rather than one solid block.
    ax.barh(range(rows), v, height=0.74, color=colors, edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(rows))
    # The arm is part of the identity, not a detail: without it four rows read
    # 'transformer_residual' and the chart cannot be used to pick one.
    ax.set_yticklabels([f"{m} [{a}]" + (f"  ({n})" if k == 'foundation' else '')
                        for m, a, n, k in zip(d['model'], d['arm'], d['name'], d['kind'])],
                       fontsize=7.5, color=_INK)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', labelsize=8, colors=_MUTED)
    ax.grid(axis='x', linestyle='--', alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(_MUTED)

    if polarity == 'bias':
        m = float(v.abs().max() or 1) * 1.22
        ax.set_xlim(-m, m)
        ax.axvline(0, color=_INK, linewidth=1.0, zorder=3)
    else:
        ax.set_xlim(0, float(v.max() or 1) * 1.16)

    # The value on every bar: this figure IS a ranking of one number, so the label is the
    # readout that replaces an axis lookup rather than decoration on a trend.
    for i, val in enumerate(v):
        left = polarity == 'bias' and val < 0
        ax.text(val, i, f'{val:,.2f}  ' if left else f'  {val:,.2f}',
                va='center', ha='right' if left else 'left',
                fontsize=6.8, color=_INK, zorder=4)

    axis = label.format(unit=unit)
    note = 'best at 0, either side' if polarity == 'bias' else 'lower is better'
    ax.set_xlabel(f'{axis}  —  {note}', fontsize=9, color=_INK)
    ax.set_title(f'{zone} — {target or "all models"}: {axis}, ranked ({note})',
                 fontsize=12, fontweight='bold', color=_INK)
    kinds = [k for k in _KIND_COLOR if k in set(d['kind'])]      # only the kinds present
    handles = [plt.Rectangle((0, 0), 1, 1, color=_KIND_COLOR[k]) for k in kinds]
    ax.legend(handles, [_KIND_LABEL[k] for k in kinds], loc='lower right', fontsize=8,
              frameon=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def plot(df, zone, evaluation_root):
    """One PNG per metric present in this root. Returns the paths written.

    A load root has eight (no wMAPE pair); a price root has ten.
    """
    d = df.dropna(subset=['MAPE'])
    if d.empty:
        return []
    unit = _unit_of(d)
    target = next((t for t in d.get('target', []) if isinstance(t, str)), '')
    out = []
    for key, label, polarity in _panels(d):
        path = os.path.join(evaluation_root, f'_scoreboard_{key}.png')
        if plot_metric(d, zone, key, label, polarity, path, unit, target):
            out.append(path)
    return out


# ---------------------------------------------------------------------------
# Segmented scoreboards: three load bands, and the most volatile quarter
# ---------------------------------------------------------------------------
# A single MAPE ranks models on a weighted average of regimes they are not equally good at.
# dom's top load band holds 4.3 % of the hours and is where every model is worst; a model that
# wins overall can be third there. These figures rank each segment on its own.

_SEG_FIGS = [
    ('load_band', 'low',    'MAPE (%) — low load band'),
    ('load_band', 'mid',    'MAPE (%) — mid load band'),
    ('load_band', 'high',   'MAPE (%) — high load band'),
    ('volatility', 'top25', 'MAPE (%) — most volatile 25% of hours'),
]


def collect_segments(evaluation_root):
    """Every *_segments.csv under results/{zone}/evaluation/ as one long DataFrame."""
    rows = []
    for path in sorted(glob.glob(os.path.join(evaluation_root, '*', '*', '*_segments.csv'))):
        parts = path.split(os.sep)
        try:
            d = pd.read_csv(path)
        except Exception:
            continue
        d['model'] = parts[-3]
        d['arm'] = arm_label(parts[-2])
        d['run'] = parts[-2]
        d['name'] = os.path.basename(path).replace('_segments.csv', '')
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_segment(seg, zone, cut, segment, label, path, kinds=None):
    """One segment, every model, ranked on that segment's own MAPE."""
    d = seg[(seg.cut == cut) & (seg.segment == segment)].dropna(subset=['MAPE'])
    if d.empty:
        return None
    # One row per model: its best arm ON THIS SEGMENT. Ranking a segment by the arm that won
    # overall would report a number the figure does not show.
    d = d.loc[d.groupby('model')['MAPE'].idxmin()].sort_values('MAPE', ascending=False)
    kinds = kinds or {}
    ks = [kinds.get(m, 'archformer' if m.startswith('archformer') else 'point')
          for m in d['model']]
    colors = [_KIND_COLOR.get(k, _C_POINT) for k in ks]
    n = len(d)
    fig, ax = plt.subplots(figsize=(11, max(4.0, 0.30 * n + 1.6)))
    ax.barh(range(n), d['MAPE'], height=0.74, color=colors, edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{m} [{a}]" for m, a in zip(d['model'], d['arm'])], fontsize=7.5)
    for i, v in enumerate(d['MAPE']):
        ax.text(v, i, f'  {v:.2f}', va='center', fontsize=7)
    ax.set_xlabel(label, fontsize=9)
    ax.set_xlim(0, d['MAPE'].max() * 1.18)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    npts = int(d['n'].iloc[0])
    ax.set_title(f'{zone} — {label}   (n = {npts:,} hours, best arm per model)',
                 fontsize=11, fontweight='bold')
    # Legend only for the kinds actually drawn, so a load-band chart with no foundation row
    # does not advertise an absent colour.
    present = [k for k in _KIND_COLOR if k in set(ks)]
    if len(present) > 1:
        ax.legend([plt.Rectangle((0, 0), 1, 1, color=_KIND_COLOR[k]) for k in present],
                  [_KIND_LABEL[k] for k in present], loc='lower right', fontsize=7.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path


def write_segments(evaluation_root, zone, kinds=None):
    """The four segmented figures plus one tidy CSV. Returns the paths written."""
    seg = collect_segments(evaluation_root)
    if seg.empty:
        return []
    csv_path = os.path.join(evaluation_root, '_scoreboard_segments.csv')
    seg.to_csv(csv_path, index=False)
    out = [csv_path]
    for cut, segment, label in _SEG_FIGS:
        p = os.path.join(evaluation_root, f'_scoreboard_seg_{cut}_{segment}.png')
        if plot_segment(seg, zone, cut, segment, label, p, kinds):
            out.append(p)
    return out


def write(evaluation_root, zone):
    df = collect(evaluation_root)
    if df.empty:
        print(f"No *_metrics.txt found under {evaluation_root} — nothing to compare yet.")
        return df
    csv_path = os.path.join(evaluation_root, '_scoreboard.csv')
    df.to_csv(csv_path, index=False)
    pngs = plot(df, zone, evaluation_root)

    # The printed table carries the same metrics as the figure, plus the two dates the figure
    # can only show as percentages. Columns absent for this target drop out on their own.
    cols = (['model', 'arm']
            + [k for k, _, _ in _METRICS if k in df.columns and df[k].notna().any()]
            + ['worst_day', 'best_day', 'days', 'kind'])
    cols = [c for c in cols if c in df.columns]
    unit = _unit_of(df)
    print(f"\n{'=' * 120}\nSCOREBOARD | zone {zone} | {evaluation_root}"
          f"{f'  |  unit {unit}' if unit else ''}\n{'=' * 120}")
    print(df[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f"\n-> {csv_path}")
    for q in pngs:
        print(f"-> {q}")
    for q in write_segments(evaluation_root, zone, dict(zip(df['model'], df['kind']))):
        print(f"-> {q}")
    return df
