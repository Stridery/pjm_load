"""Recompute segmented metrics for runs that already exist. No model is re-run.

    python scripts/backfill_segments.py                      # every zone found under results/
    python scripts/backfill_segments.py --zone dom
    python scripts/backfill_segments.py --zone dom --dry-run

Everything the segments need is already on disk: `*_detailed_errors.csv` carries the
timestamp, the truth and the prediction, and the volatility baseline comes from the cleaned
CSV. So a hundred past runs can be scored on the new cuts without touching a single weight
file -- which is the point, since several of those runs were never saved with weights at all.

Writes, per run directory:
    {MODEL}_segments.csv           the machine-readable table
    {MODEL}_detailed_errors.csv    + load_band, vol_ratio, vol_subset
    {MODEL}_metrics.txt            + the two blocks, replacing any earlier copy of them
"""

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import glob
import os
import re

import pandas as pd


def zones_present():
    """Result roots that look like a zone tree, newest layout first."""
    out = []
    for d in sorted(glob.glob('results/*/evaluation')):
        z = d.split(os.sep)[1]
        if not z.startswith('_'):
            out.append(z)
    return out


def strip_blocks(text):
    """Remove previously appended segment blocks so a re-run replaces rather than stacks."""
    return re.split(r'\n--- by (?:load band|relative volatility)', text)[0].rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zone', nargs='*', default=None,
                    help='default: every zone under results/ that has a LOAD_BANDS entry')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    import src.config as cfg
    zones = args.zone or zones_present()
    total = skipped = 0

    for zone in zones:
        if zone not in cfg.LOAD_BANDS:
            print(f"-- {zone}: no LOAD_BANDS entry, skipped "
                  f"(price/alternate-target trees are not load-banded)")
            continue
        # Re-import per zone: DATASET is read at import time, and _segments keys its volatility
        # cache on the resolved cleaned path, so the two must agree.
        os.environ['PJM_DATASET'] = zone
        for m in [k for k in list(_sys.modules) if k.startswith('src.')]:
            del _sys.modules[m]
        import src.config as zcfg
        from src.models import _segments as S

        lo, hi = zcfg.LOAD_BANDS[zone]
        files = sorted(glob.glob(f'results/{zone}/evaluation/*/*/*_detailed_errors.csv'))
        print(f"\n=== {zone}  bands {lo:,.0f} / {hi:,.0f}  ({len(files)} runs) ===")
        for f in files:
            rdir = os.path.dirname(f)
            model_name = os.path.basename(f).replace('_detailed_errors.csv', '')
            try:
                d = pd.read_csv(f, parse_dates=['datetime'])
                pcol = next(c for c in d.columns if c.endswith('_pred'))
                if len(d) % 24:
                    # Per-expert MoE directories hold an hour slice, not whole days; the load
                    # bands still mean something there but the volatility window does not line
                    # up, so they are left alone rather than half-scored.
                    print(f"  skip {model_name:<34} (not whole days: {len(d)} rows)")
                    skipped += 1
                    continue
                rows, extra = S.compute(d['datetime'], d['true_load'], d[pcol],
                                        zone=zone, cleaned_path=zcfg.CLEANED_PATH)
                if args.dry_run:
                    b = {r['segment']: round(r['MAPE'], 3) for r in rows if r['cut'] == 'load_band'}
                    v = {r['segment']: round(r['MAPE'], 3) for r in rows if r['cut'] == 'volatility'}
                    print(f"  {model_name:<38} bands {b}  vol {v}")
                    total += 1
                    continue

                for k, v in extra.items():
                    d[k] = v
                d.to_csv(f, index=False)
                S.write(rows, rdir, model_name)
                tpath = os.path.join(rdir, f'{model_name}_metrics.txt')
                if os.path.exists(tpath):
                    with open(tpath) as fh:
                        head = strip_blocks(fh.read())
                    with open(tpath, 'w') as fh:
                        fh.write(head + S.to_text(rows, zcfg.TARGET_UNIT))
                total += 1
            except Exception as exc:
                print(f"  FAIL {model_name:<34} {type(exc).__name__}: {exc}")
                skipped += 1
        print(f"  {total} written, {skipped} skipped so far")

    print(f"\n{'(dry run) ' if args.dry_run else ''}{total} runs scored, {skipped} skipped")


if __name__ == '__main__':
    main()
