"""
The committed forecast store — one wide table per zone, the site's only data source.

    web/{zone}.csv :  datetime, datetime_utc, metered, {MODEL}_pred ...

One row per real clock hour (23/24/25 on DST days). Two update rules, and they are the whole
point of the store:
  - a model's prediction for an hour is FROZEN the first time it is written. Re-running the
    daily job never rewrites a past forecast, so the real-time test shows what was actually
    predicted day-ahead, not a hindsight re-forecast.
  - metered is the truth, filled in and UPGRADED as it arrives (unverified ~2-3 d, then verified
    ~7 d). It is the only column that changes after the fact.

The store is pruned to a rolling window (default 10 days) so it stays small enough to commit
and diff each day. Row assembly (mapping a model's 24 outputs onto the day's real hours) happens
in the caller; this module only merges and prunes.
"""

import os

import pandas as pd

STORE_DIR = 'web'


def path_for(zone):
    return os.path.join(STORE_DIR, f'{zone}.csv')


def load(zone):
    """Existing store as a UTC-indexed frame, or an empty one."""
    p = path_for(zone)
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
    return df.set_index('datetime_utc').sort_index()


def update(zone, fresh, keep_days=10):
    """Merge `fresh` into the store with freeze-on-predictions, update-on-metered, then prune.

    fresh : DataFrame with columns [datetime, metered, {MODEL}_pred...] indexed by datetime_utc
            (UTC-aware). It is the freshly computed rows for the current window — brand-new days
            AND recomputed recent days. Predictions for hours already in the store are ignored
            (frozen); their metered is taken from `fresh` (the newest truth).
    """
    os.makedirs(STORE_DIR, exist_ok=True)
    fresh = fresh.sort_index()
    old = load(zone)

    if old.empty:
        merged = fresh.copy()
    else:
        cols = list(dict.fromkeys(list(old.columns) + list(fresh.columns)))
        old = old.reindex(columns=cols)
        fresh = fresh.reindex(columns=cols)
        # Union of all hours; existing rows keep their frozen predictions.
        idx = old.index.union(fresh.index)
        merged = pd.DataFrame(index=idx, columns=cols)
        merged['datetime'] = fresh['datetime'].combine_first(old['datetime'])
        # metered: newest truth wins (fresh over old).
        merged['metered'] = fresh['metered'].combine_first(old['metered'])
        # predictions: FROZEN — old wins, fresh only fills hours the store never had.
        for c in cols:
            if c.endswith('_pred'):
                merged[c] = old[c].combine_first(fresh[c])

    # Prune to the rolling window, measured back from the latest hour present.
    if keep_days is not None and len(merged):
        cutoff = merged.index.max() - pd.Timedelta(days=keep_days)
        merged = merged[merged.index > cutoff]

    out = merged.sort_index().reset_index()
    out.to_csv(path_for(zone), index=False)
    n_pred_cols = sum(c.endswith('_pred') for c in out.columns)
    scored = int(out['metered'].notna().sum()) if 'metered' in out else 0
    print(f"Store {zone}: {len(out)} rows, {n_pred_cols} models, "
          f"{scored} h with metered → {path_for(zone)}")
    return out
