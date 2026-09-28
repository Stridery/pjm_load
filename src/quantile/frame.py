"""The one input every probabilistic baseline reads, and the one split they all score on.

These two functions used to live in train_quantile.py, which the other baselines imported
from. When the scenario baseline was removed they had to move rather than be duplicated:
three entry points and two tests depend on them, and what makes the baselines comparable is
precisely that they read the same column of the same file and cut it at the same hour. A
copy in each caller is a copy that can drift.

Nothing here knows about any particular method. It is the boundary between the cleaned record
and the four baselines, and it is the only place that needs to change to point the suite at a
different dataset.
"""

import pandas as pd

import src.config as cfg


def load_frame():
    """The cleaned hourly record: (df, ept, load, temp, is_holiday, is_valid).

    `load` is the cleaned CSV's `Load` column, which is VERIFIED metered — see the note in
    src/qrnn/__init__.py for what that costs and why it is kept.
    """
    df = pd.read_csv(cfg.CLEANED_PATH, index_col=0, parse_dates=True).sort_index()
    ept = pd.DatetimeIndex(pd.to_datetime(df['Datetime_EPT']))
    return (df, ept, df['Load'].values.astype(float), df['Temp_F'].values.astype(float),
            df['is_holiday'].values.astype(float), df['is_valid'].values.astype(int))


def split_at_day_boundary(ept, test_frac):
    """Tail split, snapped back to an EPT midnight so no partial day sits at either end."""
    split = len(ept) - int(len(ept) * test_frac)
    while split > 0 and ept[split].hour != 0:
        split -= 1
    return split
