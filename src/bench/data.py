"""Datasets, the day set, and the split — one place, so all three baselines agree.

The unit of the protocol is a DAY, not a step. A target day is usable when three things hold
at once, and the third is the one that is easy to get wrong:

    1. all H of its steps are present in the record;
    2. the whole lookback window ending at its deadline is present too;
    3. all H of its steps lie inside the split's date range.

Rule 3 is why elia's test set is 746 days rather than 747: its `test_end` is 23:00, but a
15-minute day's last step is 23:45, so the final day is incomplete and is dropped. The three
counts this produces reproduce the chapter's table exactly:

    spain  869 / 292 / 291      ercot  416 / 142 / 140      elia  2239 / 748 / 746

The record is checked to be strictly regular (no gaps, no duplicate stamps) on load, which is
what lets every lag be a positional shift rather than a timestamp join.
"""

import json
import os

import numpy as np
import pandas as pd

CONFIG_DIR = 'data/quantile'
DATASETS = ('spain', 'ercot', 'elia')


class Dataset:
    """The cleaned record plus everything the protocol derives from its config."""

    def __init__(self, name, config_dir=CONFIG_DIR):
        self.name = name
        with open(os.path.join(config_dir, f'{name}_config.json')) as fh:
            self.cfg = cfg = json.load(fh)

        df = pd.read_csv(cfg['csv_path'], index_col=0)
        idx = pd.to_datetime(df.index, utc=True)
        step = pd.Timedelta(minutes=cfg['granularity_minutes'])
        d = np.unique(np.diff(idx.values))
        if len(d) != 1 or pd.Timedelta(d[0]) != step:
            raise ValueError(f"{name}: record is not on a regular {step} grid: {d}")

        self.index = idx
        self.frame = df
        self.y = df[cfg['target_col']].to_numpy(float)
        if not np.isfinite(self.y).all():
            raise ValueError(f"{name}: target column has non-finite values")

        # Column classes. The target is excluded from past_cols even though the configs list
        # it inside actual_cols, and so are the calendar codes, which are regenerated from the
        # index rather than read as columns so that every dataset encodes them identically.
        self.target_col = cfg['target_col']
        self.aux_cols = list(cfg['aux_cols'])
        self.past_cols = [c for c in cfg['actual_cols']
                          if c != self.target_col and c not in self.aux_cols]
        self.known_cols = list(cfg['forecast_cols'])
        missing = [c for c in self.past_cols + self.known_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: configured columns absent from the CSV: {missing}")

        self.step_minutes = cfg['granularity_minutes']
        self.steps_per_hour = 60 // self.step_minutes
        self.H = cfg['forecast_hours'] * self.steps_per_hour          # steps in a target day
        lb = cfg['lookback_hours']
        self.lookback_hours = min(lb) if isinstance(lb, list) else lb  # spain lists [336,168]
        self.L = self.lookback_hours * self.steps_per_hour            # steps in the window
        # Steps from the deadline step to the target day's first step. The deadline hour is
        # the last FULLY observed hour of the day before, so the gap is one full hour more
        # than the hours remaining after it.
        self.G = (24 - cfg['forecast_deadline_hour']) * self.steps_per_hour

        self.lag_min = self.G + self.H - 1
        self.lag_max = self.L - 1 + self.G

    # -- day set ---------------------------------------------------------------------

    def day_starts(self):
        """Positions of the first step of every day that satisfies rules 1 and 2."""
        first = np.flatnonzero((self.index.hour == 0) & (self.index.minute == 0))
        earliest = first - self.G - self.L + 1            # first step of the window
        ok = (earliest >= 0) & (first + self.H <= len(self.index))
        # Rule 1 in full: the H stamps really are one contiguous day.
        starts = first[ok]
        span = self.index[starts + self.H - 1] - self.index[starts]
        return starts[span == pd.Timedelta(minutes=self.step_minutes * (self.H - 1))]

    def splits(self):
        """{'train','val','test'} -> day-start positions, applying rule 3."""
        starts = self.day_starts()
        t0, tN = self.index[starts], self.index[starts + self.H - 1]
        out = {}
        for part in ('train', 'val', 'test'):
            a = pd.Timestamp(self.cfg[f'{part}_start'])
            b = pd.Timestamp(self.cfg[f'{part}_end'])
            a = a.tz_localize('UTC') if a.tzinfo is None else a.tz_convert('UTC')
            b = b.tz_localize('UTC') if b.tzinfo is None else b.tz_convert('UTC')
            out[part] = starts[(t0 >= a) & (tN <= b)]
        return out

    def steps_of(self, day_starts):
        """(n_days * H,) step positions of those days, in chronological order."""
        day_starts = np.asarray(day_starts, dtype=int)
        return (day_starts[:, None] + np.arange(self.H)[None, :]).ravel()

    # -- standardisation -------------------------------------------------------------

    def standardiser(self, train_starts):
        """(mean, sd) of the target over the training days — the reporting scale."""
        v = self.y[self.steps_of(train_starts)]
        return float(v.mean()), float(v.std())

    def column(self, name):
        """One configured column as a float array over the whole record.

        Non-finite values are carried through rather than filled: a NaN reaching the network
        is a loud failure, whereas a silently imputed one is a quiet bias. All three records
        are clean at the columns these baselines use, which is checked here.
        """
        v = self.frame[name].to_numpy(float)
        if not np.isfinite(v).all():
            raise ValueError(f"{self.name}: column {name!r} has "
                             f"{int((~np.isfinite(v)).sum())} non-finite values")
        return v

    def describe(self):
        s = self.splits()
        return (f"{self.name:6s} step {self.step_minutes:2d} min | H={self.H:3d} L={self.L:3d} "
                f"G={self.G:2d} | lags [{self.lag_min}, {self.lag_max}] | days "
                f"train {len(s['train'])} val {len(s['val'])} test {len(s['test'])}")


def load(name, config_dir=CONFIG_DIR):
    return Dataset(name, config_dir)
