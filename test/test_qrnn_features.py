"""The four things in the iQRNN baseline that would fail silently.

    PJM_DATASET=dom python test/test_qrnn_features.py

None of these would raise on their own. A lag off by one still trains, still scores, and
still produces a plausible table — it just answers a question one hour easier than the one
claimed. Likewise an objective that is not the metric: the run would converge nicely on
something the scoreboard does not measure.

    1  lag alignment      every load_lagH column IS load[t-H], and no H is below the floor
    2  split boundaries   train / validation / test partition the rows, at the stated dates
    3  loss == metric     the torch objective equals src.quantile.qeval.pinball
    4  crossing detector  zero on a sorted array, positive on a crossed one

Deliberately not covered: that RFE picks good features. That is a modelling outcome, not a
contract, and the run reports it (`rfe_features.json`, and the group counts it prints).
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qrnn import features as F                                   # noqa: E402
from src.qrnn import model as M                                      # noqa: E402
from src.quantile.qeval import QUANTILES, pinball                    # noqa: E402
from train_quantile import load_frame, split_at_day_boundary         # noqa: E402


def test_lag_alignment(lag_min=48):
    """Every lag column is the series shifted by exactly its own name, and none reaches
    into the blind spot. Checked against an index-valued series, where load[t] == t makes
    an off-by-one impossible to miss."""
    n = 3000
    ept = pd.DatetimeIndex(pd.date_range('2022-01-01', periods=n, freq='h'))
    load = np.arange(n, dtype=float)
    temp = np.arange(n, dtype=float) * -1.0
    hol = (np.arange(n) % 97 == 0).astype(float)

    X, y, rows = F.build(ept, load, temp, hol, load_lag_min=lag_min)
    names = F.candidate_names(lag_min)
    assert X.shape == (len(rows), len(names)), (X.shape, len(names))
    assert np.array_equal(y, load[rows]), "target is not the row's own load"

    checked = 0
    for j, nm in enumerate(names):
        if nm.startswith('load_lag'):
            h = int(nm[len('load_lag'):])
            assert h >= lag_min, f"{nm} reaches inside the {lag_min} h blind spot"
            assert np.array_equal(X[:, j], (rows - h).astype(np.float32)), nm
            checked += 1
        elif nm.startswith('temp_lag'):
            h = int(nm[len('temp_lag'):])
            assert np.array_equal(X[:, j], -(rows - h).astype(np.float32)), nm
            checked += 1
    assert checked == len(names) - len(F.CALENDAR_NAMES)

    # The floor is a real constraint, not a label: at lag_min the nearest observation is
    # exactly lag_min hours back, so the target hour is unreachable from anything nearer.
    nearest = min(int(nm[len('load_lag'):]) for nm in names if nm.startswith('load_lag'))
    assert nearest == lag_min, nearest
    print(f"  lag alignment      {checked} lag columns exact, nearest load lag {nearest} h")


def test_split_boundaries(val_days=365, test_frac=0.16):
    """The three periods partition the rows and start where they are said to start."""
    df, ept, load, temp, hol, valid = load_frame()
    split = split_at_day_boundary(ept, test_frac)
    test_start = ept[split].normalize()

    rows = np.arange(F.MIN_HISTORY, len(load))
    ept_r = ept[rows]
    tr, va, te = F.split_masks(ept_r, test_start, val_days)

    assert not (tr & va).any() and not (va & te).any() and not (tr & te).any(), "overlap"
    assert (tr | va | te).all(), "a row belongs to no period"
    assert ept_r[te][0] >= test_start > ept_r[va][-1], "test does not start at the split"
    assert ept_r[va][0] >= test_start - pd.Timedelta(days=val_days)
    assert ept_r[tr][-1] < ept_r[va][0], "training runs past the validation start"

    # The test period is the one every baseline in this repo shares; if this drifts, the
    # folded-in QRA and density rows stop being comparable and nothing else would say so.
    assert ept_r[te][0].normalize() == test_start, (ept_r[te][0], test_start)
    print(f"  split boundaries   train {tr.sum()} / valid {va.sum()} / test {te.sum()} h, "
          f"test opens {test_start.date()}")


def test_loss_equals_metric():
    """The trained objective and the reported score are the same number.

    The paper writes the pinball loss as a max of two branches and qeval writes it as a
    sign split; they are the same function, and this is where that is pinned down.
    """
    rng = np.random.default_rng(0)
    n = 512
    y = rng.normal(size=n) * 100 + 5000
    q = np.sort(y[None, :] + rng.normal(size=(len(QUANTILES), n)) * 300, axis=0)

    ref = float(pinball(y, q).mean())
    got = float(M.pinball_loss(torch.as_tensor(q.T, dtype=torch.float64),
                               torch.as_tensor(y, dtype=torch.float64),
                               torch.as_tensor(QUANTILES / 100.0, dtype=torch.float64)))
    assert abs(ref - got) < 1e-9, (ref, got)
    print(f"  loss == metric     {got:.10f} vs qeval {ref:.10f}")


def test_crossing_detector():
    """Zero on a monotone array; strictly positive once two levels are swapped."""
    rng = np.random.default_rng(1)
    q = np.sort(rng.normal(size=(len(QUANTILES), 200)) * 100 + 5000, axis=0)
    frac, cs = M.crossing_rate(q)
    assert frac == 0.0 and cs == 0.0, (frac, cs)

    bad = q.copy()
    bad[[40, 41], :3] = bad[[41, 40], :3]          # swap two adjacent levels on three hours
    frac, cs = M.crossing_rate(bad)
    assert frac == 3 / 200 and cs > 0, (frac, cs)

    # Sorting is what every reported score runs on, and it must not move a clean forecast.
    assert np.array_equal(np.sort(bad, axis=0), q)
    print(f"  crossing detector  clean 0.0, seeded {100 * frac:.1f}% of hours, CS {cs:.3f}")


def main():
    print("=== iQRNN baseline contracts ===")
    test_lag_alignment()
    test_lag_alignment(lag_min=24)         # the QRA-issue-time setting must hold too
    test_split_boundaries()
    test_loss_equals_metric()
    test_crossing_detector()
    print("all passed")


if __name__ == '__main__':
    main()
