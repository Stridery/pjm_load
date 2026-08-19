"""Auto-fetch PJM Day-Ahead marginal value (transmission-constraint shadow prices).

Same shape as pjm_load.py: one request per year against the Dataminer2 CSV-download endpoint
(no pagination — a year is ~76k rows and comes back in one shot), cached per year under
raw/marginal_price/, then stitched into ONE hourly table in cleaned/congestion.csv.

Day-Ahead only
--------------
rt_marginal_value is deliberately not fetched. Measured on this record, DA beats RT on every
metric (drift R2 0.086 vs 0.070; higher correlation with load in all six years), and — the
decisive part — RT cannot reach the day the features actually use. A sample cuts off at 00:00
of day d and predicts d+1, so day d is a blind spot the 168 h lookback cannot see. DA for day
d cleared at ~13:30 EPT on d-1 and is available; RT for day d is realised DURING day d, after
the cutoff. The signal decays sharply exactly there (R2 0.085 at lag 24 h -> 0.050 at 48 h),
so the one day that carries it is the one day RT cannot supply.

RTO-wide, not zone-filtered
---------------------------
The rows carry no zone field, only a monitored/contingency facility pair, and filtering them
by facility geography measures the wrong thing: congestion is a NETWORK phenomenon, so the
binding element sits wherever the flows bind, not where the load causing them is. Measured,
a hand-curated Virginia facility set scores 5x worse than the plain RTO-wide aggregate
(R2 0.019 vs 0.117) and is zero on 66% of days, while the RTO-wide aggregate is zero on 0%.
So no facility mapping is built, and the same RTO-wide table serves every zone.

That does mean the file is byte-identical between zones. It is still written per zone (~2 MB)
rather than to a shared directory, so that every data/{zone}/ path convention — and the
matrix-cache wipe in run_pipeline — keeps working untouched.
"""

import glob
import io
import logging
import os

import pandas as pd
import requests

from ._retry import with_retry

logger = logging.getLogger(__name__)

_URL = "https://api.pjm.com/api/v1/da_marginal_value"
_FIELDS = ("datetime_beginning_ept,datetime_beginning_utc,monitored_facility,"
           "contingency_facility,shadow_price")


def _headers():
    from src.config import PJM_SUBSCRIPTION_KEY
    return {
        "Ocp-Apim-Subscription-Key": PJM_SUBSCRIPTION_KEY,
        "Origin":  "https://dataminer2.pjm.com",
        "Referer": "https://dataminer2.pjm.com/",
        "Accept":  "application/json, text/plain, */*",
    }


@with_retry(max_attempts=4, backoff_base=3.0, exceptions=(requests.RequestException, ValueError))
def _get_year(year: int) -> pd.DataFrame:
    params = {
        "sort": "datetime_beginning_ept", "order": "Asc", "startRow": 1,
        "isActiveMetadata": "true", "fields": _FIELDS,
        "datetime_beginning_ept": (f"{year}-01-01T00:00:00.0000000 to "
                                   f"{year}-12-31T23:59:59.0000000"),
        "format": "csv", "download": "true",
    }
    r = requests.get(_URL, params=params, headers=_headers(), timeout=300)
    if r.status_code == 401:
        raise PermissionError(
            "PJM API returned 401 — subscription key rejected. Update PJM_SUBSCRIPTION_KEY in "
            "src/config.py (grab the current one from dataminer2.pjm.com DevTools → Network → "
            "any api.pjm.com request → 'Ocp-Apim-Subscription-Key')."
        )
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text)) if r.text.strip() else pd.DataFrame()


def fetch_marginal_value(years, raw_dir: str) -> None:
    """Fetch the listed years, overwriting/creating raw/marginal_price/da_marginal_value_{year}.csv."""
    mv_dir = os.path.join(raw_dir, "marginal_price")
    os.makedirs(mv_dir, exist_ok=True)
    logger.info("=== Fetching PJM DA marginal value  years=%s ===", list(years))
    for year in years:
        df = _get_year(year)
        if df.empty:
            logger.warning("  da_marginal_value [%d]  EMPTY — not written", year)
            continue
        df.to_csv(os.path.join(mv_dir, f"da_marginal_value_{year}.csv"), index=False)
        logger.info("  da_marginal_value [%d]  %d rows", year, len(df))


def clean_marginal_value(raw_dir: str, out_path: str) -> pd.DataFrame:
    """Stitch the per-year shards into ONE hourly table -> cleaned/congestion.csv.

    Index is the EPT hour (naive local, matching datetime_beginning_ept); columns are the
    three aggregates that survived the ablation as non-redundant:

        n        how many constraints bound in that hour   (breadth)
        sum_sp   sum of |shadow_price| over them           (total depth)
        max_sp   the largest single |shadow_price|         (worst single constraint)

    RTO-wide the largest constraint is only ~45% of the hour's total (median), so max and
    sum genuinely differ; zone-wide they would be near-identical, which is one more reason
    the aggregate is taken RTO-wide.

    Shadow prices are published negative by convention (the cost of the binding constraint);
    the absolute value is taken here so every downstream feature is a magnitude.

    Only hours that HAD a binding constraint produce a row. Absence is not missingness —
    "nothing bound this hour" is a real state — and the feature layer reindexes each day to
    its own hours and fills zero. What absence of an entire DAY means is different, and the
    feature layer treats that as missing: RTO-wide, 100% of days in 2021-2026 have at least
    one binding constraint, so a day with no rows at all is a failed fetch, not a quiet grid.
    """
    files = sorted(glob.glob(os.path.join(raw_dir, "marginal_price", "da_marginal_value_*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No da_marginal_value_*.csv under {os.path.join(raw_dir, 'marginal_price')} — "
            f"run the crawler (fetch_marginal_value) first.")

    frames = [pd.read_csv(f, usecols=["datetime_beginning_ept", "shadow_price"]) for f in files]
    df = pd.concat(frames, ignore_index=True)
    # 'mixed' because PJM writes M/D/YYYY h:MM:SS AM — a single inferred format mis-reads the
    # single-digit-month rows.
    t = pd.to_datetime(df["datetime_beginning_ept"], format="mixed").dt.floor("h")
    sp = df["shadow_price"].abs()

    g = pd.DataFrame({"t": t, "sp": sp}).groupby("t")["sp"]
    out = pd.DataFrame({"n": g.size(), "sum_sp": g.sum(), "max_sp": g.max()}).sort_index()
    out.index.name = "Datetime_EPT"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path)
    days = out.index.normalize().nunique()
    logger.info("Congestion (DA marginal value): %d binding hour(s) over %d day(s), %s → %s  →  %s",
                len(out), days, out.index[0], out.index[-1], out_path)
    return out
