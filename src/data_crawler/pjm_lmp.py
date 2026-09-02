"""Auto-fetch PJM zonal hourly LMP — Day-Ahead and Real-Time.

Same shape as pjm_marginal_value.py: one request per feed per year against the Dataminer2
CSV-download endpoint, cached per year under raw/lmp/, then filtered to the
zone and stitched into ONE hourly table in raw/lmp.csv.

Filtered by type=*zone*, and by pricing node CLIENT-SIDE
--------------------------------------------------------
Three filters look plausible and two of them are wrong:

  zone=DOM      returns every BUS inside the zone — 1788 rows an hour for DOM, 265 for BGE
                (measured) — and EXCLUDES the row actually wanted: the zonal aggregate carries
                type=ZONE and a NULL zone field, so the zone filter drops it. A year is ~15.7M
                rows of bus-level prices.
  pnode_name    not a valid filter field on these feeds; the API rejects it by name.
  pnode_id      works, but ONLY on standard data. Anything older is served from the archive,
                which rejects pnode_id, `fields` and custom sort outright:
                    "The API request contains invalid attribute(s) for archived data - Pnode_Id"

  type=*zone*   accepted on BOTH archived and standard data, and returns all 23 zonal
                aggregates — 552 rows a day, ~201k a year. Small enough to keep whole.

So the request is uniform across the archive boundary and the pricing node is selected when
the shards are stitched, not when they are fetched. Keeping all 23 zones on disk also means
adding a third zone later needs no re-fetch — the same argument pjm_marginal_value.py makes
for storing the RTO-wide table per zone.

The archive boundary is a ROLLING one (it moves with data age), so it is never hardcoded:
requests are chunked by month, and a month that straddles it comes back as
"Date range spans over archived and standard data" and is split in half and retried.

Note the zone code differs from the load feeds: BGE's LMP pnode is named 'BGE', while
config.PJM_LOAD_FETCH calls the same zone 'BC' for metered load. They are different
identifier spaces and neither is derivable from the other, so both are spelled out.

Day-Ahead and Real-Time carry different information because they arrive at different times
-----------------------------------------------------------------------------------------
A sample cuts off at 00:00 of day d and predicts d+1, so day d is a blind spot the 168 h
lookback cannot see. The Day-Ahead market for day d cleared at ~13:30 EPT on d-1, so DA
prices for the blind-spot day ARE available at the cutoff — the same availability argument
pjm_marginal_value.py makes for the congestion block. Real-Time prices for day d are realised
during day d, after the cutoff, so RT can only ever describe the lookback.

That is why this module fetches both but the feature layer uses them differently: RT as a
per-hour lookback channel, DA as a blind-spot-day static block.

Three components, not one
-------------------------
total_lmp = system_energy_price + congestion_price + marginal_loss_price, and the three behave
differently enough to be worth keeping apart:
  system_energy  RTO-wide and identical across zones — fuel cost and system-wide tightness.
                 For the blind-spot day this is the market pricing PJM's OWN day-ahead load
                 forecast, which is forward-looking information the lookback cannot contain.
  congestion     the zonal差, and largely the same signal the cong_* block already carries
                 (both descend from the constraint shadow prices).
  marginal_loss  small.

All of them are heavy-tailed (measured on dom 2025-26: total_lmp_da median 45, p99 512, max
1877) and the congestion component goes NEGATIVE (22% of RT hours), so any downstream use
needs a SIGNED log — sign(x)*log1p(|x|) — not a plain log1p. That transform is left to the
feature layer; this module stores raw prices.
"""

import glob
import io
import logging
import os

import pandas as pd
import requests

from ._retry import with_retry

logger = logging.getLogger(__name__)

_URLS = {"da": "https://api.pjm.com/api/v1/da_hrl_lmps",
         "rt": "https://api.pjm.com/api/v1/rt_hrl_lmps"}

# Where the all-zones shards land. ONE shared directory, not data/{zone}/raw/: the shards hold
# all 23 zonal nodes and are byte-identical whatever zone asked for them, so a per-zone copy
# would be ~230 MB of duplicate download. They are a CACHE — build_lmp deletes them once each
# zone's own raw/lmp.csv has been written, which is the durable artifact.
LMP_CACHE_DIR = "lmp"

# Zonal aggregate pricing nodes (type=ZONE), used to pick a zone out of the stitched table.
# Verified against the feed, not guessed. Note these are a DIFFERENT identifier space from the
# load feeds: config.PJM_LOAD_FETCH calls the Baltimore zone 'BC' for metered load, while its
# LMP pricing node is named 'BGE'. Neither is derivable from the other.
ZONE_PNODE = {"dom": 34964545, "bge": 51292}
ZONE_LMP_NAME = {"dom": "DOM", "bge": "BGE"}

# `fields` is rejected on archived data, so it is never sent — the feed's full column set comes
# back either way and the unwanted columns are dropped when the shards are stitched.
_KEEP = ("datetime_beginning_utc", "datetime_beginning_ept", "pnode_id", "pnode_name", "type")


def _headers():
    from src.config import PJM_SUBSCRIPTION_KEY
    return {
        "Ocp-Apim-Subscription-Key": PJM_SUBSCRIPTION_KEY,
        "Origin":  "https://dataminer2.pjm.com",
        "Referer": "https://dataminer2.pjm.com/",
        "Accept":  "application/json, text/plain, */*",
    }


@with_retry(max_attempts=4, backoff_base=3.0, exceptions=(requests.RequestException,))
def _request(kind: str, lo: str, hi: str):
    """One HTTP call. Retries only NETWORK failures — a 400 is an answer, not a hiccup."""
    params = {
        "startRow": 1, "rowCount": 50000, "isActiveMetadata": "true",
        "type": "*zone*",
        "datetime_beginning_ept": f"{lo} to {hi}",
        "format": "csv", "download": "true",
    }
    r = requests.get(_URLS[kind], params=params, headers=_headers(), timeout=300)
    if r.status_code == 401:
        raise PermissionError(
            "PJM API returned 401 — subscription key rejected. Update PJM_SUBSCRIPTION_KEY in "
            "src/config.py (grab the current one from dataminer2.pjm.com DevTools -> Network -> "
            "any api.pjm.com request -> 'Ocp-Apim-Subscription-Key')."
        )
    return r


def _get_range(kind: str, lo: str, hi: str) -> pd.DataFrame:
    """[lo, hi] EPT, splitting itself where it must. Deliberately NOT retry-wrapped.

    Splitting and retrying have to stay separate. When they were one function, a range that
    could not be narrowed raised ValueError, @with_retry caught it, and the whole recursion ran
    again -- four times over, at every level. The fetch thrashed instead of failing.

    Two reasons a range gets split:
      - it straddles the archive boundary, which PJM rolls with data age and reports as
        "Date range spans over archived and standard data". That boundary does NOT fall on a
        day line, so splitting has to go down to ONE HOUR, not one day.
      - it hits the 50000-row page cap.
    """
    r = _request(kind, lo, hi)
    straddles = r.status_code == 400 and "spans over archived and standard" in r.text
    if not straddles:
        r.raise_for_status()
        if not r.text.strip():
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        if len(df) < 50000:
            return df
        logger.info("  %s %s..%s hit the row cap - splitting", kind, lo[:13], hi[:13])

    a_, b_ = pd.Timestamp(lo), pd.Timestamp(hi)
    if (b_ - a_) <= pd.Timedelta(hours=1):
        raise RuntimeError(f"{kind}: {lo}..{hi} is one hour and still cannot be served "
                           f"({'archive straddle' if straddles else 'row cap'}).")
    mid = (a_ + (b_ - a_) / 2).floor("h")
    if mid <= a_:
        mid = a_ + pd.Timedelta(hours=1)
    if straddles:
        logger.info("  %s %s..%s straddles the archive boundary - splitting at %s",
                    kind, lo[:13], hi[:13], mid.strftime("%Y-%m-%d %H:%M"))
    fmt = "%Y-%m-%dT%H:%M:%S.0000000"
    return pd.concat(
        [_get_range(kind, lo, (mid - pd.Timedelta(seconds=1)).strftime(fmt)),
         _get_range(kind, mid.strftime(fmt), hi)],
        ignore_index=True)


def _get_year(kind: str, year: int) -> pd.DataFrame:
    """A year, fetched month by month. A month of 23 zones is ~17k rows, well under the cap."""
    out = []
    for m in range(1, 13):
        lo = pd.Timestamp(year=year, month=m, day=1)
        hi = (lo + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)
        df = _get_range(kind, lo.strftime("%Y-%m-%dT00:00:00.0000000"),
                        hi.strftime("%Y-%m-%dT%H:%M:%S.0000000"))
        if len(df):
            out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def fetch_lmp(years, data_root: str = "data") -> None:
    """Fetch the listed years for both feeds -> {data_root}/lmp/{da,rt}_lmp_{year}.csv.

    Zone-independent: one request set serves every zone, because type=*zone* brings back all 23
    zonal nodes at once. Selecting a zone happens in clean_lmp.
    """
    out_dir = os.path.join(data_root, LMP_CACHE_DIR)
    os.makedirs(out_dir, exist_ok=True)
    logger.info("=== Fetching PJM zonal LMP (all 23 zones)  years=%s ===", list(years))
    for kind in ("da", "rt"):
        for year in years:
            df = _get_year(kind, year)
            if df.empty:
                logger.warning("  %s_lmp [%d]  EMPTY — not written", kind, year)
                continue
            df.to_csv(os.path.join(out_dir, f"{kind}_lmp_{year}.csv"), index=False)
            logger.info("  %s_lmp [%d]  %d rows", kind, year, len(df))


def clean_lmp(zone: str, data_root: str, out_path: str) -> pd.DataFrame:
    """Stitch the per-year shards into ONE hourly table -> raw/lmp.csv.

    Lands in raw/, not cleaned/: nothing here is aligned to the load frame, gap-filled or
    transformed yet — it is the zone's slice of the feed and no more. The signed-log transform
    the heavy tails need, and the join onto the hourly index, belong to the layers above.

    EVERY column the feed returns is carried through. Nothing is dropped and nothing is
    renamed, so this file can answer a question the feature layer has not thought of yet
    (voltage, equipment, row_is_current, version_nbr all survive). The only columns that gain
    a suffix are the ones both feeds define — the DA copy keeps the bare name and the RT copy
    becomes *_rt_dup — because dropping either would be exactly the silent loss this is
    written to avoid.

    Indexed by UTC, not EPT: this table gets joined hour-by-hour onto a UTC index later, and
    the EPT hour is ambiguous on the fall-back day (01:00 appears twice), which is precisely
    the join that would silently take the wrong row.
    """
    # Selected by pnode_name, not pnode_id: the name is what the feed calls the zone and what a
    # reader can check against the data ('DOM', 'BGE'), while the id is an opaque integer. Note
    # it is a DIFFERENT identifier from the load feeds, where config.PJM_LOAD_FETCH calls the
    # Baltimore zone 'BC'.
    name = ZONE_LMP_NAME[zone.lower()]
    mv_dir = os.path.join(data_root, LMP_CACHE_DIR)

    frames = {}
    for kind in ("da", "rt"):
        files = sorted(glob.glob(os.path.join(mv_dir, f"{kind}_lmp_*.csv")))
        if not files:
            logger.warning("No %s LMP shards in %s", kind, mv_dir)
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df = df[df["pnode_name"].astype(str).str.strip() == name]
        if df.empty:
            raise ValueError(
                f"No pnode_name={name!r} rows in {mv_dir}/{kind}_lmp_*.csv — the shards hold "
                f"other zones only. Re-run fetch_lmp.")
        df["Datetime_UTC"] = pd.to_datetime(df["datetime_beginning_utc"], utc=True)
        # keep='last': a re-fetched year supersedes an older shard for the same hour.
        frames[kind] = (df.sort_values("Datetime_UTC")
                          .drop_duplicates("Datetime_UTC", keep="last")
                          .set_index("Datetime_UTC"))

    if not frames:
        raise FileNotFoundError(f"No LMP shards under {mv_dir} — run fetch_lmp first.")

    out = frames.get("da")
    if out is None:
        out = frames["rt"]
    elif "rt" in frames:
        out = out.join(frames["rt"], how="outer", rsuffix="_rt_dup")
    out = out.sort_index()

    # MERGE into whatever is already there. The shared shards are a cache that gets deleted
    # after every run, so this per-zone file is the only durable store of the history — a
    # routine crawl fetches just UPDATE_YEARS and would otherwise truncate five years of prices
    # to two. Freshly fetched hours win; untouched years survive.
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path, index_col=0, parse_dates=True)
        prev.index = pd.to_datetime(prev.index, utc=True)
        before = len(prev)
        out = pd.concat([prev[~prev.index.isin(out.index)], out]).sort_index()
        logger.info("LMP %s: merged %d refreshed hour(s) into %d existing",
                    name, len(out) - (before - prev.index.isin(out.index).sum()), before)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path)
    logger.info("LMP %s: %d hourly rows, %d columns (%s → %s) → %s",
                name, len(out), len(out.columns), out.index.min(), out.index.max(), out_path)
    return out


def clear_cache(data_root: str = "data") -> None:
    """Delete the shared all-zones shards. Called once every zone has been cleaned."""
    d = os.path.join(data_root, LMP_CACHE_DIR)
    n = len(glob.glob(os.path.join(d, "*.csv")))
    for f in glob.glob(os.path.join(d, "*.csv")):
        os.remove(f)
    if os.path.isdir(d) and not os.listdir(d):
        os.rmdir(d)
    logger.info("LMP cache cleared: %d shard(s) removed from %s", n, d)


def build_lmp(zones, years, data_root: str = "data", keep_cache: bool = False):
    """fetch once -> filter every zone -> drop the shared cache. The whole LMP pipeline.

    Every zone is refiltered on every run, not just the one that triggered it: the shards are
    zone-independent, so once they are on disk the filtering is nearly free, and leaving the
    other zone's file stale would mean its price history silently stopped at whenever it last
    ran its own crawl.
    """
    fetch_lmp(years, data_root)
    out = {}
    for z in zones:
        out[z] = clean_lmp(z, data_root, os.path.join(data_root, z, "raw", "lmp.csv"))
    if not keep_cache:
        clear_cache(data_root)
    return out


# ---------------------------------------------------------------------------
# Splitting the zone table into its two consumers
# ---------------------------------------------------------------------------
# Real-Time and Day-Ahead prices go to different places because they are available at
# different times, which is the same split pjm_marginal_value.py argues for:
#
#   RT  is realised DURING the operating day, so it can only ever describe the LOOKBACK.
#       It is joined into the cleaned load/weather frame as three more hourly channels,
#       exactly like the weather columns, and travels with them from then on.
#   DA  for the blind-spot day cleared at ~13:30 EPT the day before the cutoff, so it is the
#       one price that reaches into the gap the 168 h lookback cannot see. It goes to its own
#       per-day file, like cleaned/congestion.csv and cleaned/forecast.csv, and is read by the
#       static feature block rather than the sequence.
#
# Raw prices are carried through untransformed. The signed log the heavy tails need
# (sign(x)*log1p(|x|) — plain log1p is NaN on the 22% of RT congestion hours that are negative)
# belongs to the feature layer, where thermal_features already does the same thing.
RT_CHANNELS = {
    "total_lmp_rt":           "LMP_RT_Total",    # the zone's realised price
    "system_energy_price_rt": "LMP_RT_Energy",   # RTO-wide component — system scarcity
    "congestion_price_rt":    "LMP_RT_Cong",     # the zone's congestion premium
}
DA_COLUMNS = ["total_lmp_da", "system_energy_price_da",
              "congestion_price_da", "marginal_loss_price_da"]


def load_rt_channels(zone_lmp_path: str) -> pd.DataFrame:
    """raw/lmp.csv -> UTC-indexed frame of just the three RT lookback channels."""
    df = pd.read_csv(zone_lmp_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    missing = [c for c in RT_CHANNELS if c not in df.columns]
    if missing:
        raise KeyError(f"{zone_lmp_path} has no {missing} — re-run the LMP fetch.")
    return df[list(RT_CHANNELS)].rename(columns=RT_CHANNELS).sort_index()


def write_da_table(zone_lmp_path: str, out_path: str) -> pd.DataFrame:
    """raw/lmp.csv -> cleaned/lmp_da.csv: the Day-Ahead hourly prices, on their own index.

    Kept out of the load/weather frame for the reason clean_marginal_value gives for
    congestion: it is read per DAY by a static block, not per hour by the sequence, and folding
    it in would change the cleaned CSV that every existing model was trained on.
    """
    df = pd.read_csv(zone_lmp_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    keep = [c for c in DA_COLUMNS if c in df.columns]
    if not keep:
        raise KeyError(f"{zone_lmp_path} has none of {DA_COLUMNS} — re-run the LMP fetch.")
    ept_col = "datetime_beginning_ept"
    out = df[([ept_col] if ept_col in df.columns else []) + keep].sort_index()
    out.index.name = "Datetime_UTC"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path)
    logger.info("LMP Day-Ahead: %d hourly rows, %d price column(s) → %s",
                len(out), len(keep), out_path)
    return out
