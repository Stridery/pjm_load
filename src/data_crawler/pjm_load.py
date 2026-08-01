"""Auto-fetch PJM metered + preliminary hourly load via the Dataminer2 CSV-download URL.

Replaces the manual `raw/metered/*.csv` / `raw/preliminary/*.csv` download step. The files
written here are byte-compatible with the manual ones, so pipeline.load_metered /
load_preliminary and everything downstream are unchanged.

The CSV-download endpoint (`format=csv&download=true`) returns a WHOLE YEAR in one request
(no pagination — a leap year's 8784 hourly rows come back in one shot), and its date filter
is an EPT range string: 'M/D/YYYY 00:00to12/31/YYYY 23:59'. load_area codes and the (public,
anonymous) subscription key are hardcoded in config: PJM_LOAD_FETCH / PJM_SUBSCRIPTION_KEY.
"""

import io
import logging
import os

import pandas as pd
import requests

from ._retry import with_retry

logger = logging.getLogger(__name__)

_URL = "https://api.pjm.com/api/v1/{endpoint}"
# Fields mirror the manual PJM CSV exports so the saved files keep the same schema.
_METERED_FIELDS = ("datetime_beginning_ept,datetime_beginning_utc,is_verified,load_area,"
                   "mkt_region,mw,nerc_region,zone")
_PRELIM_FIELDS  = ("datetime_beginning_ept,datetime_beginning_utc,datetime_ending_ept,"
                   "datetime_ending_utc,load_area,prelim_load_avg_hourly")


def _headers():
    from src.config import PJM_SUBSCRIPTION_KEY
    # Origin/Referer mirror the dataminer2 web app; the subscription key is what actually
    # authorises the call (a raw keyless fetch returns 401).
    return {
        "Ocp-Apim-Subscription-Key": PJM_SUBSCRIPTION_KEY,
        "Origin":  "https://dataminer2.pjm.com",
        "Referer": "https://dataminer2.pjm.com/",
        "Accept":  "application/json, text/plain, */*",
    }


@with_retry(max_attempts=4, backoff_base=3.0, exceptions=(requests.RequestException, ValueError))
def _get_csv(endpoint: str, fields: str, load_area: str, year: int) -> pd.DataFrame:
    params = {
        "sort": "datetime_beginning_utc", "order": "Asc", "startRow": 1,
        "isActiveMetadata": "true", "fields": fields,
        "datetime_beginning_ept": f"1/1/{year} 00:00to12/31/{year} 23:59",
        "load_area": load_area, "format": "csv", "download": "true",
    }
    r = requests.get(_URL.format(endpoint=endpoint), params=params, headers=_headers(), timeout=180)
    if r.status_code == 401:
        raise PermissionError(
            "PJM API returned 401 — subscription key rejected. Update PJM_SUBSCRIPTION_KEY in "
            "src/config.py (grab the current one from dataminer2.pjm.com DevTools → Network → "
            "any api.pjm.com request → 'Ocp-Apim-Subscription-Key')."
        )
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text)) if r.text.strip() else pd.DataFrame()


def fetch_metered(zone: str, year: int, raw_dir: str) -> None:
    """Fetch metered load for one (zone, year); overwrite/create raw/metered/hrl_load_metered_{year}.csv."""
    from src.config import PJM_LOAD_FETCH
    area    = PJM_LOAD_FETCH[zone]["metered_area"]
    met_dir = os.path.join(raw_dir, "metered")
    os.makedirs(met_dir, exist_ok=True)
    met = _get_csv("hrl_load_metered", _METERED_FIELDS, area, year)
    if not met.empty:
        met.to_csv(os.path.join(met_dir, f"hrl_load_metered_{year}.csv"), index=False)
        logger.info("  metered [%s %d]  %d rows", area, year, len(met))
    else:
        logger.warning("  metered [%s %d]  EMPTY — not written", area, year)


def fetch_prelim(zone: str, year: int, raw_dir: str) -> None:
    """Fetch preliminary load for one (zone, year); overwrite/create raw/preliminary/hrl_load_prelim_{year}.csv."""
    from src.config import PJM_LOAD_FETCH
    area    = PJM_LOAD_FETCH[zone]["prelim_area"]
    pre_dir = os.path.join(raw_dir, "preliminary")
    os.makedirs(pre_dir, exist_ok=True)
    pre = _get_csv("hrl_load_prelim", _PRELIM_FIELDS, area, year)
    if not pre.empty:
        pre.to_csv(os.path.join(pre_dir, f"hrl_load_prelim_{year}.csv"), index=False)
        logger.info("  prelim  [%s %d]  %d rows", area, year, len(pre))
    else:
        logger.warning("  prelim  [%s %d]  EMPTY — not written", area, year)


def fetch_load(zone: str, metered_years, prelim_years, raw_dir: str) -> None:
    """Fetch exactly the listed years for each source (overwrite/create their files)."""
    logger.info("=== Fetching PJM load [%s]  metered=%s  prelim=%s ===",
                zone, list(metered_years), list(prelim_years))
    for year in metered_years:
        fetch_metered(zone, year, raw_dir)
    for year in prelim_years:
        fetch_prelim(zone, year, raw_dir)
