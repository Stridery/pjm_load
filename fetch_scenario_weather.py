"""
Fetch the long temperature history the weather-scenario baseline needs.

    PJM_DATASET=dom python fetch_scenario_weather.py            # 1995-2024, ~30 requests
    PJM_DATASET=bge python fetch_scenario_weather.py --start 1990 --end 2024
    PJM_DATASET=dom python fetch_scenario_weather.py --refresh  # ignore the cache

Writes data/{zone}/raw/weather_long/temp_{year}.csv, one column of hourly Temp_F on naive
local time, cached per year. train_quantile.py picks the directory up automatically.

Why this exists
---------------
Xie & Hong (2018) generate temperature scenarios from THIRTY years of history (1976-2005)
and their headline findings are all functions of that length: the quantile score falls with
k and flattens past ~15 years for the fixed-date method, and the rule of thumb
n = ceil(30/k - 1) exists to compensate a short history by shifting dates.

The load record here starts 2021, so scenarios drawn from it cap out at k=4 — four points on
a curve whose shape is the paper's actual result. Temperature does not have to come from the
load record, though; the paper's own weather history (36 years) is far longer than its load
history (10 years), and ERA5 on the same endpoint the crawler already uses reaches back to
1940. So the cap was self-imposed, and this removes it.

The scenario pool must still end strictly BEFORE the test period; that rule is enforced in
train_quantile.usable_scenario_years, not here.
"""

import argparse
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")

DEFAULT_START, DEFAULT_END = 1995, 2024
DIRNAME = 'weather_long'


def scenario_weather_dir(zone, data_root='data'):
    return os.path.join(data_root, zone.lower(), 'raw', DIRNAME)


def load_long_temperature(zone, data_root='data'):
    """(DatetimeIndex, np.ndarray) of hourly Temp_F across every cached year, or None."""
    import glob
    import numpy as np
    files = sorted(glob.glob(os.path.join(scenario_weather_dir(zone, data_root), 'temp_*.csv')))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, index_col=0, parse_dates=True) for f in files])
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df = df[df['Temp_F'].notna()]
    return pd.DatetimeIndex(df.index), df['Temp_F'].to_numpy(dtype=float)


def main():
    import src.config as cfg
    from src.data_crawler import open_meteo as om

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start', type=int, default=DEFAULT_START)
    ap.add_argument('--end', type=int, default=DEFAULT_END)
    ap.add_argument('--refresh', action='store_true', help='re-fetch years already cached')
    a = ap.parse_args()

    zone = cfg.DATASET
    out = scenario_weather_dir(zone)
    os.makedirs(out, exist_ok=True)
    loc = cfg.CRAWLER_CONFIG['location_name']
    tz = cfg.CRAWLER_CONFIG['timezone']
    lat, lon = om.geocode(loc)
    print(f"=== scenario temperature | zone {zone} | {loc} ({lat:.4f}, {lon:.4f}) ===")

    n_new = 0
    for year in range(a.start, a.end + 1):
        path = os.path.join(out, f'temp_{year}.csv')
        if os.path.exists(path) and not a.refresh:
            continue
        om.fetch_temperature_year(lat, lon, year, timezone=tz).to_csv(path)
        n_new += 1

    idx, temp = load_long_temperature(zone)
    per_year = pd.Series(temp, index=idx).groupby(idx.year).agg(['count', 'mean'])
    short = per_year[per_year['count'] < 8000]
    print(f"  {n_new} year(s) fetched, {len(per_year)} cached: {per_year.index.min()}"
          f"-{per_year.index.max()}, {len(idx)} hourly rows")
    if len(short):
        print(f"  incomplete years (excluded from any k): {list(short.index)}")
    print(f"  annual mean temperature {per_year['mean'].iloc[0]:.1f}F ({per_year.index[0]}) "
          f"-> {per_year['mean'].iloc[-1]:.1f}F ({per_year.index[-1]})")
    print(f"  -> {out}")


if __name__ == '__main__':
    main()
