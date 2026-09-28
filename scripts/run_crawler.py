"""
Run the data crawler pipeline.

Usage
-----
    python scripts/run_crawler.py                        # uses config.py defaults (zone=dom, 2021-2026)
    python scripts/run_crawler.py --zone dom2            # specify zone folder
    python scripts/run_crawler.py --zone dom2 --start 2022 --end 2023
    python scripts/run_crawler.py --zone dom2 --location Richmond --no-skip
"""

# Entry points live in scripts/ but import the package from the repo root, and running
# `python scripts/foo.py` puts scripts/ on sys.path rather than the root. Prepend the root so
# `import src...` resolves from any working directory.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

from src.data_crawler import run_pipeline

parser = argparse.ArgumentParser(description="PJM + Open-Meteo data crawler")
parser.add_argument("--zone",     type=str, help="Zone folder name, e.g. dom2 (default: from config)")
parser.add_argument("--start",    type=int, help="Start year (default: auto-detect from files)")
parser.add_argument("--end",      type=int, help="End year   (default: auto-detect from files)")
parser.add_argument("--location", type=str, help="City name for Open-Meteo geocoding (default: from config)")
parser.add_argument("--no-skip",  action="store_true", help="Re-fetch weather even if cached files exist")
parser.add_argument("--keep-matrix", action="store_true",
                    help="Do NOT wipe data/{zone}/matrix* after the crawl (default: wipe, so the "
                         "next training run rebuilds matrices from the fresh cleaned CSV)")
args = parser.parse_args()

df = run_pipeline(
    zone=args.zone,
    start_year=args.start,
    end_year=args.end,
    location_name=args.location,
    skip_existing=not args.no_skip,
    clear_matrix=not args.keep_matrix,
)

print(f"\nDone. Output shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
