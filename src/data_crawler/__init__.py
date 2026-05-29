"""
data_crawler — automated fetching of PJM load and Open-Meteo weather data.

Public API
----------
    from src.data_crawler import run_pipeline
    run_pipeline()            # uses config.CRAWLER_CONFIG defaults
    run_pipeline(start_year=2022, end_year=2023)
"""

from .pipeline import run_pipeline

__all__ = ["run_pipeline"]
