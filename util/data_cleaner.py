import sys
import os


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


if project_root not in sys.path:
    sys.path.append(project_root)


import src.config as cfg
from src.data_processor import clean_and_engineer


df_cleaned = clean_and_engineer(cfg.MERGED_PATH, cfg.CLEANED_PATH)