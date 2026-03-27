import src.config as cfg
from src.data_processor import merge_raw_data, clean_and_engineer
from src.feature_engine import build_or_load_matrix, build_timeseries_matrix

df_merged = merge_raw_data(cfg.RAW_LOAD_PATH, cfg.RAW_WEATHER_PATH, cfg.MERGED_PATH)
df_cleaned = clean_and_engineer(cfg.MERGED_PATH, cfg.CLEANED_PATH)
X_opt, y_opt = build_or_load_matrix(cfg.CLEANED_PATH, cfg.MATRIX_DIR)

X_3d, y_3d, mask_3d, timestamps_3d = build_timeseries_matrix(
    cfg.CLEANED_PATH, 
    cfg.MATRIX_DIR, 
    seq_len=cfg.TRANSFORMER_PARAMS['seq_len']
)


print("Data Preparation Complete! Ready for modeling.")