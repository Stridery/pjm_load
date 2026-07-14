"""
Forecast the days PJM has not verified yet, with the models named in PREDICT_CONFIG.

    PJM_DATASET=dom python Model_Prediction.py

Runs standalone, and is also invoked at the end of Model_Evaluation.py so a train →
evaluate pass leaves the forecasts behind without a second command.

Prerequisites: cleaned/predict.csv (written by run_crawler.py) and a trained model whose
scalers are in data/{DATASET}/matrix/ — the forecast reuses those scalers, never a new fit.
"""

import src.config as cfg
from src.model_predictor import ModelPredictor

predictor = ModelPredictor(cfg.PREDICT_CONFIG)

if any(m['enabled'] for m in cfg.PREDICT_CONFIG['models'].values()):
    if predictor.load_data():
        predictor.predict_all()
else:
    print("No models enabled in PREDICT_CONFIG.")
