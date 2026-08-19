"""Probabilistic-load-forecasting baselines: Xie & Hong (2018), IEEE TSG 9(3):1680-1687.

Temperature scenarios -> a point load forecasting model -> percentiles -> quantile score.

Deliberately isolated from src/models/. That package has a fixed train/predict/evaluate
contract bound to the per-DAY tree/sequence matrices, and its registry is what
src/serving/predict.available_models() scans; a per-HOUR MLR dropped in there would be
picked up by the daily CI forecast job and blow it up.
"""
