"""
Lightweight serving layer: forecast tomorrow from a short recent window, daily.

Separate from the training pipeline on purpose. Training builds 6-year matrices and fits the
scalers + thermal references; serving reuses those FROZEN artifacts and only ever touches the
last ~40 days of data. Nothing here re-fits anything — a re-fit on a short window would drift
silently away from what the models were trained on.
"""
