"""Probabilistic baselines on the three public datasets (spain, ercot, elia).

Completely separate from the PJM stack. Nothing here imports src.config, src.models or any
data/{zone}/ path; the only inputs are data/quantile/{name}_config.json and the cleaned CSV
each one names. The PJM baselines under src/quantile, src/density and src/qrnn are left
untouched, and the numerical cores of those packages are IMPORTED here rather than copied —
only the feature assembly, the splitting and the scoring are rewritten, because those are
what the protocol changes.

Protocol (Chapter 3): every dataset forecasts one whole calendar day at a time from a
deadline on the preceding day, using a lookback window that ENDS at that deadline. Write

    H   steps in the target day          24 (hourly) or 96 (15-minute)
    L   steps in the lookback window      168 or 672
    G   steps from the deadline to the target day's first step

then a lag k counted back from a target step is observable for EVERY step of the day, and
lies inside the window, exactly when

    G + H - 1  <=  k  <=  L - 1 + G

    spain   deadline 23:00, hourly     G=1    [24, 168]
    ercot   deadline 09:00, hourly     G=15   [38, 182]
    elia    deadline 23:00, 15-minute  G=4    [99, 675]

Reporting is in STANDARDIZED space: the target is z-scored with the mean and standard
deviation of the training days, and pinball, MPIW, Winkler, MAE, RMSE and MSE are computed
on those z units so the three datasets are on one scale. MAPE and SMAPE are scale-free but
undefined near zero in z units, so they alone are computed after mapping back to physical
units.
"""
