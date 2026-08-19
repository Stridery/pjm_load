"""Temperature-derived thermal features, split by where they belong.

(A) SEQUENCE — go INTO the 168h lookback window; every hour has its own value.
    All rolling windows are CAUSAL (backward-looking), so hour t only ever sees
    data up to t. No leakage.
      CDD_h / HDD_h          : per-hour degree-hours vs the 65F base
      CDD_cum_24/48/72h      : rolling cumulative CDD (a different value each hour)
      Temp_ma_24h / 72h      : rolling mean temperature
      Temp_delta_24h         : Temp_t - Temp_{t-24}
      HeatIndex_F, WetBulb_F : per-hour temperature/humidity combinations

(B) STATIC — one value per sample, broadcast alongside the calendar/macro features.
    These are things a rolling sequence CANNOT express:
      temp_max_prev_day           : daily EXTREME; rolling means/sums flatten the peak.
      cdd_prev_day                : previous day's total cooling demand (complements the peak).
      temp_max_3d                 : intensity ceiling of the recent heat wave.
      cdd_ratio_d1_d2             : >1 heating up, <1 cooling off — a DIRECTION that a
                                    cumulative amount cannot give.
      temp_ma_d1_minus_d3         : 3-day temperature trend slope.

Everything here is computed from the sample's OWN recent hours. There are deliberately no
train-fitted references any more: `heatwave_day_count` (a heat-wave threshold) and
`temp_anomaly_vs_climatology` (a day-of-year climatology) were dropped because they cost
0.000% and 0.011% of LightGBM-residual gain between them, while forcing a persisted
thermal_refs.pkl, a manual post-training extraction step, and a serving failure mode — serving
sees ~30 days and cannot rebuild a 6-year climatology, so the constants had to travel with the
weights. Removing them makes this module a pure function of the window it is handed.
"""

import numpy as np

BASE_TEMP_F = 65.0

THERMAL_SEQ_COLS = [
    'CDD_h', 'HDD_h',
    'CDD_cum_24h', 'CDD_cum_48h', 'CDD_cum_72h',
    'Temp_ma_24h', 'Temp_ma_72h',
    'Temp_delta_24h',
    'HeatIndex_F', 'WetBulb_F',
]

THERMAL_STATIC_NAMES = [
    'temp_max_prev_day',
    'cdd_prev_day',
    'temp_max_3d',
    'cdd_ratio_d1_d2',
    'temp_ma_d1_minus_d3',
]


# ---------------------------------------------------------------------------
# (A) Sequence columns
# ---------------------------------------------------------------------------

def _heat_index_f(T, RH):
    """NWS heat index (Rothfusz above 80F, simple formula below)."""
    simple = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + RH * 0.094)
    hi = simple.copy()
    m = ((simple + T) / 2.0) >= 80.0
    Tm, Rm = T[m], RH[m]
    hi[m] = (-42.379 + 2.04901523 * Tm + 10.14333127 * Rm - 0.22475541 * Tm * Rm
             - 6.83783e-3 * Tm ** 2 - 5.481717e-2 * Rm ** 2 + 1.22874e-3 * Tm ** 2 * Rm
             + 8.5282e-4 * Tm * Rm ** 2 - 1.99e-6 * Tm ** 2 * Rm ** 2)
    return hi


def _wet_bulb_f(T_f, RH):
    """Stull (2011) wet-bulb approximation. F -> C -> Tw -> F."""
    Tc = (T_f - 32.0) * 5.0 / 9.0
    rh = np.clip(RH, 1e-3, 100.0)
    Tw = (Tc * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
          + np.arctan(Tc + rh) - np.arctan(rh - 1.676331)
          + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)
    return Tw * 9.0 / 5.0 + 32.0


def add_thermal_sequence_cols(df):
    """Add the per-hour thermal columns to the cleaned hourly df. Returns the names."""
    T  = df['Temp_F'].astype(float)
    RH = df['RelativeHumidity_pct'].astype(float)

    cdd = np.maximum(0.0, T - BASE_TEMP_F)
    df['CDD_h'] = cdd
    df['HDD_h'] = np.maximum(0.0, BASE_TEMP_F - T)
    for w in (24, 48, 72):
        df[f'CDD_cum_{w}h'] = cdd.rolling(w, min_periods=1).sum()
    for w in (24, 72):
        df[f'Temp_ma_{w}h'] = T.rolling(w, min_periods=1).mean()
    df['Temp_delta_24h'] = T.diff(24).fillna(0.0)
    df['HeatIndex_F'] = _heat_index_f(T.values, RH.values)
    df['WetBulb_F']   = _wet_bulb_f(T.values, RH.values)
    return THERMAL_SEQ_COLS


# ---------------------------------------------------------------------------
# (B) Static features — train-fitted references
# ---------------------------------------------------------------------------

def compute_thermal_static(temp, cdd, cutoff_pos):
    """The 5 static thermal features for one sample (needs >= 72h before cutoff)."""
    c = cutoff_pos
    d1 = slice(c - 24, c)            # previous 24h
    d2 = slice(c - 48, c - 24)       # 48-24h before
    d3 = slice(c - 72, c - 48)       # 72-48h before

    cdd_d1 = float(cdd[d1].sum())
    cdd_d2 = float(cdd[d2].sum())
    t_ma_d1 = float(temp[d1].mean())
    t_ma_d3 = float(temp[d3].mean())

    return np.array([
        float(temp[d1].max()),                       # temp_max_prev_day
        cdd_d1,                                      # cdd_prev_day
        float(temp[c - 72:c].max()),                 # temp_max_3d
        cdd_d1 / (cdd_d2 + 1e-3),                    # cdd_ratio_d1_d2  (>1 = heating up)
        t_ma_d1 - t_ma_d3,                           # temp_ma_d1_minus_d3
    ], dtype='float32')
