"""Price features from PJM zonal LMP — three lookback channels and ten blind-spot-day statics.

Two blocks, split by WHEN each price becomes known, the same split
src/data_crawler/pjm_lmp.py makes when it writes the two files:

  Real-Time  ->  the LOOKBACK.  RT is realised during the operating day, so it can never
                 describe the target day; it can only say what the last 168 h were like.
  Day-Ahead  ->  a STATIC block on the BLIND-SPOT DAY.  Day d's Day-Ahead market cleared at
                 ~13:30 EPT on d-1, hours before a cutoff on day d, so it is genuinely ex-ante
                 and it is the one price that reaches into the day the lookback cannot see.
                 Day d+1's clears AFTER the cutoff and is deliberately not used.

That is the identical availability argument src/congestion_features makes, and this module
reuses its assert_available() rather than restating the posting time.


Why the signed log
------------------
Measured over 2021-2026 on dom: total_lmp_rt has median 36 and maximum 4392 — a 120x tail —
and both the total and the congestion component go NEGATIVE (RT congestion is negative ~22% of
hours, down to -199). The sequence channels are standardized, so an untransformed spike would
flatten the whole channel; and plain log1p, which congestion_features can use because shadow
prices are non-negative, is NaN below -1. Hence sign(x)*log1p(|x|): compresses the tail,
keeps the sign, defined everywhere.


Why the statics are DEPARTURES
------------------------------
src/congestion_features measured this and it was not close — the same quantity was ~50x more
useful written as a departure from its own trailing mean than as a level (cong_d0_dev carried
69% of that block's gain; three of the four level forms carried under 0.1% between them). The
reason it gives applies to price with more force, not less: about 70% of what congestion
carries is a proxy for system LOAD LEVEL that the models already see through 168 h of
Load_Estimated, and price is an even more direct function of load than a shadow price is.

So every static here is a departure, over the same three window lengths, with exactly one
absolute anchor kept — mirroring the one level form that earned its place there.


Which price, and why energy leads
---------------------------------
LMP = system_energy + congestion + marginal_loss. The block is built on two of those and a
derived third:

  energy  RTO-wide and identical across zones. This is the market pricing PJM's OWN day-ahead
          view of system tightness — genuinely forward-looking information that no amount of
          this zone's load history contains. It gets the three-window treatment.
  spread  total - energy (= congestion + loss): how much more this zone costs than the system.
          The ONLY zone-specific congestion signal in the whole feature set — cong_* is
          RTO-wide and byte-identical between dom and bge, so it says nothing about which zone
          is congested.
  total   the headline price, kept for the peak anchor and the daily maximum.

A ridge probe on the daily residual (584 days, small — direction only) ranked them
energy 0.416 > total 0.361 > raw congestion component 0.151, which is why energy is primary
and why the raw congestion component is not carried separately: cong_* already covers it, and
the spread carries the zonal part in departure form.
"""

import numpy as np
import pandas as pd

# Same posting-time constraint, same check — imported rather than restated so a change to the
# Day-Ahead posting hour cannot end up applied to one block and not the other.
from src.congestion_features import (PEAK_HOURS, W_LONG, W_PRIMARY, W_SHORT,
                                     assert_available)  # noqa: F401  (re-exported)

# --- Lookback channels ------------------------------------------------------------------
# Raw columns as the crawler joins them into the cleaned frame, and the transformed names the
# sequence actually consumes. The cleaned CSV stores raw prices; the transform happens here at
# matrix-build time, exactly as thermal_features derives its per-hour columns.
LMP_RAW_COLS = ['LMP_RT_Total', 'LMP_RT_Energy', 'LMP_RT_Cong']
LMP_SEQ_COLS = ['RT_Total_slog', 'RT_Energy_slog', 'RT_Cong_slog']


def signed_log(x):
    """sign(x) * log1p(|x|) — compresses the tail, keeps the sign, defined for negatives."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def add_lmp_sequence_cols(df):
    """Add the three per-hour price channels to the cleaned hourly df. Returns the names."""
    missing = [c for c in LMP_RAW_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"The cleaned frame has no {missing}. Re-run the crawler with PJM_LMP=1 (it joins "
            f"Real-Time LMP into cleaned_pjm_load_weather.csv), or set PJM_LMP=0 to build the "
            f"feature set without the price channels.")
    for raw, out in zip(LMP_RAW_COLS, LMP_SEQ_COLS):
        df[out] = signed_log(df[raw].astype(float))
    return LMP_SEQ_COLS


# --- Blind-spot-day statics -------------------------------------------------------------
LMP_FEATURE_NAMES = [
    # --- system energy price: the market's view of system-wide tightness, three timescales ---
    'lmp_d0_e_dev',        # day d mean - trailing W_PRIMARY mean      (the primary feature)
    'lmp_d0_e_dev_s',      # ... against a 3-day norm   — sharper, more local
    'lmp_d0_e_dev_l',      # ... against a 21-day norm  — a whole expensive stretch
    'lmp_d0_e_peak_dev',   # day d peak-hours mean - its own trailing mean
    # --- zonal spread (total - energy): this zone's own premium over the system ---
    'lmp_d0_s_dev',        # day d mean - trailing mean
    'lmp_d0_s_peak_dev',   # peak hours, same
    # --- scarcity ---
    'lmp_d0_t_max_dev',    # day d's worst single hour - its own trailing mean
    # --- medium-term drift of the norm itself (mirrors cong_w1_slope) ---
    'lmp_w1_slope',        # trailing W_PRIMARY mean - trailing W_LONG mean, on energy
    # --- the one absolute anchor, and the within-day shape ---
    'lmp_d0_t_peak',       # day d peak-hours total price, as a level
    'lmp_d0_peak_shape',   # peak-hours mean - all-day mean: how peaky the day was priced
]

N_LMP_FEATURES = len(LMP_FEATURE_NAMES)
assert N_LMP_FEATURES == 10, N_LMP_FEATURES


def active_feature_names():
    """The Day-Ahead static block. One block now: PJM_LMP=2's drift-normalised variant was
    removed once the Real-Time lookback became unconditional, which is what it existed to
    isolate."""
    return LMP_FEATURE_NAMES


def build_lmp_table(path):
    """cleaned/lmp_da.csv -> per-EPT-date DataFrame of the features, ready to look up.

    Built once per run (like build_congestion_table) because every sample reads a different day
    out of the same table.

    A day is usable only if it is COMPLETE. congestion can reindex a missing hour to zero
    because "no constraint bound" is a real state; a price has no such value — zero is a real
    and very unusual price, and a missing hour is a fetch gap, not a free hour. Incomplete days
    are marked unusable and their samples are dropped, the same way a day with no complete
    weather forecast is.
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing. Run the crawler with PJM_LMP=1 (it writes the Day-Ahead price "
            f"table), or set PJM_LMP=0 to build without the price block.")
    h = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if h.empty:
        return pd.DataFrame(columns=LMP_FEATURE_NAMES + ['_usable'])

    ept = pd.DatetimeIndex(pd.to_datetime(h['datetime_beginning_ept']))
    day = ept.normalize()

    tot = signed_log(h['total_lmp_da'].values)
    ene = signed_log(h['system_energy_price_da'].values)
    # The spread is formed on RAW dollars and then compressed: total - energy is the zone's
    # premium per MWh, which is the quantity with meaning, and it is routinely negative.
    spr = signed_log(h['total_lmp_da'].values - h['system_energy_price_da'].values)

    f = pd.DataFrame({'t': tot, 'e': ene, 's': spr}, index=day)
    peak = ept.hour.isin(PEAK_HOURS)
    g, gp = f.groupby(level=0), f[peak].groupby(level=0)

    base = pd.DataFrame({
        't_min':  g['t'].min(),
        'e_mean': g['e'].mean(),
        'e_peak': gp['e'].mean(),
        's_mean': g['s'].mean(),
        's_peak': gp['s'].mean(),
        't_mean': g['t'].mean(),
        't_peak': gp['t'].mean(),
        't_max':  g['t'].max(),
        # Raw (undivided) daily means, for the one feature that is a genuine fraction rather
        # than a log difference — a share has to be formed on the quantities themselves.
        'raw_t':  h.groupby(day)['total_lmp_da'].mean(),
        'raw_e':  h.groupby(day)['system_energy_price_da'].mean(),
    }).astype('float64')

    # min_periods=1 so the first days of the record still produce a value instead of poisoning
    # every sample whose window reaches back into them. The trailing mean INCLUDES day d, the
    # same convention congestion_features uses and justifies.
    roll = lambda col, w: base[col].rolling(w, min_periods=1).mean()

    t = pd.DataFrame({
        'lmp_d0_e_dev':      base['e_mean'] - roll('e_mean', W_PRIMARY),
        'lmp_d0_e_dev_s':    base['e_mean'] - roll('e_mean', W_SHORT),
        'lmp_d0_e_dev_l':    base['e_mean'] - roll('e_mean', W_LONG),
        'lmp_d0_e_peak_dev': base['e_peak'] - roll('e_peak', W_PRIMARY),
        'lmp_d0_s_dev':      base['s_mean'] - roll('s_mean', W_PRIMARY),
        'lmp_d0_s_peak_dev': base['s_peak'] - roll('s_peak', W_PRIMARY),
        'lmp_d0_t_max_dev':  base['t_max']  - roll('t_max',  W_PRIMARY),
        'lmp_w1_slope':      roll('e_mean', W_PRIMARY) - roll('e_mean', W_LONG),
        'lmp_d0_t_peak':     base['t_peak'],
        'lmp_d0_peak_shape': base['t_peak'] - base['t_mean'],
    })[LMP_FEATURE_NAMES]

    # A complete EPT day is 24 hours, 23 on spring-forward and 25 on fall-back.
    hours = f.groupby(level=0).size()
    t['_usable'] = t.index.map(lambda d: 23 <= hours.get(d, 0) <= 25) & t.notna().all(axis=1)
    t[LMP_FEATURE_NAMES] = t[LMP_FEATURE_NAMES].replace([np.inf, -np.inf], 0.0)
    t.index = t.index.date
    return t


def compute_lmp_features(day_table, target_day):
    """The features for one sample, in LMP_FEATURE_NAMES order, or None if unavailable.

    `target_day` is the day being forecast; everything is read off the day BEFORE it — the
    blind-spot day d, whose Day-Ahead prices cleared before the cutoff. Returning None marks
    the sample unforecastable, handled the same way as a day with no complete weather forecast:
    drop it.
    """
    d = pd.Timestamp(target_day).date() - pd.Timedelta(days=1).to_pytimedelta()
    if d not in day_table.index:
        return None
    row = day_table.loc[d]
    if not row['_usable']:
        return None
    return row[active_feature_names()].to_numpy(dtype='float32')
