# PJM DOM data for ARCHFormer — what to change so the numbers are comparable

## Files

Everything is indexed by `Datetime_UTC` (tz-aware) and also carries `Datetime_EPT`, except
the static table which is indexed by target day.

    cleaned_pjm_load_weather_dom.csv   hourly, 2021-01-01 .. 2026-08-26, 49,511 rows
                                            load, weather, RT prices, calendar
    dom_zh_config.json                     drop into cfgs_data/

  raw side tables (the bypass-variable sources)

    forecast_dom.csv                   hourly weather FORECASTS, keyed by (issue_date, lead_day)
    congestion_dom.csv                 hourly PJM Day-Ahead binding-constraint shadow prices
                                            RTO-wide, so this file is identical for dom and bge
    lmp_da_dom.csv                     hourly Day-Ahead LMP and its three components

  the same thing already computed, so you do not have to reimplement anything

    derived_hourly_dom.csv             13 engineered hourly channels
    static_by_target_day_dom.csv       64 per-target-day features, 1957 rows, 2021-03-25 .. 2026-08-25

Copy the main CSV into `data/public_cleaned_data/` and rename it to match `csv_path` in the
config, or edit `csv_path`.

Model config: your `cfgs_grid_search/archformer3.json`, unchanged.

    python run_point_forecast.py --data_cfg dom_zh_config.json --param_cfg archformer3.json

## The one setting that must change: `forecast_deadline_hour: -1`

This is the only thing you cannot guess from the data, and it is the difference between
solving our problem and solving an easier one.

Your PJM configs use `forecast_deadline_hour: 23`, which in `generate_samples_new` gives

    lookback_end = target_00:00 - 1h   ->  data through (D-1) 23:00, predict day D

Our forecast is issued a day earlier. The cutoff is the target day's **previous midnight**, so
the last observed hour is (D-2) 23:00 and day D-1 is a blind spot the model never sees:

    D-8 00:00 ............ D-2 23:00 | D-1 (unseen) | D = target
    <------- 168 h lookback -------->

`forecast_deadline_hour: -1` reproduces that exactly — `24 - (-1) = 25`, so
`lookback_end = target_00:00 - 25h = (D-2) 23:00`, and
`lookback_start = lookback_end - 167h = (D-8) 00:00`. A 168-hour window, 24-hour target,
one blind day in between. Verified against our matrix builder.

With 23 instead of -1 the model gets a full extra day of observations and the MAPE will look
much better than ours for a reason that has nothing to do with the architecture.


## Split

Ratios (what our code applies): **test = last 16% of days, validation = last 10% of what
remains, then 14 days dropped from the end of train and of val** as an embargo — our features
include 3-week rolling aggregates, so without the gap a training sample's own window would
overlap the validation period. Test is never trimmed.

Your pipeline takes explicit dates, so the config already holds the resulting windows:

| | dom |
|---|---|
| train | 2021-03-25 .. 2025-04-21 |
| val   | 2025-05-06 .. 2025-10-02 |
| test  | **2025-10-17 .. 2026-08-25**  (313 days) |

The gaps between blocks are the embargo. **The test window is the part that has to match** —
it is contiguous (313 consecutive calendar days), so your loop will score exactly the
days ours does and the numbers are directly comparable. Train/val can differ without hurting
that; our matrix additionally drops ~84 earlier days that lack a complete 48 h weather
forecast, which yours will keep. Those are all before the test window.


## Columns

`actual_cols` is every hourly physical channel in the main file; `aux_cols` is the calendar
block. Both are read per row by your `generate_samples_new`, so the config alone runs.

Target is `Load` — PJM metered load, MW. `Load_Estimated` is the preliminary series, published
in real time and available at the cutoff; the metered value is not, which is why it is only
the target.

`is_valid` is our training mask: a per (month, hour) 3-sigma rule on load, combined with label
availability. 99.6% of rows pass. Ignore it or use it to filter training rows.

### Everything our models see, beyond the main CSV

We compute these at matrix-build time and never wrote them to disk, so they were missing from
the first version of this package. They are now included, **already computed**, both as the
raw sources and as the finished features. Use whichever you prefer; you do not need to
reproduce any of the transforms.

**`derived_hourly_dom.csv` — 13 hourly channels**, aligned to the main CSV row for row.
Everything here is a function of columns already in the main file.

| block | columns |
|---|---|
| thermal (10) | per-hour CDD and HDD, rolling 24/48/72 h CDD, rolling 24 h mean temperature, 24 h temperature delta, heat index, wet-bulb temperature |
| price (3) | the three RT prices under `sign(x)·log(1+|x|)` instead of raw |

These are time-varying, so they slot straight into your `(L, F)` lookback tensor — append them
to `actual_cols`.

If you merge them in, these are the literal column names to append to `actual_cols`:

```json
"CDD_h", "HDD_h", "CDD_cum_24h", "CDD_cum_48h", "CDD_cum_72h",
"Temp_ma_24h", "Temp_ma_72h", "Temp_delta_24h", "HeatIndex_F", "WetBulb_F",
"RT_Total_slog", "RT_Energy_slog", "RT_Cong_slog"
```

If you do, drop `LMP_RT_Total` / `LMP_RT_Energy` / `LMP_RT_Cong` from `actual_cols` — the
`_slog` versions are the same three series, just transformed. Keeping both is redundant.

The config we ship does **not** include these; it runs on the main CSV alone, so it works
out of the box. Add them as a second run if you want to match our feature set.

**`static_by_target_day_dom.csv` — 64 features, one row per target day.**

| block | n | from |
|---|---|---|
| calendar | 6 | target day's month/dow sin-cos, weekend, holiday |
| macro | 4 | 3-week load trend and level, from `Load_Estimated` |
| thermal state | 5 | rolling thermal condition at the cutoff |
| **forecast weather** | 31 | `forecast_dom.csv` — departures of the target day's forecast from recent actuals, per clock hour and daily aggregate |
| **congestion** | 8 | `congestion_dom.csv` — blind-spot day's binding-constraint counts and shadow-price aggregates |
| **DA price** | 10 | `lmp_da_dom.csv` — blind-spot day's Day-Ahead price level, spread and components |

The forecast block is the largest and the one most likely to explain a gap: it is the only
place where information about the target day itself enters the model.

The catch is structural, not ours to solve for you: **these are constant across the lookback
window**, so they have no natural slot in an `(L, F)` tensor built only from the window.
We concatenate them after the encoder, as a static skip. If you would rather not touch the
architecture, broadcasting each one across all 168 positions also works and costs nothing but
width. How you feed them is up to you — we are just making sure you have the numbers.

Join key: `target_day` in the static file is the **date being predicted**, i.e. the day your
`generate_samples_new` produces the 24-hour label for. Not the last day of the window.

The raw side tables are included as well, in case you want to derive something different:

- `forecast_dom.csv` — one row per (issue_date, lead_day, hour). Pick `lead_day` so the issue
  date is at or before our cutoff, i.e. the forecast really was available. `is_complete` marks
  issues that cover the full 48 h.
- `congestion_dom.csv` — `n` binding constraints that hour, `sum_sp` / `max_sp` their shadow
  prices. This is a system-wide table, not a zonal one — the dom and bge copies are the
  same file.
- `lmp_da_dom.csv` — `total_lmp_da` and its energy / congestion / loss components.
