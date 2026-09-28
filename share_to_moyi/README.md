# PJM data for ARCHFormer — what to change so the numbers are comparable

## Files

    cleaned_pjm_load_weather_dom.csv     hourly, 2021-01-01 .. 2026-08-26, 49,511 rows
    cleaned_pjm_load_weather_bge.csv     hourly, 2021-01-01 .. 2026-07-31, 48,911 rows
    dom_zh_config.json / bge_zh_config.json    drop into cfgs_data/

Both CSVs are indexed by `Datetime_UTC` (tz-aware, contiguous hourly, no gaps) and carry
`Datetime_EPT` as a column. Copy them into `data/public_cleaned_data/` and rename to match
`csv_path` in the config, or edit `csv_path`.

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

| | dom | bge |
|---|---|---|
| train | 2021-03-25 .. 2025-04-21 | 2021-03-25 .. 2025-04-02 |
| val   | 2025-05-06 .. 2025-10-02 | 2025-04-17 .. 2025-09-11 |
| test  | **2025-10-17 .. 2026-08-25** (313 d) | **2025-09-26 .. 2026-07-31** (309 d) |

The gaps between blocks are the embargo. **The test window is the part that has to match** —
it is contiguous (313 and 309 consecutive calendar days), so your loop will score exactly the
days ours does and the numbers are directly comparable. Train/val can differ without hurting
that; our matrix additionally drops ~84 earlier days that lack a complete 48 h weather
forecast, which yours will keep. Those are all before the test window.


## Columns

`actual_cols` is every hourly physical channel in the file; `aux_cols` is the calendar block.
Both are read per row by your `generate_samples_new`, so nothing needs pre-processing.

Two notes on what our models additionally use, in case a gap shows up in the comparison:

- **Derived hourly channels.** Ours also feed 10 thermal channels (per-hour CDD/HDD, rolling
  24/48/72 h CDD, rolling mean temperature, temperature delta, heat index, wet bulb) and the
  three RT prices under `sign(x)·log(1+|x|)` rather than raw. All are computed from columns
  already in the file — none is extra data. Add them or not as you prefer.
- **Per-sample static features (40–58 of them).** Forecast-weather departures for the target
  day, Day-Ahead congestion and price aggregates for the blind-spot day, 3-week load trend,
  thermal state. These are constant across the lookback window, so they have no natural place
  in your `(L, F)` tensor, and we are not asking you to reproduce them. It does mean our point
  models see information yours will not; the forecast-weather block is the largest single part
  of it.

Target is `Load` — PJM metered load, MW. `Load_Estimated` is the preliminary series, published
in real time and available at the cutoff; the metered value is not, which is why it is only
the target.

`is_valid` is our training mask: a per (month, hour) 3-sigma rule on load, combined with label
availability. 99.6% of rows pass. Ignore it or use it to filter training rows.
