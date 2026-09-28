1.put dom_load.csv and pjm_dominionhub_hourly_2015_2025_openmeteo.csv into /data/raw/
run Data_Preparation.py and scripts/Model_Training.py

salloc -A gts-sd111 -p gpu-v100 -q inferno -N 1 -c 4 --mem=16G -t 2:00:00 --gres=gpu:1

source .venv/bin/activate

python scripts/Model_Training.py

python generate_web.py

#SBATCH --job-name=pjm_train
#SBATCH --account=
#SBATCH --partition=gpu-v100
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


## Congestion features (PJM Day-Ahead marginal value)

Eight static features built from RTO-wide transmission-constraint shadow prices, describing
the BLIND-SPOT DAY — the day between the forecast cutoff and the target day, which the 168 h
lookback cannot see. Fetched by the normal crawler run (`raw/marginal_price/` →
`cleaned/congestion.csv`), spliced in at matrix-build time like the forecast weather.

Day-Ahead only, RTO-wide only, and every feature a DEPARTURE from its own trailing mean
rather than a level — all three are measured choices, see the module docstrings in
`src/data_crawler/pjm_marginal_value.py` and `src/congestion_features.py`.

What it buys, measured on dom with `lightgbm_residual`: MAE −1.7%, peak MAPE −3.2%. The gain
is real but small and concentrated on high-stress days; roughly 70% of what congestion carries
is a system-load-level proxy the models already see through the 168 h lookback.

    PJM_DATASET=dom python scripts/run_crawler.py       # fetches marginal value along with everything else

### Run configuration

Four variables, read by `src/config.py`. The command line and the sbatch use the SAME names —
the only difference is that the sbatch accepts a space-separated LIST where a single process
takes one value, and loops.

| variable | meaning | python | sbatch | default |
|---|---|---|---|---|
| `PJM_DATASET` | zone | one | list | `dom bge` (sbatch) / `dom` (python) |
| `PJM_CONGESTION` | congestion block on/off | `0` or `1` | list | `1` |
| `PJM_TRAIN_ONLY` | which models | comma-separated names, or `residual` | same | all of `TRAIN_CONFIG` |
| `PJM_CRAWL` | refresh data first | — (run `scripts/run_crawler.py`) | `0`/`1` | `1` |

    # one process, one zone, one arm
    PJM_DATASET=dom PJM_CONGESTION=0 PJM_TRAIN_ONLY=residual python scripts/Model_Training.py

    # the same thing in the queue, plus a second zone and a second arm
    sbatch --export="ALL,PJM_CRAWL=0,PJM_CONGESTION=1 0,PJM_TRAIN_ONLY=residual" Train_Model.sbatch

Quote the WHOLE `--export` argument whenever a value contains a space: sbatch splits its
argument on commas, and the shell would otherwise end the argument at the space.

Passing a list to python raises rather than doing something surprising — `PJM_DATASET="dom bge"`
would otherwise write to `data/dom bge/`, and `PJM_CONGESTION="1 0"` would fail inside `int()`
with no hint of where the value came from.

### Switching the congestion block off

`PJM_CONGESTION=0` drops the block. It flips `MATRIX_TAG`, which every artifact whose content
depends on the static block's width carries — matrices, the three scalers, and the model run
dirs:

    data/dom/matrix/X_opt_lb168_h0.csv                    data/dom/matrix/X_opt_lb168_h0_cong.csv
    models/dom/xgboost/tail_test0.16/                     models/dom/xgboost/tail_test0.16_cong/
    results/dom/evaluation/xgboost/tail_test0.16/         .../tail_test0.16_cong/

So the two arms coexist instead of overwriting each other, and an A/B is just two runs.
`src/serving/predict` selects run dirs by tag, so it loads the weights matching the current
setting and refuses a mismatch rather than feeding a model the wrong number of features.

Note that the tag records the congestion switch, NOT a version of the feature set as a whole.
Changing what the other blocks contain therefore requires wiping `data/{zone}/matrix*` by
hand: the builders cache on filename alone, so a matrix left from the previous definition
would be silently reused. `scripts/run_crawler.py` does that wipe as its last step, so a crawl covers
it; a bare edit-then-train does not.

### Comparing across time

`run_pipeline` wipes `data/{zone}/matrix*` after every crawl, and a crawl also extends the
cleaned CSV — which moves the tail-0.16 test boundary. Metrics written under different cleaned
CSVs are not comparable, so an A/B must run both arms against the SAME crawl (`PJM_CRAWL=0` on the
second one, or no crawl at all). The durable record is the
`results/{zone}/evaluation/**/**_metrics.txt` files themselves.


## Probabilistic baselines

Three reproductions, each isolated from the point-model stack: one input
(`data/{zone}/cleaned/cleaned_pjm_load_weather.csv`), results under `results/{zone}/{name}/`,
no matrix cache, no scalers, and deliberately not registered in `src/models` (see
`src/quantile/__init__`). Both are scored by the same functions in `src/quantile/qeval.py` on
the same tail-0.16 test period, so their tables are comparable.

| baseline | paper | entry | tuning |
|---|---|---|---|
| Density combination | Li et al. 2020, TSG 11(2):1679-1690 | `scripts/train_density.py` | `DENSITY_CONFIG` |
| iQRNN | Zhang et al. 2019, TSG 10(4):4425-4434 | `scripts/train_qrnn.py` | `QRNN_CONFIG` |

    PJM_DATASET=dom python scripts/train_density.py   &&  python -m src.density.plots
    PJM_DATASET=dom python scripts/train_qrnn.py      &&  python -m src.qrnn.plots

`scripts/train_qrnn.py` folds the density row into its own scoreboard when those results are present,
labelling each row with its forecast horizon rather than assuming they answer the same
question.

Two further baselines were removed. The temperature-scenario method of Xie & Hong (2018)
needed a thirty-year weather archive, which made it the one baseline that could not be
pointed at an arbitrary load dataset. Quantile Regression Averaging on sister forecasts (Liu
et al. 2017) followed: its sister family is built from a temperature recency effect, and
driving it from lagged load instead collapses the disagreement the method converts into
quantiles — a change to the mechanism, not to the features.

The same two baselines are ported to the public datasets (`elia`, `ercot`, `spain`) under
`src/bench/`, with their own entries `scripts/train_bench_density.py` and `scripts/train_bench_iqrnn.py`.

## Foundation models (zero-shot) — `scripts/run_foundation.py`

A pre-trained univariate forecaster (TimesFM 2.5 today) is handed the recent `Load_Estimated`
history and asked for the target day. **Nothing is trained**, so there is no weights file, no
matrix cache and no train step — the run reads the cleaned CSV, calls the model, and evaluates.

    .venv/bin/pip install 'timesfm[torch]'          # optional; nothing else needs it
    export HF_HOME=/path/with/room                  # ~1 GB checkpoint, keep it off $HOME
    PJM_DATASET=dom python scripts/run_foundation.py                          # config defaults
    PJM_DATASET=dom python scripts/run_foundation.py --context 512 --mode residual
    PJM_DATASET=dom python scripts/run_foundation.py --model zero --mode residual   # harness self-check
    PJM_DATASET=dom python scripts/run_foundation.py --scoreboard             # re-read results only
    python scripts/run_foundation.py --list

Tuning lives in `FOUNDATION_CONFIG` in `src/config.py`; every flag above is a per-run override
of the same field. ~1 min per (mode, context) for a 310-day test year on CPU.

### Why it is a separate pipeline but not separate results

Not registered in `src/models`, for the reason `src/quantile` and `src/density` are not: that
registry is what `src/serving/predict.available_models()` scans, and the daily CI forecast job
loads everything it finds. A model needing an optional package and a 1 GB checkpoint the runner
does not have would break the daily forecast the moment it was registered. `scripts/Model_Training.py`,
`model_evaluator`, `serving`, `generate_web` and `Train_Model.sbatch` are all untouched.

The RESULTS are not separate — comparing them is the point. They land in the same tree as the
16 point models, with direct and residual as separate model names exactly like
`xgboost` / `xgboost_residual`, and the context length in the run tag:

    results/{zone}/evaluation/timesfm/tail_test0.16_cong_ctx512/
    results/{zone}/evaluation/timesfm_residual/tail_test0.16_cong_ctx512/
    results/{zone}/evaluation/_scoreboard.{csv,png}      # every model in one table

`_scoreboard` is a scan of every `*_metrics.txt` under `evaluation/`, so the point models need
no re-run to appear in it.

### The two things that make the comparison valid

**The day set.** Test days and truth come from `y_opt_*.csv`, not from a fresh split of the
cleaned CSV. The matrix builder drops days with no complete 48 h forecast (and, with the block
on, no Day-Ahead congestion); re-splitting the CSV would score a larger, different set of days
and report a better or worse number for reasons unrelated to the model. A run prints
`test N/310` and warns loudly if it is short.

**The 24 h blind spot.** The cutoff is the target day's PREVIOUS midnight and the lookback ends
there, so the context stops at (target − 2 d) 23:00 and the target day starts at forecast step
**25**, not step 1. The horizon is therefore 48–49, never 24. A model asked for 24 steps
forecasts the blind-spot day and scores it against the target day's actuals.

### Modes

| mode | series in | output |
|---|---|---|
| `direct` | `Load_Estimated` | forecast, then an optional Est→metered scale correction |
| `residual` | `Load_Estimated` − same-hour-last-week baseline | forecast + the baseline added back |

`residual` mirrors `src/models/*_residual.py`, and its floor is known: a model returning 0
reproduces the naive baseline exactly, so the gap to that baseline IS the model's contribution.
`--model zero --mode residual` runs precisely that and must reproduce dom 7.70% / bge 10.09%;
it is the harness's end-to-end self-check, and `test/test_foundation_context.py` asserts it.

`direct` mode's output is in `Load_Estimated` space while the target is metered load.
`est_to_metered='hourly_ratio'` fits 24 median Est/Load ratios on the TRAIN split alone and
freezes them over the test period — the same "fitted on train, frozen at inference" arrangement
the point models get. Both corrected and uncorrected scores are always reported. Residual mode
needs none of it: the baseline it adds back is in the same space, so the offset cancels.

### Context series: `estimated` vs `metered`

Preliminary load is current as of the cutoff but is a PROXY for the target: on dom it carries a
~0.8% scale offset, on bge a 3.3% *shape* mismatch (its proxy is MIDATL/8.8, a regional
aggregate whose diurnal shape is not bge's — no rescaling removes that). Metered load has no
proxy gap because it IS the target, but unverified metered trails preliminary by ~1 day
(measured: exactly 24 h behind on both zones; VERIFIED metered trails 7 d on dom and 48 d on
bge and is useless in real time).

Something has to give, and `metered_issue` says which:

| | window ends | target at | horizon | first issuable |
|---|---|---|---|---|
| `estimated` | (d−2) 23:00 | steps 25–48 | 48 | (d−1) 00:00 |
| `metered` + `day_of` | (d−2) 23:00 | steps 25–48 | 48 | **d 00:00** |
| `metered` + `day_ahead` | (d−3) 23:00 | steps 49–72 | 72 | (d−1) 00:00 |

`day_of` keeps the horizon and reads the SAME window rows as the preliminary run — only a
different column, so there is no re-pairing and nothing to rebuild. What it gives up is the
issue time: metered through (d−2) 23:00 does not exist until a day after the equivalent
preliminary does, so the forecast can first be produced at the START of the target day.

`day_ahead` keeps the point models' issue time and pays with a 72-step horizon instead.

**These are different questions and their scores are not interchangeable.** `day_of` is a
day-of forecast issued at midnight; only `day_ahead` performs the same task as the 16 point
models. The run tag (`_met` / `_met_da`), the artifact name (`*_MET_*` / `*_METDA_*`) and an
`issue time:` line in `metrics.txt` all record which was used, so a scoreboard row cannot lose
the distinction — but the scoreboard itself does not know it, so read the tag.

The horizon cost of `day_ahead` is visible before any model runs: the same naive baseline scores
7.70% on preliminary, 7.63% on metered `day_of`, and 8.06% on metered `day_ahead` (dom).

`horizon_steps: 'auto'` derives the horizon from the data rather than taking a number, so a
lagged series cannot silently drop days for want of steps. Baseline lags shift with it
(48…192 h → 72…216 h under `day_ahead`), which is what keeps the baseline from being built out
of values PJM had not published yet; `test/test_foundation_context.py` asserts that.

**Caveat.** The backtest reads the cleaned CSV's `Load`, which is VERIFIED metered. At forecast
time only the UNVERIFIED value existed, and PJM revises it. No archived snapshots exist here, so
that revision is unmeasurable and every `series='metered'` score carries a small,
one-directional optimism.

### Adding another model

One file exposing `NAME` and `forecast(contexts (N,L), horizon, params) -> (N,horizon)`, plus a
line in `src/foundation/registry.py`. The adapter never sees the frame, the split, the baseline
or the evaluation.
