1.put dom_load.csv and pjm_dominionhub_hourly_2015_2025_openmeteo.csv into /data/raw/
run Data_Preparation.py and Model_Training.py

salloc -A gts-sd111 -p gpu-v100 -q inferno -N 1 -c 4 --mem=16G -t 2:00:00 --gres=gpu:1

source .venv/bin/activate

python Model_Training.py

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

    PJM_DATASET=dom python run_crawler.py       # fetches marginal value along with everything else

### Switching it off

`PJM_CONGESTION=0` (or `CONGESTION_CONFIG['enabled'] = 0`) drops the block. It flips
`MATRIX_TAG`, which every artifact whose content depends on the static block's width carries —
matrices, the three scalers, and the model run dirs:

    data/dom/matrix/X_opt_lb168_h0.csv                    data/dom/matrix/X_opt_lb168_h0_cong.csv
    models/dom/xgboost/tail_test0.16/                     models/dom/xgboost/tail_test0.16_cong/
    results/dom/evaluation/xgboost/tail_test0.16/         .../tail_test0.16_cong/

So the two arms coexist instead of overwriting each other, and an A/B is just two runs — no
file edit, nothing left flipped by accident:

    PJM_CONGESTION=0 python Model_Training.py
    PJM_CONGESTION=1 python Model_Training.py

`src/serving/predict` selects run dirs by tag, so it loads the weights matching the current
setting and refuses a mismatch rather than feeding a model the wrong number of features.

Note that the tag records the congestion switch, NOT a version of the feature set as a whole.
Changing what the other blocks contain therefore requires wiping `data/{zone}/matrix*` by
hand: the builders cache on filename alone, so a matrix left from the previous definition
would be silently reused. `run_crawler.py` does that wipe as its last step, so a crawl covers
it; a bare edit-then-train does not.

### Training a subset

`PJM_TRAIN_ONLY` restricts a run to named models (comma-separated), overriding `TRAIN_CONFIG`
without editing it. Unknown names raise rather than silently training nothing.

    PJM_TRAIN_ONLY=lightgbm_residual python Model_Training.py
    PJM_TRAIN_ONLY=xgboost,xgboost_residual python Model_Training.py

### Comparing across time

`run_pipeline` wipes `data/{zone}/matrix*` after every crawl, and a crawl also extends the
cleaned CSV — which moves the tail-0.16 test boundary. Metrics written under different cleaned
CSVs are not comparable, so an A/B must run both arms against the SAME crawl (`CRAWL=0` on the
second one, or no crawl at all). The durable record is the
`results/{zone}/evaluation/**/**_metrics.txt` files themselves.


## Probabilistic baselines (Xie & Hong 2018, IEEE TSG 9(3):1680-1687)

Temperature scenarios -> an MLR point model -> percentiles -> quantile score (pinball loss).
CPU only, ~30 s per zone, no GPU and no sbatch needed.

    PJM_DATASET=dom python train_quantile.py        # T-cube / Vanilla / Hong-2014
    PJM_DATASET=dom python -m src.quantile.plots    # figures from what the run wrote

Reads ONE input, data/{zone}/cleaned/cleaned_pjm_load_weather.csv, and writes to
results/{zone}/quantile/. It shares nothing else with the point-model stack: no matrix
cache, no scalers, and deliberately not registered in src/models (see src/quantile/__init__).
Tuning lives in QUANTILE_CONFIG in src/config.py.