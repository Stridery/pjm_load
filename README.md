1.put dom_load.csv and pjm_dominionhub_hourly_2015_2025_openmeteo.csv into /data/raw/
run Data_Preparation.py and Model_Training.py

salloc -A gts-sd111 -p gpu-v100 -q inferno -N 1 -c 4 --mem=16G -t 2:00:00 --gres=gpu:1

source .venv/bin/activate

python Model_Training.py

dom
xgboost
Global -> MAPE: 6.89% | MAE: 1160.68 | RMSE: 1666.93 | ME: -472.28 MW (under)
         BRS:  -0.4384 MW/MW (under↓ at peak, near 0 = unbiased across load range)

Global -> MAPE: 6.58% | MAE: 1073.76 | RMSE: 1494.81 | ME: -95.75 MW (under)
         BRS:  -0.3324 MW/MW (under↓ at peak, near 0 = unbiased across load range)

transformer
Global -> MAPE: 7.42% | MAE: 1232.09 | RMSE: 1680.37 | ME: -390.70 MW (under)
         BRS:  -0.4443 MW/MW (under↓ at peak, near 0 = unbiased across load range)


Global -> MAPE: 6.62% | MAE: 1092.35 | RMSE: 1524.87 | ME: -206.63 MW (under)
         BRS:  -0.2767 MW/MW (under↓ at peak, near 0 = unbiased across load range)

Global -> MAPE: 6.71% | MAE: 1083.85 | RMSE: 1463.98 | ME: +44.30 MW (over)
         BRS:  -0.2593 MW/MW (under↓ at peak, near 0 = unbiased across load range)