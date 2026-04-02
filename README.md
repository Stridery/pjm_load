1.put dom_load.csv and pjm_dominionhub_hourly_2015_2025_openmeteo.csv into /data/raw/
run Data_Preparation.py and Model_Training.py

salloc -A gts-sd111 -p gpu-v100 -q inferno -N 1 -c 4 --mem=16G -t 2:00:00 --gres=gpu:1