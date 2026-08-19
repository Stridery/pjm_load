#!/bin/bash
# Retrain models across zones and feature-set arms.
#
#   sbatch Train_Model.sbatch                                                # everything, congestion on
#   sbatch --export=ALL,ARMS="1 0",PJM_TRAIN_ONLY=residual Train_Model.sbatch  # both arms, residual models
#   sbatch --export=ALL,DATASETS=dom,CRAWL=0 Train_Model.sbatch              # one zone, no data refresh
#
# The job holds one GPU — it sits in the queue until one is free.
#
# KNOBS (all optional, all read from the environment)
#   DATASETS        zones to run, space separated          default "dom bge"
#   ARMS            PJM_CONGESTION values to run, in order  default "1"
#   PJM_TRAIN_ONLY  comma-separated model names, or the
#                   shorthand `residual` for every *_residual   default: all of TRAIN_CONFIG
#   CRAWL           1 = refresh data first                  default 1
#
# WHY ARMS IS A LOOP AND THE CRAWL IS OUTSIDE IT
#   The test split is a FRACTION of the cleaned CSV (tail 0.16), so extending the CSV moves the
#   test boundary. Two arms crawled separately would be scored on different test sets and the
#   difference between them would no longer be the feature set. Crawling once per zone, then
#   running every arm against that same CSV, is what makes the comparison mean anything.
#
# WHERE THE RESULTS LAND
#   Each arm carries its own MATRIX_TAG, so nothing overwrites anything:
#     models/{zone}/{model}/tail_test0.16[_cong]/          weights
#     results/{zone}/evaluation/{model}/tail_test0.16[_cong]/*_metrics.txt

#SBATCH --job-name=pjm_train
#SBATCH --account=gts-sd111
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source .venv/bin/activate

DATASETS="${DATASETS:-dom bge}"
ARMS="${ARMS:-1}"
CRAWL="${CRAWL:-1}"

# `residual` expands from the registry rather than a hand-typed list, so a model added to
# TRAIN_CONFIG later is picked up automatically — and passing one word sidesteps the fact that
# sbatch --export uses commas to separate variables, which a model list also uses.
if [ "$PJM_TRAIN_ONLY" = "residual" ]; then
    PJM_TRAIN_ONLY=$(python -c \
        "from src.config import TRAIN_CONFIG; print(','.join(m for m in TRAIN_CONFIG if m.endswith('_residual')))") \
        || { echo "Could not expand PJM_TRAIN_ONLY=residual"; exit 1; }
fi
export PJM_TRAIN_ONLY="${PJM_TRAIN_ONLY:-}"

echo "============================================================"
echo "zones: $DATASETS   arms(PJM_CONGESTION): $ARMS   CRAWL: $CRAWL"
echo "models: ${PJM_TRAIN_ONLY:-all (TRAIN_CONFIG)}"
echo "start $(date)"
echo "============================================================"

for DATASET in $DATASETS; do
    if [ "$CRAWL" = "1" ]; then
        echo "--- Crawl + rebuild cleaned data ($DATASET) ---"
        PJM_DATASET=$DATASET python run_crawler.py \
            || { echo "Crawl failed for $DATASET — aborting."; exit 1; }
    else
        echo "--- Skipping crawl (CRAWL=0) for $DATASET ---"
    fi

    for ARM in $ARMS; do
        echo "------------------------------------------------------------"
        echo "Zone $DATASET | PJM_CONGESTION=$ARM | $(date)"
        PJM_DATASET=$DATASET PJM_CONGESTION=$ARM python -c \
            "import src.config as c; print(f'  MATRIX_TAG={c.MATRIX_TAG!r}')"
        echo "------------------------------------------------------------"

        # Fail before spending GPU hours if the congestion arm has nothing to read.
        if [ "$ARM" != "0" ] && [ ! -f "data/$DATASET/cleaned/congestion.csv" ]; then
            echo "data/$DATASET/cleaned/congestion.csv missing — run run_crawler.py (it fetches"
            echo "Day-Ahead marginal value), or drop this arm. Aborting."
            exit 1
        fi

        PJM_DATASET=$DATASET PJM_CONGESTION=$ARM python Model_Training.py \
            || { echo "Training failed: zone=$DATASET arm=$ARM — aborting."; exit 1; }

        echo "Zone $DATASET arm $ARM done | $(date)"
    done
done

echo "============================================================"
echo "All training complete: $(date)"
echo "Next: ./bundle_serving.sh   (needs an authenticated gh CLI, so run it from a login node)"
echo "============================================================"
