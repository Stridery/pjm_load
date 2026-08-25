#!/bin/bash
# Retrain models across zones and feature-set arms. Sits in the queue until a node is free.
#
# ONE VARIABLE PER CONCEPT — the same names src/config.py reads, so a command line and a
# submission differ only in whether a variable holds one value or several. There is no
# sbatch-only spelling of a config knob to get out of sync with:
#
#   PJM_DATASET     zones      one for python, a space-separated list here   default "dom bge"
#   PJM_CONGESTION  arms       0 or 1 for python, a list here                default "1"
#   PJM_TRAIN_ONLY  models     comma-separated names, or `residual`          default: all
#   PJM_CRAWL       1 = refresh the data first (sbatch only — the CLI just
#                   runs run_crawler.py directly)                            default 1
#
# Quote the WHOLE --export when any value contains a space; sbatch splits its argument on
# commas, and the shell would otherwise end the argument at the space:
#
#   sbatch Train_Model.sbatch                                          # all zones, congestion on
#   sbatch --export="ALL,PJM_CRAWL=0,PJM_CONGESTION=1 0" Train_Model.sbatch          # both arms
#   sbatch --export="ALL,PJM_CRAWL=0,PJM_CONGESTION=0" Train_Model.sbatch            # baseline only
#   sbatch --export="ALL,PJM_DATASET=dom,PJM_TRAIN_ONLY=residual" Train_Model.sbatch # one zone, residuals
#
# WHY THE CRAWL SITS OUTSIDE THE ARM LOOP
#   The test split is a FRACTION of the cleaned CSV (tail 0.16), so extending the CSV moves the
#   test boundary. Two arms crawled separately would be scored on different test sets and the
#   difference between them would no longer be the feature set. Crawl once per zone, then run
#   every arm against that same CSV.
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

# Read once, then unset: the loops below hand each child ONE value per variable, and
# src/config.py rejects a list outright. Leaving the list exported would mean a child that
# forgot its per-iteration assignment would inherit the list and fail with config's guard
# rather than quietly running the wrong thing.
_zones="${PJM_DATASET:-dom bge}"
_arms="${PJM_CONGESTION:-1}"
_crawl="${PJM_CRAWL:-1}"
unset PJM_DATASET PJM_CONGESTION
export PJM_TRAIN_ONLY="${PJM_TRAIN_ONLY:-}"   # config resolves this one, including `residual`

echo "============================================================"
echo "zones: $_zones   arms(PJM_CONGESTION): $_arms   crawl: $_crawl"
echo "models: ${PJM_TRAIN_ONLY:-all (TRAIN_CONFIG)}"
echo "start $(date)"
echo "============================================================"

for ZONE in $_zones; do
    if [ "$_crawl" = "1" ]; then
        echo "--- Crawl + rebuild cleaned data ($ZONE) ---"
        PJM_DATASET=$ZONE python run_crawler.py \
            || { echo "Crawl failed for $ZONE — aborting."; exit 1; }
    else
        echo "--- Skipping crawl (PJM_CRAWL=0) for $ZONE ---"
    fi

    for ARM in $_arms; do
        echo "------------------------------------------------------------"
        echo "Zone $ZONE | PJM_CONGESTION=$ARM | $(date)"
        PJM_DATASET=$ZONE PJM_CONGESTION=$ARM python -c \
            "import src.config as c; print(f'  MATRIX_TAG={c.MATRIX_TAG!r}')" \
            || { echo "Config rejected zone=$ZONE arm=$ARM — aborting."; exit 1; }
        echo "------------------------------------------------------------"

        # Fail before spending compute if the congestion arm has nothing to read.
        if [ "$ARM" != "0" ] && [ ! -f "data/$ZONE/cleaned/congestion.csv" ]; then
            echo "data/$ZONE/cleaned/congestion.csv missing — run run_crawler.py (it fetches"
            echo "Day-Ahead marginal value), or drop this arm. Aborting."
            exit 1
        fi

        PJM_DATASET=$ZONE PJM_CONGESTION=$ARM python Model_Training.py \
            || { echo "Training failed: zone=$ZONE arm=$ARM — aborting."; exit 1; }

        echo "Zone $ZONE arm $ARM done | $(date)"
    done
done

echo "============================================================"
echo "All training complete: $(date)"
echo "Next: ./bundle_serving.sh   (needs an authenticated gh CLI, so run it from a login node)"
echo "============================================================"
