#!/bin/bash
#SBATCH --job-name=pjm_train
#SBATCH --account=gts-sd111
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=6:00:00
#SBATCH -q inferno
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate
mkdir -p logs

WORKDIR=$SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Dataset list — add/remove zone abbreviations here.
# ---------------------------------------------------------------------------
DATASETS="dom2"

for DATASET in $DATASETS; do
    echo "=============================="
    echo "Dataset: $DATASET"
    echo "=============================="

    # -- Data pipeline: fetch/cache Open-Meteo weather, build joined+cleaned CSV.
    echo "--- Data pipeline for $DATASET ---"
    PJM_DATASET=$DATASET python run_crawler.py
    if [ $? -ne 0 ]; then
        echo "Data pipeline failed for $DATASET — aborting."
        exit 1
    fi

    # -- Try to get a GPU node (20-minute timeout).
    #    If allocation is granted, training runs on the GPU node.
    #    If the queue wait exceeds 20 minutes, fall back to CPU training.
    echo "--- Requesting GPU allocation (20-min timeout) ---"
    ALLOC_FLAG="/tmp/pjm_gpu_alloc_${SLURM_JOB_ID}_${DATASET}"

    timeout 1200 salloc \
        -A gts-sd111 -p gpu-v100 -q inferno \
        -N 1 -c 4 --mem=16G -t 2:00:00 --gres=gpu:1 \
        bash -c "touch $ALLOC_FLAG \
                 && cd $WORKDIR \
                 && source .venv/bin/activate \
                 && PJM_DATASET=$DATASET python Model_Training.py"

    if [ ! -f "$ALLOC_FLAG" ]; then
        echo "GPU allocation timed out after 20 minutes — training on CPU."
        PJM_DATASET=$DATASET python Model_Training.py
    fi
    rm -f "$ALLOC_FLAG"

done

echo "All training complete."
