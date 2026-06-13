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

# ---------------------------------------------------------------------------
# Dataset list.
#   Single zone  : 'dom', 'bge', 'dom2', ...
#   Joint model  : 'joint'  →  zones are taken from JOINT_ZONES in src/config.py
# ---------------------------------------------------------------------------
DATASETS="dom2"

for DATASET in $DATASETS; do
    echo "=============================="
    echo "Dataset: $DATASET"
    echo "=============================="

    if [ "$DATASET" = "joint" ]; then
        # Joint model reads JOINT_ZONES from config; no per-zone crawler needed
        # (individual zone cleaned CSVs must already exist)
        echo "--- Joint training (zones from config.JOINT_ZONES) ---"
        ALLOC_FLAG="/tmp/pjm_gpu_alloc_${SLURM_JOB_ID}_joint"
        timeout 1200 salloc \
            -A gts-sd111 -p gpu-v100 -q inferno \
            -N 1 -c 4 --mem=16G -t 2:00:00 --gres=gpu:1 \
            bash -c "touch $ALLOC_FLAG \
                     && cd $SLURM_SUBMIT_DIR \
                     && source .venv/bin/activate \
                     && python train_joint.py"
        if [ ! -f "$ALLOC_FLAG" ]; then
            echo "GPU timed out — running joint training on CPU."
            python train_joint.py
        fi
        rm -f "$ALLOC_FLAG"

    else
        # Single-zone: fetch weather (cached) then train
        echo "--- Data pipeline for $DATASET ---"
        PJM_DATASET=$DATASET python run_crawler.py
        if [ $? -ne 0 ]; then
            echo "Data pipeline failed for $DATASET — aborting."
            exit 1
        fi

        echo "--- Training models for $DATASET ---"
        ALLOC_FLAG="/tmp/pjm_gpu_alloc_${SLURM_JOB_ID}_${DATASET}"
        timeout 1200 salloc \
            -A gts-sd111 -p gpu-v100 -q inferno \
            -N 1 -c 4 --mem=16G -t 2:00:00 --gres=gpu:1 \
            bash -c "touch $ALLOC_FLAG \
                     && cd $SLURM_SUBMIT_DIR \
                     && source .venv/bin/activate \
                     && PJM_DATASET=$DATASET python Model_Training.py"
        if [ ! -f "$ALLOC_FLAG" ]; then
            echo "GPU timed out — training on CPU."
            PJM_DATASET=$DATASET python Model_Training.py
        fi
        rm -f "$ALLOC_FLAG"
    fi

done

echo "All training complete."
