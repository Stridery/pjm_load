#!/bin/bash
# Full retrain: crawl fresh data + train every model, on both zones.
# Submit with:  sbatch Train_Model.sbatch
# The job itself holds one GPU — it sits in the queue until a GPU is free (no timeout,
# no CPU fallback), then runs the whole pipeline for dom and bge in turn.

#SBATCH --job-name=pjm_train
#SBATCH --account=gts-sd111
#SBATCH --partition=gpu-v100
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source .venv/bin/activate

DATASETS="dom bge"

for DATASET in $DATASETS; do
    echo "============================================================"
    echo "Zone: $DATASET   |   start $(date)"
    echo "============================================================"

    echo "--- Crawl + rebuild cleaned data ($DATASET) ---"
    PJM_DATASET=$DATASET python run_crawler.py \
        || { echo "Crawl failed for $DATASET — aborting."; exit 1; }

    echo "--- Train + evaluate all models ($DATASET) ---"
    PJM_DATASET=$DATASET python Model_Training.py \
        || { echo "Training failed for $DATASET — aborting."; exit 1; }

    echo "Zone $DATASET done   |   $(date)"
done

echo "============================================================"
echo "All training complete: $(date)"
