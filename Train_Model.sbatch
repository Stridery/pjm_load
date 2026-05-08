#!/bin/bash
#SBATCH --job-name=pjm_train
#SBATCH --account=gts-sd111
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2:00:00
#SBATCH -q inferno
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

cd $SLURM_SUBMIT_DIR

source .venv/bin/activate

echo "=============================="
echo "Training: DOM dataset"
echo "=============================="
PJM_DATASET=dom python Model_Training.py

echo "=============================="
echo "Training: BGE dataset"
echo "=============================="
PJM_DATASET=bge python Model_Training.py

echo "All training complete."
