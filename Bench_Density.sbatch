#!/bin/bash
# Baseline A (Li, Wang & Zhang 2020, density combination) on the three public datasets.
#
# One job per dataset, submitted as an array, because the three differ by ~20x in size and a
# single job would have to be sized for elia. Each task runs all 15 base models across 3 seeds.
#
#   sbatch Bench_Density.sbatch                     # all three datasets
#   sbatch --array=0 Bench_Density.sbatch           # spain only
#   sbatch --array=1,2 Bench_Density.sbatch         # ercot and elia
#
#   BENCH_SEEDS   space-separated seed list         default "0 1 2"
#   BENCH_MODELS  subset of base model names        default: all 15
#
#   sbatch --export=ALL,BENCH_SEEDS="0" --array=0 Bench_Density.sbatch    # one seed, spain
#
#SBATCH --job-name=bench_density
#SBATCH --account=gts-sd111
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/bench_density_%A_%a.out
#SBATCH --error=logs/bench_density_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source .venv/bin/activate

DATASETS=(spain ercot elia)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
SEEDS="${BENCH_SEEDS:-0 1 2}"

# One worker per allocated core; each pins BLAS to one thread, so the parallelism is across
# base models and never inside them.
export PYTHONPATH=.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "=== ${DS} | seeds ${SEEDS} | ${SLURM_CPUS_PER_TASK} workers | $(date) ==="
python3 -u scripts/train_bench_density.py \
        --dataset "$DS" \
        --seeds $SEEDS \
        --workers "$SLURM_CPUS_PER_TASK" \
        ${BENCH_MODELS:+--models $BENCH_MODELS}
echo "=== done $(date) ==="
