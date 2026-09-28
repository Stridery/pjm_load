#!/bin/bash
# Baseline B (Zhang, Quan & Srinivasan 2019, iQRNN) on the three public datasets.
#
# One job per dataset, submitted as an array. Unlike the density baseline, the seeds run
# SEQUENTIALLY inside one process: feature selection happens once and is cached, and the
# three trainings then reuse it, so a job is RFE + 3 fits rather than 3 independent runs.
#
#   sbatch Bench_IQRNN.sbatch                      # all three datasets
#   sbatch --array=0 Bench_IQRNN.sbatch            # spain only
#   sbatch --array=1,2 Bench_IQRNN.sbatch          # ercot and elia
#
#   BENCH_SEEDS       space-separated seed list          default "0 1 2"
#   BENCH_SPEC        iQRNN-500 | iQRNN-500-ZD | iQRNN-4000 | ... | bQRNN-400
#                                                        default iQRNN-500
#   BENCH_NFEAT       override the RFE target            default min(100, pool/3)
#   BENCH_REFRESH_RFE 1 = recompute the cached selection  default 0
#
#   sbatch --export=ALL,BENCH_SEEDS="0" --array=1 Bench_IQRNN.sbatch      # one seed, ercot
#   sbatch --export=ALL,BENCH_SPEC="bQRNN-400" Bench_IQRNN.sbatch         # the paper's control
#
# GPU: this asks for CPU only. The network is small enough that CPU is workable, and RFE is
# sklearn and stays on CPU regardless. To use a GPU node, add the cluster's own --partition
# and --gres lines and pass BENCH_DEVICE=cuda.
#
#SBATCH --job-name=bench_iqrnn
#SBATCH --account=gts-sd111
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/bench_iqrnn_%A_%a.out
#SBATCH --error=logs/bench_iqrnn_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source .venv/bin/activate

DATASETS=(spain ercot elia)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
SEEDS="${BENCH_SEEDS:-0 1 2}"
SPEC="${BENCH_SPEC:-iQRNN-500}"
DEVICE="${BENCH_DEVICE:-auto}"

# torch and the RFE forest are the only threaded stages here, and they do not compete: RFE
# uses n_jobs=-1 and finishes before training starts. Let both use the whole allocation.
export PYTHONPATH=.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "=== ${DS} | ${SPEC} | seeds ${SEEDS} | device ${DEVICE} | $(date) ==="
python3 -u scripts/train_bench_iqrnn.py \
        --dataset "$DS" \
        --seeds $SEEDS \
        --spec "$SPEC" \
        --device "$DEVICE" \
        ${BENCH_NFEAT:+--n-features $BENCH_NFEAT} \
        ${BENCH_REFRESH_RFE:+--refresh-rfe}
echo "=== done $(date) ==="
