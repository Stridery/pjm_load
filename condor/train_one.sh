#!/bin/bash
# STAGE 2, one job per (target, zone, arm, MODEL). Trains and evaluates exactly one model.
#
# The matrices are already on disk (stage 1), so this process only reads them. Every artifact
# it writes is under models/{zone}{tag}/{model}/... and results/{zone}{tag}/evaluation/{model}/...
# — no two jobs in this stage share an output path, which is what makes 360 of them safe to
# run at once.
set -u
source .venv/bin/activate

# torch and the boosters both size their thread pools from the PHYSICAL core count, which on
# these machines is many times what the slot actually granted. Left alone, a 4-core slot runs
# ~100 threads over 4 cores and is slower than single-threaded. Condor exports the request in
# _CONDOR_ASSIGNED_CPUS / OMP_NUM_THREADS on newer versions; pin it either way.
_cpus="${OMP_NUM_THREADS:-${_CONDOR_ASSIGNED_CPUS:-4}}"
export OMP_NUM_THREADS="$_cpus" MKL_NUM_THREADS="$_cpus" OPENBLAS_NUM_THREADS="$_cpus"

echo "=== $(hostname) $(date) $PJM_TARGET/$PJM_DATASET cong=$PJM_CONGESTION lmp=$PJM_LMP"
echo "    model=$PJM_TRAIN_ONLY  threads=$_cpus"

# Fail before spending a slot if the arm has nothing to read. Same guard the sbatch had.
if [ "$PJM_CONGESTION" != 0 ] && [ ! -f "data/$PJM_DATASET/cleaned/congestion.csv" ]; then
    echo "data/$PJM_DATASET/cleaned/congestion.csv missing — run the crawler first."; exit 1
fi
if [ "$PJM_LMP" != 0 ] && [ ! -f "data/$PJM_DATASET/cleaned/lmp_da.csv" ]; then
    echo "data/$PJM_DATASET/cleaned/lmp_da.csv missing — run the crawler first."; exit 1
fi

python scripts/Model_Training.py
rc=$?
echo "=== done rc=$rc $(date)"
exit $rc
