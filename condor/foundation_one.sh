#!/bin/bash
# STAGE 2 sibling, one job per (target, zone, arm, mode). Zero-shot TimesFM — nothing is
# trained, so it does not wait for the training jobs, only for their matrices.
#
# NEEDS THE CHECKPOINT ALREADY ON DISK. requirement.txt leaves timesfm commented out and the
# ~1 GB checkpoint is a network fetch; execute nodes may have no route out. Fetch it once on
# the submit host into a shared HF_HOME (see condor/README.md) — HF_HUB_OFFLINE=1 below then
# makes a missing checkpoint an immediate, legible failure rather than a hung download.
set -u
source .venv/bin/activate
export HF_HUB_OFFLINE=1
_cpus="${OMP_NUM_THREADS:-4}"
export OMP_NUM_THREADS="$_cpus" MKL_NUM_THREADS="$_cpus"

echo "=== timesfm $(hostname) $(date) $PJM_TARGET/$PJM_DATASET cong=$PJM_CONGESTION lmp=$PJM_LMP mode=$PJM_FMODE"
python run_foundation.py --model timesfm --mode "$PJM_FMODE"
rc=$?
echo "=== done rc=$rc $(date)"
exit $rc
