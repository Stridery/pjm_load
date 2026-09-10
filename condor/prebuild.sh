#!/bin/bash
# STAGE 1, one job per (target, zone, arm). Builds the two matrices that combination's
# eighteen training jobs will all read, and does nothing else.
#
# WHY THIS STAGE EXISTS
#   build_or_load_matrix and build_timeseries_matrix cache on FILENAME (src/feature_engine),
#   so eighteen training jobs starting on a cold cache would all decide to build and all write
#   the same X_3d_...npy at once. One writer first, then eighteen readers — that is the whole
#   content of the PARENT/CHILD line in train.dag.
set -u
source .venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "=== prebuild $(hostname) $(date) $PJM_TARGET/$PJM_DATASET cong=$PJM_CONGESTION lmp=$PJM_LMP"

# The arm's extra feature blocks are SEPARATE cleaned files, spliced into the static block at
# build time — they are not columns of the cleaned load CSV. The crawler is gated on the same
# two switches, so a crawl run with PJM_LMP=0 leaves nothing here for the twelve `_lmp` arms.
# Say so now, in one line, rather than as a KeyError from inside the builder.
if [ "$PJM_CONGESTION" != 0 ] && [ ! -f "data/$PJM_DATASET/cleaned/congestion.csv" ]; then
    echo "data/$PJM_DATASET/cleaned/congestion.csv missing — re-crawl with PJM_CONGESTION=1."; exit 1
fi
if [ "$PJM_LMP" != 0 ] && [ ! -f "data/$PJM_DATASET/cleaned/lmp_da.csv" ]; then
    echo "data/$PJM_DATASET/cleaned/lmp_da.csv missing — re-crawl with PJM_LMP=1."; exit 1
fi
python - <<'PY'
import os, shutil
import src.config as cfg
from src.feature_engine import build_or_load_matrix, build_timeseries_matrix

# BUILD INTO A SCRATCH DIRECTORY, THEN RENAME INTO PLACE.
#
# The builders write with to_csv/np.save — no temp file, no fsync, no rename — and decide
# "already built" from os.path.exists alone. A job evicted during the ~2 s it takes to write
# the 82 MB 2D CSV therefore leaves a TRUNCATED file that the next attempt reads as a cache
# hit: pandas parses a short CSV without complaining and the column-count guard still passes.
# (A truncated .npy at least fails loudly — np.load checks the length its header declares.)
#
# matrix_dir is a parameter, so the fix needs no change to feature_engine: build somewhere
# private, then os.replace each file into the real directory. rename(2) is atomic within a
# filesystem, NFS included, so a matrix is either absent or complete — never half-written.
# Building into an empty directory also means this stage always rebuilds, which is what a
# retry should do.
real = cfg.MATRIX_DIR
tmp  = os.path.join(real, f'.building_{os.getpid()}')
os.makedirs(tmp, exist_ok=True)
try:
    # Both builders take the module defaults, exactly as Model_Training.py calls them at all
    # ten of its call sites — so a combination has exactly one 2D and one 3D matrix.
    X, y = build_or_load_matrix(cfg.CLEANED_PATH, tmp)
    print(f'2D {X.shape} {y.shape}')
    X3, y3, mask, ts = build_timeseries_matrix(cfg.CLEANED_PATH, tmp)
    print(f'3D {X3.shape} {y3.shape}')

    for name in sorted(os.listdir(tmp)):
        os.replace(os.path.join(tmp, name), os.path.join(real, name))
        print('  ->', os.path.join(real, name))
finally:
    # Concurrent prebuild jobs for other combinations of the same zone share this directory,
    # so only ever remove our own scratch — never the directory itself.
    shutil.rmtree(tmp, ignore_errors=True)
PY
rc=$?
echo "=== prebuild done rc=$rc $(date)"
exit $rc
