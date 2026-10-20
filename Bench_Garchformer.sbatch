#!/bin/bash
# GARCHFormer (ARCHFormer3) grid search on the three public datasets, on PACE.
#
# One array task per dataset. Inside a task the 64 legal grid cells run EIGHT AT A TIME, one
# thread each, through xargs -P: a cell uses a single core (attention runs over seven tokens,
# so a second thread has nothing to do, and src/bench/garchformer.py pins torch to 1 thread),
# and an 8-core allocation running them serially would leave seven cores idle.
#
#   sbatch Bench_Garchformer.sbatch                                   # 3 datasets, seed 0
#   sbatch --array=2 Bench_Garchformer.sbatch                         # elia only
#   sbatch --export=GF_SEEDS="0 1 2" Bench_Garchformer.sbatch         # three seeds
#   sbatch --export=GF_SEEDS="1 2" --array=0 Bench_Garchformer.sbatch # top up spain
#
#   GF_SEEDS      space-separated seed list                     default "0"
#   GF_PARALLEL   cells running at once                          default $SLURM_CPUS_PER_TASK
#   GF_MODE       gaussian | sigma_scaled                        default gaussian
#                 interval construction only; point metrics and the val_MAPE ranking are
#                 identical in both. Each mode has its own result tree (garchformer/ vs
#                 garchformer_sigma_scaled/), so running one never touches the other.
#   GF_CELLS      optional file of cells to run instead of the whole grid, one per line:
#                 "d_model num_heads layers lambda" -- e.g. just the three best cells when
#                 all that is wanted is their sigma_scaled row.
#
#   sbatch --export=GF_MODE=sigma_scaled,GF_SEEDS="0 1 2" Bench_Garchformer.sbatch
#
# --export WITHOUT "ALL", on purpose. This script sets everything it needs itself (cwd, venv,
# PYTHONPATH, thread counts), so nothing from the submitting shell has to be carried in. And
# `--export=ALL,GF_SEEDS=...` has a trap this cluster's man page spells out: ALL takes
# precedence, so a GF_SEEDS already exported in your shell silently overrides the value on
# the command line. Listing only the variable you mean to pass avoids both.
#
# The per-cell entry point, the output layout and the collector are the ones the condor port
# uses (scripts/bench_garchformer.py, scripts/bench_garchformer_collect.py); nothing here is
# a second implementation. The grid itself lives in src/bench/garchformer.py and is read from
# there, so this file cannot drift from the collector's completeness check.
#
# Idempotent. A cell writes one JSON named after itself, atomically, and exits 0 without
# retraining if that JSON already exists -- so a task that is killed or times out can simply
# be resubmitted and will pick up only the cells it is missing. Seeds can be added the same
# way: cells already on disk for seed 0 are skipped when GF_SEEDS="0 1 2" is run later.
#
# Expected wall clock at 8-way parallelism, from the single-threaded per-cell timings in
# condor/make_bench_gf_jobs.sh (the heaviest cell, d_model 128 / 2 layers, is the ceiling):
#
#     ercot   ~2 min/cell    64 cells   ~16 min per seed
#     spain   ~5 min/cell               ~40 min per seed
#     elia   ~14 min/cell               ~2 h   per seed
#
# Memory: peak RSS 747 MB per cell (elia). Eight at once is ~6 GB; 32G is comfortable.
#
#SBATCH --job-name=bench_gf
#SBATCH --account=gts-sd111
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/bench_gf_%A_%a.out
#SBATCH --error=logs/bench_gf_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source .venv/bin/activate

DATASETS=(spain ercot elia)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
SEEDS="${GF_SEEDS:-0}"
PAR="${GF_PARALLEL:-$SLURM_CPUS_PER_TASK}"
export GF_MODE="${GF_MODE:-gaussian}"
MODEL_DIR="garchformer"; [ "$GF_MODE" = gaussian ] || MODEL_DIR="garchformer_${GF_MODE}"

export PYTHONPATH=.
# ONE THREAD PER CELL. The parallelism here is across processes, so the BLAS libraries and
# torch's intra-op pool must not each try to use the whole allocation -- eight cells times
# eight threads on eight cores would thrash. src/bench/garchformer.py also calls
# torch.set_num_threads(1) itself; these exports cover the BLAS libraries, which read them at
# load time.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

[ -f "data/quantile/${DS}_config.json" ] || {
    echo "data/quantile/${DS}_config.json missing -- the datasets are fixed inputs, not fetched."
    exit 1
}

# -- the work list: one line per (dataset, cell, seed), straight from the grid definition ----
LIST="logs/bench_gf_${SLURM_ARRAY_JOB_ID}_${DS}_${GF_MODE}.cells"
python - "$DS" "$SEEDS" "${GF_CELLS:-}" <<'PY' > "$LIST"
import sys
sys.path.insert(0, '.')
from src.bench.garchformer import cells
ds, seeds, subset = sys.argv[1], sys.argv[2].split(), sys.argv[3]
if subset:                       # GF_CELLS: explicit list, e.g. the best cell per dataset
    grid = [dict(zip(('d_model', 'num_heads', 'layers', 'lambda_cons'),
                     (int(a), int(b), int(c), float(d))))
            for a, b, c, d in (l.split() for l in open(subset) if l.strip())]
else:
    grid = list(cells())
for s in seeds:
    for c in grid:
        print(ds, c['d_model'], c['num_heads'], c['layers'], f"{c['lambda_cons']:g}", s)
PY
N=$(wc -l < "$LIST")
# Counting helpers. NOT a bare `find ... | wc -l` / `grep -l ... | wc -l`: under `set -o
# pipefail` a find on a directory that does not exist yet (first run) or a grep with no match
# (every cell succeeded) exits non-zero, the pipeline inherits it, and `set -e` then kills the
# script -- silently, since stderr is discarded. `|| true` goes INSIDE the braces, on the
# producer, so the pipeline's status is wc's. The first version of this file died exactly
# this way, at this line, 11 seconds into all three tasks.
_ncells() { { find "results/bench/${DS}/${MODEL_DIR}/cells" -maxdepth 1 -name '*.json' 2>/dev/null || true; } | wc -l; }
_nfailed() { { grep -l '^=== done rc=[^0]' logs/gf_${GF_MODE}_${DS}_*.out 2>/dev/null || true; } | wc -l; }
DONE0=$(_ncells)

echo "============================================================"
echo "dataset   $DS      seeds  $SEEDS      mode  $GF_MODE  ->  results/bench/$DS/$MODEL_DIR/"
echo "cells     $N in this run, $DONE0 already on disk (those are skipped)"
echo "parallel  $PAR cells at a time, 1 thread each, on $SLURM_CPUS_PER_TASK cores"
echo "host      $(hostname)    start $(date)"
echo "============================================================"

# -- run them: xargs keeps PAR processes alive, refilling from the list as each one exits ----
# -L1 hands one line's six fields to bash -c as $0..$5. Each cell gets its own log so eight
# concurrent writers never interleave; the cell JSON is written via .tmp + os.replace, so
# concurrent cells cannot corrupt each other's output either. The worker always exits 0 so
# one failed cell cannot take xargs and the 63 others down with it; the real rc is in the log.
xargs -P "$PAR" -L1 bash -c '
    log="logs/gf_${GF_MODE}_${0}_d${1}_h${2}_L${3}_lam${4}_s${5}.out"
    python scripts/bench_garchformer.py \
        --dataset "$0" --d-model "$1" --num-heads "$2" --layers "$3" \
        --lambda-cons "$4" --seed "$5" --device cpu --mode "$GF_MODE" > "$log" 2>&1
    rc=$?                         # captured BEFORE anything else can reset it
    echo "=== done rc=$rc $(date)" >> "$log"
    exit 0                        # one failed cell must not take down xargs and the rest
' < "$LIST"

# -- account for every cell, then collect --------------------------------------------------
DONE1=$(_ncells)
FAILED=$(_nfailed)
echo ""
echo "------------------------------------------------------------"
echo "cells on disk  $DONE0 -> $DONE1     failed logs  $FAILED     $(date)"
if [ "$FAILED" -gt 0 ]; then
    grep -l '^=== done rc=[^0]' logs/gf_${GF_MODE}_${DS}_*.out | sed 's/^/  /'
fi
echo "------------------------------------------------------------"

python scripts/bench_garchformer_collect.py --dataset "$DS" --mode "$GF_MODE" || true

echo ""
echo "done $(date)"
echo "results:  results/bench/${DS}/${MODEL_DIR}/{grid.csv,grid_summary.csv,best.json}"
[ "$FAILED" -eq 0 ]
