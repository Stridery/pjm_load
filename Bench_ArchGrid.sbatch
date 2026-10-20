#!/bin/bash
# GARCHFormer grid search on the PJM point-forecast task, two wirings, on PACE.
#
# One array task per (zone, mode, wiring) -- 8 tasks. Inside a task the 65 cells x 5 seeds run
# EIGHT AT A TIME, one thread each (xargs -P), through scripts/grid_archformer.py. The grid is
# the public-dataset one (src/bench/garchformer.cells) plus the shipped reference cell.
#
#   sbatch Bench_ArchGrid.sbatch                              # dom+bge x direct+residual x hp+skip
#   sbatch --array=0-3 Bench_ArchGrid.sbatch                  # dom only
#   sbatch --array=1,5 Bench_ArchGrid.sbatch                  # residual/hp on both zones
#   sbatch --export=AG_SEEDS="0 1 2" Bench_ArchGrid.sbatch    # fewer seeds
#   sbatch --export=PJM_CONGESTION=1,PJM_LMP=1 Bench_ArchGrid.sbatch   # the _cong_lmp arm
#
#   task index = zone*4 + mode*2 + wiring    with zone (dom, bge), mode (direct, residual),
#                                            wiring (hp, skip)
#   AG_SEEDS      seed list, default "0 1 2 3 4" -- FIVE, because the comparison the collector
#                 makes is over the 5-seed ensemble; single ARCHFormer runs on PJM move by
#                 0.1-0.3 MAPE between seeds
#   AG_PARALLEL   cells at once, default $SLURM_CPUS_PER_TASK
#   AG_CELLS      optional file "d_model num_heads layers lambda" per line, instead of the grid
#
# Idempotent: a cell with its .json and .npz on disk is skipped, so a killed task is resumed by
# resubmitting. Writes only under results/{zone}/_tuning/archgrid_*/ -- no weights, nothing in
# evaluation/, so the scoreboard is untouched.
#
# Cost: ~2 min per run single-threaded (estimate; 32-66 s was measured on 8 threads), 325 runs
# per task, 8 at a time -> ~1.5 h per task. Memory: the dom matrix is ~90 MB per process.
#
#SBATCH --job-name=archgrid
#SBATCH --account=gts-sd111
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/archgrid_%A_%a.out
#SBATCH --error=logs/archgrid_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source .venv/bin/activate

ZONES=(dom bge); MODES=(direct residual); WIRINGS=(hp skip)
i=$SLURM_ARRAY_TASK_ID
export PJM_DATASET="${ZONES[$((i / 4))]}"
MODE="${MODES[$(((i / 2) % 2))]}"
WIRING="${WIRINGS[$((i % 2))]}"
export PJM_CONGESTION="${PJM_CONGESTION:-1}" PJM_LMP="${PJM_LMP:-0}"
SEEDS="${AG_SEEDS:-0 1 2 3 4}"
PAR="${AG_PARALLEL:-$SLURM_CPUS_PER_TASK}"
export PYTHONPATH=.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ARM=$(python -c "import src.config as c; print(c.MATRIX_TAG)")
CDIR="results/${PJM_DATASET}/_tuning/archgrid_${MODE}${ARM}/cells"
[ -f "data/${PJM_DATASET}/matrix/X_3d_lb168_h0${ARM}.npy" ] || {
    echo "matrix data/${PJM_DATASET}/matrix/X_3d_lb168_h0${ARM}.npy missing -- build it first"; exit 1; }

LIST="logs/archgrid_${SLURM_ARRAY_JOB_ID}_${PJM_DATASET}_${MODE}_${WIRING}.cells"
python - "$MODE" "$WIRING" "$SEEDS" "${AG_CELLS:-}" <<'PY' > "$LIST"
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from grid_archformer import all_cells
mode, wiring, seeds, subset = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4]
if subset:
    grid = [dict(zip(('d_model', 'num_heads', 'layers', 'lambda_cons'), (int(a), int(b), int(c), float(d))))
            for a, b, c, d in (l.split() for l in open(subset) if l.strip())]
else:
    grid = all_cells()
for s in seeds:
    for c in grid:
        print(mode, wiring, c['d_model'], c['num_heads'], c['layers'], f"{c['lambda_cons']:g}", s)
PY
N=$(wc -l < "$LIST")
_ncells()  { { find "$CDIR" -maxdepth 1 -name "${WIRING}_*.json" 2>/dev/null || true; } | wc -l; }
_nfailed() { { grep -l '^=== done rc=[^0]' logs/ag_${PJM_DATASET}_${MODE}_${WIRING}_*.out 2>/dev/null || true; } | wc -l; }
DONE0=$(_ncells)

echo "============================================================"
echo "zone $PJM_DATASET  arm $ARM  mode $MODE  wiring $WIRING  seeds [$SEEDS]"
echo "runs $N in this task, $DONE0 already on disk (skipped) | $PAR at a time, 1 thread each"
echo "host $(hostname)  start $(date)"
echo "============================================================"

export MODE WIRING
xargs -P "$PAR" -L1 bash -c '
    log="logs/ag_${PJM_DATASET}_${0}_${1}_d${2}_h${3}_L${4}_lam${5}_s${6}.out"
    python scripts/grid_archformer.py --mode "$0" --wiring "$1" --d-model "$2" --num-heads "$3" \
        --layers "$4" --lambda-cons "$5" --seed "$6" > "$log" 2>&1
    rc=$?; echo "=== done rc=$rc $(date)" >> "$log"; exit 0
' < "$LIST"

DONE1=$(_ncells); FAILED=$(_nfailed)
echo; echo "cells on disk $DONE0 -> $DONE1    failed logs $FAILED    $(date)"
if [ "$FAILED" -gt 0 ]; then grep -l '^=== done rc=[^0]' logs/ag_${PJM_DATASET}_${MODE}_${WIRING}_*.out | sed 's/^/  /'; fi
python scripts/grid_archformer_collect.py --mode "$MODE" || true
echo; echo "done $(date)   results: results/${PJM_DATASET}/_tuning/archgrid_${MODE}${ARM}/{grid.csv,grid_summary.csv,best.json}"
[ "$FAILED" -eq 0 ]
