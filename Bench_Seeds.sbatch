#!/bin/bash
# ARCHFormer seed study: how much of a single run's score is the model and how much is the
# draw, and what averaging several draws gives. Measurement only -- no model weights.
#
# ONE VARIABLE PER CONCEPT — the same names src/config.py reads, so a command line and a
# submission differ only in whether a variable holds one value or several:
#
#   PJM_DATASET     zones     one for python, a space-separated list here    default "dom bge"
#   PJM_CONGESTION  arms      0 or 1 for python, a list here                 default "1"
#   PJM_LMP         arms      0 or 1 for python, a list here                 default "0"
#   SEED_MODES      modes     "direct residual" | "residual" | "direct"      default both
#   SEED_N          seeds     how many per variant; default from config      default ""
#   SEED_ONLY       variants  archformer | archformer_hp                     default both
#
#   sbatch Bench_Seeds.sbatch                                          # dom+bge, _cong, both modes
#   sbatch --export="ALL,PJM_DATASET=bge" Bench_Seeds.sbatch           # one zone
#   sbatch --export="ALL,SEED_N=10" Bench_Seeds.sbatch                 # tighter error bars
#   sbatch --export="ALL,PJM_CONGESTION=1 0,PJM_LMP=1 0" Bench_Seeds.sbatch   # all four arms
#
# WHY THIS EXISTS
#   The shipped runs under results/{zone}/evaluation/ set no torch seed -- src/models/
#   _seq_trainer.py never calls manual_seed -- so every sequence-model row in the scoreboard,
#   not just ARCHFormer's, is one undocumented draw. ARCHFormer's spread is sd 0.115 on dom
#   against transformer's 0.029, and the gap between the two models is 0.22 pp, so single-seed
#   rankings between them are not decidable. This job makes the distribution explicit.
#
#   It also measures the one intervention that has worked: averaging the seeds' PREDICTIONS
#   (not their scores) gains 0.23-0.60 pp in every zone and mode measured so far, and on dom
#   puts archformer_hp_residual at 2.987, below every single-seed transformer_residual run.
#
# TWO VARIANTS, NOT THREE
#   archformer      Moyi Guo's recipe, the baseline. Architecture untouched.
#   archformer_hp   same architecture, this dataset's recipe (huber / 5e-4 / wd 1e-4 / dr 0.2).
#   archformer_skip is deliberately excluded: it scored 3.380 against hp's 3.242 on the same
#   arm, and it is the one variant that changes where features enter, so repeating it would be
#   measuring a design that already lost.
#
# COST
#   2 variants x 2 modes x SEED_N runs per zone-arm. ARCHFormer takes 32-66 s per run, so the
#   default (2 zones, 1 arm, 5 seeds) is 40 runs, well under an hour.

#SBATCH --job-name=pjm_seeds
#SBATCH --account=gts-sd111
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source .venv/bin/activate

# Read once, then unset: the loops hand each child ONE value per variable, and src/config.py
# rejects a list outright.
_zones="${PJM_DATASET:-dom bge}"
_congs="${PJM_CONGESTION:-1}"
_lmps="${PJM_LMP:-0}"
_modes="${SEED_MODES:-direct residual}"
_n="${SEED_N:-}"
_only="${SEED_ONLY:-}"
unset PJM_DATASET PJM_CONGESTION PJM_LMP

_combos=$(( $(echo $_zones | wc -w) * $(echo $_congs | wc -w) * $(echo $_lmps | wc -w) ))
echo "============================================================"
echo "zones:   $_zones   congestion: $_congs   lmp: $_lmps"
echo "modes:   $_modes   seeds: ${_n:-from ARCHFORMER_PARAMS['n_seeds']}"
echo "variants: ${_only:-archformer, archformer_hp}"
echo "arm combinations: $_combos"
echo "start $(date)"
echo "============================================================"

_seed_flag=""; [ -n "$_n" ]    && _seed_flag="--seeds $_n"
_only_flag=""; [ -n "$_only" ] && _only_flag="--variant $_only"

_done=0
_failed=0
for ZONE in $_zones; do
  for CONG in $_congs; do
    for LMP in $_lmps; do
      _done=$((_done + 1))
      echo ""
      echo "------------------------------------------------------------"
      echo "[$_done/$_combos] zone=$ZONE cong=$CONG lmp=$LMP | $(date)"
      PJM_DATASET=$ZONE PJM_CONGESTION=$CONG PJM_LMP=$LMP python -c \
          "import src.config as c; print(f'  MATRIX_TAG={c.MATRIX_TAG!r}  n_seq={c.N_SEQ_FEATURES}  n_seeds={c.ARCHFORMER_PARAMS[\"n_seeds\"]}  results->{c.RESULT_ROOT}')" \
          || { echo "Config rejected zone=$ZONE cong=$CONG lmp=$LMP — aborting."; exit 1; }
      echo "------------------------------------------------------------"

      PJM_DATASET=$ZONE PJM_CONGESTION=$CONG PJM_LMP=$LMP \
          python -u scripts/bench_seeds.py \
              --mode $_modes \
              $_seed_flag \
              $_only_flag \
          || { echo "!! bench_seeds failed for zone=$ZONE cong=$CONG lmp=$LMP"; _failed=$((_failed + 1)); }
    done
  done
done

# --- Scoreboard, once per zone, after everything ----------------------------------------
for ZONE in $_zones; do
    echo ""
    echo "--- Scoreboard $ZONE | $(date) ---"
    PJM_DATASET=$ZONE python scripts/run_foundation.py --scoreboard || true
done

echo ""
echo "============================================================"
echo "done $(date)   failures: $_failed"
echo "results:  results/{zone}/_tuning/seed_study_{direct,residual}{arm}.csv"
echo "          results/{zone}/evaluation/{variant}_ens{N}/"
echo "============================================================"
[ "$_failed" -eq 0 ]
