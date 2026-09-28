#!/bin/bash
# Soft-gated Mixture-of-Experts vs hard calendar routing vs no mixture. Measurement only:
# no model weights are written, and nothing already in evaluation/ is overwritten.
#
# ONE VARIABLE PER CONCEPT — the same names src/config.py reads, so a command line and a
# submission differ only in whether a variable holds one value or several:
#
#   PJM_DATASET     zones     one for python, a space-separated list here    default "dom bge"
#   PJM_CONGESTION  arms      0 or 1 for python, a list here                 default "1"
#   PJM_LMP         arms      0 or 1 for python, a list here                 default "0"
#   SMOE_MODES      modes     "direct residual" | "residual" | "direct"      default both
#   SMOE_SEEDS      seeds     how many per variant                           default 3
#   SMOE_ONLY       variants  subset, space separated                        default all four
#
# Quote the WHOLE --export when any value contains a space:
#
#   sbatch Bench_SMoE.sbatch                                              # dom+bge, _cong, both modes
#   sbatch --export="ALL,PJM_DATASET=dom" Bench_SMoE.sbatch               # one zone
#   sbatch --export="ALL,SMOE_SEEDS=5" Bench_SMoE.sbatch
#   sbatch --export="ALL,SMOE_ONLY=smoe_z smoe_cal" Bench_SMoE.sbatch     # skip the controls
#
# WHAT IS BEING COMPARED, AND WHY TWO OF THE FOUR ARE CONTROLS
#   transformer        the plain model, 119,576 parameters
#   transformer_wide   ONE head widened to ~173,000 parameters. Without it a win by the
#                      mixtures is unreadable: they carry ~54k more parameters than the plain
#                      head, all of it in the extra experts, so beating `transformer` could
#                      just be capacity. Beating THIS is what would mean the mixture helps.
#   smoe_z             soft gate reading all of z (encoder output + every static feature)
#   smoe_cal           soft gate reading ONLY the 6 target-day calendar channels. This is the
#                      softened form of moe_transformer's month rule: still calendar-driven,
#                      but the boundaries are learned and no sample is withheld from any
#                      expert.
#
#   The incumbent moe_transformer hard-routes on month and splits the 24-hour output between
#   experts by hour band, which divides 1461 training days twelve ways -- 1,428 to 7,018
#   target scalars per expert against 35,064 for a single head. It loses to the plain
#   transformer by 0.087 pp in residual mode and 0.827 pp in direct mode. Its rows are already
#   in the scoreboard; this job does not re-run it.
#
# READ THE GATE BEFORE THE MAPE
#   Each smoe variant prints, after its seeds:
#       gate mean [.. .. .. ..]   per-sample max 0.xx   [verdict]
#   Both numbers matter and the mean alone misleads. A flat mean with a HIGH per-sample max
#   means the gate does route decisively, just not to the same expert every day -- which is
#   the outcome the whole idea is betting on. A flat mean with a flat per-sample max means no
#   routing happened, and that is a real result: it says the premise that different regimes
#   want different predictors is unsupported here, however the split is drawn.
#
# WHY BOTH MODES
#   Hard routing loses ~10x more in direct mode than in residual mode, so direct is where a
#   fix has the most room to show. Dropping it to save an hour drops the informative half.
#
# COST
#   4 variants x 2 modes x SEEDS runs PER zone-arm. The transformer takes 280-390 s per run,
#   so the default (2 zones, 1 arm, 3 seeds) is 48 runs, about 5 h. --time is set past that.

#SBATCH --job-name=pjm_smoe
#SBATCH --account=gts-sd111
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source .venv/bin/activate

# Read once, then unset: the loops hand each child ONE value per variable, and src/config.py
# rejects a list outright. A child that forgot its per-iteration assignment would otherwise
# inherit the list and die in config's guard rather than quietly run the wrong thing.
_zones="${PJM_DATASET:-dom bge}"
_congs="${PJM_CONGESTION:-1}"
_lmps="${PJM_LMP:-0}"
_modes="${SMOE_MODES:-direct residual}"
_seeds="${SMOE_SEEDS:-3}"
_only="${SMOE_ONLY:-}"
unset PJM_DATASET PJM_CONGESTION PJM_LMP

_n=$(( $(echo $_zones | wc -w) * $(echo $_congs | wc -w) * $(echo $_lmps | wc -w) ))
echo "============================================================"
echo "zones:   $_zones   congestion: $_congs   lmp: $_lmps"
echo "modes:   $_modes   seeds: $_seeds"
echo "variants: ${_only:-all four (transformer, transformer_wide, smoe_z, smoe_cal)}"
echo "arm combinations: $_n"
echo "start $(date)"
echo "============================================================"

_only_flag=""
[ -n "$_only" ] && _only_flag="--only $_only"

_done=0
_failed=0
for ZONE in $_zones; do
  for CONG in $_congs; do
    for LMP in $_lmps; do
      _done=$((_done + 1))
      echo ""
      echo "------------------------------------------------------------"
      echo "[$_done/$_n] zone=$ZONE cong=$CONG lmp=$LMP | $(date)"
      # Print the resolved arm before training, so a log shows which matrix was read rather
      # than which switches were set.
      PJM_DATASET=$ZONE PJM_CONGESTION=$CONG PJM_LMP=$LMP python -c \
          "import src.config as c; print(f'  MATRIX_TAG={c.MATRIX_TAG!r}  n_seq={c.N_SEQ_FEATURES}  results->{c.RESULT_ROOT}')" \
          || { echo "Config rejected zone=$ZONE cong=$CONG lmp=$LMP — aborting."; exit 1; }
      echo "------------------------------------------------------------"

      PJM_DATASET=$ZONE PJM_CONGESTION=$CONG PJM_LMP=$LMP \
          python -u scripts/bench_smoe.py \
              --mode $_modes \
              --seeds $_seeds \
              $_only_flag \
          || { echo "!! bench_smoe failed for zone=$ZONE cong=$CONG lmp=$LMP"; _failed=$((_failed + 1)); }
    done
  done
done

# --- Scoreboard, once per zone, after everything ----------------------------------------
# The ensemble rows land in evaluation/{variant}_ens{N}/ and are picked up by the ordinary
# scan, so the table and the per-metric chart both refresh with no special case.
for ZONE in $_zones; do
    echo ""
    echo "--- Scoreboard $ZONE | $(date) ---"
    PJM_DATASET=$ZONE python scripts/run_foundation.py --scoreboard || true
done

echo ""
echo "============================================================"
echo "done $(date)   failures: $_failed"
echo "results:  results/{zone}/_tuning/smoe_{direct,residual}{arm}.csv"
echo "          results/{zone}/evaluation/{variant}_ens${_seeds}/"
echo "============================================================"
[ "$_failed" -eq 0 ]
