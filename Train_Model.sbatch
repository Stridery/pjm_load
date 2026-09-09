#!/bin/bash
# Retrain models across TARGETS, zones and feature-set arms. Sits in the queue until a node
# is free.
#
# ONE VARIABLE PER CONCEPT — the same names src/config.py reads, so a command line and a
# submission differ only in whether a variable holds one value or several. There is no
# sbatch-only spelling of a config knob to get out of sync with:
#
#   PJM_TARGET      classes    one for python, a space-separated list here
#                              default "load load_met rt_price"
#   PJM_DATASET     zones      one for python, a list here                   default "dom bge"
#   PJM_CONGESTION  arms       0 or 1 for python, a list here                default "1 0"
#   PJM_LMP         arms       0 or 1 for python, a list here                default "1 0"
#   PJM_TRAIN_ONLY  models     comma-separated names, or `residual`          default: all
#   PJM_CRAWL       1 = refresh the data first                               default 0
#
# Quote the WHOLE --export when any value contains a space; sbatch splits its argument on
# commas, and the shell would otherwise end the argument at the space:
#
#   sbatch Train_Model.sbatch                                            # everything: 3x2x4
#   sbatch --export="ALL,PJM_TARGET=load_met" Train_Model.sbatch         # one class, 2 zones, 4 arms
#   sbatch --export="ALL,PJM_TARGET=rt_price,PJM_DATASET=dom" Train_Model.sbatch
#   sbatch --export="ALL,PJM_CRAWL=1" Train_Model.sbatch                 # refresh the data first
#
# THE THREE TARGET CLASSES  (src/config.TARGET_SPECS)
#   load      metered load from PRELIMINARY load history. The original task.
#   load_met  metered load from METERED history, on TimesFM's information geometry: the
#             lookback window steps back one day because PJM publishes metered about a day
#             after preliminary. Same issue time, so the forecast-weather and Day-Ahead
#             blocks are unchanged.
#   rt_price  Real-Time zonal price, same cutoff and target day.
#
#   Artifacts carry the class in their ROOT, so the three never collide:
#       models/{zone}{TARGET_TAG}/...        results/{zone}{TARGET_TAG}/...
#   TARGET_TAG is '' for load, '_met' for load_met, '_rtprice' for rt_price.
#
# RESIDUAL MODELS
#   Nine of the eighteen regress on the deviation from a same-hour-last-week baseline built
#   from the class's own load channel. That baseline is in MW, so it is meaningful for the
#   two LOAD classes and a category error against a $/MWh target — src/config marks that with
#   `residual_ok` and Model_Training.py drops them for rt_price with a printed reason. Nothing
#   here needs to know: the loop asks for every model and the config decides.
#
# WHY THE CRAWL SITS OUTSIDE EVERY LOOP
#   The test split is a FRACTION of the cleaned CSV (tail 0.16), so extending the CSV moves
#   the test boundary. Anything crawled mid-run would be scored on a different test set and
#   the difference between two runs would no longer be the thing under test. Crawl once per
#   zone, up front, then run every (target, arm) against that same CSV.
#
# THE FOUR ARMS
#   PJM_CONGESTION and PJM_LMP are independent switches and the loops take their cross
#   product:  ''  |  _cong  |  _lmp  |  _cong_lmp
#   Each also runs the zero-shot TimesFM benchmark on the same arm, so the foundation rows
#   land in the same results tree and the scoreboard at the end covers everything.

#SBATCH --job-name=pjm_train
#SBATCH --account=gts-sd111
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source .venv/bin/activate

# Read once, then unset: the loops below hand each child ONE value per variable, and
# src/config.py rejects a list outright. Leaving a list exported would mean a child that
# forgot its per-iteration assignment inherits the list and fails with config's guard rather
# than quietly running the wrong thing.
_targets="${PJM_TARGET:-load load_met rt_price}"
_zones="${PJM_DATASET:-dom bge}"
_congs="${PJM_CONGESTION:-1 0}"
_lmps="${PJM_LMP:-1 0}"
_crawl="${PJM_CRAWL:-0}"
unset PJM_TARGET PJM_DATASET PJM_CONGESTION PJM_LMP
export PJM_TRAIN_ONLY="${PJM_TRAIN_ONLY:-}"   # config resolves this one, including `residual`

_n=$(( $(echo $_targets | wc -w) * $(echo $_zones | wc -w) \
       * $(echo $_congs | wc -w) * $(echo $_lmps | wc -w) ))
echo "============================================================"
echo "targets: $_targets"
echo "zones:   $_zones   congestion: $_congs   lmp: $_lmps"
echo "models:  ${PJM_TRAIN_ONLY:-all (TRAIN_CONFIG)}"
echo "combinations: $_n   crawl first: $_crawl"
echo "start $(date)"
echo "============================================================"

# --- 1. Crawl, ONCE per zone, before any training ---------------------------------------
# Outside every loop on purpose (see the header). run_crawler also wipes data/{zone}/matrix*,
# so all three target classes rebuild against the same cleaned CSV.
if [ "$_crawl" = "1" ]; then
    for ZONE in $_zones; do
        echo "--- Crawling $ZONE | $(date) ---"
        PJM_DATASET=$ZONE python run_crawler.py \
            || { echo "Crawl failed for $ZONE — aborting."; exit 1; }
    done
else
    echo "--- Skipping crawl (PJM_CRAWL=$_crawl); using the cleaned CSVs already on disk ---"
fi

# --- 2. Train: target x zone x arm -------------------------------------------------------
_done=0
for TARGET in $_targets; do
  for ZONE in $_zones; do
    for CONG in $_congs; do
      for LMP in $_lmps; do
        _done=$((_done + 1))
        echo "------------------------------------------------------------"
        echo "[$_done/$_n] target=$TARGET zone=$ZONE cong=$CONG lmp=$LMP | $(date)"
        PJM_TARGET=$TARGET PJM_DATASET=$ZONE PJM_CONGESTION=$CONG PJM_LMP=$LMP python -c \
            "import src.config as c; print(f'  TARGET_TAG={c.TARGET_TAG!r}  MATRIX_TAG={c.MATRIX_TAG!r}  "\
"target={c.TARGET_COL} ({c.TARGET_UNIT})  series={c.SERIES_COL}  lag={c.TARGET_LAG_DAYS}d  "\
"residual_ok={c.TARGET_SPEC[\"residual_ok\"]}')" \
            || { echo "Config rejected target=$TARGET zone=$ZONE cong=$CONG lmp=$LMP — aborting."; exit 1; }
        echo "------------------------------------------------------------"

        # Fail before spending compute if an arm has nothing to read.
        if [ "$CONG" != "0" ] && [ ! -f "data/$ZONE/cleaned/congestion.csv" ]; then
            echo "data/$ZONE/cleaned/congestion.csv missing — run with PJM_CRAWL=1. Aborting."
            exit 1
        fi
        if [ "$LMP" != "0" ] && [ ! -f "data/$ZONE/cleaned/lmp_da.csv" ]; then
            echo "data/$ZONE/cleaned/lmp_da.csv missing — run with PJM_CRAWL=1. Aborting."
            exit 1
        fi

        # One combination is ~100 min of the 18 models. A failure here is reported and the
        # loop continues: losing one arm should not cost the twenty-three that follow.
        PJM_TARGET=$TARGET PJM_DATASET=$ZONE PJM_CONGESTION=$CONG PJM_LMP=$LMP \
            python Model_Training.py \
            || echo "!! Training FAILED: target=$TARGET zone=$ZONE cong=$CONG lmp=$LMP — continuing."

        # Zero-shot TimesFM on the same arm. Nothing is trained, so a failure must not cost
        # the models that just succeeded. Its matrix is already built by the run above, which
        # is why it comes second. rt_price has no residual mode (its baseline would be the
        # price's own weekly mean, which is left for a separate experiment), so only the
        # modes that class supports are asked for.
        if [ "$TARGET" = "rt_price" ]; then _fmodes="direct"; else _fmodes="direct residual"; fi
        for FMODE in $_fmodes; do
            PJM_TARGET=$TARGET PJM_DATASET=$ZONE PJM_CONGESTION=$CONG PJM_LMP=$LMP \
                python run_foundation.py --model timesfm --mode $FMODE \
                || echo "!! TimesFM $FMODE failed: target=$TARGET zone=$ZONE cong=$CONG lmp=$LMP — continuing."
        done

        echo "[$_done/$_n] done | $(date)"
      done
    done
  done
done

echo "============================================================"
echo "All training complete: $(date)"
echo "Next: ./bundle_serving.sh   (needs an authenticated gh CLI, so run it from a login node)"
echo "============================================================"

# --- 3. One scoreboard per (target, zone) ------------------------------------------------
# Read back out of the metrics files rather than kept in memory, so a partial run still
# produces a scoreboard for whatever finished.
for TARGET in $_targets; do
  for ZONE in $_zones; do
    echo "============================================================"
    PJM_TARGET=$TARGET PJM_DATASET=$ZONE python run_foundation.py --scoreboard \
        || echo "Scoreboard failed for target=$TARGET zone=$ZONE — the per-model metrics files are still on disk."
  done
done
echo "all done $(date)"
