#!/bin/bash
# Write the three itemdata lists the .sub files consume. Run it on the submit host before
# every condor_submit_dag: the lists ARE the run configuration, and regenerating them is how
# you narrow a run to one zone or one model.
#
# SAME VARIABLE NAMES AS THE SBATCH, same "one value for python, a list here" rule, so a PACE
# submission and a Condor submission differ only in which scheduler reads the list:
#
#   PJM_TARGET      default "load load_met rt_price"
#   PJM_DATASET     default "dom bge"
#   PJM_CONGESTION  default "1 0"
#   PJM_LMP         default "1 0"
#   PJM_TRAIN_ONLY  comma-separated model names, or `residual`   default: all eighteen
#
#   ./condor/make_jobs.sh                                   # everything
#   PJM_TARGET=load_met ./condor/make_jobs.sh               # one class
#   PJM_TRAIN_ONLY=archformer,archformer_residual ./condor/make_jobs.sh
#
# WHY ONE JOB PER MODEL RATHER THAN ONE PER COMBINATION
#   The sbatch trained eighteen models in one process because it held one node for the whole
#   run and nothing was gained by splitting. Condor hands out slots, not nodes, so the wall
#   clock of a run is the wall clock of its SLOWEST UNIT — one 18-model process is ~100
#   minutes, one model is ~5-20. The split is only safe because every model writes into its
#   own results/.../{model}/{run_tag}/ directory and reads a matrix that the prebuild stage
#   already put on disk; see train.dag.

set -eu
cd "$(dirname "$0")/.."

_targets="${PJM_TARGET:-load load_met rt_price}"
_zones="${PJM_DATASET:-dom bge}"
_congs="${PJM_CONGESTION:-1 0}"
_lmps="${PJM_LMP:-1 0}"

# Two classes, because they want different SLOTS and different matrices. Trees fit 24 hourly
# boosters off the 2D CSV and use every core they are given; the nets train one small model
# off the 3D .npy and are fine with four. Memory is the same for both — see below.
_tree="xgboost lightgbm xgboost_residual lightgbm_residual"
_seq="transformer lstm moe_transformer mstnn moe_mstnn moe_lstm archformer
      transformer_residual lstm_residual moe_transformer_residual mstnn_residual
      moe_mstnn_residual moe_lstm_residual archformer_residual"

# PJM_TRAIN_ONLY is resolved HERE so the generated list is the whole truth about what will
# run. `residual` is the same shorthand src/config.py understands.
_only="${PJM_TRAIN_ONLY:-}"
_keep() {   # $1 = model name -> 0 if it should be generated
    [ -z "$_only" ] && return 0
    if [ "$_only" = residual ]; then case "$1" in *_residual) return 0;; *) return 1;; esac; fi
    case ",$_only," in *",$1,"*) return 0;; *) return 1;; esac
}

mkdir -p condor logs
: > condor/jobs_prebuild.txt
: > condor/jobs_train.txt
: > condor/jobs_foundation.txt

for TARGET in $_targets; do
  for ZONE in $_zones; do
    for CONG in $_congs; do
      for LMP in $_lmps; do
        echo "$TARGET, $ZONE, $CONG, $LMP" >> condor/jobs_prebuild.txt

        for M in $_tree $_seq; do
            _keep "$M" || continue
            # rt_price DOES have a residual baseline — the per-clock-hour mean of its own
            # price history over the lookback window (src/config TARGET_SPECS.baseline_col)
            # — but that history is the RT_Total_slog channel, which rides in the LMP
            # lookback block. With PJM_LMP=0 there is no price in the window to average, so
            # config.residual_baseline_reason() refuses and Model_Training drops the nine
            # residual models with a printed reason. Filter them here instead of queueing
            # 36 processes whose only output is that reason.
            if [ "$TARGET" = rt_price ] && [ "$LMP" = 0 ]; then
                case "$M" in *_residual) continue;; esac
            fi
            # MEASURED, not guessed. condor records what each job actually used, and the
            # first 52 jobs of this set reported:
            #
            #     MemoryUsage   0-977 MB   (trees 733 MB, nets 977 MB)
            #     DiskUsage          2 KB  — the shell's own log lines, nothing else
            #
            # The 2 KB is the whole point of should_transfer_files = NO: the job reads and
            # writes the project over NFS and never touches the execute node's scratch.
            #
            # This is not free money. A request is a FILTER — condor only considers slots
            # with at least that much free — so the original 32G/20G matched 43 of 235
            # slots and capped the run at ~22 concurrent. 8G is still 8x the measured peak.
            #
            # Re-measure after a full run and tighten again if the heavy models stay small:
            #   condor_history -constraint 'ClusterId == <id>' \
            #       -af RequestMemory MemoryUsage RequestCpus RemoteWallClockTime
            # Too low is not silent: the job is HELD with `memory usage exceeded`, and
            # `condor_q -hold` names it.
            case " $_tree " in
                *" $M "*) CPUS=8; MEM=8G ;;
                *)        CPUS=4; MEM=8G ;;
            esac
            echo "$TARGET, $ZONE, $CONG, $LMP, $M, $CPUS, $MEM" >> condor/jobs_train.txt
        done

        # Zero-shot TimesFM on the same arm. Nothing is trained, so it is a sibling of the
        # training jobs rather than something they wait for.
        #
        # TWO TARGET CLASSES, NOT THREE. src/foundation/_bench.py pairs each context
        # series with the one PJM_TARGET it may be scored against:
        #
        #     SERIES_TARGET = {'estimated': 'load', 'metered': 'load', 'rt_price': 'rt_price'}
        #
        # There is no entry for load_met, so run_one() refuses every load_met run with
        # "series='metered' forecasts Load, but PJM_TARGET='load_met' scores Load" — the
        # guard against scoring a forecast of one series against another series' y matrix.
        # Generating those 16 jobs anyway is how the first full run lost its TimesFM
        # stage: they failed, DAGMan failed the node, and 36 queued jobs were removed.
        #
        # NOTE this contradicts src/config._FOUNDATION_BY_TARGET, which carries a
        # 'load_met' entry and a comment saying load_met IS TimesFM's geometry. One of
        # the two is stale. Skipping here is the conservative reading: it changes no
        # benchmark semantics, where adding load_met to SERIES_TARGET would.
        if [ "$TARGET" != load_met ]; then
            if [ "$TARGET" = rt_price ]; then _fmodes="direct"; else _fmodes="direct residual"; fi
            for FMODE in $_fmodes; do
                echo "$TARGET, $ZONE, $CONG, $LMP, $FMODE" >> condor/jobs_foundation.txt
            done
        fi
      done
    done
  done
done

printf 'prebuild   %4d jobs\ntrain      %4d jobs\nfoundation %4d jobs\n' \
    "$(wc -l < condor/jobs_prebuild.txt)" \
    "$(wc -l < condor/jobs_train.txt)" \
    "$(wc -l < condor/jobs_foundation.txt)"
