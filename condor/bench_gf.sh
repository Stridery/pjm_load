#!/bin/bash
# One driver for the GARCHFormer grid search on the three public datasets.
#
#     ./condor/bench_gf.sh              # jobs -> submit, then tell you how to watch
#
# Steps can be run one at a time, and every one of them is idempotent:
#
#     ./condor/bench_gf.sh jobs         # regenerate condor/jobs_bench_gf.txt only
#     ./condor/bench_gf.sh submit       # generate the list and submit it
#     ./condor/bench_gf.sh status       # what is queued / running / held right now
#     ./condor/bench_gf.sh pool [jobid] # are there execute slots, and why is a job idle
#     ./condor/bench_gf.sh rm           # condor_rm this grid's cluster
#     ./condor/bench_gf.sh local        # LAST RESORT, see below — run the list on THIS machine
#     ./condor/bench_gf.sh progress     # how many cells have landed on disk
#     ./condor/bench_gf.sh failures     # which jobs exited non-zero
#     ./condor/bench_gf.sh collect      # read the cells off disk into one table (run at the end)
#     ./condor/bench_gf.sh clean        # delete the cell JSONs and start the grid over
#
# The run can be narrowed with the same two variables condor/make_bench_gf_jobs.sh reads:
#
#     GF_DATASET=spain ./condor/bench_gf.sh submit            # one dataset, 64 jobs
#     GF_SEEDS="0 1 2" ./condor/bench_gf.sh submit            # three seeds, 576 jobs
#
# WHERE THIS MAY RUN — the ISyE cluster guide's roles, which are not interchangeable
#
#     Gateway Servers    castle, keep        connect from outside; SCP/SFTP. NOT for compute.
#     Central Manager    isye-hps0400        collector + negotiator
#     Submit Server      compute01           where you edit and submit jobs
#     Execution Machines `condor_status -compact` lists them; condor puts jobs there for you
#
# So: submit from compute01, and let condor place the work. A gateway is for getting in and
# moving files — a grid search is ~15 CPU-hours and does not belong on one. `local` exists
# only for the case where the pool itself is unavailable, and it refuses to run on a gateway
# or on the submit server; see step_local.
#
# WHY THIS IS NOT A DAG
#   condor/run_all.sh submits condor/train.dag because the PJM matrices have to be built
#   before any model can read them — a real dependency, and a DAG is how you express it.
#   Here there is none: the three datasets under data/quantile/ are fixed inputs that every
#   job reads directly, nothing is crawled, nothing is cached and no weights are written. So
#   this is one flat cluster of independent jobs, which also means it needs no POST script —
#   condor/node_report.sh exists to stop one bad proc from aborting a DAG node, and there is
#   no node here to abort.
#
# WHAT IS DISTRIBUTED, EXACTLY
#   One condor job per GRID CELL: 64 legal cells x 3 datasets x the seed count. Each job
#   trains one ARCHFormer3 and writes one JSON named after its own cell, so no two jobs share
#   an output path and the whole set can run at once. A cell that already has its JSON exits 0
#   without retraining, so resubmitting after a partial run picks up only what is missing.

set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${GF_OUT:-results/bench}"
JOBS="${GF_JOBS:-condor/jobs_bench_gf.txt}"

_say() { printf '\n=== %s | %s\n' "$*" "$(date '+%H:%M:%S')"; }
_die() { printf '\n!!! %s\n' "$*" >&2; exit 1; }

_need_condor() {
    command -v condor_submit >/dev/null \
        || _die "no condor_submit on $(hostname).
     The ISyE guide's Submit Server is compute01 (= isye-hps0401): ssh to castle or keep,
     then \`ssh compute01\`. castle and keep are Gateway Servers and carry no condor
     binaries by design. compute02 (= isye-hps0402) also runs a schedd and will accept a
     submission, but it is not the documented submit host — prefer compute01."
}

_datasets() { echo "${GF_DATASET:-spain ercot elia}"; }

# The guide's Gateway Servers, by the hostname each one reports.
_GATEWAYS='isye-castle isye-keep castle keep'

_host_role() {
    local h; h=$(hostname -s 2>/dev/null || hostname)
    case " $_GATEWAYS " in *" $h "*) echo gateway; return;; esac
    case "$h" in isye-hps040[12]|compute0[12]) echo submit; return;; esac
    echo other
}

# Count a dataset's finished cells. NOT `ls ... | wc -l`: with `set -o pipefail` a glob that
# matches nothing makes ls exit non-zero, the pipeline inherits it, and `set -e` then kills
# the script -- so an empty grid printed a header and silently quit. find always exits 0.
# `|| true` INSIDE the braces, not after the pipe: with `set -o pipefail` the pipeline takes
# the worst status in it, so a find that fails on a not-yet-created directory would fail the
# assignment that calls this and `set -e` would kill the script -- which is exactly how an
# empty grid managed to print a header and quit.
_ncells() { { find "$OUT/$1/garchformer/cells" -maxdepth 1 -name '*.json' 2>/dev/null || true; } | wc -l; }

# ---------------------------------------------------------------------------------------

step_jobs() {
    _say "generating condor/jobs_bench_gf.txt"
    ./condor/make_bench_gf_jobs.sh
}

step_submit() {
    _need_condor
    [ -f data/quantile/spain_config.json ] \
        || _die "data/quantile/ is empty. The three public datasets are FIXED INPUTS and are
     not fetched by any job here — put the configs and cleaned CSVs in place first."
    step_jobs
    _say "submitting"
    condor_submit condor/BenchGarch.sub
    cat <<EOF

    watch it:     ./condor/bench_gf.sh status
    progress:     ./condor/bench_gf.sh progress
    when done:    ./condor/bench_gf.sh collect

    A cell that is EVICTED goes back to Idle and re-runs from the start — vanilla jobs hold
    no checkpoint, which is the argument for one job per cell. Slowest cell is elia at about
    14 min single-threaded; see condor/make_bench_gf_jobs.sh for the per-dataset timings.
EOF
}

step_local() {
    # LAST RESORT. Run the SAME job list on this one machine, N cells at a time.
    #
    # It exists because the pool's central manager (isye-hps0400) can go down while a schedd
    # stays up: condor_submit then succeeds, condor_q shows the queue, and nothing ever
    # matches because there is no negotiator. Jobs sit Idle with no error and no ETA.
    #
    # IT REFUSES TO RUN ON A GATEWAY OR ON THE SUBMIT SERVER, and that is the point of the
    # guard rather than a politeness. The ISyE guide gives castle and keep one job -- getting
    # in from outside, and SCP/SFTP -- and gives compute01 one job, editing and submitting.
    # Neither is an execution machine; condor puts work on those for you. A ~15 CPU-hour grid
    # on a gateway is simply the wrong machine, however idle it looks. GF_ALLOW_LOCAL=1 is
    # available for a host you know you are entitled to compute on.
    #
    # Not a second implementation of anything: condor/jobs_bench_gf.txt is the work list
    # either way, the per-cell entry point is the same script, and the cells are idempotent --
    # so a local run and a condor run can be mixed, each skipping what the other finished.
    #
    # ONE THREAD PER CELL, N CELLS IN PARALLEL -- the opposite of what looks natural, and it
    # is forced: torch's multi-threaded CPU kernels SIGILL on these hosts (see
    # DEFAULT_PARAMS['threads'] in src/bench/garchformer.py). Parallelism across PROCESSES is
    # therefore the only parallelism available, and it happens to be the better kind anyway --
    # 7 attention tokens give a second thread almost nothing to do.
    #
    # Measured per cell: peak RSS 747 MB (elia, the heaviest), so N=16 is ~12 GB.
    local role; role=$(_host_role)
    if [ "$role" != other ] && [ "${GF_ALLOW_LOCAL:-0}" != 1 ]; then
        case "$role" in
          gateway) _die "$(hostname -s) is a GATEWAY SERVER (ISyE guide: castle / keep are for
     connecting in and for SCP/SFTP). This grid is ~15 CPU-hours and does not belong here.
     Submit it through condor from compute01 instead:  ./condor/bench_gf.sh submit" ;;
          submit)  _die "$(hostname -s) is the SUBMIT SERVER (ISyE guide: compute01 is for
     editing and submitting jobs). Running the grid on it instead of through it defeats the
     scheduler and competes with every other user's shadow processes.
     Submit it:  ./condor/bench_gf.sh submit" ;;
        esac
    fi
    [ "${GF_ALLOW_LOCAL:-0}" = 1 ] && [ "$role" != other ] \
        && echo "    GF_ALLOW_LOCAL=1 set — overriding the $role-host guard on $(hostname -s)."
    local par="${GF_PARALLEL:-}"
    if [ -z "$par" ]; then par=$(( $(nproc) * 2 / 3 )); [ "$par" -lt 1 ] && par=1; fi
    [ -s "$JOBS" ] || step_jobs
    local n; n=$(wc -l < "$JOBS")
    mkdir -p logs
    _say "$n cells on $(hostname), $par at a time, 1 thread each"
    echo "    logs:     logs/gf_local_*.out  (one per cell, like condor's)"
    echo "    progress: ./condor/bench_gf.sh progress   (from any host -- results are on NFS)"
    echo "    stop:     Ctrl-C, or kill the xargs. Finished cells are already banked."
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    # Fields 1-6 of the itemdata; 7 and 8 are the condor slot size and mean nothing here.
    # -L1 hands one line's words to bash -c, so $0..$5 are dataset..seed.
    awk -F', *' '{print $1, $2, $3, $4, $5, $6}' "$JOBS" \
      | xargs -P "$par" -L1 bash -c '
            log="logs/gf_local_$0_d$1_h$2_L$3_lam$4_s$5.out"
            # nice 19: castle is a SHARED login host -- 2 other users and a 15-minute load
            # average of 4.6 while this was written. At the lowest priority the run takes only
            # cycles nobody else wants, so another user'"'"'s interactive work preempts it
            # instead of queueing behind it. It costs us nothing when the box is idle.
            nice -n 19 .venv/bin/python scripts/bench_garchformer.py \
                --dataset "$0" --d-model "$1" --num-heads "$2" --layers "$3" \
                --lambda-cons "$4" --seed "$5" --device cpu --out "'"$OUT"'" > "$log" 2>&1
            echo "=== done rc=$? $(date)" >> "$log"'
    _say "local run finished"
    step_progress
}

step_pool() {
    # Why a job is not running, in the order the answers come. `condor_status -totals` does
    # not exist -- the flag is `-total`, singular -- and plain `condor_status` with no slots
    # prints an empty table rather than saying "there are none", which is easy to misread.
    _need_condor
    _say "execute slots in the pool"
    local n; n=$(condor_status -af Name 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "    NONE. No startd is advertising, so nothing can ever match and the jobs will"
        echo "    sit Idle forever. compute01 (.191) and compute03 (.190) were both unreachable"
        echo "    from castle; if they were the execute nodes, this pool has no workers."
    else
        condor_status -compact || true
        echo
        condor_status -total || true
    fi
    _say "why the first job is idle"
    # The guide calls this `condor_q -better`; -better-analyze is its full spelling.
    condor_q -better-analyze "${1:-}" 2>/dev/null | tail -30 || true
}

step_status() {
    _need_condor
    condor_q
    echo
    _say "held (a held job is NOT going to run until you look at it)"
    condor_q -hold || true
}

step_progress() {
    # The real progress signal: one JSON lands per finished cell. condor_q only says what is
    # still queued, and a cell that finished before an eviction is already banked.
    local total=0 want seeds nds
    want=$(.venv/bin/python -c 'import sys; sys.path.insert(0, "."); from src.bench.garchformer import cells; print(len(list(cells())))')
    seeds=$(echo "${GF_SEEDS:-0}" | wc -w)
    nds=$(_datasets | wc -w)
    _say "cells on disk"
    for ds in $(_datasets); do
        local n; n=$(_ncells "$ds")
        total=$((total + n))
        printf '    %-6s %4d / %-4d %s\n' "$ds" "$n" "$((want * seeds))" \
               "$([ "$n" -eq $((want * seeds)) ] && echo complete || echo ...)"
    done
    printf '    %-6s %4d / %-4d\n' TOTAL "$total" "$((want * seeds * nds))"
}

step_rm() {
    # The guide's `condor_rm job_id`. Removing the queue loses nothing: finished cells are
    # already on disk and a resubmission skips them.
    _need_condor
    local cl="${1:-}"
    [ -n "$cl" ] || cl=$(condor_q -af ClusterId 2>/dev/null | sort -un | tail -1)
    [ -n "$cl" ] || _die "nothing of yours is queued"
    _say "condor_rm $cl"
    condor_rm "$cl"
}

step_failures() {
    _say "logs reporting a non-zero rc"
    grep -l '^=== done rc=[^0]' logs/gf_*.out 2>/dev/null || echo "  none"
    _say "condor's own record"
    if command -v condor_history >/dev/null; then
        condor_history -constraint 'ExitCode =!= 0' \
            -af ClusterId ProcId Args ExitCode 2>/dev/null | head -40 || true
    else
        echo "  (no condor_history on $(hostname))"
    fi
    cat <<'EOF'

    rc=2 is deliberate: an indivisible (d_model, num_heads) cell. make_bench_gf_jobs.sh does
    not generate any, so seeing one means the itemdata was hand-edited.
    rc=1 is the data guard in condor/bench_garch_one.sh — data/quantile/ not visible from
    that execute node, which usually means it did not mount the NFS home.
EOF
}

step_collect() {
    _say "collecting"
    .venv/bin/python scripts/bench_garchformer_collect.py \
        --dataset $(_datasets) --out "$OUT"
}

step_clean() {
    # Deliberately noisy and deliberately not -f. Deleting the cells throws away the whole
    # grid, and the only reason to do it is a change that invalidates every number.
    _say "this deletes every GARCHFormer cell under $OUT"
    for ds in $(_datasets); do
        printf '    %-6s %s cells\n' "$ds" "$(_ncells "$ds")"
    done
    read -r -p "    type 'yes' to delete: " ans
    [ "$ans" = yes ] || _die "not deleted"
    for ds in $(_datasets); do
        rm -f "$OUT/$ds/garchformer/cells"/*.json
    done
    _say "deleted"
}

case "${1:-all}" in
    jobs)      step_jobs ;;
    submit)    step_submit ;;
    status)    step_status ;;
    pool)      shift || true; step_pool "${1:-}" ;;
    local)     step_local ;;
    rm)        shift || true; step_rm "${1:-}" ;;
    progress)  step_progress ;;
    failures)  step_failures ;;
    collect)   step_collect ;;
    clean)     step_clean ;;
    all)       step_submit ;;
    *)         _die "unknown step '$1'. One of: jobs submit status pool rm local progress failures collect clean" ;;
esac
