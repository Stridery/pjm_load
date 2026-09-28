#!/bin/bash
# One driver for the whole retrain on the ISyE HTCondor pool.
#
#     ./condor/run_all.sh              # everything: venv -> crawl -> submit the full DAG
#
# Every step is idempotent and skips itself when its output already exists, so re-running
# after a failure resumes rather than redoing. Steps can also be run one at a time:
#
#     ./condor/run_all.sh setup        # build .venv (python3.11) and install requirements
#     ./condor/run_all.sh crawl        # fetch data for both zones, superset, on THIS host
#     ./condor/run_all.sh submit       # generate the job lists and submit the full DAG
#     ./condor/run_all.sh smoke        # optional: submit 1 prebuild + 1 model and WAIT
#
# There is no smoke test in the default path on purpose. The DAG is its own safety net: the
# 24 prebuild jobs are short and every training job is their child, so if the execute nodes
# turn out not to mount the NFS home, stage 1 fails and stage 2 is never submitted. Nothing
# is lost but a few minutes of queue time. `smoke` stays available for when a single job's
# output is easier to read than 24 of them.
#     ./condor/run_all.sh status       # what is queued / running right now
#     ./condor/run_all.sh failures     # which jobs exited non-zero
#     ./condor/run_all.sh scoreboard   # read the metrics off disk (run when the DAG is done)
#
# The run can be narrowed with the SAME variables the sbatch used — they are passed straight
# through to condor/make_jobs.sh, which turns each into a list of jobs:
#
#     PJM_TARGET=rt_price ./condor/run_all.sh submit
#     PJM_DATASET=dom PJM_TRAIN_ONLY=archformer ./condor/run_all.sh submit
#
# The whole thing is long (the crawl alone downloads ~230 MB of LMP shards), so for the full
# path run it detached:
#
#     nohup ./condor/run_all.sh > logs/run_all.log 2>&1 &
#     tail -f logs/run_all.log

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

# Throttle for the full submission. 360 jobs in the queue is fine for the schedd, but every
# RUNNING job holds a shadow process on this submit host, and its banner asks for light use.
MAXJOBS="${MAXJOBS:-100}"
MAXIDLE="${MAXIDLE:-30}"

_say()  { printf '\n=== %s | %s\n' "$*" "$(date '+%H:%M:%S')"; }
_die()  { printf '\n!!! %s\n' "$*" >&2; exit 1; }

_need_condor() {
    command -v condor_submit >/dev/null \
        || _die "no condor_submit on $(hostname). Submit from compute01 or isye-hps0401 — castle has no condor binaries."
}

# ---------------------------------------------------------------------------------------
UV=$HOME/.local/bin/uv

# THE VENV MUST NOT DEPEND ON A SYSTEM INTERPRETER.
#
# A venv holds packages and a SYMLINK; it does not hold a Python. Built the ordinary way on a
# submit host, .venv/bin/python3.12 points at /usr/bin/python3.12 — and the execute nodes do
# not have that file. (ISyE's own example, /opt/examples/basic_python_example/basic.cmd, puts
# Python at /opt/python3/latest instead, but that is 3.7.3 and this project needs >=3.9 for
# pandas 2, numpy 1.24 and torch 2.) The symptom is not a clean error: bash skips the dangling
# symlink, falls through to the system `python`, and the job dies on `No module named numpy`
# — or on `python: command not found` where there is no system python at all.
#
# So the interpreter itself has to live on the shared filesystem. `uv python install` fetches
# a standalone CPython build into ~/.local/share/uv/python/, which every node sees over NFS,
# and `uv venv --python` points the venv at THAT instead of at /usr/bin.
_venv_is_portable() {
    [ -x .venv/bin/python ] || return 1
    local target; target=$(readlink -f .venv/bin/python 2>/dev/null) || return 1
    case "$target" in "$HOME"/*) return 0 ;; *) return 1 ;; esac
}

step_setup() {
    if _venv_is_portable; then
        _say "setup: .venv already points into \$HOME ($(readlink -f .venv/bin/python)) — skipping"
        return
    fi
    [ -e .venv ] && _say "setup: existing .venv resolves to $(readlink -f .venv/bin/python 2>&1) — outside \$HOME, rebuilding"

    if [ ! -x "$UV" ]; then
        _say "setup: installing uv into ~/.local/bin"
        curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 sh
        [ -x "$UV" ] || _die "uv install failed — expected $UV"
    fi

    _say "setup: fetching a standalone CPython 3.12 into ~/.local/share/uv"
    "$UV" python install 3.12

    _say "setup: building .venv"
    # RENAME, then delete in the background. Two things make a plain `rm -rf .venv` wrong
    # here: $HOME is NFS, so deleting ~10k files is slow, and if anything else is walking the
    # same tree (a second run of this script, an editor, a running job) the kernel hands back
    # ESTALE on entries the other process already unlinked. Under `set -e` that failed rm
    # takes the whole run with it. A rename is one atomic operation, frees the path
    # immediately, and cannot half-succeed.
    if [ -e .venv ]; then
        local old=".venv.old.$$"
        mv .venv "$old" || _die "could not move the old .venv aside"
        ( rm -rf "$old" >/dev/null 2>&1 & )
    fi
    "$UV" venv --python 3.12 .venv
    "$UV" pip install --python .venv/bin/python -r requirement.txt
    # CPU wheels on purpose: no stage of this run asks for a GPU (see condor/README.md).
    "$UV" pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
    "$UV" pip install --python .venv/bin/python 'timesfm[torch]' \
        || echo "  (timesfm not installed — the foundation stage will be skipped)"

    # The whole point of the rebuild. Check it rather than trusting it: this is the one
    # property that made 384 jobs fail last time, and it fails silently on the submit host,
    # where /usr/bin/python3.12 does exist.
    _venv_is_portable || _die "the rebuilt .venv still resolves to $(readlink -f .venv/bin/python) — it must live under \$HOME"
    _say "setup: done — .venv/bin/python -> $(readlink -f .venv/bin/python)"
    .venv/bin/python -c 'import numpy, pandas, torch; print("  imports OK  torch", torch.__version__)'
}

# ---------------------------------------------------------------------------------------
step_crawl() {
    local need=0
    for z in dom bge; do
        for f in cleaned_pjm_load_weather.csv forecast.csv congestion.csv lmp_da.csv; do
            [ -f "data/$z/cleaned/$f" ] || need=1
        done
    done
    if [ "$need" = 0 ] && [ "${FORCE_CRAWL:-0}" != 1 ]; then
        _say "crawl: all four cleaned files present for both zones — skipping (FORCE_CRAWL=1 to redo)"
        return
    fi

    # BOTH SWITCHES ON. The crawler is gated on them too (src/data_crawler/pipeline.py steps
    # 0b/0c), so PJM_LMP=0 would leave no lmp_da.csv and no Real-Time price columns in the
    # cleaned CSV — and then the twelve `_lmp` arms have nothing to build from. Crawl the
    # SUPERSET once; each arm selects from it at matrix-build time.
    #
    # On a login host, never in a job: extending the CSV moves the test boundary (tail 0.16),
    # so it must happen once before anything trains, and execute nodes may have no route to
    # the PJM and Open-Meteo APIs.
    for z in dom bge; do
        _say "crawl: $z"
        PJM_DATASET=$z PJM_CONGESTION=1 PJM_LMP=1 .venv/bin/python scripts/run_crawler.py \
            || _die "crawl failed for $z"
    done
    _say "crawl: done"
    ls -la data/dom/cleaned/ data/bge/cleaned/
}

# ---------------------------------------------------------------------------------------
# Submit one .sub, block until its jobs leave the queue, return their exit status.
_submit_and_wait() {
    local sub=$1 logpat=$2 outpat=$3
    local cluster
    cluster=$(condor_submit -terse "$sub" | head -1 | cut -d. -f1)
    [ -n "$cluster" ] || _die "condor_submit $sub produced no cluster id"
    _say "submitted $sub as cluster $cluster — waiting"
    # condor_wait blocks on the job event log until every job in it has finished.
    condor_wait "${logpat/CLUSTER/$cluster}" || _die "condor_wait failed for cluster $cluster"
    local out
    out=$(ls ${outpat/CLUSTER/$cluster} 2>/dev/null | head -1) || true
    [ -n "$out" ] || _die "no output file matching ${outpat/CLUSTER/$cluster}"
    echo "--- $out"
    cat "$out"
    grep -q 'rc=0' "$out" || _die "job did not report rc=0 — see $out and its .err"
}

step_smoke() {
    _need_condor
    _say "smoke: one prebuild + one model on the widest arm (load/dom/_cong_lmp)"
    # lightgbm is the cheapest of the eighteen, and _cong_lmp exercises the splice of BOTH
    # side files into the static block — the path the other 23 combinations depend on. The
    # matrix it builds is one of the 24 the full run needs, so nothing here is throwaway.
    PJM_TARGET=load PJM_DATASET=dom PJM_CONGESTION=1 PJM_LMP=1 PJM_TRAIN_ONLY=lightgbm \
        ./condor/make_jobs.sh

    # This is the step that proves the two things that cannot be checked from a submit host:
    # that the execute nodes mount the NFS home (should_transfer_files = NO), and that the
    # venv's interpreter exists there.
    _submit_and_wait condor/Prebuild.sub 'logs/prebuild.CLUSTER.log' 'logs/prebuild.*.CLUSTER.out'
    _submit_and_wait condor/Train.sub    'logs/train.CLUSTER.log'    'logs/*_lightgbm.CLUSTER.out'

    _say "smoke: artifacts"
    ls -R results/dom/evaluation/lightgbm/ 2>/dev/null | head -20
    _say "smoke: PASSED"
}

# ---------------------------------------------------------------------------------------
step_submit() {
    _need_condor
    _say "submit: generating job lists"
    ./condor/make_jobs.sh

    # Costs a second and catches a malformed .sub NOW. Without it a bad Train.sub would only
    # surface after the 24 prebuild jobs had already run, since DAGMan submits each node's
    # file only when its turn comes.
    _say "submit: validating submit-file syntax"
    condor_submit -dry-run /dev/null condor/Prebuild.sub >/dev/null || _die "condor/Prebuild.sub rejected"
    condor_submit -dry-run /dev/null condor/Train.sub    >/dev/null || _die "condor/Train.sub rejected"
    if [ -s condor/jobs_foundation.txt ]; then
        condor_submit -dry-run /dev/null condor/Foundation.sub >/dev/null \
            || _die "condor/Foundation.sub rejected"
    fi

    # The foundation stage needs the timesfm package AND its ~1 GB checkpoint already on
    # disk — foundation_one.sh sets HF_HUB_OFFLINE=1 so a missing checkpoint fails fast
    # instead of hanging on a node that has no route out. Fetch it here, on the submit host,
    # into the shared HF_HOME all forty jobs read; drop the stage if that is not possible.
    #
    # DOWNLOAD ONLY — do not run a benchmark to trigger the fetch. scripts/run_foundation.py scores
    # against the target matrix, which stage 1 of this very DAG has not built yet, so using it
    # as a warm-up fails on a missing matrix and drops a stage that was perfectly fine.
    # from_pretrained is the call that actually downloads, and it needs nothing on disk.
    local dag=condor/train.run.dag
    cp condor/train.dag "$dag"
    local ok=0
    if .venv/bin/python -c 'import timesfm3' 2>/dev/null; then
        _say "submit: warming the TimesFM checkpoint into $HOME/hf_cache"
        HF_HOME="$HOME/hf_cache" .venv/bin/python -c "
import src.config as cfg
from timesfm3 import TimesFM3Forecaster
ckpt = cfg.FOUNDATION_CONFIG['adapters']['timesfm']['checkpoint']
print('  fetching', ckpt)
TimesFM3Forecaster.from_pretrained(ckpt, device='cpu')
print('  cached')
" && ok=1
    else
        _say "submit: timesfm not installed"
    fi
    if [ "$ok" != 1 ]; then
        _say "submit: dropping the foundation stage (prebuild and train are unaffected)"
        sed -i 's/^JOB foundation/# JOB foundation/; s/CHILD train foundation/CHILD train/' "$dag"
        : > condor/jobs_foundation.txt
    fi

    _say "submit: condor_submit_dag -maxjobs $MAXJOBS -maxidle $MAXIDLE $dag"
    condor_submit_dag -maxjobs "$MAXJOBS" -maxidle "$MAXIDLE" -force "$dag"

    cat <<EOF

Submitted. Watch it with:

    cd $ROOT
    ./condor/run_all.sh status
    condor_watch_q
    condor_q -better-analyze <ClusterId>     # still idle? this says which requirement failed
    condor_q -hold                            # held? usually over request_memory
    tail -f logs/load_dom_c1_l1_archformer.*.out

When the DAG is done:

    ./condor/run_all.sh failures
    ./condor/run_all.sh scoreboard
EOF
}

# ---------------------------------------------------------------------------------------
step_status()  { _need_condor; condor_q; echo; condor_q -hold || true; }

step_failures() {
    _say "logs reporting a non-zero rc"
    grep -l 'rc=[^0]' logs/*.out 2>/dev/null || echo "  none"
    _say "condor's own record"
    condor_history -constraint 'ExitCode =!= 0' -af ClusterId ProcId Args ExitCode 2>/dev/null | head -40 || true
}

step_scoreboard() {
    # Reads the metrics files off disk rather than anything held in memory, so a partial run
    # still scores whatever finished.
    for t in ${PJM_TARGET:-load rt_price}; do
        for z in ${PJM_DATASET:-dom bge}; do
            _say "scoreboard: $t / $z"
            PJM_TARGET=$t PJM_DATASET=$z .venv/bin/python scripts/run_foundation.py --scoreboard \
                || echo "  no metrics yet for $t/$z"
        done
    done
}

# ---------------------------------------------------------------------------------------
mkdir -p logs condor

case "${1:-all}" in
    setup)      step_setup ;;
    crawl)      step_setup; step_crawl ;;
    smoke)      step_smoke ;;
    submit)     step_submit ;;
    status)     step_status ;;
    failures)   step_failures ;;
    scoreboard) step_scoreboard ;;
    all)        step_setup; step_crawl; step_submit ;;
    *)          sed -n '2,30p' "$0"; exit 1 ;;
esac
