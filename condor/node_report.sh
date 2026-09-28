#!/bin/bash
# DAGMan POST script. Run once, after every proc of a node has finished; ITS exit status
# becomes the node's status, and that is the whole point.
#
# WHY A NODE MUST NOT FAIL ON ONE BAD PROC
#   A DAG node here is not one job, it is a whole fan-out: `train` is 360 independent models,
#   `foundation` is 40 independent benchmarks. DAGMan's default is that ANY failed proc fails
#   the node — and a failed node makes it stop scheduling that node's remaining procs, then
#   abort the DAG and condor_rm everything of that node still queued.
#
#   That is exactly what happened on the first full run: one TimesFM job hit a config guard at
#   19:53, and at 20:47 DAGMan removed the 36 TimesFM jobs that had never got a slot. 36 good
#   runs lost to one bad one. For a pipeline stage whose output everything downstream needs
#   that behaviour is right; for a fan-out of independent results it is not.
#
#   So this script reports and exits 0. Nothing is swallowed: each job's real exit code is
#   still in its .out (`=== done rc=N`) and in condor_history, and
#   `./condor/run_all.sh failures` reads both. prebuild deliberately has NO post script —
#   if the matrices are not built, nothing downstream can run and stopping is correct.
NODE=${1:-?}
RET=${2:-?}

echo "=== POST $NODE | DAGMan reported node return $RET | $(date '+%F %T')"

# One line per job that did not report rc=0, taken from the logs the wrappers write.
_bad=$(grep -l '^=== done rc=[^0]' logs/*.out 2>/dev/null || true)
if [ -n "$_bad" ]; then
    echo "    jobs that did not exit 0:"
    for f in $_bad; do
        printf '      %-58s %s\n' "$(basename "$f")" "$(grep -o 'rc=[0-9]*' "$f" | tail -1)"
    done
    echo "    the DAG continues — see ./condor/run_all.sh failures"
else
    echo "    every job reported rc=0"
fi
exit 0
