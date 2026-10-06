#!/bin/bash
# Write condor/jobs_bench_gf.txt — one line per (dataset, d_model, num_heads, layers,
# lambda_cons, seed). Run it on the submit host before condor_submit; the list IS the run
# configuration, so narrowing a run means regenerating it.
#
#   ./condor/make_bench_gf_jobs.sh                        # 3 datasets x 64 cells x 1 seed
#   GF_DATASET=spain ./condor/make_bench_gf_jobs.sh       # one dataset
#   GF_SEEDS="0 1 2" ./condor/make_bench_gf_jobs.sh       # three seeds -> 576 jobs
#
#   GF_DATASET   default "spain ercot elia"
#   GF_SEEDS     default "0"
#
# THE GRID ITSELF IS NOT HERE. It lives in src/bench/garchformer.py (D_MODELS, NUM_HEADS,
# LAYERS, LAMBDAS, cells()), and this script asks that module to enumerate it. One
# definition, so the job list and scripts/bench_garchformer_collect.py's completeness check
# cannot drift apart — a bash copy of the grid is exactly the kind of duplicate that gets
# edited once.
#
# WHY ONE JOB PER CELL
#   Same argument as condor/make_jobs.sh. A run's wall clock is its slowest UNIT, and the
#   slowest cell here is ~9 min (elia, d_model 128, 2 layers) against ~14 h for the whole
#   grid in one process. Nothing is shared between cells: no data is written, no weights are
#   saved, and each one writes a single JSON named after itself, so no two jobs touch the
#   same path.

set -eu
cd "$(dirname "$0")/.."

_datasets="${GF_DATASET:-spain ercot elia}"
_seeds="${GF_SEEDS:-0}"

mkdir -p condor logs
: > condor/jobs_bench_gf.txt

for DS in $_datasets; do
  # Slots. MEASURED single-threaded on one cell each (d_model 128, 2 layers, the heaviest):
  #     ercot   2.6 s/epoch   416 train days   X (416,168,15)    ~2 min for 50 epochs
  #     spain   6.5 s/epoch   869 train days   X (869,168,27)    ~5 min
  #     elia   17.3 s/epoch  2239 train days   X (2239,672,8)   ~14 min
  # The tensors are 4-48 MB, the model at most 1.26 M parameters, and attention runs over
  # SEVEN tokens — so this is a small job and 4G is already generous. Re-measure and tighten
  # after the first cluster:
  #     condor_history -constraint 'ClusterId == <id>' \
  #         -af RequestMemory MemoryUsage RequestCpus RemoteWallClockTime
  # Too low is not silent: the job is HELD with `memory usage exceeded`.
  # ONE core per job. Not a thrift: torch's multi-threaded CPU kernels SIGILL on these
  # hosts (5 of 6 spain cells died at 2 threads, 0 of 6 at 1), so the jobs run
  # single-threaded either way -- see DEFAULT_PARAMS['threads'] in src/bench/garchformer.py.
  # Asking for 1 is then strictly better: a request is a FILTER, and on the PJM run an
  # oversized one cut the matching slots from 235 to 53.
  CPUS=1
  MEM=4G
  for SEED in $_seeds; do
    .venv/bin/python - "$DS" "$SEED" "$CPUS" "$MEM" <<'PY' >> condor/jobs_bench_gf.txt
import sys
sys.path.insert(0, '.')
from src.bench.garchformer import cells
ds, seed, cpus, mem = sys.argv[1:5]
for c in cells():
    print(f"{ds}, {c['d_model']}, {c['num_heads']}, {c['layers']}, "
          f"{c['lambda_cons']:g}, {seed}, {cpus}, {mem}")
PY
  done
done

# What the divisibility rule removed, printed rather than left to be inferred from a count.
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, '.')
from src.bench.garchformer import D_MODELS, NUM_HEADS, LAYERS, LAMBDAS, cells, skipped
sk = skipped()
print(f"grid: d_model {list(D_MODELS)} x heads {list(NUM_HEADS)} x layers {list(LAYERS)} "
      f"x lambda {list(LAMBDAS)}")
print(f"      {len(list(cells()))} legal cells per dataset per seed"
      + (f"; dropped as indivisible: {sk}" if sk else ""))
PY

printf 'bench_gf  %4d jobs\n' "$(wc -l < condor/jobs_bench_gf.txt)"
