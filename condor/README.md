# Running the retrain on the ISyE HTCondor pool

The Condor port of `Train_Model.sbatch`. Same variables, same artifacts, same scoreboard —
the only thing that changes is that the sbatch's four nested loops become a list of jobs the
pool runs at the same time.

    sbatch:  1 job  x 24 combinations x 18 models, serial     ~40 h
    condor:  396 jobs, one per (target, zone, arm, model)     ~ the slowest single model

## What one job actually gets

The four arms are FOUR DIFFERENT MATRICES on disk, not one matrix that a model takes columns
from. The cleaned CSV holds the base superset (load, weather, and the three Real-Time price
columns the crawler joins in); the forecast weather, the congestion statics and the Day-Ahead
LMP statics live in their own `cleaned/*.csv` and are spliced into the static block at
MATRIX-BUILD time, gated by `PJM_CONGESTION` / `PJM_LMP`. The result is a fixed-width matrix
whose width `_expected_widths()` derives from the config, tagged `''` / `_cong` / `_lmp` /
`_cong_lmp`, and a model's first layer is built for exactly that width — `_reject_stale()`
exists to stop a wider matrix reaching a narrower model.

So the 24 prebuild jobs are 3 targets x 2 zones x 4 arms, and none of them is redundant.

## Why one job per MODEL

Condor hands out slots carved from machines, not whole nodes, so a run's wall clock is the
wall clock of its slowest unit. Splitting is safe because of two facts about this codebase:

* every model writes into its own `models/{zone}{tag}/{model}/` and
  `results/{zone}{tag}/evaluation/{model}/{run_tag}/` — no two training jobs share an output
  path (`src/models/_eval_utils.py`);
* the only shared state is the matrix cache, and `src/feature_engine` keys it on FILENAME.
  Eighteen jobs starting cold would all build and all write the same `.npy`. Stage 1 of the
  DAG builds it once; stages 2 only read.

## No GPU

`Train_Model.sbatch` requests none either — it asks for 8 CPUs and 32 GB. The models are
small (200 epochs of ~113 steps at batch 32) and the ~100 min/combination figure in its
header is a CPU number. The pool has nine GPU machines and thousands of idle CPU cores, so
CPU jobs start now while GPU jobs queue. `request_gpus = 1` in `Train.sub` is one commented
line away if a net turns out to be worth it.

## Run it

One command, from the project root on a submit host:

    nohup ./condor/run_all.sh > logs/run_all.log 2>&1 &
    tail -f logs/run_all.log

`run_all.sh` walks the whole flow — build `.venv`, crawl both zones, then submit the DAG.
Every step skips itself when its output is already there, so re-running after a failure
resumes instead of redoing. The steps are also individually callable (`setup`, `crawl`,
`submit`, `smoke`, `status`, `failures`, `scoreboard`) and the same `PJM_*` variables narrow
a run:

    PJM_DATASET=dom PJM_TRAIN_ONLY=archformer ./condor/run_all.sh submit

There is no smoke test in the default path: the DAG is its own safety net. The 24 prebuild
jobs are short and every training job is their child, so if the execute nodes turn out not to
mount the NFS home, stage 1 fails and stage 2 is never submitted — the cost of finding out is
a few minutes of queue time. `./condor/run_all.sh smoke` runs a single prebuild + a single
model and blocks on them, for when one job's output is easier to read than 24.

## The same thing by hand

    ssh compute01                              # or isye-hps0401; castle has no condor binaries
    cd /home/zyou46/pjm_load

    # 0. ONCE: environment. castle's default python3 is 3.8; use 3.11.
    python3.11 -m venv .venv
    .venv/bin/pip install -U pip
    .venv/bin/pip install -r requirement.txt
    .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

    # 1. Crawl on a LOGIN host, never in a job. Extending the CSV moves the test boundary
    #    (tail 0.16), so it must happen once, before anything trains — and execute nodes may
    #    have no route to the PJM and Open-Meteo APIs.
    #
    #    BOTH SWITCHES ON, ALWAYS. The crawler is gated on them too
    #    (src/data_crawler/pipeline.py: steps 0b and 0c), so PJM_LMP=0 means no lmp_da.csv and
    #    no Real-Time price columns in the cleaned CSV — and then the twelve `_lmp` arms have
    #    nothing to build from. Crawl the SUPERSET once; each arm selects from it at
    #    matrix-build time. (Train_Model.sbatch got this by `unset`-ing both before the crawl
    #    so config fell back to its defaults of 1; spelling it out is the same thing, minus the
    #    chance that an exported PJM_LMP=0 in your shell quietly narrows the data.)
    for z in dom bge; do
      PJM_DATASET=$z PJM_CONGESTION=1 PJM_LMP=1 .venv/bin/python run_crawler.py
    done

    # 2. Generate the job lists, then submit the DAG.
    ./condor/make_jobs.sh                       # -> condor/jobs_*.txt
    condor_submit_dag condor/train.dag

    # 3. Watch
    condor_q                                    # DAGMan node + its jobs
    condor_watch_q
    condor_q -better-analyze <ClusterId>        # why is it still idle
    condor_q -hold                              # why was it held (usually: over request_memory)
    tail -f logs/load_dom_c1_l1_archformer.*.out

    # 4. Scoreboard, after the DAG reports done. Reads the metrics files off disk, so a
    #    partial run still scores whatever finished.
    for t in load load_met rt_price; do for z in dom bge; do
      PJM_TARGET=$t PJM_DATASET=$z .venv/bin/python run_foundation.py --scoreboard
    done; done

### TimesFM jobs

`requirement.txt` leaves `timesfm` commented out and the checkpoint is a ~1 GB download that
an execute node probably cannot reach. Fetch it once on the submit host, or drop
`condor/jobs_foundation.txt` to zero lines and skip the stage:

    .venv/bin/pip install 'timesfm[torch]'
    HF_HOME=/home/zyou46/hf_cache .venv/bin/python run_foundation.py --model timesfm --mode direct

## Narrowing a run

`make_jobs.sh` reads the same variables as the sbatch, one value or a space-separated list:

    PJM_TARGET=load_met ./condor/make_jobs.sh                    # one class
    PJM_DATASET=dom PJM_CONGESTION=1 ./condor/make_jobs.sh       # one zone, one arm
    PJM_TRAIN_ONLY=archformer,archformer_residual ./condor/make_jobs.sh

## Right-sizing the requests

A request is a FILTER, not a reservation: condor only considers slots with at least that much
free. Ask for more than the job uses and the run throttles itself. The first version of this
directory asked for 32G/20G per job and ran ~22 jobs at once out of 424; `condor_q
-better-analyze` showed why:

    Step   Slots Matched  Condition
    [0]        235        Arch == X86_64
    [3]         53        TARGET.Disk   >= RequestDisk        <- 20G cut 235 to 53
    [5]        127        TARGET.Memory >= RequestMemory
    [6]         43        [3] && [5]

What the jobs actually used, from `condor_history`:

    MemoryUsage   0-977 MB   (trees 733 MB, nets 977 MB)
    DiskUsage          2 KB  -- just the shell's log lines
    wall clock    46-1238 s  per model

The 2 KB is `should_transfer_files = NO` working as intended: the job reads and writes the
project over NFS and never touches the execute node's scratch. The current requests (8G
memory, 1G disk, 8 cores for trees / 4 for nets) keep ~8x headroom over that and stop being
the binding constraint.

Re-measure after a run and tighten again if the heavy models stay small:

    condor_history -constraint 'ClusterId == <id>' \
        -af RequestMemory MemoryUsage RequestDisk DiskUsage RequestCpus RemoteWallClockTime | head -40

Too small is not silent: the job is HELD with `memory usage exceeded` and `condor_q -hold`
names it. Disk carries no such risk here, since nothing is written to scratch.

To fix a run that is already queued without resubmitting it, edit the idle jobs in place:

    condor_qedit -constraint 'JobStatus == 1' RequestDisk 1048576     # KB
    condor_qedit -constraint 'JobStatus == 1 && RequestMemory == 16384' RequestMemory 8192

## Finding failures

A failed job does not stop the other 359 — but only because `train` and `foundation` carry a
POST script. DAGMan's default is that one failed proc fails the whole node, and a failed node
stops being scheduled and has its queued procs removed when the DAG winds up. On the first
full run one TimesFM job hit a config guard at 19:53 and 36 TimesFM jobs that had never got a
slot were removed at 20:47. `condor/node_report.sh` prints what failed and exits 0, so the
node succeeds and its siblings run. `prebuild` has no POST script on purpose: without the
matrices nothing downstream means anything.

After the DAG finishes:

    ./condor/run_all.sh failures
    grep -l "rc=[^0]" logs/*.out
    condor_history -constraint 'ExitCode =!= 0' -af ClusterId ProcId Args ExitCode

## Reading the results

Each target class gets its OWN results root, so `results/dom` holds `load` only:

    results/{zone}            load        18 models x 4 arms
    results/{zone}_met        load_met    18 models x 4 arms
    results/{zone}_rtprice    rt_price    18 models on the two _lmp arms, 9 on the others

rt_price's nine residual models need the price history their baseline averages, and that
channel rides in the LMP lookback block — so they exist on the `_lmp` and `_cong_lmp` arms
only. See "Residual models" in the repo README.

The scoreboard reads whichever root `PJM_TARGET` selects, so asking for it without the
variable silently reports `load`:

    PJM_TARGET=rt_price PJM_DATASET=bge .venv/bin/python run_foundation.py --scoreboard
    ./condor/run_all.sh scoreboard        # all three classes, both zones

## TimesFM covers two target classes, not three

`src/foundation/_bench.py` maps each context series to the single `PJM_TARGET` it may be
scored against — `{'estimated': 'load', 'metered': 'load', 'rt_price': 'rt_price'}` — and has
no entry for `load_met`, so every `load_met` benchmark is refused by that guard.
`make_jobs.sh` therefore generates 24 foundation jobs (16 `load` + 8 `rt_price`), not 40.

This contradicts `src/config._FOUNDATION_BY_TARGET`, which has a `load_met` entry and a
comment saying load_met is TimesFM's own geometry. One of the two is stale. Skipping is the
conservative reading; adding `load_met` to `SERIES_TARGET` would change what the benchmark
measures and is a decision for whoever owns that comparison.
