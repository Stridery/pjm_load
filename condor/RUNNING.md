# Running training on the ISyE cluster (HTCondor)

Two machines. You never log into a compute node — condor puts the jobs there for you.

> **The roles are fixed by the ISyE cluster guide and are not interchangeable.**
>
> | role | machine | what it is for |
> |---|---|---|
> | Gateway Server | `castle`, `keep` | connecting in from outside; SCP/SFTP |
> | Central Manager | `isye-hps0400` | collector + negotiator |
> | Submit Server | `compute01` (= `isye-hps0401`) | editing and submitting jobs |
> | Execution Machines | `condor_status -compact` | condor puts your jobs here |
>
> Step 1 below runs on castle because it is **network and filesystem work** — fetching data
> and unpacking a venv — not computation. A gateway is not an execution machine: anything
> that would otherwise be a condor job belongs in a condor job. `condor/bench_gf.sh local`
> enforces this and refuses to run on a gateway or on the submit server.

---

## Step 1 — on `castle`: environment and data

One-time. Neither step needs condor, and both are too slow for the submit host.

```bash
ssh castle.isye.gatech.edu
git clone <repo> && cd <clone>          # or cd into an existing clone

./condor/run_all.sh setup               # ~2 min   uv + standalone CPython 3.12 + .venv
./condor/run_all.sh crawl               # ~25 min  both zones, needs internet
```

Both skip themselves if already done, so re-running is free.

Do **not** build the venv by hand with `python3 -m venv`: that points at
`/usr/bin/python3.12`, which the execute nodes do not have, and every job then dies with
`No module named numpy`. `setup` uses `uv` so the interpreter lives in `$HOME`.

Check it worked:

```bash
ls data/dom/cleaned/        # 4 csv files
ls -l .venv/bin/python      # must resolve into $HOME, not /usr/bin
```

---

## Step 2 — on the submit host: train

```bash
ssh compute01
cd <clone>
```

**First time (no matrices yet):**

```bash
./condor/run_all.sh submit
```

`submit` is a subcommand: it ONLY submits. It does not repeat step 1 — that is what the bare
`./condor/run_all.sh` (setup + crawl + submit) would do. What it submits is the DAG, whose
first stage is 24 condor jobs that build the matrices **on the execute nodes**, and only then
396 training jobs. Building matrices is compute work, so it belongs on condor rather than on
castle.

**Every later run (matrices already on disk):**

```bash
./condor/make_jobs.sh               # 396 jobs
condor_submit condor/Train.sub      # training only — skips the matrix stage
```

Which case you are in:

```bash
ls data/dom/matrix/         # empty or missing -> use run_all.sh submit
```

---

## Train a subset

Same two commands as above, with variables in front of `make_jobs.sh`:

```bash
PJM_TARGET=rt_price         ./condor/make_jobs.sh   # load | rt_price          
PJM_DATASET=dom             ./condor/make_jobs.sh   # dom | bge
PJM_CONGESTION=1 PJM_LMP=1  ./condor/make_jobs.sh   # one arm; each is 1 or 0
PJM_TRAIN_ONLY=archformer   ./condor/make_jobs.sh   # one model, or `residual` for all nine

condor_submit condor/Train.sub
```

Combine freely. Anything you leave out runs all of its values.

---

## Watch it (submit host)

```bash
condor_watch_q
./condor/run_all.sh status          # queue + anything held
./condor/run_all.sh failures        # after it finishes
```

## Read results (either machine)

```bash
./condor/run_all.sh scoreboard      # tables + per-metric PNGs, all six result roots
```

---

## Example: archformer

```bash
# --- castle, once ---
ssh castle.isye.gatech.edu && cd <clone>
./condor/run_all.sh setup
./condor/run_all.sh crawl

# --- submit host ---
ssh compute01 && cd <clone>

PJM_TARGET=load PJM_TRAIN_ONLY=archformer,archformer_residual ./condor/make_jobs.sh
#   train  16 jobs   = 2 models x 2 regions x 4 arms

condor_submit condor/Train.sub
condor_watch_q

# --- results ---
PJM_TARGET=load PJM_DATASET=dom .venv/bin/python scripts/run_foundation.py --scoreboard
```

---

## Gotchas

- Always run `condor_submit` **from the repo root** — the submit files use relative paths.
- `logs/` must exist before submitting, or jobs go on hold.
- Do not raise `request_memory` / `request_disk` "for safety". They are filters: padding them
  shrinks the set of matching slots and slows the whole run down.
