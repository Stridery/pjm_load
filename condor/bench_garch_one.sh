#!/bin/bash
# One GARCHFormer grid cell. Trains, scores, writes one JSON, exits.
#
# Arguments come from the environment, set by BenchGarch.sub out of the itemdata, so the
# submit file and this script share no argument ORDER that could silently shift.
#
# Nothing is fetched and nothing is cached. The three datasets under data/quantile/ are
# fixed inputs, read-only — there is no crawler step here and no prebuild stage, which is
# why this is a single .sub rather than a DAG like condor/train.dag.
set -u
source .venv/bin/activate

# ONE THREAD, and this is not slot hygiene -- it is a crash. torch's multi-threaded CPU
# kernels fault on these hosts: a spain cell at d_model 128 died with SIGILL in 5 of 6 runs at
# OMP_NUM_THREADS=2 and 0 of 6 at 1. The fault is inside `param.addcdiv_`
# (torch/optim/adam.py:544), an ATen vectorised kernel, and ATEN_CPU_CAPABILITY=default does
# not stop it. See DEFAULT_PARAMS['threads'] in src/bench/garchformer.py for the full
# measurement and for what it costs (1.4-1.8x per cell, against far more matching slots).
#
# src/bench/garchformer.py calls torch.set_num_threads(1) itself, so a direct python run is
# covered too; these exports are for the BLAS libraries, which read them at load time.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
_cpus=1

echo "=== $(hostname) $(date) $GF_DATASET"
echo "    d_model=$GF_DMODEL heads=$GF_HEADS layers=$GF_LAYERS lambda=$GF_LAMBDA" \
     "seed=$GF_SEED threads=$_cpus"

# Fail before spending a slot if the dataset is not on disk. src/bench/data.py retries the
# config's csv_path basename inside data/quantile/, so the config file is the thing to check.
if [ ! -f "data/quantile/${GF_DATASET}_config.json" ]; then
    echo "data/quantile/${GF_DATASET}_config.json missing — the three public datasets are"
    echo "fixed inputs and are not fetched by this job. Put them in place first."
    exit 1
fi

python scripts/bench_garchformer.py \
    --dataset     "$GF_DATASET" \
    --d-model     "$GF_DMODEL" \
    --num-heads   "$GF_HEADS" \
    --layers      "$GF_LAYERS" \
    --lambda-cons "$GF_LAMBDA" \
    --seed        "$GF_SEED" \
    --device      cpu
rc=$?
echo "=== done rc=$rc $(date)"
exit $rc
