"""Process-pool helpers for the QRA baseline.

The two expensive stages are embarrassingly parallel — eight sister models that never see
each other, and the (m, L) configurations scored independently — so a pool computes
BIT-IDENTICAL results and only changes how long they take. That property is why this is the
one optimisation applied: it needs no numerical argument to defend, unlike replacing the
per-day `lstsq` with incremental normal equations, which squares the condition number of a
~650-column design and would have to be justified rather than merely timed.

Thread oversubscription is the trap here. This node hands out 4 CPUs while OpenBLAS defaults
to 4 threads, so four workers would launch sixteen threads onto four cores and finish SLOWER
than the serial version. Every worker therefore pins BLAS to one thread: the parallelism is
across tasks, never inside them.
"""

import os

DEFAULT_MAX = 8


def n_workers(requested=None):
    """How many processes to run.

    Honours the SLURM allocation over the node's core count — cpu_count() reports 24 on a
    node where the job was granted 4, and spawning 24 there would thrash.
    """
    if requested:
        return max(1, int(requested))
    import multiprocessing as mp
    alloc = os.environ.get('SLURM_CPUS_ON_NODE') or os.environ.get('SLURM_CPUS_PER_TASK')
    n = int(alloc) if alloc and str(alloc).isdigit() else mp.cpu_count()
    return max(1, min(n, DEFAULT_MAX))


def single_threaded():
    """Context manager pinning BLAS to one thread inside a worker."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(1)
    except Exception:               # threadpoolctl absent — accept the oversubscription
        import contextlib
        return contextlib.nullcontext()


def run(fn, items, workers, label=None):
    """map(fn, items) across a process pool; a plain serial map when that is not worth it.

    Results come back in submission order, so callers can zip them against `items` without
    tagging every task. `fn` must be a module-level callable (a pool pickles it).

    `label` turns on a completion counter. That is not cosmetic: a silent stage that runs for
    tens of minutes is indistinguishable from a hung one, and the natural response to a hung
    job is Ctrl-C.
    """
    import sys
    import time
    items = list(items)
    n = len(items)
    t0 = time.time()

    def tick(done):
        if not label:
            return
        el = time.time() - t0
        eta = el / done * (n - done) if done else 0
        sys.stdout.write(f"\r    {label}: {done}/{n} done, {el / 60:.1f} min elapsed"
                         f"{f', ~{eta / 60:.0f} min left' if done < n else ''}   ")
        sys.stdout.flush()
        if done == n:
            sys.stdout.write("\n")

    if workers <= 1 or n <= 1:
        out = []
        for i, x in enumerate(items, 1):
            out.append(fn(x))
            tick(i)
        return out

    from concurrent.futures import ProcessPoolExecutor, as_completed
    out = [None] * n
    with ProcessPoolExecutor(max_workers=min(workers, n)) as ex:
        futs = {ex.submit(fn, x): i for i, x in enumerate(items)}
        for done, f in enumerate(as_completed(futs), 1):
            out[futs[f]] = f.result()
            tick(done)
    return out
