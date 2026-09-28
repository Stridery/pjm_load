"""
Benchmark zero-shot foundation forecasters against the 16 trained point models.

    PJM_DATASET=dom python scripts/run_foundation.py                       # config defaults
    PJM_DATASET=dom python scripts/run_foundation.py --model seasonal_naive --mode direct
    PJM_DATASET=dom python scripts/run_foundation.py --context 512 --mode residual
    PJM_DATASET=dom python scripts/run_foundation.py --no-calibrate
    PJM_DATASET=dom python scripts/run_foundation.py --scoreboard          # re-read results, run nothing
    python scripts/run_foundation.py --list

A pre-trained univariate model is handed the recent Load_Estimated history and asked for the
target day. Nothing is trained, so there is no weights file and no train step — the run reads
the cleaned CSV and the target matrix, calls the model, and evaluates.

Everything is configured in FOUNDATION_CONFIG (src/config.py); the flags below are per-run
overrides of the same fields, in the style of the other standalone pipelines.

Results land in results/{zone}/evaluation/ NEXT TO the 16 point models, with direct and residual
under separate model names, so the scoreboard is a single scan of one tree:

    results/{zone}/evaluation/timesfm/tail_test0.16_cong_ctx512/
    results/{zone}/evaluation/timesfm_residual/tail_test0.16_cong_ctx512/
    results/{zone}/evaluation/_scoreboard.{csv,png}

Deliberately NOT registered in src/models — see src/foundation/__init__.py. That keeps the daily
CI forecast job, scripts/Model_Training.py and Train_Model.sbatch untouched.
"""

# Entry points live in scripts/ but import the package from the repo root, and running
# `python scripts/foo.py` puts scripts/ on sys.path rather than the root. Prepend the root so
# `import src...` resolves from any working directory.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import sys

import pandas as pd

import src.config as cfg
from src.foundation import _bench, _scoreboard
from src.foundation.registry import ADAPTERS


def main():
    fcfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.FOUNDATION_CONFIG.items()}
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=fcfg['model'], choices=sorted(ADAPTERS),
                    help='adapter to run (default from FOUNDATION_CONFIG)')
    ap.add_argument('--mode', nargs='*', default=None, choices=list(_bench.MODES),
                    help="'direct', 'residual', or both (default from config)")
    ap.add_argument('--context', nargs='*', type=int, default=None,
                    help='context length(s) in hours; each becomes its own run dir')
    ap.add_argument('--series', nargs='*', default=None, choices=sorted(_bench.SERIES),
                    help="context series: 'estimated' (preliminary, current) or 'metered' "
                         "(the target itself, ~1 d staler so the horizon grows to 72)")
    ap.add_argument('--metered-issue', default=None, choices=['day_of', 'day_ahead'],
                    help="for --series metered: 'day_of' keeps horizon 48 and issues at the "
                         "start of the target day; 'day_ahead' keeps the point models' issue "
                         "time and grows the horizon to 72")
    ap.add_argument('--horizon', type=int, default=None,
                    help='forecast steps; omit to derive from the data (recommended)')
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--no-calibrate', action='store_true',
                    help='skip the Est->metered scale correction in direct mode')
    ap.add_argument('--eval-train', action='store_true',
                    help='also evaluate the train split (5x the samples; off by default)')
    ap.add_argument('--scoreboard', action='store_true',
                    help='rebuild the comparison table from what is already on disk, run nothing')
    ap.add_argument('--list', action='store_true', help='list available adapters and exit')
    args = ap.parse_args()

    if args.list:
        print("Available foundation adapters:")
        for name in sorted(ADAPTERS):
            print(f"  {name:16s} {ADAPTERS[name].__doc__.splitlines()[0]}")
        return

    if args.scoreboard:
        _scoreboard.write(fcfg['result_root'], cfg.DATASET)
        return

    modes = args.mode or fcfg['modes']
    contexts = args.context or fcfg['context_hours']
    all_series = args.series or [fcfg.get('series', 'estimated')]
    if args.horizon is not None:
        fcfg['horizon_steps'] = args.horizon
    if args.batch_size is not None:
        fcfg['batch_size'] = args.batch_size
    if args.metered_issue:
        fcfg['metered_issue'] = args.metered_issue
    if args.no_calibrate:
        fcfg['est_to_metered'] = 'none'
    if args.eval_train:
        fcfg['eval_train'] = True

    print(f"zone {cfg.DATASET} | model {args.model} | modes {modes} | contexts {contexts} "
          f"| series {all_series}"
          + (f" | metered_issue {fcfg['metered_issue']}" if 'metered' in all_series else ""))
    print(f"matrix tag {cfg.MATRIX_TAG!r} | est_to_metered {fcfg['est_to_metered']} | "
          f"horizon {fcfg['horizon_steps']}")

    rows, failures = [], []
    for sr in all_series:
        for context_hours in contexts:
            for mode in modes:
                try:
                    rows.append(_bench.run_one(args.model, mode, context_hours, fcfg, series=sr))
                except Exception as e:                  # one bad cell must not lose the rest
                    failures.append((f'{sr}/{mode}', context_hours, repr(e)))
                    print(f"\n!! {args.model}/{sr}/{mode}/ctx{context_hours} FAILED: {e}",
                          file=sys.stderr)

    if rows:
        print(f"\n{'=' * 72}\nthis run\n{'=' * 72}")
        print(pd.DataFrame(rows)[['name', 'series', 'mode', 'context_hours', 'MAPE', 'MAE',
                                  'RMSE', 'ME', 'test_days']].to_string(index=False,
                                                          float_format=lambda v: f'{v:.3f}'))
    for mode, ctx, err in failures:
        print(f"FAILED  {mode}/ctx{ctx}: {err}", file=sys.stderr)

    _scoreboard.write(fcfg['result_root'], cfg.DATASET)
    if failures and not rows:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
