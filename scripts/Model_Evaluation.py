# Entry points live in scripts/ but import the package from the repo root, and running
# `python scripts/foo.py` puts scripts/ on sys.path rather than the root. Prepend the root so
# `import src...` resolves from any working directory.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import src.config as cfg
from src.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(cfg.EVAL_CONFIG)
evaluator.load_data()

if any(m['enabled'] for m in cfg.EVAL_CONFIG['models'].values()):
    evaluator.evaluate_all()

if cfg.EVAL_CONFIG['single_day']['enabled']:
    sd = cfg.EVAL_CONFIG['single_day']
    evaluator.show_single_day(sd['model'], sd['model_path'], sd['date'])
