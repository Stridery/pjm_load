# Entry points live in scripts/ but import the package from the repo root, and running
# `python scripts/foo.py` puts scripts/ on sys.path rather than the root. Prepend the root so
# `import src...` resolves from any working directory.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import src.config as cfg
from src.joint_model_evaluator import JointModelEvaluator

evaluator = JointModelEvaluator(cfg.JOINT_EVAL_CONFIG, cfg.JOINT_ZONES)
evaluator.load_data()

if any(m['enabled'] for m in cfg.JOINT_EVAL_CONFIG['models'].values()):
    evaluator.evaluate_all()

if cfg.JOINT_EVAL_CONFIG['single_day']['enabled']:
    sd = cfg.JOINT_EVAL_CONFIG['single_day']
    evaluator.show_single_day(sd['model_type'], sd['model_path'], sd['date'])
