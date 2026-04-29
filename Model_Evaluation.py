import src.config as cfg
from src.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(cfg.EVAL_CONFIG)
evaluator.load_data()

if any(m['enabled'] for m in cfg.EVAL_CONFIG['models'].values()):
    evaluator.evaluate_all()

if cfg.EVAL_CONFIG['single_day']['enabled']:
    sd = cfg.EVAL_CONFIG['single_day']
    evaluator.show_single_day(sd['model'], sd['model_path'], sd['date'])
