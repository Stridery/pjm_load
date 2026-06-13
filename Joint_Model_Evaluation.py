import src.config as cfg
from src.joint_model_evaluator import JointModelEvaluator

evaluator = JointModelEvaluator(cfg.JOINT_EVAL_CONFIG, cfg.JOINT_ZONES)
evaluator.load_data()

if any(m['enabled'] for m in cfg.JOINT_EVAL_CONFIG['models'].values()):
    evaluator.evaluate_all()

if cfg.JOINT_EVAL_CONFIG['single_day']['enabled']:
    sd = cfg.JOINT_EVAL_CONFIG['single_day']
    evaluator.show_single_day(sd['model_type'], sd['model_path'], sd['date'])
