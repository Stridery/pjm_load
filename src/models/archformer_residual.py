# src/models/archformer_residual.py
"""GARCHFormer trained on the residual against a naive same-hour-last-week baseline.

Same ARCHFormer3 / same params / same split as src/models/archformer.py -- only the
target moves. The residual machinery is shared
(src/models/_residual.make_residual_model); this file is just the registration,
exactly like mstnn_residual.py.

Worth stating because it interacts with the variance head: the residual target is a
SIGNED deviation, so sigma^2 is now the variance of the residual rather than of the
load level. The consistency loss is unchanged (it is fit against (y - y_hat)^2 in
whatever space the target lives), and sigma stays positive by construction.
"""

from src.models._residual import make_residual_model
from src.models.archformer import ARCHFormerNet, predict as _net_predict
from src.config import ARCHFORMER_FEATURE_CONFIG, ARCHFORMER_RESIDUAL_PARAMS

train, predict, evaluate = make_residual_model(
    name='ARCHFORMER_RESIDUAL',
    model_type='archformer_residual',
    filename='archformer_residual_best.pth',
    feature_cfg=ARCHFORMER_FEATURE_CONFIG,
    params_default=ARCHFORMER_RESIDUAL_PARAMS,
    model_cls=ARCHFormerNet,
    base_predict=_net_predict,
    is_moe=False,
)
