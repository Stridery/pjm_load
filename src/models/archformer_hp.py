# src/models/archformer_hp.py
"""ARCHFormer on the author's architecture, with this dataset's training recipe.

Identical to src/models/archformer.py in every structural respect -- same 504,725 parameters,
same patch_len 24, same d_model 64, same forward pass. The only difference is four training
knobs, and that is the point: the architecture was never the problem.

    his        mse   / 2e-3 / wd 0    / dropout 0.1
    here       huber / 5e-4 / wd 1e-4 / dropout 0.2

His values are reasonable on the data he tuned them for -- 5-28 feature columns. Here the
model sees 68-89 and 1461 training samples, and the validation curve bounces instead of
descending, which is what too large a step looks like. A one-at-a-time sweep found that no
single knob helps: huber alone is worse, weight decay alone is much worse, a lower learning
rate alone is flat. Only the combination pays, because regularisation cannot help a run that
is still overshooting. See ARCHFORMER_HP_PARAMS in src/config.py for the measured table.

Trains to its own directory, so it is a separate scoreboard row beside the baseline rather
than a replacement for it.
"""

from src.models.archformer import make_variant
from src.config import ARCHFORMER_HP_PARAMS

NAME = 'ARCHFORMER_HP'
MODEL_TYPE = 'archformer_hp'
FILENAME = 'archformer_hp_best.pth'

train, predict, evaluate = make_variant(
    name=NAME, model_type=MODEL_TYPE, filename=FILENAME,
    params_default=ARCHFORMER_HP_PARAMS,
)
