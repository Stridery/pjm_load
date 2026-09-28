# src/models/archformer_skip.py
"""ARCHFormer with the tuned recipe AND the static block routed past the patch tokenizer.

Two changes from the baseline, so read its row against archformer_hp (same recipe, no skip)
rather than against the baseline, or the two are confounded.

THE SKIP
The vendored ARCHFormer3 has no notion of a static feature: every column in the author's
configs is read per row. This repo broadcasts 40-64 constants across the window, so a 24-hour
patch carries each of them 24 times and 72% of the tokenizer's input is those copies. Here the
time-varying channels go through the tokenizer and the constants get their own out_dim head,
added to the forecast -- which is how the other six sequence models treat them.

The head has to emit all out_dim hours, not one value. A first version concatenated the
statics into FinalProjectionHead, which is Linear(C -> 1) applied per position: the static
contribution was then the SAME scalar at all 168 positions (measured std 0.0), so 54 features
-- including the 24 hourly forecast-weather departures that carry the target day's shape --
collapsed into a single offset. The model underfit, train loss 0.29 against the baseline's
0.14, and scored 4.65%. Emitting out_dim fixes that.

CAVEAT ON THE RECIPE
The four training values come from a sweep run on the BASELINE, whose tokenizer compresses
~28:1. This variant's compresses ~8:1, which is a different optimisation problem. Re-run
`python scripts/tune_archformer.py --group 3` against this variant before treating them as tuned.
"""

from src.models.archformer import make_variant
from src.config import ARCHFORMER_SKIP_PARAMS

NAME = 'ARCHFORMER_SKIP'
MODEL_TYPE = 'archformer_skip'
FILENAME = 'archformer_skip_best.pth'

train, predict, evaluate = make_variant(
    name=NAME, model_type=MODEL_TYPE, filename=FILENAME,
    params_default=ARCHFORMER_SKIP_PARAMS,
)
