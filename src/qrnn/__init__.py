"""Probabilistic baseline 4 — Zhang, Quan & Srinivasan (2019), IEEE TSG 10(4):4425-4434.

An improved quantile regression neural network (iQRNN): ONE network trained on the summed
pinball loss emits all 99 percentiles at once, instead of fitting 99 separate quantile
models. The paper's contribution is not the architecture — a two-hidden-layer MLP with
parametric ReLU — but the training machinery brought over from deep learning: batch
training, early stopping, dropout, GAUSSIAN NOISE LAYERS and a rule-based decaying learning
rate. Its headline claim is that this beats quantile regression averaging on sister
forecasts, which is baseline 2 in this repo.

Where this sits next to the other three baselines. Hong & Fan's tutorial splits probabilistic
load forecasting into three sources; the first three baselines here take one each, and this
one is the deep-learning route that the later literature (ITFT, FPSeq2Q, Bayesian MDN) all
benchmark against:

    src/quantile/scenarios.py   Xie & Hong 2018   simulating predictors
    src/quantile/qra.py         Liu et al. 2017   combining point forecasts into quantiles
    src/density/                Li et al. 2020    combining DENSITIES
    src/qrnn/                   Zhang et al. 2019 one NN -> 99 quantiles directly

Isolated from the point-model stack on exactly the same terms as the other three: reads only
data/{zone}/cleaned/cleaned_pjm_load_weather.csv, writes only under results/{zone}/qrnn/, and
is invisible to src/models, the matrix cache and the serving path. That registry is what
src/serving/predict.available_models() scans and the daily CI forecast job loads, so a torch
model with its own feature pipeline dropped in there would break the daily forecast.

Shared evaluation helpers (pinball, Winkler, coverage, CRPS, the figure style) are imported
from src.quantile rather than duplicated, so all four baselines are scored by one set of
rules and land in comparable tables.


Two things that make the comparison to the other baselines valid
---------------------------------------------------------------
**The test period.** The same tail-0.16 split, snapped to an EPT midnight, that
train_quantile.py / train_qra.py / train_density.py use — imported from train_quantile
rather than re-derived, so the four are scored on identical hours.

**The 48 h blind spot.** The repo's cutoff is the target day's previous midnight, so the last
observed load is (D-1) 23:00 and target hour k of day D+1 sits 25+k hours later. A load lag
valid for EVERY target hour must therefore be at least 48. The paper's shortest load lag is
24 h, which is simply not observable here, so the candidate pool starts at 48 — the same
shift src/density/features.py documents, for the same reason.


The load series is metered, on purpose
--------------------------------------
Both the lag block and the target are the cleaned CSV's `Load`, which is VERIFIED metered.
That column trails real time badly — 7 days on dom, 48 days on bge — so `Load` at the cutoff
hour does not exist when the forecast would be issued, and every score here carries that
optimism.

It is kept because it is what makes this a BASELINE rather than a PJM model. All three of the
other probabilistic baselines fit on the same column (train_quantile's MLR, sisters.py,
density/features.py), so switching one of them to preliminary load would take it out of their
table. More to the point, the public datasets this framework is meant to be pointed at next —
GEFCom2014, ISO-NE, UK-SMEC — have no preliminary/metered distinction at all, just one load
series. A feature definition that depends on a PJM-specific data-vintage column cannot be run
against them without a special case.

The alternative, for the record: `Load_Estimated` is preliminary load, current as of the
cutoff and therefore honest about availability. It costs a proxy gap — measured on the full
record, dom 0.82% (a near-constant 1.0079 level offset, hourly median ratios spanning
0.23 pp) and bge 3.30% (ratios spanning 2.18 pp — a SHAPE mismatch, since bge's proxy is the
MIDATL aggregate over 8.8, and no rescaling removes it). Using it would need the Est->metered
correction FOUNDATION_CONFIG['est_to_metered'] applies, and would trade comparability with the
other three baselines for comparability with the 16 point models.

Note that the 48 h lag floor is NOT a consequence of this choice. It comes from the horizon —
target hour k sits 25+k hours after the last observation, so one lag list valid for all 24
hours must start at 48 — and it would be identical on any load column.


Temperature is ex-post, on purpose
----------------------------------
Temperature enters at lag 0 (the target hour's own realised value). That is the paper's own
arrangement — GEFCom2014 competitors used "the actual temperatures ... as a proxy for
temperature forecasts" — and it is also what src/quantile/qra.py does. Keeping it makes this
baseline comparable with QRA and the scenario benchmark, which is the point of adding it.

It does mean every number here is ex-post in the weather and NOT comparable with the 16 point
models, which read archived real forecasts from cleaned/forecast.csv. Splicing those in here
would answer a harder and more realistic question; it is deliberately not done, because a
fourth baseline whose weather assumption differs from the other three cannot go in their
table. See QRNN_CONFIG in src/config.py.
"""
