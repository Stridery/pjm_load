# Porting three probabilistic baselines to elia / ercot / spain

Design review draft. Software architecture, the three model architectures, how features are
fed in, and every deviation from the published specifications.

Fully separate from the PJM stack: new package `src/bench/`, new entry `train_bench.py`, new
tree `results/bench/`. `src/config.py`, `train_qra.py`, `scripts/train_density.py`, `scripts/train_qrnn.py`
and `results/{zone}/` are untouched.

---

## 1. Software architecture

### 1.1 Modules

```
src/bench/
    __init__.py      package contract and the isolation rationale
    config.py        JSON schema, loading, validation, derived quantities
    dataset.py       the Dataset object: index, target, column classes, split masks,
                     lag floor, forecast-alignment probe
    lags.py          lag-grid construction; the one place the floor rule is applied
    matrix.py        lazy design-matrix builder (build only the rows or columns asked for)
    metrics.py       Pinball / PICP / MPIW / Winkler / MAE / RMSE / MAPE, (N,H) reshape,
                     per-step / per-day / per-month breakdowns
    density.py       Model A driver
    iqrnn.py         Model B driver
    qra.py           Model C driver + the autoregressive sister family
    plots.py         figures
train_bench.py       CLI entry
```

### 1.2 What is reused, what is replaced

| existing module | | reason |
|---|---|---|
| `src/quantile/qeval.py` | **reuse** | `pinball` and `winkler` already match the target formulas term for term |
| `src/quantile/_pool.py` | **reuse** | process pool, dataset-agnostic |
| `src/quantile/plots.py` | **reuse** | style helpers + the three generic figures |
| `src/quantile/qra.py` | **reuse** | `quantreg_irls`, `rolling_windows`, `qra` — stage 2 is untouched |
| `src/quantile/design.py` | **partial** | `calendar_block` reused; `temp_block` replaced by a load-driven variant |
| `src/density/{base_models,combine,crps_gmm,gaussfit}.py` | **reuse** | they take plain `(X, y)` arrays and know nothing about PJM |
| `src/qrnn/model.py` | **reuse** | network, loss, schedule all take `(X, y, masks, cfg)` |
| `src/density/features.py` | **replace** | PJM lag list hardcoded at module level |
| `src/qrnn/features.py` | **replace** | PJM candidate pool; also builds the full matrix eagerly |
| `src/quantile/sisters.py` | **replace** | temperature-driven family |
| `src/quantile/frame.py` | **replace** | single hardcoded CSV schema |

So the three papers' *methods* are reused unchanged; what is rewritten is the layer that
decides which columns at which lags become a matrix.

### 1.3 The Dataset object

```python
@dataclass(frozen=True)
class Dataset:
    name: str
    frame: pd.DataFrame          # the cleaned CSV, index sorted, no reindexing applied
    index: pd.DatetimeIndex
    step: pd.Timedelta           # 15 min or 60 min
    steps_per_day: int           # H = 96 or 24
    target_col: str
    past_cols:     list[str]     # actual_cols minus target; usable only at lag >= floor
    known_cols:    list[str]     # forecast_cols; usable at lag 0
    aux_cols:      list[str]     # calendar; always usable
    lag_floor:     int           # in STEPS, derived (see 2.2)
    lookback:      int           # in STEPS
    horizon:       int           # in STEPS
    train / val / test: np.ndarray   # boolean masks over index
    shift_steps:   dict[str, int]    # per-column forecast alignment, default 0

    def y(self) -> np.ndarray
    def day_block(self, mask) -> tuple[int, int]   # (N days, H) for the (d,h) reshape
    def probe_alignment(self) -> pd.DataFrame      # section 7.1
```

Construction validates: splits are contiguous, non-overlapping, day-aligned and ordered; the
index is uniform at `step` with no duplicates; every configured column exists; `target_col` is
not in `known_cols`.

### 1.4 Execution flow

```
train_bench.py --dataset spain --models density iqrnn qra
   │
   ├─ config.load("spain")          JSON → validated spec
   ├─ Dataset.build(spec)           masks, floor, alignment probe → printed + written
   │
   ├─ for each model:
   │     lags.grid(dataset, model)  the model's lag specification, floor applied
   │     matrix.build(...)          lazily, only the rows/columns that stage needs
   │     <model>.run(...)           fit → predict test → (3, n_test) quantiles
   │     metrics.report(...)        reshape to (N,H) → scores + breakdowns
   │
   └─ scoreboard across models
```

Every model emits the same object — a `(3, n_test)` array of sorted quantiles plus the test
timestamps — so `metrics.py` and `plots.py` have one input contract.

### 1.5 Config changes needed

`csv_path` in all three JSONs points at `data/public_cleaned_data/`, which does not exist;
files are in `data/quantile/`. Two fields are added:

```jsonc
"csv_path": "data/quantile/cleaned_spain_data.csv",   // corrected
"shift_steps": {},                                     // per-column forecast alignment, §7.1
"quantiles": [0.05, 0.50, 0.95]
```

`spain_config.json` has `"lookback_hours": [336, 168]` where the others have a scalar; taken
as **168** per your instruction, and rewritten as a scalar.

---

## 2. Datasets

| | elia | ercot | spain |
|---|---|---|---|
| granularity | 15 min | 60 min | 60 min |
| steps per day `H` | **96** | 24 | 24 |
| target | `Total Load` | `ACTUAL_NetLoad` | `total load actual` |
| lookback | 672 steps (168 h) | 168 steps | 168 steps |
| horizon | 96 steps (24 h) | 24 steps | 24 steps |
| `forecast_deadline_hour` | 23 | **9** | 23 |
| **lag floor** | **99 steps** | **38 steps** | **24 steps** |
| samples (after 5 w history) | 355,868 | 16,118 | 34,199 |
| train / val / test (steps) | 214,944 / 71,808 / 71,709 | **10,176** / 3,408 / 3,374 | 21,024 / 7,008 / 6,984 |
| test days | 747 | 141 | 291 |
| `past_cols` / `known_cols` | 0 / 4 | 11 / 0 | 22 / 1 |

All three are clean: no missing timestamps, no duplicates, no NaNs, uniform step, UTC or
naive — so unlike PJM there is no DST irregularity to normalise, and `(N, H)` reshaping is
exact.

### 2.1 Sample definition

One sample is **one target time step**, not one day — the granularity all three papers use. A
day-level layout would leave ercot with about 424 training days, which none of the three
models can be fitted on.

### 2.2 Lag floor

The information cutoff for target day `D` is `cutoff(D) = (D − 1 day) + deadline_hour`. A lag
valid for *every* step of the target day must reach from the cutoff to the day's last step:

```
floor = ( last step of D  −  cutoff(D) ) / step
```

| | cutoff | last step of D | gap | floor |
|---|---|---|---|---|
| elia | (D−1) 23:00 | D 23:45 | 24 h 45 min | 99 steps |
| ercot | (D−1) 09:00 | D 23:00 | 38 h | 38 steps |
| spain | (D−1) 23:00 | D 23:00 | 24 h | 24 steps |

elia's floor is 99 rather than a round 96 because the final 15-minute step runs 45 minutes
past the whole day. This is what pushes the density lag list one day further out (§3.2).

ercot's deadline of 9 gives a blind spot of 15–38 hours, far wider than the other two. That is
its market rule (the day-ahead market closes the previous morning), not an error.

### 2.3 Feature availability

One rule, shared by all three models:

| column class | usable lags | reason |
|---|---|---|
| `target_col` | ≥ floor | realised value, known only after the cutoff |
| `past_cols` (`actual_cols` minus target) | ≥ floor | same |
| `known_cols` (`forecast_cols`) | **lag 0** | day-ahead forecast, describes the target day, published before the cutoff |
| `aux_cols` (HoD / DoW / MoY) | always | calendar |

`target_col` also appearing inside `actual_cols` is harmless — it is handled as the target.

---

## 3. Model A — density combination (Li, Wang & Zhang 2020)

### 3.1 Architecture

```
15 base learners, three families of five
    GPR  ×5    ARD Matérn 3/2 + RBF + White kernel
               diversity: a different 2000-point random subsample of T1 each (seeds 100-104)
    QRNN ×5    diversity: architecture × feature subset × width
               NN1 bn/all/64   NN2 bn/lags/128   NN3 l2/all/64
               NN4 l2/all/128  NN5 l2/recent/96
               feature subsets: all=15, lags=11 lag columns, recent=short lags + calendar
    QRGB ×5    diversity: max_depth / min_child_weight / subsample / colsample_bytree
               XGB1 4/1/0.9/0.9   XGB2 6/5/0.8/0.8   XGB3 8/10/0.7/0.7
               XGB4 3/20/0.9/0.6  XGB5 6/2/0.6/1.0
         │
         ├─ GPR emits a Gaussian (μ, σ) natively
         └─ QRNN / QRGB emit 9 deciles → Gaussian by probit regression:
                if y_q = μ + Φ⁻¹(q)·σ then OLS of y_q on Φ⁻¹(q)
                gives intercept = μ, slope = σ
         ↓
    Gaussian mixture  F = Σ wᵢ Fᵢ
         ↓
    weights minimise CRPS. For a Gaussian mixture the CRPS is closed form
    (Baringhaus–Franz):  CRPS(F,y) = E|Y−y| − ½E|Y−Y'| = ΣΣ αᵢₖ wᵢwₖ + Σ βᵢ wᵢ
    quadratic in w → a small QP on the simplex, not a simulation.
    wᵢ = 0 drops model i, so selection and weighting happen in one solve.
    Q is indefinite (αᵢᵢ = −σᵢ/√π); the paper eliminates the equality constraint and
    reports the reduced matrix positive definite — computed here as a certificate,
    multi-start SLSQP either way.
         ↓
    reported alongside: MAPE-guided weights, simple average, best single model on T3
```

### 3.2 Features — 15, fixed by the paper, no exogenous variables

`11 target lags + hour + dayofweek + month + is_weekend`

The published lag list (hours) is `24, 25, 26, 27, 28, 48, 72, 96, 120, 144, 168`: the same
step of day on the most recent fully observed day plus the four before it, then six daily lags
out to a week. Instantiated per dataset against the floor:

| | lag list (steps) | note |
|---|---|---|
| **spain** | `24,25,26,27,28,48,72,96,120,144,168` | **the paper's list verbatim** — floor is exactly 24 |
| **ercot** | `48,49,50,51,52,72,96,120,144,168,192` | shifted +24. With deadline 9, day D−1 is not fully observed, so the most recent fully observed day is D−2 = 48 h |
| **elia** | `192,196,200,204,208,288,384,480,576,672,768` | the list in steps is `96,100,…,672`; its minimum 96 is below floor 99, so shifted one day (+96) |

Note this model gets **no** `known_cols` and **no** `past_cols` — Li's explicit design choice
is that the base learners are generic, so the combination step is what is being tested.

### 3.3 Periods and input shape

```
T1 = train            fit the 15 base models
T2 = first half val   report base models (the paper's Table I)
T3 = second half val  optimise the 15 weights
T4 = test             reported

X : (n_samples, 15)  float32
y : (n_samples,)
```

### 3.4 Deviations

| paper | here | forced? |
|---|---|---|
| lags `24…168` h | shifted to clear the floor (spain unchanged) | **yes** — lag 24 is not observable on ercot/elia |
| T1:T2:T3:T4 = 10:1:1:1 | T1 = train, T2/T3 = val halved, T4 = test | **yes** — the configs define three periods, not four |
| CVXOPT | scipy SLSQP, multi-start | no — equivalent at 15 variables; the convexity certificate is computed rather than assumed |
| deciles as the base-learner output | unchanged | — |

---

## 4. Model B — iQRNN (Zhang, Quan & Srinivasan 2019)

### 4.1 Architecture

```
input (103)
  ├─ Linear(103 → 500)
  ├─ PReLU(500)                  per-channel learnable slope
  ├─ Dropout(p)                  p = 0.5, or 0 for the -ZD variant
  ├─ GaussianNoise(sd = 0.3)     training only; the paper's largest single ablation effect
  ├─ Linear(500 → 500)
  ├─ PReLU(500)
  ├─ Dropout(p)
  ├─ GaussianNoise(sd = 0.3)
  └─ Linear(500 → 3)             output width = the quantile set {5%, 50%, 95%}

loss = Σ_q pinball(y, ŷ_q)       the output width IS the quantile set and the objective IS
                                 the reported metric, so one fit produces every quantile

schedule (paper Fig. 2)
    Adam, lr 1e-3, batch 168
    one `bad` counter = epochs since the best validation pinball, driving two rules:
        bad ≥ 20                 stop
        every 5 bad epochs       lr ×= 0.4, floored at 1e-5
    best weights restored at the end
```

Two configurations are run (`p = 0.5` and `p = 0`), 5 seeds each, reported as mean and SD.

### 4.2 Features — candidate pool → RFE → 103

The paper's pool is `calendar + holiday + load lags (1 d … 5 w) + temperature (now … 5 w)`,
1760 candidates, cut to 100 by RFE plus 3 manual add-backs.

**None of these datasets has temperature.** Temperature is the paper's exogenous driver known
at the deadline, so it is replaced by whatever each dataset provides, under §2.3:

| | target lags | `past_cols` | `known_cols` | calendar | **pool** |
|---|---|---|---|---|---|
| **elia** | 3,262 (99…3360, every step) | none | 4 × lag 0 | 11 | **3,277** |
| **ercot** | 803 (38…840) | 11 cols × 8 daily echoes = 88 | none | 11 | **902** |
| **spain** | 817 (24…840) | 22 cols × 7 daily echoes = 154 | 1 × lag 0 | 11 | **983** |

- the **target** gets a dense grid (floor → 5 weeks, every step) — the paper's treatment of load
- **`past_cols`** get a daily-echo grid `{floor} ∪ {⌈floor/H⌉·H + k·H, k = 0…6}`. The paper
  affords a dense grid because it has one exogenous series; spain has 22, and a dense grid
  there would be an 18,000-column pool
- **calendar**: HoD / DoW / MoY raw codes + three sin-cos pairs + is_weekend

RFE: `RandomForestRegressor(n_estimators=40, criterion='absolute_error', max_depth=12,
max_features=0.3, min_samples_leaf=20)`, `step=0.35`, on a 12,000-row subsample of the training
rows only, cached by a signature covering the dataset, floor, pool definition and seed.

### 4.3 The candidate matrix is never materialised

```
elia, if built whole:   355,868 × 3,277 × 4 B = 4.66 GB
  RFE stage, subsample:  12,000 × 3,277 × 4 B = 0.157 GB
  after selection:      355,868 ×   103 × 4 B = 0.147 GB
                                        peak  = 0.16 GB
```

`matrix.py` builds only the rows or columns a stage asks for, so peak memory is essentially
independent of dataset size. RFE costs about 10 minutes and is cached, so elia can use the
full 3,262-lag grid with no subsampling.

### 4.4 Input shape

```
X : (n_samples, 103)  float32, standardised on training rows only, then frozen
y : (n_samples,)      standardised the same way, inverted after prediction
out : (n_samples, 3)  sorted to remove crossing
```

### 4.5 Deviations

| paper | here | forced? |
|---|---|---|
| temperature lags 0…5 w | each dataset's own exogenous columns | **yes** — no temperature exists |
| dense lag grid for the exogenous series | daily-echo grid for `past_cols` | **yes** — 22 series × dense grid is not tractable |
| 99 quantiles | {5, 50, 95} | no — your specification |
| per-sample L2 norm then per-column standardisation | per-column only | no — sample-wise norm makes the target scale unrecoverable from its own row |
| validation year selects, then retrain with an internal 90/10 split | one validation block does both early stopping and configuration choice | **yes** — retraining on the selection set leaves nothing to stop on |
| 100 independent runs | 5 | no — cost; mean and SD still reported |
| bQRNN control included | off by default | no — already run on PJM (250.5 vs 404.2 pinball); adds ~3.6 h on elia and no new information. One switch |
| 1760 candidates → 100 + 3 | pool size varies by dataset → 100 + 3 | **yes** — pool size follows from the available columns |

---

## 5. Model C — QRA on sister forecasts: **dropped**

Not built, and not deferred. Liu et al. build a family of sister point forecasts and convert
their *disagreement* into quantiles. The members differ only in how they treat a
**temperature** recency effect, and none of these three datasets has a temperature column.
Driving the family from lagged load instead leaves all eight sisters reading one signal, which
is a change to the mechanism rather than to the features. The PJM port, still temperature
driven, already strained: dispersion 259.6 MW against an error spread of 726.2 MW on `dom`
(2.80x short, median pairwise error correlation 0.88); 28.9 against 199.3 on `bge` (6.89x, 0.96).

Each dataset then breaks it a second time, differently:

| | history before validation | 2 y | 3 y | consequence |
|---|---|---|---|---|
| `spain` | 2.38 y | yes | no | 8 sisters collapse to 4 |
| `ercot` | 1.14 y | no | no | no feasible window at all |
| `elia` | 6.13 y | yes | yes | family intact, but 87 h of `lstsq` unless two concessions are made |

Models A and B each survive the missing temperature intact, so the baseline set is those two.
`src/quantile/{qra,sisters,design,qra_plots}.py` and `train_qra.py` have been deleted;
`src/quantile/__init__.py` records why.

## 6. Metrics

Quantile set {5 %, 50 %, 95 %}, interval `[q₀₅, q₉₅]`, **α = 0.10**. Over `N` days × `H` steps:

```
Pinball = (1/(|Q|·N·H)) Σ_q Σ_{d,h}  max{ (q−1)(y − ŷ_q),  q(y − ŷ_q) }
PICP    = (1/(N·H))     Σ_{d,h}  1{ ℓ ≤ y ≤ u }
MPIW    = (1/(N·H))     Σ_{d,h}  (u − ℓ)
Winkler = (1/(N·H))     Σ_{d,h}  [ (u−ℓ) + (2/α)(ℓ−y)1{y<ℓ} + (2/α)(y−u)1{y>u} ]
```

Point metrics use the median q₅₀:

```
MAE  = (1/(N·H)) Σ |ŷ₅₀ − y|
RMSE = sqrt( (1/(N·H)) Σ (ŷ₅₀ − y)² )
MAPE = (100/(N·H)) Σ |ŷ₅₀ − y| / |y|
```

`qeval.pinball` and `qeval.winkler` match these term for term and are reused; PICP and MPIW
are thin new wrappers. The test block reshapes to `(N, H)` — exact here, since no dataset has
DST or missing steps — so the per-step and per-day breakdowns fall out of the same array.

A partial day at either end of the test period is dropped, with the count printed, rather than
padded: `Σ_{d,h}` assumes complete days.

---

## 7. Known problems

### 7.1 The forecast columns are shifted relative to their timestamps

```
spain  total load forecast     lag 0  8.76%  →  +24 steps   1.11%
ercot  ERC_Load                lag 0  4.45%  →  +24 steps   2.46%
ercot  ERC_Wind                lag 0 70.32%  →  +24 steps  17.25%
ercot  ERC_Solar               lag 0 49.07%  →  +24 steps  29.38%
elia   Day-ahead 6PM forecast  lag 0 13.04%  →  +24 steps (= 6 h)   2.63%
elia   Week-ahead forecast     lag 0 15.24%  → +168 steps (= 42 h)  2.94%
elia   Most recent forecast    lag 0  1.92%  →   +0 steps  (not shifted)
```

It is the forecast columns and not the actuals that moved: spain's summed generation
correlates best with `total load actual` at lag 0 (0.8972, against 0.6335 at lag 24). On elia
the shifts are 24 and 168 — the *hourly* lag constants applied to a 15-minute series, giving
6 h and 42 h instead of 24 h and 168 h — and `Most recent forecast` was not shifted at all,
which an intentional scheme would not do.

Running as configured (lag 0) is not a leak, only a degradation. The cost lands hardest on
ercot, whose target is **net load** and whose wind and solar day-ahead forecasts are its main
exogenous signal. `Dataset.probe_alignment()` runs at load time, prints this table and writes
`alignment_probe.txt`; `shift_steps` is a per-column config field, so confirming a value is a
one-line change with no code edit.

### 7.2 QRA on elia is not tractable as specified

See §5.5. Two concessions, elia only.

### 7.3 ercot has 10,176 training samples against the paper's 42,984

The 103→500→500 network carries ~350 k parameters. The paper's own size sweep found 500
sufficient at 43 k samples; 200–300 is probably right here. A short width sweep on the
validation block before fixing the number is cheap (ercot trains in ~25 s per run).

### 7.4 The three datasets do not pose the same task

elia's `known_cols` contains the official day-ahead load forecast; spain's and ercot's do not
(spain's `total load forecast` is excluded from both lists, which looks deliberate — including
it would reduce the task to copying the TSO). elia's scores will look better and part of that
is the TSO's, not the model's. Printed in each dataset's output header.

### 7.5 Feature sets are not symmetric across models

Following each paper, density gets no exogenous variables at all (Li's explicit design
choice), while iQRNN's pool does — so on elia iQRNN sees the official forecast and density does
not. Faithful, but it needs saying in the results. The alternative, giving all three the same
feature set, would take density away from its published specification.

---

## 8. Build order

Each step ends with something checkable, so a mistake in the data layer surfaces before three
models are built on it.

| | deliverable | validated by |
|---|---|---|
| 1 | `config.py`, `dataset.py`, `lags.py`, `matrix.py` | contract tests (§9); alignment probe printed for all three datasets |
| 2 | `metrics.py` | reproduce `qeval` on a fixed array; PICP/MPIW against a hand-computed case |
| 3 | `density.py` | **spain first** — its lag list is the paper's verbatim, so a wrong number here is the port, not the adaptation |
| 4 | `iqrnn.py` | spain, then ercot (small, fast), then elia (memory path) |
| 5 | `qra.py` | spain, then ercot; elia last, with the §5.5 concessions |
| 6 | `plots.py`, scoreboard | — |

Density first because it is the only one of the three that needs no conceptual adaptation on
spain — it is the cleanest test that the dataset layer is right.

## 9. Contract tests

`test/test_bench_dataset.py`, in the style of `test/test_qrnn_features.py` — things that would
fail silently rather than raise:

1. **lag alignment** — every `lag{k}` column equals the source series shifted by exactly `k`,
   checked against an index-valued series where an off-by-one is impossible to miss
2. **floor respected** — no `target`/`past` column appears below the dataset's floor, and the
   floor equals the value derived from `deadline_hour` for all three datasets
3. **split integrity** — train/val/test partition the index, are day-aligned, ordered, and
   match the JSON dates
4. **reshape exactness** — `(N, H)` reshape round-trips the test block with no reordering
5. **metrics** — `Pinball` and `Winkler` agree with `qeval` to machine precision; PICP/MPIW on
   a hand-built case
6. **lazy matrix** — the lazily built subset is bit-identical to the corresponding slice of an
   eagerly built matrix on a small dataset
