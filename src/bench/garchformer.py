"""GARCHFormer (ARCHFormer3) on the three public datasets — one grid cell per call.

The ARCHITECTURE is Moyi Guo's, imported from src/models/_garchformer/ exactly as the PJM
adapter imports it; `src/models/_garchformer/__init__.py` records what was and was not
vendored. Nothing in that package is touched here. What this module is, is the layer that
hands it the three public datasets instead of PJM: the (N, L, F) window tensor, the
standardisation, a training loop, and the scoring.

Why not reuse src/models/archformer.py
--------------------------------------
That adapter exists to make ARCHFormer3 look like this repo's other six sequence models, so
it is wired to the PJM matrix, the tail-0.16 split, the 3-sigma validity mask, the on-disk
scalers and `src.config`. None of those exist on these datasets — the split is explicit
dates from the config, there is no mask, and the scaling is per-column on the train period.
So the adapter is bypassed and `ARCHFormer3` is constructed directly. It already consumes
(B, L, F) and emits ((B, out_len), (B, out_len)), which is exactly what is needed.

One exception to the package's isolation rule
---------------------------------------------
`src/bench/__init__.py` says nothing here imports `src.models`. This module does, for the
vendored architecture alone — and importing it executes `src/models/__init__.py`, which
pulls in `src.config`, xgboost and lightgbm as a side effect (~6 s). That is the whole of
the coupling: no PJM path is read, no PJM config value is used, and every number below comes
from this module or the dataset's own JSON.

The window
----------
A target day starting at step `s` is forecast from the window that ENDS at the deadline:

    steps [s - G - L + 1, s - G]        L steps, the last one the deadline step
    target steps [s, s + H - 1]         H steps, the whole calendar day

which is `Dataset.lag_max` and `Dataset.lag_min` expressed as a contiguous block rather than
as a lag list. The columns are the upstream `select_cols` — `actual_cols + forecast_cols +
aux_cols` — so the target's own history is in the window, which is not a leak: the window
stops at the deadline.

`aux_cols` duplicated inside `actual_cols` are dropped. ercot's config lists HoD / DoW / MoY
in both, and upstream's `select_cols` concatenation therefore feeds the network three
columns twice. Identical inputs to a Linear are harmless but not free — they widen
`Linear(patch_len*F, d_model)` for nothing — so the names are de-duplicated in order and the
drop is printed. ercot F is 15 here, not upstream's 18.

patch_len = one day
-------------------
`patch_len = ds.H`, so one token is one calendar day on all three datasets: 24 steps on the
hourly two, **96 on elia**. Upstream's `archformer3.json` pins `patch_len: 24` for every
dataset, which on a 15-minute series makes a token six hours and gives 28 tokens instead of
7 — a different model, not a different dataset. Tying it to H keeps the three comparable.

It also keeps the decoder legal. `MuCrossAttentionBlock.forward` reads ONE `L` out of
`x_dec.shape` and applies it to `mu_enc` as well (src/models/_garchformer/layers.py:418-425),
so the encoder and decoder token counts MUST match or the `.view()` raises. With
`seq_length_dec = seq_length_enc` and one patch length, both are `L // H` = 7 everywhere.

Standardisation
---------------
Per column, mean and sd over the rows inside the TRAIN date range, applied everywhere —
upstream `standardize_data_new`. `std_cols` is `actual_cols + forecast_cols`; the calendar
codes go in raw, at 0-23 / 0-6 / 1-12 against z-scored neighbours. That is the author's
recipe and it is kept, because the grid below varies the architecture and a second change
underneath it would make the rows measure two things at once.

Reporting is the bench's, not upstream's: predictions are inverted to physical units, MAPE
and SMAPE computed there, and the rest computed after re-z-scoring with
`Dataset.standardiser` — the mean and sd of the target over the training DAYS, which is what
density and iQRNN report against. So a GARCHFormer row sits on the same axis as theirs.

Two deviations from ArchFormer3Trainer, both deliberate
------------------------------------------------------
**The validation loss excludes the lambda term.** Upstream's `evaluate()` returns
`mse + lambda_cons * L1(sigma^2, eps^2)`, so raising lambda raises the validation number
mechanically and model selection drifts with it. A lambda sweep scored that way compares
stopping times, not lambdas. Validation here is the point loss alone.

**The best epoch is kept.** Upstream tracks `best_val_loss` but its `save_model` call is
commented out, so what gets tested is the LAST epoch. Here the best state_dict is held in
memory, restored before scoring, and never written to disk — the grid wants performance, not
weights.
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._garchformer.pointmodels import ARCHFormer3

LEVELS = np.array([0.05, 0.50, 0.95])

# How the interval is built from the model's own (y_hat, sigma_sq). Both are REPORTED; neither
# touches training or model selection, so the two modes share every point metric and the
# grid's val_MAPE ranking, and differ only in PICP / MPIW / Winkler / Pinball.
#
#   gaussian      y_hat +- z * sigma, sigma straight from the variance head. The head is
#                 trained with L1(sigma^2, eps^2) on TRAINING residuals, which is not a
#                 calibration objective (it targets a median of eps^2 and never sees held-out
#                 error), and the measured result is under-coverage: PICP 60-78 against a
#                 nominal 90, with sigma nearly constant and uncorrelated with |residual|.
#   sigma_scaled  the same sigma times ONE scalar c fitted on the validation split,
#                 c = RMSE(val residuals) / RMS(sigma_val), applied unchanged to test. This
#                 is the post-hoc variance scaling the chapter's own table labels
#                 "sigma-scaled" for ProbGF and SparseTF, so the row lands in that column.
MODES = ('gaussian', 'sigma_scaled')
# Gaussian quantiles of those levels. sigma_sq is trained to match the squared error, so
# (y_hat, sqrt(sigma_sq)) is a conditional Gaussian and these are its 5/50/95 points. The
# interval metrics derived from it are REPORTED, not optimised -- nothing in the loss asks
# for calibration -- but they are free and they put the row next to density and iQRNN.
ZS = np.array([-1.6448536269514722, 0.0, 1.6448536269514722])

# Upstream cfgs_grid_search/archformer3.json, plus the two knobs its trainer takes as
# constructor arguments. The four grid dimensions are NOT here: they come in per cell.
DEFAULT_PARAMS = {
    'intermediate_dim': 32,     # MLPConnectionBlock hidden temporal size
    'dropout': 0.1,
    'lr': 2e-3,                 # archformer3.json "lr"
    'epochs': 50,               # archformer3.json "epochs"
    'batch_size': 32,           # run_point_forecast.py's prepare_model_and_trainer default
    'patience': 15,             # ours: upstream never stops early. Must be < epochs.
    'grad_clip': 1.0,           # ours: the sigma path's L1 term is unbounded at init
    # SINGLE-THREADED ON PURPOSE -- torch's multi-threaded CPU kernels crash on the hosts
    # this runs on. Measured on castle (Xeon E5-2630, 24 cores, AVX but no AVX2), one spain
    # cell at d_model 128 / 2 layers, repeated six times per setting:
    #
    #     OMP_NUM_THREADS=2                                1 ok / 5 SIGILL
    #     OMP_NUM_THREADS=2  ATEN_CPU_CAPABILITY=default    3 ok / 3 SIGILL
    #     OMP_NUM_THREADS=1                                6 ok / 0
    #
    # The fault lands in `param.addcdiv_` inside torch/optim/adam.py:544 -- an ATen
    # vectorised kernel, not this repo's code -- and an Adam-only reproduction on tensors of
    # the same size does NOT crash, so it is the intra-op thread pool and not the optimiser.
    # `ATEN_CPU_CAPABILITY=default` does not fix it, which rules out the obvious
    # "wheel built for AVX2" explanation on its own.
    #
    # The cost is small and the models are why: attention runs over SEVEN tokens, so there is
    # very little for a second thread to do. Per epoch, 1 thread against 2-4:
    #
    #     ercot  2.63 s vs 1.87 s      spain  6.48 s vs 4.45 s      elia  17.3 s vs 9.5 s
    #
    # 1.4-1.8x slower per cell, against `request_cpus = 1` matching far more slots -- so the
    # GRID's wall clock is expected to improve, not worsen. Raise `threads` only on a host
    # where the loop above comes back clean.
    'threads': 1,
}


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------
# THE one definition. The condor itemdata generator and the collector both read it from here,
# so a job list and the table that checks the job list for completeness cannot disagree.
#
#   num_heads   encoder and decoder share the count -- ARCHFormer3 takes a single
#               `num_heads` and hands it to both stacks (pointmodels.py:45,52).
#   layers      likewise tied: one value for num_encoder_layers and num_decoder_layers.
#   d_model     the PatchTokenizer's output width, Linear(patch_len*F -> d_model).
#   lambda_cons the variance-consistency weight. 0 is the ablation: the sigma head keeps its
#               weights but loses its training signal, which is ARCHFormer2's behaviour.
D_MODELS = (32, 64, 128)
NUM_HEADS = (16, 32, 64)
LAYERS = (1, 2)
LAMBDAS = (0.0, 0.05, 0.1, 0.2)


def cells():
    """Every LEGAL cell, in a stable order.

    (32, 64) is the one dropped pair: ArchQAttentionBlock and MuCrossAttentionBlock both
    assert `d_model % num_heads == 0`, and 32 % 64 != 0. So the grid is 8 (d_model, heads)
    pairs rather than 9, and 8 x 2 x 4 = 64 cells. `skipped()` names the dropped pairs so
    the count is never a surprise -- upstream's driver discards them with a bare `continue`.
    """
    for d in D_MODELS:
        for h in NUM_HEADS:
            if d % h:
                continue
            for n in LAYERS:
                for lam in LAMBDAS:
                    yield {'d_model': d, 'num_heads': h, 'layers': n, 'lambda_cons': lam}


def skipped():
    """[(d_model, num_heads)] pairs the divisibility rule removes."""
    return [(d, h) for d in D_MODELS for h in NUM_HEADS if d % h]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def feature_cols(ds, verbose=False):
    """Upstream `select_cols`, de-duplicated in order. See the module docstring."""
    raw = list(ds.cfg['actual_cols']) + list(ds.cfg['forecast_cols']) + list(ds.cfg['aux_cols'])
    cols, dropped = [], []
    for c in raw:
        (cols if c not in cols else dropped).append(c)
    if dropped and verbose:
        print(f"  de-duplicated {len(dropped)} repeated column(s): {dropped}")
    return cols


def train_range_mask(ds):
    """Record steps inside the config's train date range — upstream's standardisation rows.

    Deliberately the DATE RANGE and not the training day set: upstream slices the frame with
    `df.loc[train_start:train_end]`, and the statistics are what the author's recipe is.
    """
    a = _utc(ds.cfg['train_start'])
    b = _utc(ds.cfg['train_end'])
    return (ds.index >= a) & (ds.index <= b)


def _utc(ts):
    import pandas as pd
    t = pd.Timestamp(ts)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def column_stats(ds, cols):
    """(mean, sd) per column over the train date range. Calendar columns are left alone.

    sd == 0 would divide the column to infinity; upstream replaces it with 1e-8, which turns
    a constant column into a huge one instead. A constant column carries no information
    either way, so the scale is set to 1 and the column becomes a constant 0 after centring.
    """
    m = train_range_mask(ds)
    std_cols = set(ds.cfg['actual_cols']) | set(ds.cfg['forecast_cols'])
    mean = np.zeros(len(cols)); sd = np.ones(len(cols))
    for j, c in enumerate(cols):
        if c not in std_cols:
            continue                           # aux: raw codes, the author's choice
        v = ds.column(c)[m]
        mean[j] = v.mean()
        s = v.std()
        sd[j] = s if s > 0 else 1.0
    return mean, sd


def build(ds, day_starts, cols, mean, sd):
    """(X, y) for those days.

        X  (n_days, L, F)  float32, standardised
        y  (n_days, H)     float32, in the target column's standardised units

    The window is the contiguous block ending at the deadline step, so it is assembled as one
    fancy-index per column rather than per lag.
    """
    starts = np.asarray(day_starts, dtype=int)
    win = starts[:, None] - ds.G - ds.L + 1 + np.arange(ds.L)[None, :]   # (n, L)
    tgt = starts[:, None] + np.arange(ds.H)[None, :]                     # (n, H)

    X = np.empty((len(starts), ds.L, len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        X[:, :, j] = ((ds.column(c)[win] - mean[j]) / sd[j]).astype(np.float32)

    jt = cols.index(ds.target_col)
    y = ((ds.y[tgt] - mean[jt]) / sd[jt]).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------

def make_model(ds, cell, params):
    """ARCHFormer3 for this dataset and this grid cell.

    `num_encoder_layers` and `num_decoder_layers` both take `cell['layers']`: the two counts
    are independent in the architecture (the decoder reads only the LAST encoder layer's mu
    and sigma, src/models/_garchformer/pointmodels.py:80-82) but the grid ties them.
    """
    d_model, heads = int(cell['d_model']), int(cell['num_heads'])
    if d_model % heads:
        raise ValueError(f"d_model={d_model} is not divisible by num_heads={heads}; "
                         "ArchQAttentionBlock and MuCrossAttentionBlock both assert it")
    return ARCHFormer3(
        seq_length_enc=ds.L,
        seq_length_dec=ds.L,          # must equal the encoder's; see the module docstring
        num_features=len(feature_cols(ds)),
        patch_len=ds.H,               # one token = one day
        d_model=d_model,
        num_heads=heads,
        num_encoder_layers=int(cell['layers']),
        num_decoder_layers=int(cell['layers']),
        intermediate_dim=int(params['intermediate_dim']),
        out_len=ds.H,
        dropout=float(params['dropout']),
    )


def _forward_all(model, X, device, batch=256):
    """(y_hat, sigma_sq) over X, in chunks, no grad."""
    model.eval()
    yh, ss = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            a, b = model(xb)
            yh.append(a.cpu().numpy()); ss.append(b.cpu().numpy())
    return np.concatenate(yh), np.concatenate(ss)


def run(ds, cell, params=None, seed=0, device='auto', verbose=True, mode='gaussian'):
    """Train one grid cell and score it on validation and test. Writes nothing.

    Returns a flat dict: the cell, the run facts, and `val_*` / `test_*` metrics. `mode` picks
    how the interval is formed (see MODES); it changes nothing before scoring.
    """
    from src.bench import metrics as M

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    params = {**DEFAULT_PARAMS, **(params or {})}
    dev = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                       if device == 'auto' else device)

    # Set here and not only in the condor wrapper, so that a direct `python scripts/...` run
    # gets the same protection. OMP_NUM_THREADS is read by the BLAS libraries at load time
    # and is already pinned by the wrapper; this is torch's own intra-op pool, which it sizes
    # from the core count regardless. See DEFAULT_PARAMS['threads'].
    if dev.type == 'cpu':
        torch.set_num_threads(int(params['threads']))

    torch.manual_seed(seed)
    np.random.seed(seed)

    cols = feature_cols(ds, verbose=verbose)
    mean, sd = column_stats(ds, cols)
    sp = ds.splits()
    Xtr, ytr = build(ds, sp['train'], cols, mean, sd)
    Xva, yva = build(ds, sp['val'], cols, mean, sd)
    Xte, yte = build(ds, sp['test'], cols, mean, sd)

    model = make_model(ds, cell, params).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  F={len(cols)} patch={ds.H} tokens={ds.L // ds.H} | "
              f"d_model={cell['d_model']} heads={cell['num_heads']} "
              f"layers={cell['layers']} lambda={cell['lambda_cons']} | "
              f"{n_params:,} params | X {Xtr.shape} / {Xva.shape} / {Xte.shape} | {dev}")

    lam = float(cell['lambda_cons'])
    opt = torch.optim.Adam(model.parameters(), lr=float(params['lr']))
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t, yva_t = torch.from_numpy(Xva).to(dev), torch.from_numpy(yva).to(dev)
    bs = int(params['batch_size'])

    best_val, best_state, best_epoch, bad = float('inf'), None, -1, 0
    t0 = time.time()
    for epoch in range(int(params['epochs'])):
        model.train()
        order = torch.randperm(len(Xtr_t))
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            xb, yb = Xtr_t[idx].to(dev), ytr_t[idx].to(dev)
            opt.zero_grad()
            y_hat, sigma_sq = model(xb)
            loss = F.mse_loss(y_hat, yb)
            if lam > 0:
                # ArchFormer3Trainer's variance-consistency term. NOT detached: upstream lets
                # it push the mean path too, and that is part of the method.
                loss = loss + lam * F.l1_loss(sigma_sq, (yb - y_hat) ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(params['grad_clip']))
            opt.step()

        # Point loss ALONE -- see the module docstring on why the lambda term is excluded.
        model.eval()
        with torch.no_grad():
            v = F.mse_loss(model(Xva_t)[0], yva_t).item()
        if v < best_val - 1e-12:
            best_val, best_epoch, bad = v, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= int(params['patience']):
                break
    secs = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    # The reporting scale: the bench's own, so this row sits next to density and iQRNN.
    mean_d, sd_d = ds.standardiser(sp['train'])
    jt = cols.index(ds.target_col)

    out = {**{k: cell[k] for k in ('d_model', 'num_heads', 'layers', 'lambda_cons')},
           'dataset': ds.name, 'seed': seed, 'n_params': n_params,
           'epochs_run': epoch + 1, 'best_epoch': best_epoch,
           'val_mse_model_units': best_val, 'seconds': secs,
           'n_features': len(cols), 'patch_len': ds.H, 'device': str(dev),
           'mode': mode, 'sigma_scale': 1.0}

    # val is scored first on purpose: sigma_scaled fits its one scalar there and test reuses it.
    for part, (X, y) in (('val', (Xva, yva)), ('test', (Xte, yte))):
        y_hat, sigma_sq = _forward_all(model, X, dev)
        # model units -> physical -> bench z
        phys_hat = y_hat * sd[jt] + mean[jt]
        phys_sig = np.sqrt(np.maximum(sigma_sq, 0.0)) * sd[jt]
        phys_act = y * sd[jt] + mean[jt]
        z_hat = ((phys_hat - mean_d) / sd_d).ravel()
        z_sig = (phys_sig / sd_d).ravel()
        z_act = ((phys_act - mean_d) / sd_d).ravel()
        if mode == 'sigma_scaled':
            if part == 'val':
                resid_rms = float(np.sqrt(np.mean((z_act - z_hat) ** 2)))
                sig_rms = float(np.sqrt(np.mean(z_sig ** 2)))
                out['sigma_scale'] = resid_rms / max(sig_rms, 1e-12)
            z_sig = z_sig * out['sigma_scale']
        q = z_hat[None, :] + ZS[:, None] * z_sig[None, :]
        q = np.sort(q, axis=0)
        s = M.score(z_act, q, LEVELS, scale=(mean_d, sd_d))
        out.update({f'{part}_{k}': v for k, v in s.items()})
        out[f'n_{part}_days'] = len(X)
    return out
