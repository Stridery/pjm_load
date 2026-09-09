"""GARCHFormer (ARCHFormer3) — vendored from Moyi Guo's `garchformer` repository.

Two files are copied verbatim from `garchformer/models/archformer3/`:

    layers.py       PatchTokenizer, ARCHEncoder/Decoder, VarianceFiLMFusion, ...
    pointmodels.py  ARCHFormer3, the assembled model

The ONLY edit is the import path at the top of pointmodels.py
(`models.archformer3.layers` -> `src.models.archformer.layers`). Nothing else is
touched, so the architecture and its hyperparameters are the author's.

What is NOT vendored, and why
-----------------------------
The upstream repo also ships its own data pipeline (utils/data_utils.py),
splitter, trainer, metrics and grid-search driver. None of that is used here,
because this repo already has all of it AND because the upstream versions answer
a DIFFERENT question:

  * upstream `forecast_deadline_hour: 23` reads data through (D-1) 23:00 and
    predicts day D. This repo's 16 point models read through (D-1) 23:00 and
    predict day D+1 -- one day further out, with the blind-spot day in between.
    Scoring the two together would flatter GARCHFormer for a reason that has
    nothing to do with the architecture.
  * upstream splits on hardcoded calendar dates; this repo splits tail-0.16 with
    a 14-day embargo, so the test periods would not match and the row could not
    go on the shared scoreboard.
  * upstream reports MAE/MSE/RMSE in STANDARDIZED units (only MAPE/SMAPE are
    rescaled); this repo reports MW.
  * upstream has no 3-sigma validity mask and no DST normalisation to 24 EPT
    slots.

So the model is imported and everything around it stays this repo's own. See
src/models/archformer.py for the adapter.
"""
