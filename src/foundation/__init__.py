"""Zero-shot foundation-model forecasters, benchmarked against the 16 point models.

A pre-trained univariate model (TimesFM and friends) is handed the recent Load_Estimated
history and asked for the target day. Nothing is fitted here, so there is no train step and
no weights of ours to save.

Why this is a separate pipeline
-------------------------------
Deliberately isolated from src/models/, for the same reason src/quantile and src/density are:
that package's registry (TREE_MODELS / SEQ_MODELS) is what src/serving/predict.available_models()
scans, and anything registered there is loaded by the daily CI forecast job. A model that needs
an optional package and a ~1 GB checkpoint the CI runner does not have would break the daily
forecast the moment it was registered. Keeping it out means scripts/Model_Training.py, model_evaluator,
serving, generate_web and Train_Model.sbatch are all untouched.

The RESULTS are not isolated. They are written into results/{zone}/evaluation/ next to the 16
point models — comparing them is the whole point — with direct and residual under separate
model names, exactly like xgboost / xgboost_residual:

    results/{zone}/evaluation/timesfm/tail_test0.16_cong_ctx512/
    results/{zone}/evaluation/timesfm_residual/tail_test0.16_cong_ctx512/

What is shared with the point-model stack
-----------------------------------------
Everything except the model call itself:

    feature_engine._split_indices        the identical tail-0.16 split
    feature_engine._normalize_to_24h     DST days normalised to 24 EPT slots
    _residual.baseline_from_windows      the same-hour-last-week baseline, bit-for-bit
    _eval_utils.EvalUtils.evaluate_one   the metrics file and all seven figures

The test DAYS come from the y matrix rather than from a fresh split of the cleaned CSV, so the
day set carries the matrix builder's drop rules (no complete forecast / no congestion -> day
dropped). Without that the two would be scored on different days and the comparison would mean
nothing.
"""
