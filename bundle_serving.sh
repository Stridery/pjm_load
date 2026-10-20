#!/bin/bash
# Bundle the serving artifacts (model weights + scalers) and upload them as a
# GitHub Release asset, so the daily CI job can download them without the ~180 MB living in git
# history. Run this ONCE after every retrain.
#
#   ./bundle_serving.sh
#
# Needs the `gh` CLI authenticated (gh auth login). The daily workflow downloads the asset with
# `gh release download serving-latest`.
set -euo pipefail

TAG="serving-latest"
BUNDLE="serving_bundle.tar.gz"

# Everything the serving predict path loads, and nothing else (no raw data — serving fetches
# that fresh; no training matrices — only the scalers are needed).
# Only what the daily job loads: the SERVING_ARM run directory of each SERVING_MODELS entry,
# direct and residual, plus that arm's scalers. models/ also holds the other three feature
# arms, the MoE heads and the superseded GARCHFormer variants -- shipping those would multiply
# the download for weights src.serving.run will never open.
ARM_TAG="_cong"
RUN_SEQ="tail_test0.16_tail_val0.1${ARM_TAG}"      # sequence models carry a validation tag
RUN_TREE="tail_test0.16${ARM_TAG}"                 # tree models do not
MODELS="xgboost lightgbm transformer lstm mstnn archformer_skip"

PATHS=()
for z in dom bge; do
    for m in $MODELS; do
        for name in "$m" "${m}_residual"; do
            for run in "$RUN_SEQ" "$RUN_TREE"; do
                d="models/$z/$name/$run"
                [ -d "$d" ] && PATHS+=("$d")
            done
        done
    done
    PATHS+=(data/$z/matrix/*lb168_h0${ARM_TAG}.pkl)
done
[ ${#PATHS[@]} -eq 0 ] && { echo "Nothing to bundle — train first."; exit 1; }
echo "Bundling ${#PATHS[@]} paths"
tar -czf "$BUNDLE" "${PATHS[@]}"

echo "Bundle: $BUNDLE  ($(du -h "$BUNDLE" | cut -f1))"

# Create the release if it does not exist, then overwrite the asset.
gh release view "$TAG" >/dev/null 2>&1 || \
    gh release create "$TAG" --title "Serving artifacts (latest)" \
        --notes "Model weights + scalers + thermal refs for the daily forecast job. Overwritten each retrain."
gh release upload "$TAG" "$BUNDLE" --clobber

echo "Uploaded $BUNDLE to release '$TAG'. The daily workflow will pick it up on its next run."
