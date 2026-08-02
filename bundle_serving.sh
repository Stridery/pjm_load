#!/bin/bash
# Bundle the serving artifacts (model weights + scalers + thermal refs) and upload them as a
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
# that fresh; no training matrices — only the scalers + thermal_refs.pkl are needed).
tar -czf "$BUNDLE" \
    models/dom models/bge \
    data/dom/matrix/*.pkl data/bge/matrix/*.pkl

echo "Bundle: $BUNDLE  ($(du -h "$BUNDLE" | cut -f1))"

# Create the release if it does not exist, then overwrite the asset.
gh release view "$TAG" >/dev/null 2>&1 || \
    gh release create "$TAG" --title "Serving artifacts (latest)" \
        --notes "Model weights + scalers + thermal refs for the daily forecast job. Overwritten each retrain."
gh release upload "$TAG" "$BUNDLE" --clobber

echo "Uploaded $BUNDLE to release '$TAG'. The daily workflow will pick it up on its next run."
