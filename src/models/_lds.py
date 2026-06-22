"""Label Distribution Smoothing (LDS) for imbalanced regression.

Ref: Westphal (2025), Deep Imbalanced Regression.

Steps:
  1. Bin training labels into equal-width buckets.
  2. Count empirical frequency per bucket.
  3. Smooth counts with a Gaussian kernel (reduces noise at sparse boundary bins).
  4. Weight = 1 / smoothed_freq, normalised so mean weight = 1.

Rare labels (e.g. peak-load days) receive higher weights; common labels lower.
"""

import numpy as np
from scipy.ndimage import convolve1d


def _gaussian_kernel(ks: int, sigma: float) -> np.ndarray:
    half = ks // 2
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-x ** 2 / (2.0 * sigma ** 2))
    return k / k.sum()


def compute_lds_weights(
    labels: np.ndarray,
    n_bins: int = 100,
    ks: int = 5,
    sigma: float = 2.0,
    min_freq_ratio: float = 0.05,
) -> np.ndarray:
    """Return per-sample LDS weights (mean = 1) for a 1-D label array.

    Args:
        labels:         1-D array of target values (MW or scaled).
        n_bins:         Number of equal-width histogram bins.
        ks:             Gaussian kernel half-width (kernel size = 2*ks+1 ... actually ks).
                        Must be odd; defaults to 5.
        sigma:          Gaussian kernel std-dev (in bin units).
        min_freq_ratio: Bins whose smoothed count falls below
                        max_count * min_freq_ratio are clamped to avoid
                        extreme weights for near-empty edge bins.

    Returns:
        weights: np.ndarray of shape (len(labels),), normalised so mean = 1.
    """
    labels = np.asarray(labels, dtype=float)
    lo, hi = labels.min(), labels.max()

    edges = np.linspace(lo, hi, n_bins + 1)
    # digitize returns 1-indexed; subtract 1 for 0-indexed bins, clip to [0, n_bins-1]
    bin_idx = np.clip(np.digitize(labels, edges[1:-1]), 0, n_bins - 1)

    counts = np.bincount(bin_idx, minlength=n_bins).astype(float)

    kernel = _gaussian_kernel(ks, sigma)
    smoothed = convolve1d(counts, weights=kernel, mode='reflect')

    # Clamp near-empty bins to avoid exploding weights at the tails
    smoothed = np.clip(smoothed, smoothed.max() * min_freq_ratio, None)

    bin_weights = 1.0 / smoothed
    sample_weights = bin_weights[bin_idx]

    # Normalise: mean weight = 1  →  preserves effective learning rate
    return sample_weights / sample_weights.mean()
