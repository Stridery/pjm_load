"""Feature Distribution Smoothing (FDS) for imbalanced regression.

Ref: Westphal (2025), Deep Imbalanced Regression.

At each training epoch:
  1. Accumulate (feature, label) pairs from every training batch.
  2. After the batch loop: bin labels, compute per-bin mean/var of features.
  3. Apply a Gaussian kernel to smooth statistics across adjacent bins.
  4. Next epoch's forward pass calibrates each sample's feature vector via
     an affine transform that maps the bin's raw distribution to the
     smoothed one:

       f_cal = (f - mean_raw[b]) / std_raw[b] * std_smooth[b] + mean_smooth[b]

Rare samples (e.g. peak-load days) have noisy feature statistics; smoothing
borrows stability from neighbouring label-adjacent samples.

FDS is applied during training only.  Inference uses the standard forward pass.
"""

import numpy as np
import torch
from scipy.ndimage import convolve1d


def _gaussian_kernel(ks: int, sigma: float) -> np.ndarray:
    half = ks // 2
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-x ** 2 / (2.0 * sigma ** 2))
    return k / k.sum()


class FDSModule:
    """Stateful FDS manager — one instance per training run.

    Args:
        feature_dim:  Dimensionality of encoder output (hidden_size or d_model).
        n_bins:       Number of equal-width label bins.
        ks:           Gaussian kernel size (number of points, should be odd).
        sigma:        Gaussian kernel std-dev in bin units.
        eps:          Numerical stability for std computation.
    """

    def __init__(
        self,
        feature_dim: int,
        n_bins: int = 50,
        ks: int = 5,
        sigma: float = 2.0,
        eps: float = 1e-6,
    ):
        self.feature_dim = feature_dim
        self.n_bins = n_bins
        self.eps = eps
        self.kernel = _gaussian_kernel(ks, sigma)

        # Per-bin raw statistics — updated every epoch
        self._mean = np.zeros((n_bins, feature_dim), dtype=np.float64)
        self._var  = np.zeros((n_bins, feature_dim), dtype=np.float64)

        # Smoothed statistics — used for calibration
        self.mean_smooth = np.zeros((n_bins, feature_dim), dtype=np.float64)
        self.var_smooth  = np.zeros((n_bins, feature_dim), dtype=np.float64)

        # Label range (set from first epoch's labels, fixed thereafter)
        self._lo: float | None = None
        self._hi: float | None = None

        # Calibration is skipped until at least one smooth() call has run
        self._ready = False

        # Batch accumulators — reset at start of each update call
        self._acc_features: list[np.ndarray] = []
        self._acc_labels:   list[np.ndarray] = []

    # ------------------------------------------------------------------ #
    # Per-batch accumulation                                               #
    # ------------------------------------------------------------------ #

    def collect(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Append detached (feature, label) arrays from one training batch.

        Call once per batch, after optimizer.step().
        features: (batch, feature_dim)
        labels:   (batch,) — representative scalar per sample (e.g. mean load)
        """
        self._acc_features.append(features.detach().cpu().float().numpy())
        self._acc_labels.append(labels.detach().cpu().float().numpy())

    # ------------------------------------------------------------------ #
    # Epoch-end statistics update                                          #
    # ------------------------------------------------------------------ #

    def update_and_smooth(self) -> None:
        """Compute per-bin stats from accumulated epoch data, then Gaussian-smooth.

        Call once at the END of every training epoch.
        """
        if not self._acc_features:
            return

        features = np.concatenate(self._acc_features, axis=0)  # (N, feat_dim)
        labels   = np.concatenate(self._acc_labels,   axis=0)  # (N,)

        # Clear for next epoch
        self._acc_features = []
        self._acc_labels   = []

        # Fix label range from first epoch
        if self._lo is None:
            self._lo = float(labels.min())
            self._hi = float(labels.max())

        edges   = np.linspace(self._lo, self._hi, self.n_bins + 1)
        bin_idx = np.clip(np.digitize(labels, edges[1:-1]), 0, self.n_bins - 1)

        for b in range(self.n_bins):
            mask = bin_idx == b
            n = int(mask.sum())
            if n >= 1:
                self._mean[b] = features[mask].mean(axis=0)
            if n >= 2:
                self._var[b] = features[mask].var(axis=0)
            # n == 0: leave previous stats (or zeros on first epoch)

        # Smooth each feature dimension independently across bins
        for d in range(self.feature_dim):
            self.mean_smooth[:, d] = convolve1d(
                self._mean[:, d], self.kernel, mode='reflect'
            )
            self.var_smooth[:, d] = convolve1d(
                np.maximum(self._var[:, d], 0.0), self.kernel, mode='reflect'
            )

        self._ready = True

    # ------------------------------------------------------------------ #
    # Calibration (training forward pass only)                             #
    # ------------------------------------------------------------------ #

    def calibrate(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Affine-calibrate encoder features using smoothed per-bin statistics.

        Gradient flows through the affine transform back into the encoder.
        Smoothed stats are treated as constants (no gradient).

        Args:
            features: (batch, feature_dim) — raw encoder output, on any device.
            labels:   (batch,) — representative label per sample.

        Returns:
            Calibrated feature tensor (same shape and device as input).
        """
        if not self._ready:
            return features

        device    = features.device
        labels_np = labels.detach().cpu().float().numpy()

        edges   = np.linspace(self._lo, self._hi, self.n_bins + 1)
        bin_idx = np.clip(np.digitize(labels_np, edges[1:-1]), 0, self.n_bins - 1)

        def to_t(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(arr[bin_idx], dtype=torch.float32, device=device)

        mean_raw    = to_t(self._mean)
        std_raw     = to_t(np.sqrt(np.maximum(self._var,       0.0) + self.eps))
        mean_smooth = to_t(self.mean_smooth)
        std_smooth  = to_t(np.sqrt(np.maximum(self.var_smooth, 0.0) + self.eps))

        return (features - mean_raw) / std_raw * std_smooth + mean_smooth
