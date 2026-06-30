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


def _fill_sparse_bins(arr: np.ndarray, counts: np.ndarray, min_n: int) -> np.ndarray:
    """Replace rows in arr where counts < min_n with nearest-neighbour row.

    Operates on a copy; does not mutate arr.
    """
    out = arr.copy()
    sparse = np.where(counts < min_n)[0]
    dense  = np.where(counts >= min_n)[0]
    if len(dense) == 0 or len(sparse) == 0:
        return out
    for b in sparse:
        nearest = dense[np.argmin(np.abs(dense - b))]
        out[b] = arr[nearest]
    return out


class FDSModule:
    """Stateful FDS manager — one instance per training run.

    Args:
        feature_dim:  Dimensionality of encoder output (hidden_size or d_model).
        bin_width:    Physical width of each label bin (same units as labels, e.g. MW).
                      Bins are anchored to multiples of bin_width and fixed after
                      the first epoch so EMA statistics are consistent across epochs.
        ks:           Gaussian kernel size (number of points, should be odd).
        sigma:        Gaussian kernel std-dev in bin units.
        momentum:     EMA coefficient for per-bin statistics (weight on current epoch).
                      0 = never update; 1 = replace history each epoch.
        eps:          Floor added to raw std before division (prevents amplification
                      in single-sample bins). Set to ~1 % of expected feature std.
    """

    def __init__(
        self,
        feature_dim: int,
        bin_width: float = 200.0,
        ks: int = 5,
        sigma: float = 2.0,
        momentum: float = 0.1,
        eps: float = 1e-4,
    ):
        self.feature_dim = feature_dim
        self.bin_width   = bin_width
        self.momentum    = momentum
        self.eps         = eps
        self.kernel      = _gaussian_kernel(ks, sigma)

        # Bin edges — fixed from first epoch's physical grid, reused every subsequent epoch
        self._edges: np.ndarray | None = None
        self.n_bins: int = 0

        # Per-bin raw statistics (allocated after first epoch when n_bins is known)
        self._mean: np.ndarray | None = None
        self._var:  np.ndarray | None = None
        self._counts: np.ndarray | None = None   # how many samples seen per bin (for fill)

        # Smoothed statistics — used for calibration
        self.mean_smooth: np.ndarray | None = None
        self.var_smooth:  np.ndarray | None = None

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
        labels:   (batch,) — representative scalar per sample (e.g. peak load)
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

        self._acc_features = []
        self._acc_labels   = []

        # First epoch: build fixed physical-width edges and allocate stat arrays
        if self._edges is None:
            lo = np.floor(labels.min() / self.bin_width) * self.bin_width
            hi = np.ceil(labels.max()  / self.bin_width) * self.bin_width
            self._edges  = np.arange(lo, hi + self.bin_width, self.bin_width)
            self.n_bins  = len(self._edges) - 1
            self._mean   = np.zeros((self.n_bins, self.feature_dim), dtype=np.float64)
            self._var    = np.zeros((self.n_bins, self.feature_dim), dtype=np.float64)
            self._counts = np.zeros(self.n_bins, dtype=np.int64)
            self.mean_smooth = np.zeros((self.n_bins, self.feature_dim), dtype=np.float64)
            self.var_smooth  = np.zeros((self.n_bins, self.feature_dim), dtype=np.float64)

        bin_idx = np.clip(np.digitize(labels, self._edges[1:-1]), 0, self.n_bins - 1)

        m = self.momentum
        epoch_counts = np.zeros(self.n_bins, dtype=np.int64)
        for b in range(self.n_bins):
            mask = bin_idx == b
            n    = int(mask.sum())
            epoch_counts[b] = n
            if n >= 1:
                cur_mean = features[mask].mean(axis=0)
                if self._ready:
                    self._mean[b] = (1 - m) * self._mean[b] + m * cur_mean
                else:
                    self._mean[b] = cur_mean
            if n >= 2:
                cur_var = features[mask].var(axis=0)
                if self._ready:
                    self._var[b] = (1 - m) * self._var[b] + m * cur_var
                else:
                    self._var[b] = cur_var
            # n == 0: leave previous EMA stats untouched

        self._counts = epoch_counts

        # Fill sparse bins (n < 2) with nearest-neighbour before smoothing
        # to prevent zero-var bins from pulling down neighbours via convolution
        mean_filled = _fill_sparse_bins(self._mean, epoch_counts, min_n=1)
        var_filled  = _fill_sparse_bins(self._var,  epoch_counts, min_n=2)

        for d in range(self.feature_dim):
            self.mean_smooth[:, d] = convolve1d(
                mean_filled[:, d], self.kernel, mode='reflect'
            )
            self.var_smooth[:, d] = convolve1d(
                np.maximum(var_filled[:, d], 0.0), self.kernel, mode='reflect'
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

        bin_idx = np.clip(np.digitize(labels_np, self._edges[1:-1]), 0, self.n_bins - 1)

        def to_t(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(arr[bin_idx], dtype=torch.float32, device=device)

        mean_raw    = to_t(self._mean)
        std_raw     = to_t(np.sqrt(np.maximum(self._var,       0.0) + self.eps))
        mean_smooth = to_t(self.mean_smooth)
        std_smooth  = to_t(np.sqrt(np.maximum(self.var_smooth, 0.0) + self.eps))

        return (features - mean_raw) / std_raw * std_smooth + mean_smooth
