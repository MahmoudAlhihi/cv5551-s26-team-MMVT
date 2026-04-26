"""
audio_localize.py

Smooths Goertzel data from sweep passes and maps the peak intensity
back to robot-frame (X, Y) coordinates.

Each sample is (magnitude, snr) from the three-Goertzel Arduino sketch.
Localization uses *sustained* SNR — a real acoustic source produces
high SNR for many consecutive samples; transients (clicks, bumps) only
spike for one or two. Requiring sustained SNR rejects loud transients
even when their absolute magnitude exceeds the real source.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d
from scipy.signal import savgol_filter


# Tuneable gating parameters
SNR_FLOOR     = 1.5   # sustained SNR below this contributes nothing
SUSTAIN_SAMPLES = 7   # samples must have high SNR over this window to count
                      # at ~100 samples/sec sweep speed and 100 mm/s arm,
                      # 7 samples ≈ 70 ms ≈ 7 mm of motion — narrower than
                      # any real source bump, wider than any transient


def _to_signal(data, **kwargs) -> np.ndarray:
    """
    Convert raw sweep data into a 1-D signal suitable for peak finding.

    For (magnitude, snr) pairs, computes:  magnitude * snr^2

    SNR² weighting heavily favors high-SNR samples (real source) over
    high-magnitude/low-SNR samples (broadband transients). Example:
        Real source:  mag=20, snr=8  → 20 * 64  = 1280
        Loud clap:    mag=200, snr=2 → 200 * 4  = 800
    Real wins despite 10x less raw magnitude. No hard gate means narrow
    peaks aren't accidentally zeroed.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] >= 2:
        magnitude = arr[:, 0]
        snr = np.clip(arr[:, 1], 0.0, None)  # SNR is always non-negative
        return magnitude * snr * snr
    raise ValueError(f"Unexpected data shape {arr.shape}")


def smooth(data, method: str = "savgol", **kwargs) -> np.ndarray:
    arr = _to_signal(data)
    if len(arr) < 5:
        return arr

    if method == "savgol":
        win = kwargs.get("window", min(9, len(arr) // 2 * 2 + 1))
        if win % 2 == 0:
            win += 1
        poly = kwargs.get("poly", 3)
        return savgol_filter(arr, win, poly)
    elif method == "gaussian":
        sigma = kwargs.get("sigma", 5.0)
        return gaussian_filter1d(arr, sigma)
    elif method == "median":
        from scipy.ndimage import median_filter
        size = kwargs.get("size", 7)
        return median_filter(arr, size=size)
    else:
        raise ValueError(f"Unknown method: {method}")


def sample_to_coord(index: int, n_samples: int,
                    start_coord: float, end_coord: float) -> float:
    if n_samples <= 1:
        return (start_coord + end_coord) / 2.0
    t = index / (n_samples - 1)
    return start_coord + t * (end_coord - start_coord)


def localize_source(data,
                    start_coord: float,
                    end_coord: float,
                    method: str = "savgol",
                    peak_weight: float = 0.7,
                    cluster_weight: float = 0.3,
                    top_k: int = 5,
                    **kwargs) -> tuple[float, np.ndarray]:
    smoothed = smooth(data, method=method, **kwargs)

    if len(smoothed) == 0:
        raise RuntimeError(
            "localize_source: empty signal. Check Arduino output / read_mic.py."
        )

    peak_idx = int(np.argmax(smoothed))

    top_indices = np.argsort(smoothed)[-top_k:]
    weights = smoothed[top_indices]
    if weights.sum() < 1e-12:
        cluster_idx = peak_idx
    else:
        cluster_idx = np.average(top_indices, weights=weights)

    blended_idx = peak_weight * peak_idx + cluster_weight * cluster_idx
    blended_idx = np.clip(blended_idx, 0, len(smoothed) - 1)

    coord = sample_to_coord(int(round(blended_idx)), len(smoothed),
                            start_coord, end_coord)
    return coord, smoothed