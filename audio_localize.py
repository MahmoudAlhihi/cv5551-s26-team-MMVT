"""
audio_localize.py

Smooths raw 2.5 kHz Goertzel magnitude data from sweep passes and
maps the peak intensity back to robot-frame (X, Y) coordinates.

Usage:
    from audio_localize import localize_source

    x_coord = localize_source(data_x, X_MAX, X_MIN)   # X sweep goes MAX->MIN
    y_coord = localize_source(data_y, Y_MIN, Y_MAX)   # Y sweep goes MIN->MAX
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


def smooth(data: list[float], method: str = "savgol", **kwargs) -> np.ndarray:
    """
    Smooth a 1-D signal.

    Methods
    -------
    savgol   – Savitzky-Golay (preserves peaks well, good default)
    gaussian – Gaussian kernel smoothing
    median   – sliding median (good for spike removal)

    Returns the smoothed array (same length as input).
    """
    arr = np.asarray(data, dtype=float)
    if len(arr) < 5:
        return arr

    if method == "savgol":
        win = kwargs.get("window", min(9, len(arr) // 2 * 2 + 1))  # must be odd
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
    """Map a sample index to a robot-frame coordinate via linear interpolation."""
    if n_samples <= 1:
        return (start_coord + end_coord) / 2.0
    t = index / (n_samples - 1)
    return start_coord + t * (end_coord - start_coord)


def localize_source(data: list[float],
                    start_coord: float,
                    end_coord: float,
                    method: str = "savgol",
                    peak_weight: float = 0.7,
                    cluster_weight: float = 0.3,
                    top_k: int = 5,
                    **kwargs) -> tuple[float, np.ndarray]:
    smoothed = smooth(data, method=method, **kwargs)
    
    # --- Highest peak (primary) ---
    peak_idx = int(np.argmax(smoothed))
    
    # --- Cluster of top-k peaks (secondary) ---
    top_indices = np.argsort(smoothed)[-top_k:]
    weights = smoothed[top_indices]
    cluster_idx = np.average(top_indices, weights=weights)
    
    # --- Blend ---
    blended_idx = peak_weight * peak_idx + cluster_weight * cluster_idx
    blended_idx = np.clip(blended_idx, 0, len(smoothed) - 1)
    
    coord = sample_to_coord(int(round(blended_idx)), len(smoothed),
                            start_coord, end_coord)
    return coord, smoothed