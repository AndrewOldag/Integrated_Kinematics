"""
Outlier detection and removal for velocity / REGR profile data.

Provides array-level filtering (no file I/O) so it integrates cleanly
into the profile loading pipeline.  Two methods are available:

  - **MAD** (median absolute deviation) — robust, recommended for
    biological data where the distribution may not be perfectly normal.
  - **IQR** (inter-quartile range) — classic box-plot fences.

Both methods operate on the *y* values (velocity or REGR) and return a
boolean mask of the same length indicating which points to **keep**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class OutlierSettings:
    """User-configurable outlier filtering settings."""

    enabled: bool = True
    method: str = "mad"  # "mad" or "iqr"
    mad_thresh: float = 4.0  # robust z-score cutoff for MAD
    iqr_k: float = 3.0  # fence multiplier for IQR
    keep_nonfinite: bool = False  # keep NaN / Inf rows?


def outlier_mask(
    y: np.ndarray,
    method: str = "mad",
    mad_thresh: float = 4.0,
    iqr_k: float = 3.0,
    keep_nonfinite: bool = False,
) -> np.ndarray:
    """Return a boolean *keep* mask for *y* based on the chosen method.

    Parameters
    ----------
    y : 1-D array of float
        The values to check (typically velocity or REGR).
    method : ``"mad"`` | ``"iqr"``
        Detection algorithm.
    mad_thresh : float
        For MAD: reject points whose |robust z-score| exceeds this.
    iqr_k : float
        For IQR: reject points outside ``[Q1 - k*IQR, Q3 + k*IQR]``.
    keep_nonfinite : bool
        If *True*, NaN / Inf values are kept (masked as True).

    Returns
    -------
    np.ndarray of bool, same length as *y*.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return np.ones(0, dtype=bool)

    finite = np.isfinite(y)
    mask = np.ones(n, dtype=bool)

    if not keep_nonfinite:
        mask &= finite

    x = y[finite]
    if x.size < 5:
        # Too few finite points to reliably detect outliers — keep all.
        return mask

    if method.lower() == "mad":
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad == 0:
            # MAD is zero (e.g. many identical values) — fall back to SD.
            sd = np.std(x)
            if sd == 0:
                return mask  # all identical → nothing is an outlier
            z = (y - med) / sd
        else:
            z = 0.6745 * (y - med) / mad  # 0.6745 ≈ norm.ppf(0.75)
        good = np.abs(z) <= mad_thresh

    elif method.lower() == "iqr":
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        lo = q1 - iqr_k * iqr
        hi = q3 + iqr_k * iqr
        good = (y >= lo) & (y <= hi)

    else:
        raise ValueError(f"method must be 'mad' or 'iqr', got '{method}'")

    if keep_nonfinite:
        good = good | (~finite)

    mask &= good
    return mask


def filter_profile_arrays(
    x: np.ndarray,
    y: np.ndarray,
    settings: OutlierSettings,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Filter paired x/y arrays according to *settings*.

    Returns
    -------
    (x_clean, y_clean, n_removed)
    """
    if not settings.enabled:
        return x, y, 0

    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]

    keep = outlier_mask(
        y,
        method=settings.method,
        mad_thresh=settings.mad_thresh,
        iqr_k=settings.iqr_k,
        keep_nonfinite=settings.keep_nonfinite,
    )

    n_removed = int(np.sum(~keep))
    return x[keep], y[keep], n_removed
