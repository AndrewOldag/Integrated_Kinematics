"""Disk-domain generation for patch tracking."""

from __future__ import annotations

import numpy as np


def create_disk_domain(radius: int, n_angular: int = 200) -> np.ndarray:
    """Generate sample offsets inside a disk as (drow, dcol)."""
    r_vals = np.linspace(0, radius, radius + 1)
    theta_vals = np.linspace(-np.pi, np.pi, n_angular)
    rr, tt = np.meshgrid(r_vals, theta_vals, indexing="ij")
    drow = (rr * np.cos(tt)).ravel()
    dcol = (rr * np.sin(tt)).ravel()
    return np.column_stack([drow, dcol])
