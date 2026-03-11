"""Affine patch tracking for propagated midline points."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.ndimage import map_coordinates

from .domain import create_disk_domain


def track_points(
    images: np.ndarray,
    initial_points: np.ndarray,
    disk_radius: int = 28,
    threshold: float = 10.0,
    n_angular: int = 200,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    n_frames = images.shape[0]
    n_points = initial_points.shape[0]
    point_list = np.zeros((n_points, 2, n_frames))
    point_list[:, :, 0] = initial_points

    domain = create_disk_domain(disk_radius, n_angular)
    clip_radius = disk_radius + 20
    tol = 1e-6

    for t in range(n_frames - 1):
        i_frame = images[t]
        g_frame = images[t + 1]

        for pt in range(n_points):
            pos = point_list[pt, :, t]
            if pos[1] > threshold:
                displacement = _affine_patch_match(i_frame, g_frame, pos, clip_radius, domain, tol)
            else:
                displacement = np.array([0.0, 0.0])
            point_list[pt, :, t + 1] = pos + displacement

        if progress_callback:
            progress_callback(t + 1, n_frames - 1)

    return point_list


def _affine_patch_match(
    i_frame: np.ndarray,
    g_frame: np.ndarray,
    point: np.ndarray,
    radius: int,
    domain: np.ndarray,
    tol: float = 1e-6,
) -> np.ndarray:
    r, c = point
    row_range = np.arange(r - radius, r + radius + 1)
    col_range = np.arange(c - radius, c + radius + 1)
    grid_r, grid_c = np.meshgrid(row_range, col_range, indexing="ij")

    i_patch = map_coordinates(
        i_frame, [grid_r.ravel(), grid_c.ravel()], order=1, mode="nearest"
    ).reshape(grid_r.shape)
    g_patch = map_coordinates(
        g_frame, [grid_r.ravel(), grid_c.ravel()], order=1, mode="nearest"
    ).reshape(grid_r.shape)

    d1, d2 = np.gradient(i_patch)
    centre = radius
    sample_r = domain[:, 0] + centre
    sample_c = domain[:, 1] + centre
    ii = map_coordinates(i_patch, [sample_r, sample_c], order=1, mode="nearest")
    d1i = map_coordinates(d1, [sample_r, sample_c], order=1, mode="nearest")
    d2i = map_coordinates(d2, [sample_r, sample_c], order=1, mode="nearest")
    x = domain.copy()

    t_params = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    norms: list[float] = []
    for _ in range(50):
        tr = t_params.reshape(2, 3)
        xh = np.hstack([x, np.ones((x.shape[0], 1))])
        xt = (tr @ xh.T).T
        gi = map_coordinates(
            g_patch, [xt[:, 0] + centre, xt[:, 1] + centre], order=1, mode="nearest"
        )

        dy = x[:, 0]
        dx = x[:, 1]
        mi = np.column_stack([d1i * dy, d1i * dx, d1i, d2i * dy, d2i * dx, d2i])
        residual = ii - gi
        d_t, *_ = np.linalg.lstsq(mi, residual, rcond=None)
        t_params = t_params + d_t
        current_norm = float(np.linalg.norm(residual))
        norms.append(current_norm)

        if np.linalg.norm(d_t) <= tol:
            break
        if len(norms) >= 2 and norms[-1] >= norms[-2]:
            t_params = t_params - d_t
            break

    drow = t_params[2]
    dcol = t_params[5]
    return np.array([drow, dcol])
