"""Interactive tools for manual tracing and auto-midline review."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button


def trace_midline_on_image(image: np.ndarray, spacing_px: float = 15.0) -> np.ndarray:
    """Collect a manual tip->base polyline and resample it to fixed spacing."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(image, cmap="gray")
    ax.set_title(
        "Manual midline trace: click tip -> base. "
        "Left click add points, right click finish."
    )
    ax.axis("off")

    points_xy: list[tuple[float, float]] = []
    line_artist = {"line": None}
    done = {"value": False}

    def _redraw() -> None:
        if line_artist["line"] is not None:
            line_artist["line"].remove()
        if points_xy:
            xs = [p[0] for p in points_xy]
            ys = [p[1] for p in points_xy]
            (line_artist["line"],) = ax.plot(xs, ys, "g.-", linewidth=1.5)
        fig.canvas.draw_idle()

    def _on_click(event) -> None:
        if event.inaxes != ax:
            return
        if event.button == 1:
            points_xy.append((float(event.xdata), float(event.ydata)))
            _redraw()
        elif event.button == 3 and len(points_xy) >= 2:
            done["value"] = True
            plt.close(fig)

    cid = fig.canvas.mpl_connect("button_press_event", _on_click)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    if len(points_xy) < 2:
        raise ValueError("Manual tracing requires at least 2 points.")
    return resample_polyline_xy(np.array(points_xy, dtype=float), spacing_px=spacing_px)


def review_auto_midline_overlay(
    image: np.ndarray,
    midline_xy: list[tuple[float, float]],
    qc_xy: tuple[float, float],
    title: str = "Auto midline review",
) -> bool:
    """Show first-frame auto overlay with Approve / Deny buttons."""
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.subplots_adjust(bottom=0.2)
    ax.imshow(image, cmap="gray")
    if len(midline_xy) > 1:
        coords = np.array(midline_xy)
        ax.plot(coords[:, 0], coords[:, 1], "g-", linewidth=2)
    ax.plot(qc_xy[0], qc_xy[1], "r*", markersize=14)
    ax.set_title(title)
    ax.axis("off")

    approve_ax = plt.axes([0.28, 0.05, 0.18, 0.08])
    deny_ax = plt.axes([0.54, 0.05, 0.18, 0.08])
    approve_btn = Button(approve_ax, "Approve")
    deny_btn = Button(deny_ax, "Deny")

    decision = {"approved": False, "set": False}

    def _approve(_event) -> None:
        decision["approved"] = True
        decision["set"] = True
        plt.close(fig)

    def _deny(_event) -> None:
        decision["approved"] = False
        decision["set"] = True
        plt.close(fig)

    approve_btn.on_clicked(_approve)
    deny_btn.on_clicked(_deny)
    plt.show()
    return bool(decision["approved"] if decision["set"] else False)


def resample_polyline_xy(polyline_xy: np.ndarray, spacing_px: float = 15.0) -> np.ndarray:
    if polyline_xy.shape[0] < 2:
        raise ValueError("Need at least 2 points to resample.")

    diffs = np.diff(polyline_xy, axis=0)
    seg_len = np.sqrt(np.sum(diffs**2, axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = float(cum_len[-1])
    n_points = max(int(round(total_len / max(spacing_px, 1.0))), 2)
    sample_dists = np.linspace(0.0, total_len, n_points)
    xs = np.interp(sample_dists, cum_len, polyline_xy[:, 0])
    ys = np.interp(sample_dists, cum_len, polyline_xy[:, 1])
    return np.column_stack([xs, ys])


def coords_xy_to_rowcol(coords_xy: np.ndarray) -> np.ndarray:
    """Convert sampled (x, y) points to tracker's (row, col)."""
    return np.column_stack([coords_xy[:, 1], coords_xy[:, 0]])
