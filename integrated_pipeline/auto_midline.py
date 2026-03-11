"""Automatic first-frame midline extraction with fallback logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class AutoMidlineResult:
    midline_coords_xy: list[tuple[float, float]]
    qc_point_xy: tuple[float, float]
    overlay_image: np.ndarray
    method: str
    confidence: float
    notes: str = ""


def extract_auto_midline(
    first_frame: np.ndarray,
    checkpoint_path: Optional[str] = None,
) -> AutoMidlineResult:
    """Extract ordered (x, y) midline coordinates from a first frame.

    If deep-learning dependencies/checkpoint are unavailable, falls back to a
    robust classical extraction based on root thresholding and per-column medians.
    """
    if checkpoint_path and Path(checkpoint_path).exists():
        deep = _try_deep_model_path(first_frame, checkpoint_path)
        if deep is not None:
            return deep

    return _classical_midline(first_frame)


def _classical_midline(first_frame: np.ndarray) -> AutoMidlineResult:
    gray_u8 = _to_uint8(first_frame)
    bbox = _find_root_bbox(gray_u8)
    bx, by, bw, bh = bbox
    crop = gray_u8[by : by + bh, bx : bx + bw]

    blur = cv2.GaussianBlur(crop, (7, 7), 0)
    to_thresh = (255 - blur) if _needs_inversion(blur) else blur
    _, thresh = cv2.threshold(to_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    ys, xs = np.where(clean > 0)
    coords: list[tuple[float, float]] = []
    if xs.size > 0:
        x_min = int(xs.min())
        x_max = int(xs.max())
        for x_local in range(x_min, x_max + 1):
            y_hits = ys[xs == x_local]
            if y_hits.size == 0:
                continue
            y_local = float(np.median(y_hits))
            coords.append((float(x_local + bx), float(y_local + by)))

    if not coords:
        # Last-resort fallback: straight line through image center.
        h, w = gray_u8.shape
        xs_lin = np.linspace(0, w - 1, 50)
        ys_lin = np.full_like(xs_lin, h / 2.0)
        coords = [(float(x), float(y)) for x, y in zip(xs_lin, ys_lin)]
        confidence = 0.0
        method = "fallback_centerline"
    else:
        confidence = min(1.0, len(coords) / 200.0)
        method = "classical_threshold_midline"

    qc_point = max(coords, key=lambda pt: pt[0])
    overlay = _build_overlay(gray_u8, coords, qc_point)
    return AutoMidlineResult(
        midline_coords_xy=coords,
        qc_point_xy=qc_point,
        overlay_image=overlay,
        method=method,
        confidence=confidence,
    )


def _try_deep_model_path(first_frame: np.ndarray, checkpoint_path: str) -> Optional[AutoMidlineResult]:
    """Best-effort adapter for DL path; returns None if unavailable."""
    try:
        import importlib.util
        import tempfile
        from pathlib import Path as _Path

        import torch

        # Load legacy predictor module directly from file path, without editing it.
        legacy_root = _Path("D:/nolan_lab/root_midline_extraction")
        predict_py = legacy_root / "predict.py"
        if not predict_py.exists():
            return None

        spec = importlib.util.spec_from_file_location("legacy_predict", str(predict_py))
        if spec is None or spec.loader is None:
            return None
        legacy_predict = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy_predict)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = legacy_predict.build_model(device)
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        image_u8 = _to_uint8(first_frame)
        result = legacy_predict.predict_single(model, image_u8, device)
        coords = [(float(x), float(y)) for x, y in result["midline_coords"]]
        if not coords:
            return None
        qc_point = (float(result["qc_point"][0]), float(result["qc_point"][1]))
        overlay = _build_overlay(image_u8, coords, qc_point)
        confidence = min(1.0, len(coords) / 200.0)
        return AutoMidlineResult(
            midline_coords_xy=coords,
            qc_point_xy=qc_point,
            overlay_image=overlay,
            method="deep_model_predict_single",
            confidence=confidence,
            notes="Loaded from legacy checkpoint path.",
        )
    except Exception:
        return None


def _build_overlay(
    image_u8: np.ndarray, coords: list[tuple[float, float]], qc_point: tuple[float, float]
) -> np.ndarray:
    overlay = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
    if len(coords) > 1:
        poly = np.array([[int(x), int(y)] for x, y in coords], dtype=np.int32)
        cv2.polylines(overlay, [poly], isClosed=False, color=(0, 255, 0), thickness=2)
    qx, qy = int(qc_point[0]), int(qc_point[1])
    cv2.drawMarker(overlay, (qx, qy), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
    return overlay


def _to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = arr.astype(np.float64)
    lo, hi = arr.min(), arr.max()
    if hi <= 1.0:
        arr = arr * 255.0
    elif hi > 255.0:
        # 16-bit or other wide-range data: stretch to full [0, 255]
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else arr * 0.0
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def _needs_inversion(image_u8: np.ndarray) -> bool:
    _, otsu_mask = cv2.threshold(image_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_mean = float(np.mean(image_u8[otsu_mask == 255])) if np.any(otsu_mask == 255) else 128.0
    bg_mean = float(np.mean(image_u8[otsu_mask == 0])) if np.any(otsu_mask == 0) else 128.0
    return bool(fg_mean < bg_mean)


def _find_root_bbox(image_u8: np.ndarray) -> tuple[int, int, int, int]:
    h_img, w_img = image_u8.shape[:2]
    blurred = cv2.GaussianBlur(image_u8, (51, 51), 0)
    to_thresh = (255 - blurred) if _needs_inversion(blurred) else blurred
    _, binary = cv2.threshold(to_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_k)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if n_labels <= 1:
        return (0, 0, w_img, h_img)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas) + 1)
    x = int(stats[largest, cv2.CC_STAT_LEFT])
    y = int(stats[largest, cv2.CC_STAT_TOP])
    w = int(stats[largest, cv2.CC_STAT_WIDTH])
    h = int(stats[largest, cv2.CC_STAT_HEIGHT])
    pad_x = max(int(w * 0.15), 100)
    pad_y = max(int(h * 0.15), 100)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(w_img - x, w + 2 * pad_x)
    h = min(h_img - y, h + 2 * pad_y)
    return (x, y, w, h)
