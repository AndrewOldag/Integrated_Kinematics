"""Image loading utilities for TIFF stacks and TIFF folders."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import numpy as np


def natural_sort_key(value: str) -> list[object]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", value)]


def load_image_stack(path: str | Path) -> np.ndarray:
    import tifffile

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    raw = tifffile.imread(str(path))
    return _normalise_stack(raw)


def load_image_folder(folder: str | Path) -> np.ndarray:
    import tifffile

    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    tiffs = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff")],
        key=lambda p: natural_sort_key(p.name),
    )
    if not tiffs:
        raise FileNotFoundError(f"No TIFF files found in {folder}")

    frames: list[np.ndarray] = []
    for tif_path in tiffs:
        raw = tifffile.imread(str(tif_path))
        gray = _to_gray_2d(raw)
        frames.append(gray)

    stacked = np.stack(frames, axis=0).astype(np.float64)
    if stacked.max() > 1.0:
        stacked = stacked / _dtype_max_from_values(stacked)
    return stacked


def load_from_array(path_or_arr: Union[str, Path, np.ndarray]) -> np.ndarray:
    if isinstance(path_or_arr, (str, Path)):
        arr = np.load(str(path_or_arr))
    else:
        arr = np.asarray(path_or_arr)
    return _normalise_stack(arr)


def _normalise_stack(raw: np.ndarray) -> np.ndarray:
    if raw.ndim == 2:
        raw = raw[np.newaxis, ...]
    elif raw.ndim == 3:
        if raw.shape[2] in (3, 4) and raw.shape[0] > 4 and raw.shape[1] > 4:
            raw = _to_gray_2d(raw)[np.newaxis, ...]
    elif raw.ndim == 4:
        frames = [_to_gray_2d(raw[t]) for t in range(raw.shape[0])]
        raw = np.stack(frames, axis=0)

    if raw.ndim != 3:
        raise ValueError(f"Unsupported image shape: {raw.shape}")

    arr = raw.astype(np.float64)
    if arr.max() > 1.0:
        arr = arr / _dtype_max_from_values(arr)
    return arr


def _to_gray_2d(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] in (3, 4):
        r = frame[:, :, 0].astype(np.float64)
        g = frame[:, :, 1].astype(np.float64)
        b = frame[:, :, 2].astype(np.float64)
        return 0.2125 * r + 0.7154 * g + 0.0721 * b
    if frame.ndim == 3:
        return frame[:, :, 0].astype(np.float64)
    raise ValueError(f"Unexpected frame shape: {frame.shape}")


def _dtype_max_from_values(arr: np.ndarray) -> float:
    if np.issubdtype(arr.dtype, np.integer):
        return float(np.iinfo(arr.dtype).max)
    mx = float(arr.max())
    if mx <= 1.0:
        return 1.0
    if mx <= 255.0:
        return 255.0
    if mx <= 65535.0:
        return 65535.0
    return mx
