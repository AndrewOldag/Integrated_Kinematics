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


def load_sldy_stack(
    path: str | Path,
    acquisition_index: int = 0,
    channel_index: int = 0,
) -> np.ndarray:
    """Load a SlideBook .sldy file into a (T, H, W) float64 stack.

    Args:
        path: Path to the .sldy file.
        acquisition_index: Which acquisition (capture) to load (default 0).
        channel_index: Which fluorescence channel to load (default 0).

    Returns:
        Normalised float64 array of shape (T, H, W).
    """
    try:
        from sld import SlideBook
    except ImportError as exc:
        raise ImportError(
            "slidebook-python is required to read .sldy files: pip install slidebook-python"
        ) from exc

    path = Path(path)
    sb = SlideBook(path)
    if not sb.images:
        raise ValueError(f"No acquisitions found in {path}")
    if acquisition_index >= len(sb.images):
        raise IndexError(
            f"acquisition_index {acquisition_index} out of range "
            f"({len(sb.images)} acquisitions in file)"
        )

    acq = sb.images[acquisition_index]
    if not acq.channels:
        raise ValueError(f"No channels found in acquisition {acquisition_index}")
    if channel_index >= len(acq.channels):
        raise IndexError(
            f"channel_index {channel_index} out of range "
            f"({len(acq.channels)} channels in acquisition)"
        )

    ch_key = f"ch_{acq.channels[channel_index]}"
    frames_raw = acq.data[ch_key]  # list of np.ndarray, one per file/timepoint

    # Sort by natural filename order so timepoints are in sequence
    ch_paths = sorted(
        acq._directory.glob(f"ImageData_Ch{acq.channels[channel_index]}*.npy"),
        key=lambda p: natural_sort_key(p.name),
    )
    frames = [np.load(str(p)) for p in ch_paths]

    if not frames:
        raise ValueError(f"No image data found for channel {ch_key}")

    # Each frame may be (H, W) or (Z, H, W) — flatten Z by max projection if needed
    normalised = []
    for f in frames:
        arr = np.asarray(f)
        if arr.ndim == 3:
            arr = arr.max(axis=0)  # Z max projection → (H, W)
        normalised.append(arr)

    stack = np.stack(normalised, axis=0)  # (T, H, W)
    return _normalise_stack(stack)


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
