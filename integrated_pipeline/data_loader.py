"""Load integrated pipeline output folders for interface display."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class OutputProfile:
    dataset_id: str
    output_dir: Path
    initialization_mode: str
    l_domain: np.ndarray
    velocity: np.ndarray
    regr: np.ndarray
    raw_domain: np.ndarray
    raw_velocity: np.ndarray
    regr_peak: Optional[float] = None
    regr_peak_location: Optional[float] = None


def discover_output_profiles(output_root: str | Path) -> list[OutputProfile]:
    root = Path(output_root)
    if not root.exists():
        return []
    profiles: list[OutputProfile] = []
    for child in sorted([p for p in root.iterdir() if p.is_dir()]):
        fit_path = child / "fitData.csv"
        raw_path = child / "rawData.csv"
        if not fit_path.exists() or not raw_path.exists():
            continue
        profiles.append(load_output_profile(child))
    return profiles


def load_output_profile(output_dir: str | Path) -> OutputProfile:
    out = Path(output_dir)
    fit = np.loadtxt(out / "fitData.csv", delimiter=",")
    raw = np.loadtxt(out / "rawData.csv", delimiter=",")
    if fit.ndim == 1:
        fit = fit[np.newaxis, :]
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]

    meta = _load_json(out / "run_metadata.json") or {}
    init_mode = str(meta.get("initialization_mode", "unknown"))
    peak_val = _load_scalar(out / "REGR_peak_value.csv")
    peak_loc = _load_scalar(out / "REGR_peak_location.csv")

    return OutputProfile(
        dataset_id=out.name,
        output_dir=out,
        initialization_mode=init_mode,
        l_domain=fit[:, 0],
        velocity=fit[:, 1],
        regr=fit[:, 2] if fit.shape[1] > 2 else np.array([], dtype=float),
        raw_domain=raw[:, 0],
        raw_velocity=raw[:, 1] if raw.shape[1] > 1 else np.array([], dtype=float),
        regr_peak=peak_val,
        regr_peak_location=peak_loc,
    )


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_scalar(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    arr = np.loadtxt(path, delimiter=",")
    if np.ndim(arr) == 0:
        return float(arr)
    return float(np.ravel(arr)[0])
