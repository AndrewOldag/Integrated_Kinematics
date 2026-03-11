"""
Scaling utilities and scope-type handling for RootKinematicsViewer.

All logic here is intentionally simple and explicit so biologists can
adjust pixel-to-micron scaling without touching the rest of the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class ScopeType(str, Enum):
    AUTOMATIC = "Auto"
    KINEMATIC = "Kinematic"
    PLANT = "Plant"


# Default values match the MATLAB/notebook pipeline and can be overridden in the UI.
# Units: micron per pixel (µm / px)
UM_PER_PX_KINEMATIC: float = 0.325
UM_PER_PX_PLANT: float = 2.105


@dataclass
class ScopeDetectionResult:
    scope_type: Optional[ScopeType]
    reason: str


KINEMATIC_TOKENS = {"kinematic", "kine", "root_scope"}
PLANT_TOKENS = {"plant", "wholeplant", "macro"}


def _tokenize_path(path: Path) -> set[str]:
    tokens: set[str] = set()
    for part in path.parts:
        base = part.lower()
        for t in base.replace("-", "_").replace(".", "_").split("_"):
            if t:
                tokens.add(t)
    return tokens


def infer_scope_type(path: Path, df: Optional[pd.DataFrame] = None) -> ScopeDetectionResult:
    """
    Best-effort detection of whether a CSV comes from kinematic or plant scope.

    Heuristics:
    - explicit "scope" column
    - tokens in filename or parent folders
    """
    # 1) Explicit "scope" column
    if df is not None:
        for col in df.columns:
            if col.lower() == "scope":
                val = str(df[col].iloc[0]).strip().lower()
                if "kin" in val:
                    return ScopeDetectionResult(ScopeType.KINEMATIC, 'scope column == "kinematic"')
                if "plant" in val:
                    return ScopeDetectionResult(ScopeType.PLANT, 'scope column == "plant"')

    # 2) Path tokens
    tokens = _tokenize_path(path)
    if tokens & KINEMATIC_TOKENS:
        return ScopeDetectionResult(ScopeType.KINEMATIC, "filename/folder tokens suggest kinematic")
    if tokens & PLANT_TOKENS:
        return ScopeDetectionResult(ScopeType.PLANT, "filename/folder tokens suggest plant")

    return ScopeDetectionResult(None, "no clear scope indicator")


def guess_x_is_microns(column_name: str, values: np.ndarray) -> bool:
    """
    Decide whether x is already in microns based on column naming and magnitude.
    """
    name_lower = column_name.lower()
    if "um" in name_lower or "micron" in name_lower:
        return True

    if values.size == 0:
        return False

    vmax = float(np.nanmax(values))
    # Very rough heuristic: pixels are rarely > 5000, microns can be.
    if vmax > 5000:
        return True

    return False


def px_to_um(values_px: np.ndarray, scope_type: ScopeType) -> np.ndarray:
    """
    Convert pixel values to microns using the configured scaling constants.
    """
    if scope_type == ScopeType.KINEMATIC:
        factor = UM_PER_PX_KINEMATIC
    elif scope_type == ScopeType.PLANT:
        factor = UM_PER_PX_PLANT
    else:
        # Fallback if somehow called with automatic.
        factor = UM_PER_PX_KINEMATIC
    return values_px.astype(float) * float(factor)
