"""
Data loading and metadata inference for RootKinematicsViewer.

This module is deliberately written in a very explicit style to make it easy
for non-specialists to reason about how CSV files are interpreted.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .ki_models import LogisticParams, LOGISTIC_PARAM_NAMES
from .ki_scaling import (
    ScopeDetectionResult,
    ScopeType,
    UM_PER_PX_KINEMATIC,
    UM_PER_PX_PLANT,
    infer_scope_type,
    guess_x_is_microns,
    px_to_um,
)
from .ki_outlier_filter import OutlierSettings, filter_profile_arrays


# Column candidates, in priority order
X_COLUMNS = ["x_um", "x (um)", "x", "pos_um", "position_um", "x_px", "pos_px", "position_px"]
VELOCITY_COLUMNS = [
    "velocity",
    "vel",
    "v",
    "V",
    "velocity_um_min",
    "velocity_um_per_min",
    "velocity_px",
]

# Columns known to be in raw pixel/frame units (need px→µm and frame→min conversion)
_PIXEL_VELOCITY_NAMES = {"velocity_px"}
REGR_COLUMNS = [
    "regr",
    "REGR",
    "regr_1_min",
    "regr_per_min",
]


LOGISTIC_CANDIDATE_COLUMNS = [
    "a",
    "b",
    "c",
    "L",
    "k",
    "x0",
    "v0",
    "amplitude",
    "slope",
    "center",
    "baseline",
]


TIME_REGEXES = [
    re.compile(
        r"(?i)(?:t|time|hr|h|min|m)[ _-]*([0-9]+(?:\.[0-9]+)?)\s*(h|hr|hrs|m|min|mins)?"
    ),
    re.compile(r"(?i)([0-9]+(?:\.[0-9]+)?)\s*(h|hr|hrs|m|min|mins)"),
]

CONTROL_TOKENS = {"control", "ctrl", "mock", "untreated", "wt"}
PERTURBED_TOKENS = {"perturbed", "stress", "mannitol", "salt", "treated", "drug"}

# Structured filename convention: scopetype_condition_timepoint_replicate
# e.g.  k_d_1_0.csv   p_c_0_2.csv   k_m_0_0.csv
_SCOPE_CODES: Dict[str, ScopeType] = {
    "k": ScopeType.KINEMATIC,
    "kin": ScopeType.KINEMATIC,
    "kinematic": ScopeType.KINEMATIC,
    "p": ScopeType.PLANT,
    "plant": ScopeType.PLANT,
}

_CONDITION_CODES: Dict[str, str] = {
    "c": "Control",
    "ctrl": "Control",
    "control": "Control",
    "wt": "WT",
    "w": "WT",
    "m": "Mutant",
    "mut": "Mutant",
    "mutant": "Mutant",
    "d": "Drought",
    "drought": "Drought",
    "s": "Salt",
    "salt": "Salt",
    "t": "Treated",
    "treated": "Treated",
    "p": "Perturbed",
    "pert": "Perturbed",
    "perturbed": "Perturbed",
}


@dataclass
class FilenameInfo:
    """Parsed fields from the structured filename convention."""
    scope: Optional[ScopeType] = None
    condition: Optional[str] = None
    time_min: Optional[float] = None
    replicate: Optional[int] = None


def parse_structured_filename(name: str) -> Optional[FilenameInfo]:
    """
    Parse filenames of the form:  scopetype_condition_timepoint_replicate[_extra...].csv

    Examples:
        k_d_1_0.csv  → Kinematic, Drought, time=1 min, replicate 0
        p_c_0_2.csv  → Plant, Control, time=0 min, replicate 2
        k_m_0_0.csv  → Kinematic, Mutant, time=0 min, replicate 0

    Also handles prefixed names like  rawData_k_d_1_0.csv  by searching for
    the scope code after splitting on underscores.
    """
    base = os.path.splitext(os.path.basename(name))[0]
    parts = base.split("_")

    # Try to locate the scope code in the parts list
    scope_idx: Optional[int] = None
    for i, tok in enumerate(parts):
        if tok.lower() in _SCOPE_CODES:
            scope_idx = i
            break

    if scope_idx is None or scope_idx + 3 >= len(parts):
        return None  # doesn't match the convention

    scope_tok = parts[scope_idx].lower()
    cond_tok = parts[scope_idx + 1].lower()
    time_tok = parts[scope_idx + 2]
    repl_tok = parts[scope_idx + 3]

    scope = _SCOPE_CODES.get(scope_tok)
    condition = _CONDITION_CODES.get(cond_tok)

    try:
        time_val = float(time_tok)
    except ValueError:
        time_val = None

    try:
        repl_val = int(repl_tok)
    except ValueError:
        repl_val = None

    if scope is None and condition is None:
        return None  # not enough info to be useful

    return FilenameInfo(
        scope=scope,
        condition=condition,
        time_min=time_val,
        replicate=repl_val,
    )


def infer_time_from_name(name: str) -> Optional[float]:
    """
    Try to infer time (in minutes) from a filename.
    First tries structured convention, then falls back to regex.
    """
    info = parse_structured_filename(name)
    if info is not None and info.time_min is not None:
        return info.time_min

    base = os.path.basename(name)
    for regex in TIME_REGEXES:
        m = regex.search(base)
        if not m:
            continue
        value = float(m.group(1))
        unit = (m.group(2) or "").lower()
        if unit in ("h", "hr", "hrs") or "h" in unit:
            return value * 60.0
        # default to minutes
        return value
    return None


def infer_condition_from_name(name: str) -> Optional[str]:
    """
    Try to infer condition from a filename.
    First tries structured convention, then falls back to token matching.
    """
    info = parse_structured_filename(name)
    if info is not None and info.condition is not None:
        return info.condition

    tokens = re.split(r"[^A-Za-z0-9]+", name.lower())
    token_set = set(t for t in tokens if t)
    if token_set & CONTROL_TOKENS:
        return "Control"
    if token_set & PERTURBED_TOKENS:
        return "Perturbed"
    return None


def infer_scope_from_name(name: str) -> Optional[ScopeType]:
    """Try to infer scope type from structured filename convention."""
    info = parse_structured_filename(name)
    if info is not None and info.scope is not None:
        return info.scope
    return None


@dataclass
class Profile:
    path: Path
    condition: Optional[str] = None  # WT/Mutant or Control/Perturbed
    time_min: Optional[float] = None
    scope_type: Optional[ScopeType] = None
    scope_reason: str = ""

    x_um: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    velocity: Optional[np.ndarray] = None
    regr: Optional[np.ndarray] = None
    regr_x: Optional[np.ndarray] = None  # smooth x-grid for REGR (when computed via logistic fit)

    logistic_params: Optional[LogisticParams] = None
    logistic_fit_3p: Optional[np.ndarray] = None  # cached 3-param logistic fit [a, b, c]

    # Bookkeeping
    load_warnings: List[str] = field(default_factory=list)

    @property
    def filename(self) -> str:
        return self.path.name


def _select_first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _extract_logistic_params(df: pd.DataFrame) -> Optional[LogisticParams]:
    """
    Attempt to extract scalar logistic parameters from a CSV table.

    Strategy:
    - look for columns whose names match known candidate names
    - require that the column has effectively a single unique value
    - build a mapping for canonical keys v0, L, k, x0 if possible
    """
    mapping: Dict[str, float] = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower not in {c.lower() for c in LOGISTIC_CANDIDATE_COLUMNS}:
            continue

        series = df[col]
        uniques = pd.unique(series.dropna())
        if len(uniques) == 0:
            continue
        if len(uniques) == 1:
            value = float(uniques[0])
        else:
            # if it's almost constant, still allow it
            value = float(series.mean())

        # map heuristically to canonical keys
        if col_lower in ("v0", "baseline"):
            mapping.setdefault("v0", value)
        elif col_lower in ("l", "a", "amplitude"):
            mapping.setdefault("L", value)
        elif col_lower in ("k", "slope", "b"):
            mapping.setdefault("k", value)
        elif col_lower in ("x0", "center", "c"):
            mapping.setdefault("x0", value)

    try:
        if all(name in mapping for name in LOGISTIC_PARAM_NAMES):
            return LogisticParams.from_mapping(mapping)
    except Exception:
        return None
    return None


def load_profile(
    path: Path,
    global_scope_override: ScopeType = ScopeType.AUTOMATIC,
    outlier_settings: Optional[OutlierSettings] = None,
) -> Profile:
    df = pd.read_csv(path)

    # Heuristic: handle headerless CSV files where the first row of numeric
    # data was interpreted as a header. Column names end up as numbers
    # (e.g. "10.034", "0") so we never match "x_um" / "velocity". If all
    # column names look numeric, re-read with no header and assign default
    # names so the pipeline can find position and velocity/REGR.
    try:
        all_numeric_headers = True
        for col in df.columns:
            try:
                float(str(col).strip())
            except (ValueError, TypeError):
                all_numeric_headers = False
                break
        if all_numeric_headers and df.shape[1] >= 1:
            df = pd.read_csv(path, header=None)
            ncols = df.shape[1]
            new_cols: List[str] = []
            if ncols >= 1:
                new_cols.append("x_px")  # raw pixel distance, needs conversion
            if ncols >= 2:
                new_cols.append("velocity_px")  # raw px/frame, needs conversion
            for idx in range(2, ncols):
                new_cols.append(f"col_{idx}")
            df.columns = new_cols
    except Exception:
        pass

    # Scope detection — structured filename first, then heuristics
    if global_scope_override == ScopeType.AUTOMATIC:
        fname_scope = infer_scope_from_name(path.name)
        if fname_scope is not None:
            scope = fname_scope
            reason = f"filename convention: {scope.value}"
        else:
            inferred = infer_scope_type(path, df)
            scope = inferred.scope_type or ScopeType.KINEMATIC
            reason = inferred.reason or "defaulted to kinematic"
    else:
        scope = global_scope_override
        reason = f"user override: {global_scope_override.value}"

    profile = Profile(path=path, scope_type=scope, scope_reason=reason)

    # X coordinate
    x_col = _select_first_existing_column(df, X_COLUMNS)
    if x_col is None:
        profile.load_warnings.append("No x/position column found.")
        profile.x_um = np.array([], dtype=float)
    else:
        x_values = df[x_col].to_numpy(dtype=float)
        if guess_x_is_microns(x_col, x_values):
            profile.x_um = x_values
        else:
            profile.x_um = px_to_um(x_values, scope)

    # Velocity
    vel_col = _select_first_existing_column(df, VELOCITY_COLUMNS)
    if vel_col is not None:
        v_raw = df[vel_col].to_numpy(dtype=float)
        if vel_col.lower() in _PIXEL_VELOCITY_NAMES:
            # Raw data is in pixels/frame — convert to µm/min
            # Infer frame interval from filename (same logic as backend)
            fname_lower = path.name.lower()
            frame_interval_sec = 60.0 if "subset" in fname_lower else 30.0
            um_per_px = (
                UM_PER_PX_PLANT if scope == ScopeType.PLANT else UM_PER_PX_KINEMATIC
            )
            v_raw = v_raw * um_per_px * (60.0 / frame_interval_sec)
        profile.velocity = v_raw

    # --- Outlier filtering (applied to velocity before fitting) ---
    if outlier_settings is None:
        outlier_settings = OutlierSettings(enabled=False)

    if (
        outlier_settings.enabled
        and profile.velocity is not None
        and profile.x_um.size >= 2
    ):
        x_clean, v_clean, n_removed = filter_profile_arrays(
            profile.x_um, profile.velocity, outlier_settings,
        )
        if n_removed > 0:
            profile.load_warnings.append(
                f"Outlier filter ({outlier_settings.method.upper()}): "
                f"removed {n_removed}/{profile.x_um.size} points."
            )
            profile.x_um = x_clean
            profile.velocity = v_clean

    # REGR
    regr_col = _select_first_existing_column(df, REGR_COLUMNS)
    if regr_col is not None:
        profile.regr = df[regr_col].to_numpy(dtype=float)

    # Fallback: compute REGR via logistic fit + gradient (matches notebook pipeline)
    if profile.regr is None and profile.velocity is not None and profile.x_um.size >= 2:
        try:
            from scipy.optimize import curve_fit

            def _logistic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
                return a / (1.0 + np.exp(-b * (x - c)))

            v = np.asarray(profile.velocity, dtype=float)
            x = np.asarray(profile.x_um, dtype=float)
            n = min(v.size, x.size)
            v = v[:n]
            x = x[:n]
            mask = np.isfinite(x) & np.isfinite(v)
            x_clean = x[mask]
            v_clean = v[mask]
            if x_clean.size >= 10:
                # Downsample for faster fitting (2000 pts max)
                if x_clean.size > 2000:
                    step = x_clean.size // 2000
                    x_ds = x_clean[::step]
                    v_ds = v_clean[::step]
                else:
                    x_ds = x_clean
                    v_ds = v_clean
                p0 = [np.max(np.abs(v_ds)), 0.002, np.median(x_ds)]
                bounds = ([0, -np.inf, 0], [np.inf, np.inf, np.inf])
                popt, _ = curve_fit(_logistic, x_ds, v_ds, p0=p0, bounds=bounds, maxfev=5000)
                profile.logistic_fit_3p = np.asarray(popt)
                L_eval = np.linspace(float(np.min(x_clean)), float(np.max(x_clean)), 500)
                V_fit = _logistic(L_eval, *popt)
                profile.regr = np.gradient(V_fit, L_eval)
                profile.regr_x = L_eval
        except Exception:
            # If logistic fit fails, fall back to raw gradient
            v = np.asarray(profile.velocity, dtype=float)
            x = np.asarray(profile.x_um, dtype=float)
            n = min(v.size, x.size)
            if n >= 2:
                profile.regr = np.gradient(v[:n], x[:n])

    # Logistic parameters (if any)
    profile.logistic_params = _extract_logistic_params(df)

    return profile


def discover_csv_files(root: Path) -> List[Path]:
    paths: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".csv"):
                paths.append(Path(dirpath) / name)
    return sorted(paths)


def _auto_infer_from_filename(p: Profile) -> None:
    """Apply structured filename convention, then fall back to heuristics."""
    info = parse_structured_filename(p.filename)
    if info is not None:
        if info.condition is not None and p.condition is None:
            p.condition = info.condition
        if info.time_min is not None and p.time_min is None:
            p.time_min = info.time_min
        # scope is already handled during load_profile
        return

    # Legacy fallback
    t = infer_time_from_name(p.filename)
    if t is not None and p.time_min is None:
        p.time_min = t
    cond = infer_condition_from_name(p.filename)
    if cond is not None and p.condition is None:
        p.condition = cond


def load_genetic_mode(
    wt_folder: Path,
    mutant_folder: Optional[Path],
    scope_override: ScopeType = ScopeType.AUTOMATIC,
    outlier_settings: Optional[OutlierSettings] = None,
) -> List[Profile]:
    profiles: List[Profile] = []

    # WT
    for path in discover_csv_files(wt_folder):
        p = load_profile(path, global_scope_override=scope_override,
                         outlier_settings=outlier_settings)
        p.condition = "WT"
        _auto_infer_from_filename(p)  # may refine condition from naming convention
        profiles.append(p)

    # Mutant (optional)
    if mutant_folder is not None:
        for path in discover_csv_files(mutant_folder):
            p = load_profile(path, global_scope_override=scope_override,
                             outlier_settings=outlier_settings)
            p.condition = "Mutant"
            _auto_infer_from_filename(p)
            profiles.append(p)

    return profiles


def load_environmental_mode(
    folder: Path,
    scope_override: ScopeType = ScopeType.AUTOMATIC,
    auto_infer: bool = True,
    outlier_settings: Optional[OutlierSettings] = None,
) -> List[Profile]:
    profiles: List[Profile] = []
    for path in discover_csv_files(folder):
        p = load_profile(path, global_scope_override=scope_override,
                         outlier_settings=outlier_settings)
        if auto_infer:
            _auto_infer_from_filename(p)
        profiles.append(p)
    return profiles


def group_key_genetic(profile: Profile) -> Tuple[str]:
    return (profile.condition or "Unknown",)


def group_key_environmental(profile: Profile) -> Tuple[str, str]:
    cond = profile.condition or "Unknown"
    time_label = "Unknown"
    if profile.time_min is not None:
        time_label = f"{profile.time_min:g} min"
    return cond, time_label
