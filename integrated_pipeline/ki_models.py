"""
Logistic models for velocity and REGR used by RootKinematicsViewer.

Default assumptions:
- x in microns (µm)
- velocity in µm / min
- REGR in 1 / min

Parameterisation (default, editable by developers later):
    v(x) = v0 + L / (1 + exp(-k * (x - x0)))
    regr(x) = dv/dx = (L * k * exp(-k * (x - x0))) / (1 + exp(-k * (x - x0)))**2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

import numpy as np


LOGISTIC_PARAM_NAMES = ("v0", "L", "k", "x0")


@dataclass
class LogisticParams:
    v0: float
    L: float
    k: float
    x0: float

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float]) -> "LogisticParams":
        """
        Build LogisticParams from an arbitrary mapping of names to values,
        expecting the canonical keys v0, L, k, x0 to be present.
        """
        missing = [name for name in LOGISTIC_PARAM_NAMES if name not in mapping]
        if missing:
            raise ValueError(f"Missing logistic parameters: {', '.join(missing)}")
        return cls(
            v0=float(mapping["v0"]),
            L=float(mapping["L"]),
            k=float(mapping["k"]),
            x0=float(mapping["x0"]),
        )

    def as_dict(self) -> Dict[str, float]:
        return {"v0": self.v0, "L": self.L, "k": self.k, "x0": self.x0}


def logistic_velocity(x: np.ndarray, params: LogisticParams) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    v0, L, k, x0 = params.v0, params.L, params.k, params.x0
    return v0 + L / (1.0 + np.exp(-k * (x - x0)))


def logistic_regr(x: np.ndarray, params: LogisticParams) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    L, k, x0 = params.L, params.k, params.x0
    exp_term = np.exp(-k * (x - x0))
    denom = (1.0 + exp_term) ** 2
    return (L * k * exp_term) / denom


def aggregate_params(
    params_list: Iterable[LogisticParams],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Given a sequence of LogisticParams, compute per-parameter mean and std.

    Returns:
        {
          "mean": np.array([...]),  # in order (v0, L, k, x0)
          "std":  np.array([...]),
        }
    """
    arr = np.array([[p.v0, p.L, p.k, p.x0] for p in params_list], dtype=float)
    if arr.size == 0:
        raise ValueError("No logistic parameter sets provided.")
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=0)
    return {"mean": mean, "std": std}


def params_from_array(values: Iterable[float]) -> LogisticParams:
    vals = list(values)
    if len(vals) != 4:
        raise ValueError("Expected 4 values for logistic parameters (v0, L, k, x0).")
    return LogisticParams(v0=vals[0], L=vals[1], k=vals[2], x0=vals[3])
