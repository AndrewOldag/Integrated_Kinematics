"""Logistic model fitting helpers."""

from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import curve_fit


def veloc_spec(x: np.ndarray, a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    arg = np.clip(-c * (x - e), -500, 500)
    return a + (b - a) / (1.0 + np.exp(arg)) ** (1.0 / d)


def fit_logistic(x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, np.ndarray | float]:
    x = np.asarray(x_data, dtype=np.float64).ravel()
    y = np.asarray(y_data, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 5:
        raise ValueError("Not enough points for logistic fit.")

    c_grid = np.linspace(0.001, 0.01, 3)
    d_grid = np.linspace(0.5, 8, 3)
    y_sorted = np.sort(y)
    v_max = float(np.mean(y_sorted[-3:])) if y_sorted.size >= 3 else float(y.max())
    a_init = float(np.mean(y))
    e_init = float(np.mean(x))

    best_params = None
    best_mse = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ci in c_grid:
            for di in d_grid:
                p0 = [a_init, v_max, ci, di, e_init]
                try:
                    popt, _ = curve_fit(veloc_spec, x, y, p0=p0, maxfev=10000, method="lm")
                    if np.any(np.isnan(popt)) or np.any(np.isinf(popt)):
                        continue
                    residuals = y - veloc_spec(x, *popt)
                    mse = float(np.mean(residuals**2))
                    if mse < best_mse:
                        best_mse = mse
                        best_params = popt
                except Exception:
                    continue
    if best_params is None:
        raise RuntimeError("Logistic fit failed for all initial guesses.")
    return {"parameters": best_params, "mse": best_mse}


def eval_model(params: np.ndarray, domain: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    params = np.asarray(params, dtype=np.float64)
    x = np.asarray(domain, dtype=np.float64)
    velocity = veloc_spec(x, *params)
    dv = np.gradient(velocity)
    ds = np.gradient(x)
    ds[ds == 0] = np.finfo(float).eps
    strain = dv / ds
    return velocity, strain
