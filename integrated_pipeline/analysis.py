"""Steady-state kinematics analysis and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np

from .fitting import eval_model, fit_logistic
from .image_io import load_from_array, load_image_folder, load_image_stack
from .tracking import track_points


def steady_state_analysis(
    point_list: np.ndarray,
    fit_domain_max: float = 2000.0,
    fit_domain_n: int = 2000,
    time_interval: float = 1.0,
    time_unit: str = "frame",
) -> Dict:
    pts = np.transpose(point_list, (0, 2, 1)).copy()
    pts = pts - pts[0:1, :, :]
    d_x = np.diff(pts, n=1, axis=0)
    d_v = np.diff(pts, n=1, axis=1)
    d_l = np.sqrt(np.sum(d_x**2, axis=2))
    l_vals = np.cumsum(d_l, axis=0)
    l_vals = l_vals[:, :-1]
    d_l_safe = d_l.copy()
    d_l_safe[d_l_safe == 0] = np.finfo(float).eps
    nd_x = d_x / d_l_safe[:, :, np.newaxis]
    d_v_tm = np.sum(nd_x[:, :-1, :] * d_v[:-1, :, :], axis=2)

    raw_l = l_vals.ravel(order="F")
    raw_v = d_v_tm.ravel(order="F")
    fit_result = fit_logistic(raw_l, raw_v)
    l_domain = np.linspace(0, fit_domain_max, fit_domain_n)
    velocity_profile, strain_profile = eval_model(fit_result["parameters"], l_domain)

    return {
        "velocity_profile": velocity_profile,
        "strain_profile": strain_profile,
        "l_domain": l_domain,
        "model_params": fit_result["parameters"],
        "model_mse": fit_result["mse"],
        "raw_domain": raw_l,
        "raw_velocity": raw_v,
        "time_interval": time_interval,
        "time_unit": time_unit,
        "type": "steadyState",
    }


def compute_growth_zone_metrics(
    profile: Dict, rel_cutoff: float = 0.1, per_cutoff: float = 0.5, abs_cutoff: float = 0.001
) -> Dict:
    strain = profile["strain_profile"]
    l_domain = profile["l_domain"]
    ds = np.gradient(l_domain)
    peak = np.max(strain)
    evans_zone = (strain > (peak - rel_cutoff * peak)).astype(float)
    per_zone = np.zeros_like(strain)
    sorted_indices = np.argsort(strain)[::-1]
    sorted_strain = strain[sorted_indices]
    cum = np.cumsum(sorted_strain)
    if cum[-1] != 0:
        cum = cum / cum[-1]
    per_zone[sorted_indices[cum < per_cutoff]] = 1.0
    abs_zone = (strain > abs_cutoff).astype(float)

    profile["EVANSZONE"] = evans_zone
    profile["EVANSZONE_WIDTH"] = float(np.sum(ds * evans_zone))
    profile["PERZONE"] = per_zone
    profile["PERZONE_WIDTH"] = float(np.sum(ds * per_zone))
    profile["ABSZONE"] = abs_zone
    profile["ABSZONE_WIDTH"] = float(np.sum(ds * abs_zone))
    max_idx = int(np.argmax(strain))
    profile["maxREGR"] = float(strain[max_idx])
    profile["maxLocation"] = float(l_domain[max_idx])
    return profile


def run_steady_state_kinematics(
    images: Union[np.ndarray, str, Path],
    initial_points: np.ndarray,
    disk_radius: int = 28,
    threshold: float = 10.0,
    rel_cutoff: float = 0.1,
    per_cutoff: float = 0.5,
    abs_cutoff: float = 0.001,
    fit_domain_max: float = 2000.0,
    fit_domain_n: int = 2000,
    time_interval: float = 1.0,
    time_unit: str = "frame",
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict:
    if isinstance(images, (str, Path)):
        path = Path(images)
        if path.is_dir():
            img_stack = load_image_folder(path)
        elif path.suffix.lower() == ".npy":
            img_stack = load_from_array(path)
        else:
            img_stack = load_image_stack(path)
    else:
        img_stack = load_from_array(images)

    def _tracking_cb(frame: int, total: int) -> None:
        if progress_callback:
            progress_callback("tracking", frame, total)

    point_list = track_points(
        img_stack,
        initial_points,
        disk_radius=disk_radius,
        threshold=threshold,
        progress_callback=_tracking_cb,
    )

    if progress_callback:
        progress_callback("analysis", 0, 1)
    profile = steady_state_analysis(
        point_list,
        fit_domain_max=fit_domain_max,
        fit_domain_n=fit_domain_n,
        time_interval=time_interval,
        time_unit=time_unit,
    )
    if progress_callback:
        progress_callback("metrics", 0, 1)
    profile = compute_growth_zone_metrics(
        profile, rel_cutoff=rel_cutoff, per_cutoff=per_cutoff, abs_cutoff=abs_cutoff
    )
    profile["point_list"] = point_list
    if progress_callback:
        progress_callback("done", 1, 1)
    return profile


def save_results(profile: Dict, output_dir: Union[str, Path]) -> None:
    from .plotting import plot_steady_state_profile

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fmt = "%.5g"
    np.savetxt(
        out / "rawData.csv",
        np.column_stack([profile["raw_domain"], profile["raw_velocity"]]),
        delimiter=",",
        fmt=fmt,
    )
    np.savetxt(
        out / "fitData.csv",
        np.column_stack([profile["l_domain"], profile["velocity_profile"], profile["strain_profile"]]),
        delimiter=",",
        fmt=fmt,
    )
    _save_scalar(out / "REGR_peak_location.csv", profile.get("maxLocation"))
    _save_scalar(out / "REGR_peak_value.csv", profile.get("maxREGR"))
    _save_scalar(out / "EVANS_zone_width.csv", profile.get("EVANSZONE_WIDTH"))
    _save_scalar(out / "PERCENT_zone_width.csv", profile.get("PERZONE_WIDTH"))
    _save_scalar(out / "ABS_zone_width.csv", profile.get("ABSZONE_WIDTH"))

    for key, filename in [
        ("EVANSZONE", "EVANS_zone.csv"),
        ("PERZONE", "PERCENT_zone.csv"),
        ("ABSZONE", "ABS_zone.csv"),
    ]:
        if key in profile:
            np.savetxt(out / filename, profile[key], delimiter=",", fmt=fmt)

    if "point_list" in profile:
        pl = profile["point_list"]
        t_count = pl.shape[2]
        flat = pl.transpose(2, 0, 1).reshape(t_count, -1)
        np.savetxt(out / "xyCoord.csv", flat, delimiter=",", fmt=fmt)

    with open(out / "time_interval.csv", "w", encoding="utf-8") as fh:
        fh.write("time_interval,time_unit\n")
        fh.write(f"{profile.get('time_interval', 1.0)},{profile.get('time_unit', 'frame')}\n")

    fig = plot_steady_state_profile(profile)
    fig.savefig(out / "REGR.png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _save_scalar(path: Path, value: float | None) -> None:
    if value is not None:
        np.savetxt(path, [value], delimiter=",", fmt="%.5g")
