"""Plotting helper for kinematics outputs."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_steady_state_profile(profile: Dict) -> Figure:
    l_domain = profile["l_domain"]
    velocity = profile["velocity_profile"]
    strain = profile["strain_profile"]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(profile["raw_domain"], profile["raw_velocity"], ".", color="steelblue", alpha=0.3, markersize=2)
    ax1.plot(l_domain, velocity, color="red", linewidth=1.5)
    ax1.set_xlabel("Along Root (px)")
    ax1.set_ylabel("Velocity (px / frame)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    growth_rate = float(np.mean(velocity[-10:])) if velocity.size >= 10 else float(np.max(velocity))
    vel_range = max(abs(growth_rate), abs(float(np.min(velocity))), 1e-9)
    ax1.set_ylim(-0.1 * vel_range, 2.0 * growth_rate if growth_rate != 0 else 1.0)
    ax1.set_xlim(0, l_domain[-1])

    ax2 = ax1.twinx()
    ax2.plot(l_domain, strain, color="black", linewidth=1.0)
    ax2.set_ylabel("Strain (REGR, 1 / frame)", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    max_strain = float(np.max(strain)) if float(np.max(strain)) > 0 else 1.0
    ax2.set_ylim(0, max_strain * 1.2)

    _overlay_zone(ax2, l_domain, profile, "EVANSZONE", "red", max_strain)
    _overlay_zone(ax2, l_domain, profile, "PERZONE", "cyan", max_strain)
    _overlay_zone(ax2, l_domain, profile, "ABSZONE", "gold", max_strain)

    fig.tight_layout()
    return fig


def _overlay_zone(ax, l_domain: np.ndarray, profile: Dict, key: str, color: str, max_strain: float) -> None:
    if key not in profile:
        return
    zone = profile[key]
    ax.fill_between(l_domain, 0, max_strain * zone, color=color, alpha=0.15)
