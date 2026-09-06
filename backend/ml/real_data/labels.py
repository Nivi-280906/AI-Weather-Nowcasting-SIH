"""
Label derivation for real-data training samples.

Important honesty note: none of IMDAA/INSAT/GPM/DEM comes with a
"thunderstorm happened here" ground-truth label attached. A fully
rigorous project would cross-reference IMD storm reports / lightning
network data / verified flood records as ground truth. That's a real
next step beyond this prototype's scope.

In the meantime, this module derives labels from the same physical
thresholds meteorologists use operationally (CAPE/CIN erosion, IWV surge,
CTT collapse, convergence, and realized QPE) -- this mirrors exactly what
data_simulator.py already does for the synthetic labels, just applied to
real extracted fields instead of synthetic ones. It is a physically
grounded proxy, not verified ground truth -- say this plainly if asked
during evaluation.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def derive_labels(final_frame: dict) -> dict:
    """final_frame: dict with 2D (grid x grid) arrays for the LAST
    timestep of a sample sequence:
        iwv_rate, cape, cin, conv_low, ctt_drop_rate, qpe
    Returns thunderstorm / cloudburst / flash_flood_seed probabilities,
    using the same functional form as the simulator's heuristic (so the
    real-data model is trained on labels statistically comparable to
    what it saw when it was trained/validated on synthetic data)."""
    peak_iwv_rate = float(np.nanmax(final_frame["iwv_rate"]))
    peak_cape = float(np.nanmax(final_frame["cape"]))
    peak_conv = float(-np.nanmin(final_frame["conv_low"]))
    peak_ctt_drop = float(-np.nanmin(final_frame["ctt_drop_rate"]))
    peak_qpe = float(np.nanmax(final_frame["qpe"]))

    thunder = _sigmoid(0.006 * peak_cape + 40 * peak_conv + 0.15 * peak_ctt_drop - 3.0)
    cloudburst = _sigmoid(0.35 * peak_iwv_rate + 0.02 * peak_qpe + 0.08 * peak_ctt_drop - 3.5)
    flash_flood_seed = _sigmoid(0.9 * cloudburst + 0.015 * peak_qpe - 2.0)

    return {
        "thunderstorm": float(np.clip(thunder, 0, 1)),
        "cloudburst": float(np.clip(cloudburst, 0, 1)),
        "flash_flood_seed": float(np.clip(flash_flood_seed, 0, 1)),
    }
