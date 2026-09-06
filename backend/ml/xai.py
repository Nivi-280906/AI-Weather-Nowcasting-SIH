"""
Lightweight, dependency-free explainability: occlusion / perturbation-based
feature attribution. For each physical variable (IWV, CAPE, CIN, shear,
convergence, CTT drop, QPE) we zero-out that channel across the input
sequence and measure how much each task's probability drops. Larger drop
= that variable was a bigger driver of the alert -> shown to disaster
managers as "why the model raised this alert."

This is cheaper than Integrated Gradients / SHAP but needs no extra
dependency and runs in milliseconds, which matters for a real-time
dashboard.
"""
import torch
import numpy as np
from ml.data_simulator import FEATURES


@torch.no_grad()
def attribute(model, x: torch.Tensor):
    """
    x: [1, T, C, H, W] single sample
    returns: dict feature_name -> {"thunderstorm":.., "cloudburst":.., "flash_flood":..}
             (positive = removing this feature dropped the probability,
              i.e. this feature was pushing the risk UP)
    """
    model.eval()
    base_t, base_c, base_f, _ = model(x)
    base_t, base_c, base_f = base_t.item(), base_c.item(), base_f.item()

    attributions = {}
    for i, name in enumerate(FEATURES):
        x_occ = x.clone()
        # replace this feature channel with its own mean (removes signal, keeps scale)
        mean_val = x_occ[:, :, i, :, :].mean()
        x_occ[:, :, i, :, :] = mean_val

        t, c, f, _ = model(x_occ)
        attributions[name] = {
            "thunderstorm": round(float(base_t - t.item()), 4),
            "cloudburst": round(float(base_c - c.item()), 4),
            "flash_flood": round(float(base_f - f.item()), 4),
        }
    return attributions


def top_drivers(attributions: dict, task: str, k: int = 3):
    """Return the top-k feature names driving a given task's probability up."""
    scored = sorted(attributions.items(), key=lambda kv: kv[1][task], reverse=True)
    return [{"feature": name, "contribution": vals[task]} for name, vals in scored[:k]]
