"""
Real QPE (Quantitative Precipitation Estimation) ingestion.

The problem statement names QPE as its own data source: "Satellite-derived
precipitation estimates ... a reliable, openly accessible alternative to
ground-based radar." The free, no-request-queue source is NASA GPM IMERG
(HDF5/NetCDF, half-hourly), available via NASA Earthdata / GES DISC with
a free account.

IMERG's `precipitationCal` variable is rainfall rate in mm/hr, which is
exactly the `qpe` feature already used in the model -- no derivation
needed here, just regridding onto each station's patch and resampling to
match the IMDAA/INSAT time cadence (accumulate/average half-hourly IMERG
into each 3-hourly step).
"""
from __future__ import annotations

import numpy as np
import xarray as xr

QPE_VAR_CANDIDATES = ["precipitationCal", "precipitation", "precip"]


def _find_var(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


def load_imerg(filepaths: list[str]) -> xr.Dataset:
    datasets = [xr.open_dataset(fp) for fp in filepaths]
    merged = xr.merge(datasets, compat="override", join="outer")

    var_name = _find_var(merged, QPE_VAR_CANDIDATES)
    if var_name:
        merged = merged.rename({var_name: "qpe"})

    coord_rename = {}
    for cand in ("lat",):
        if cand in merged.coords:
            coord_rename[cand] = "latitude"
    for cand in ("lon",):
        if cand in merged.coords:
            coord_rename[cand] = "longitude"
    return merged.rename(coord_rename)


def extract_patch(ds: xr.Dataset, lat: float, lon: float, grid: int, box_deg: float = 1.5) -> xr.Dataset:
    lat_target = np.linspace(lat - box_deg, lat + box_deg, grid)
    lon_target = np.linspace(lon - box_deg, lon + box_deg, grid)
    return ds.interp(latitude=lat_target, longitude=lon_target, method="linear")


def resample_to_3hourly(qpe_halfhourly: np.ndarray) -> np.ndarray:
    """qpe_halfhourly: [T_half, H, W] with T_half a multiple of 6
    (six half-hour IMERG steps per 3-hour IMDAA/INSAT step).
    Returns [T_half // 6, H, W] mean rainfall rate per 3-hour step."""
    n_half, h, w = qpe_halfhourly.shape
    n_3h = n_half // 6
    trimmed = qpe_halfhourly[: n_3h * 6]
    return trimmed.reshape(n_3h, 6, h, w).mean(axis=1).astype(np.float32)
