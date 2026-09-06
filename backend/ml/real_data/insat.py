"""
Real INSAT-3D/3DR ingestion (via MOSDAC).

INSAT-3DR Imager gives brightness temperature for the Water Vapour (WV,
6.5-7.0 micron) and Thermal Infrared (TIR-1, 10.2-11.2 micron) channels.

  - TIR-1 brightness temperature IS, to a first approximation, the cloud
    top temperature (CTT) -- this is the standard operational usage, not
    an approximation we're introducing.
  - True Integrated Water Vapour (IWV) requires a physical retrieval
    algorithm (radiative transfer inversion) that needs auxiliary
    atmospheric-profile input beyond what a prototype can implement from
    scratch. We use WV-channel brightness temperature directly as a
    MOISTURE PROXY (colder WV brightness temperature ~ more upper-level
    moisture): this is a documented simplification, not a claim of doing
    full physical IWV retrieval. Flag this honestly in any report/PPT.

Files are typically HDF5 (read via h5netcdf/netCDF4 engine through
xarray) or NetCDF, depending on what MOSDAC's order system delivers --
this loader tries both.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

WV_VAR_CANDIDATES = ["WV", "wv", "IMG_WV", "brightness_temperature_wv"]
TIR_VAR_CANDIDATES = ["TIR1", "tir1", "IMG_TIR1", "brightness_temperature_tir1"]


def _open_any(filepath: str) -> xr.Dataset:
    for engine in ("h5netcdf", "netcdf4", None):
        try:
            return xr.open_dataset(filepath, engine=engine)
        except Exception:
            continue
    raise IOError(f"Could not open {filepath} as NetCDF4 or HDF5")


def _find_var(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in ds.data_vars:
            return name
    lower_map = {v.lower(): v for v in ds.data_vars}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def load_insat(filepaths: list[str]) -> xr.Dataset:
    """Opens and merges INSAT-3D/3DR files, renaming the WV and TIR-1
    channels to `wv_bt` and `tir1_bt` (both in Kelvin)."""
    datasets = [_open_any(fp) for fp in filepaths]
    merged = xr.merge(datasets, compat="override", join="outer")

    rename = {}
    wv_name = _find_var(merged, WV_VAR_CANDIDATES)
    if wv_name:
        rename[wv_name] = "wv_bt"
    tir_name = _find_var(merged, TIR_VAR_CANDIDATES)
    if tir_name:
        rename[tir_name] = "tir1_bt"
    merged = merged.rename(rename)

    coord_rename = {}
    for cand in ("lat", "Latitude"):
        if cand in merged.coords:
            coord_rename[cand] = "latitude"
    for cand in ("lon", "Longitude"):
        if cand in merged.coords:
            coord_rename[cand] = "longitude"
    return merged.rename(coord_rename)


def extract_patch(ds: xr.Dataset, lat: float, lon: float, grid: int, box_deg: float = 1.5) -> xr.Dataset:
    lat_target = np.linspace(lat - box_deg, lat + box_deg, grid)
    lon_target = np.linspace(lon - box_deg, lon + box_deg, grid)
    return ds.interp(latitude=lat_target, longitude=lon_target, method="linear")


# Empirical rescale for the WV-brightness-temperature -> IWV-proxy mapping.
# WV channel brightness temp typically ranges ~200K (very dry/cold, high
# cloud) to ~260K (moist, low-level view). We map that range onto a
# 0-70 kg/m^2 IWV-like scale so downstream feature semantics/units match
# the rest of the pipeline (kg/m^2, same as the simulator's `iwv` field).
_WV_BT_DRY_K = 260.0
_WV_BT_MOIST_K = 200.0
_IWV_PROXY_MAX = 70.0


def iwv_proxy_from_wv(wv_bt_kelvin: np.ndarray) -> np.ndarray:
    frac = (np.asarray(wv_bt_kelvin) - _WV_BT_MOIST_K) / (_WV_BT_DRY_K - _WV_BT_MOIST_K)
    frac = np.clip(1.0 - frac, 0.0, 1.0)  # colder BT (drier term inverted) -> higher proxy
    return (frac * _IWV_PROXY_MAX).astype(np.float32)


def ctt_from_tir1(tir1_bt_kelvin: np.ndarray) -> np.ndarray:
    """TIR-1 brightness temperature used directly as cloud top temperature."""
    return np.asarray(tir1_bt_kelvin, dtype=np.float32)
