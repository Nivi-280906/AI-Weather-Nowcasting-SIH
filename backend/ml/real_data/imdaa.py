"""
Real IMDAA reanalysis ingestion.

Reads the NetCDF4 files downloaded from rds.ncmrwf.gov.in (3-hourly,
pressure-level: U wind, V wind, temperature, geopotential height,
relative humidity) and derives, for a small lat/lon patch around each
station:

    cape, cin            (metpy surface-based parcel calculation)
    shear_06km           (bulk wind shear between ~1000 hPa and ~500 hPa)
    conv_low             (low-level horizontal wind convergence, ~850 hPa)

This is the "thermodynamic baseline" data source named in the problem
statement (IMDAA -> temperature, humidity, geopotential, U/V wind ->
CAPE/CIN, shear, convergence).

Expects files where the variable names follow IMDAA's usual short names;
adjust VAR_NAME_MAP below if your downloaded files use different keys
(open one file with xarray and check `ds.data_vars` if extraction fails).
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from metpy.calc import cape_cin, dewpoint_from_relative_humidity, parcel_profile
from metpy.units import units

# IMDAA NetCDF variable names -> our internal names. IMDAA files sometimes
# ship one variable per file (as ordered from rds.ncmrwf.gov.in) rather than
# one file with all variables, so we merge them by time+level+lat+lon.
VAR_NAME_MAP = {
    "u": ["u", "ugrd", "u_component_of_wind"],
    "v": ["v", "vgrd", "v_component_of_wind"],
    "t": ["t", "tmp", "temperature"],
    "z": ["z", "hgt", "gh", "geopotential_height"],
    "r": ["r", "rh", "relative_humidity"],
}


def _find_var(ds: xr.Dataset, candidates: list[str]) -> str:
    for name in candidates:
        if name in ds.data_vars:
            return name
    lower_map = {v.lower(): v for v in ds.data_vars}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise KeyError(
        f"None of {candidates} found in dataset variables {list(ds.data_vars)}"
    )


def load_imdaa(filepaths: list[str]) -> xr.Dataset:
    """Opens and merges one or more IMDAA NetCDF4 files into a single
    Dataset indexed by (time, level, latitude, longitude), with variables
    renamed to u, v, t, z, r."""
    datasets = [xr.open_dataset(fp) for fp in filepaths]
    merged = xr.merge(datasets, compat="override", join="outer")

    rename = {}
    for internal_name, candidates in VAR_NAME_MAP.items():
        try:
            found = _find_var(merged, candidates)
            rename[found] = internal_name
        except KeyError:
            pass  # variable not present in this batch of files; skip
    merged = merged.rename(rename)

    # Standardize coordinate names (IMDAA sometimes uses lat/lon vs
    # latitude/longitude, or "isobaricInhPa" vs "level" vs "plev").
    coord_rename = {}
    for cand in ("lat", "latitude"):
        if cand in merged.coords and cand != "latitude":
            coord_rename[cand] = "latitude"
    for cand in ("lon", "longitude"):
        if cand in merged.coords and cand != "longitude":
            coord_rename[cand] = "longitude"
    for cand in ("level", "isobaricInhPa", "plev", "pressure_level"):
        if cand in merged.coords and cand != "level":
            coord_rename[cand] = "level"
    merged = merged.rename(coord_rename)

    return merged


def extract_patch(
    ds: xr.Dataset, lat: float, lon: float, grid: int, box_deg: float = 1.5
) -> xr.Dataset:
    """Extracts (and regrids to grid x grid) a square lat/lon patch centered
    on (lat, lon). box_deg is the half-width of the box in degrees --
    default 1.5 deg (~165km) covers a realistic convective-scale domain at
    IMDAA's native resolution."""
    lat_target = np.linspace(lat - box_deg, lat + box_deg, grid)
    lon_target = np.linspace(lon - box_deg, lon + box_deg, grid)
    return ds.interp(latitude=lat_target, longitude=lon_target, method="linear")


def compute_cape_cin(patch: xr.Dataset, time_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Surface-based CAPE/CIN per grid cell at one timestep.
    Loops per-column (metpy's parcel calc is 1D) -- fine for a 16x16 patch
    used offline during dataset construction."""
    t_slice = patch["t"].isel(time=time_idx)  # [level, lat, lon], Kelvin
    r_slice = patch["r"].isel(time=time_idx)  # [level, lat, lon], %
    levels_hpa = patch["level"].values  # hPa
    order = np.argsort(levels_hpa)[::-1]  # surface (highest hPa) first
    levels_hpa = levels_hpa[order]

    ny, nx = t_slice.shape[1], t_slice.shape[2]
    cape = np.zeros((ny, nx), dtype=np.float32)
    cin = np.zeros((ny, nx), dtype=np.float32)

    p = levels_hpa * units.hPa
    for iy in range(ny):
        for ix in range(nx):
            t_profile = t_slice.values[order, iy, ix] * units.kelvin
            r_profile = np.clip(r_slice.values[order, iy, ix], 0, 100) * units.percent
            try:
                td_profile = dewpoint_from_relative_humidity(t_profile, r_profile)
                prof = parcel_profile(p, t_profile[0], td_profile[0]).to("kelvin")
                c, i = cape_cin(p, t_profile, td_profile, prof)
                cape[iy, ix] = c.magnitude if np.isfinite(c.magnitude) else 0.0
                cin[iy, ix] = i.magnitude if np.isfinite(i.magnitude) else 0.0
            except Exception:
                # Missing/degenerate profile (e.g. supersaturated or
                # non-monotonic sounding at this cell) -> neutral fallback.
                cape[iy, ix] = 0.0
                cin[iy, ix] = 0.0
    return cape, cin


def compute_shear_06km(patch: xr.Dataset, time_idx: int) -> np.ndarray:
    """Bulk wind shear between ~1000 hPa (near-surface) and ~500 hPa
    (~5.5-6km altitude in a standard atmosphere) -- a standard nowcasting
    proxy for full hodograph-based 0-6km shear."""
    levels = patch["level"].values
    low_lvl = levels[np.argmin(np.abs(levels - 1000))]
    high_lvl = levels[np.argmin(np.abs(levels - 500))]

    u_low = patch["u"].isel(time=time_idx).sel(level=low_lvl).values
    v_low = patch["v"].isel(time=time_idx).sel(level=low_lvl).values
    u_high = patch["u"].isel(time=time_idx).sel(level=high_lvl).values
    v_high = patch["v"].isel(time=time_idx).sel(level=high_lvl).values

    return np.sqrt((u_high - u_low) ** 2 + (v_high - v_low) ** 2).astype(np.float32)


def compute_convergence_low(patch: xr.Dataset, time_idx: int, level_hpa: float = 850.0) -> np.ndarray:
    """Low-level (~850 hPa) horizontal wind convergence: -(du/dx + dv/dy).
    Positive output = convergence (rising motion trigger)."""
    levels = patch["level"].values
    lvl = levels[np.argmin(np.abs(levels - level_hpa))]

    u = patch["u"].isel(time=time_idx).sel(level=lvl).values
    v = patch["v"].isel(time=time_idx).sel(level=lvl).values

    lat_vals = patch["latitude"].values
    lon_vals = patch["longitude"].values
    dy_m = np.gradient(lat_vals) * 111_000.0  # deg -> meters (lat)
    mean_lat_rad = np.radians(np.mean(lat_vals))
    dx_m = np.gradient(lon_vals) * 111_000.0 * np.cos(mean_lat_rad)  # deg -> meters (lon)

    dudx = np.gradient(u, axis=1) / dx_m[np.newaxis, :]
    dvdy = np.gradient(v, axis=0) / dy_m[:, np.newaxis]

    return (-(dudx + dvdy)).astype(np.float32)
