"""
Real DEM ingestion (SRTM 1-arc-second GeoTIFF from USGS EarthExplorer, or
CartoDEM from Bhuvan -- problem statement names both as acceptable).

Replaces the hand-typed elevation_m / slope_deg / drainage_order columns
in seed.py with values sampled from an actual elevation raster.

Slope is computed properly (real terrain gradient). Drainage/stream order
is approximated by counting how many nearby higher-elevation cells drain
toward this one within a local window -- a lightweight proxy for full D8
flow-accumulation routing (a full flow-accumulation library like
`richdem` or `pysheds` would refine this further; noted here rather than
silently overstating precision, matching the existing disclosure already
in dem_utils.py).
"""
from __future__ import annotations

import math

import numpy as np
import rasterio
from rasterio.windows import from_bounds


def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    m_per_deg_lat = 111_132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    m_per_deg_lon = 111_412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
    return m_per_deg_lat, m_per_deg_lon


def sample_station_terrain(
    dem_path: str, lat: float, lon: float, window_deg: float = 0.02
) -> dict:
    """Reads a small window of the DEM around (lat, lon) and returns
    elevation_m, slope_deg, drainage_order for that station."""
    with rasterio.open(dem_path) as src:
        window = from_bounds(
            lon - window_deg, lat - window_deg, lon + window_deg, lat + window_deg,
            transform=src.transform,
        )
        elev = src.read(1, window=window).astype(np.float32)
        elev[elev <= -1000] = np.nan  # SRTM void/no-data sentinel

        if elev.size == 0 or np.all(np.isnan(elev)) or min(elev.shape) < 3:
            # Window too small/degenerate (off the tile's edge, or a
            # window_deg too tight for this raster's resolution) to
            # compute a gradient safely -- report as out-of-coverage
            # rather than crashing the whole batch.
            return {"elevation_m": None, "slope_deg": None, "drainage_order": None}

        px_h = window.height or elev.shape[0]
        px_w = window.width or elev.shape[1]
        m_per_deg_lat, m_per_deg_lon = _meters_per_degree(lat)
        dy_m = (2 * window_deg * m_per_deg_lat) / max(px_h, 1)
        dx_m = (2 * window_deg * m_per_deg_lon) / max(px_w, 1)

        gy, gx = np.gradient(elev, dy_m, dx_m)
        slope_rad = np.arctan(np.sqrt(gy**2 + gx**2))
        slope_deg = float(np.nanmean(np.degrees(slope_rad)))

        center_elev = float(np.nanmean(elev[elev.shape[0] // 2 - 1: elev.shape[0] // 2 + 1,
                                             elev.shape[1] // 2 - 1: elev.shape[1] // 2 + 1]))

        # Drainage-order proxy: fraction of surrounding cells higher than
        # the center cell, scaled to roughly match the 1-5 integer scale
        # already used in seed.py/dem_utils.py.
        higher_frac = float(np.nanmean(elev > center_elev))
        drainage_order = int(np.clip(round(1 + higher_frac * 4), 1, 5))

        elevation_m = int(round(center_elev)) if not np.isnan(center_elev) else None

    return {
        "elevation_m": elevation_m,
        "slope_deg": round(slope_deg, 1) if not np.isnan(slope_deg) else None,
        "drainage_order": drainage_order,
    }


def sample_all_stations(dem_path: str, stations: list[dict]) -> list[dict]:
    """stations: list of dicts with at least 'id', 'lat', 'lon'.
    Returns a new list of dicts: {id, elevation_m, slope_deg, drainage_order}
    ready to be written back into seed.py or a DB update."""
    results = []
    for s in stations:
        terrain = sample_station_terrain(dem_path, s["lat"], s["lon"])
        results.append({"id": s["id"], "name": s.get("name"), **terrain})
    return results
