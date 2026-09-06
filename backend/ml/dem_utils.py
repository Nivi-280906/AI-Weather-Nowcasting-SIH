"""
Topographic flood-propagation module.

Most nowcasting demos stop at "cloudburst probability at point X."
The actual disaster (flash flood) frequently happens DOWNSTREAM of the
storm cell, minutes to an hour later, as water channels through the
drainage network. This module is the project's core novelty:

    propagated_flood_risk(station) =
        f( own cloudburst_prob,
           upstream neighbours' cloudburst_prob weighted by slope/distance,
           local slope, drainage (stream) order )

It is a lightweight stand-in for full hydrological flow-accumulation
(D8/D-infinity routing on a DEM), using a static list of upstream
neighbours per station (precomputed from DEM in `seed.py`).
"""
from typing import Dict, List


def propagate_flood_risk(
    station_id: int,
    own_cloudburst_prob: float,
    own_slope_deg: float,
    own_drainage_order: int,
    upstream_probs: List[float],
) -> float:
    """
    Combines a station's own cloudburst risk with upstream contributions.
    Steeper slope + higher stream order => water arrives faster & concentrates
    more => higher amplification of upstream risk into local flash-flood risk.
    """
    slope_factor = min(1.0, own_slope_deg / 25.0)          # 0..1, saturates at 25deg
    order_factor = min(1.0, own_drainage_order / 5.0)       # higher order = more channelized

    amplification = 0.4 + 0.6 * (0.5 * slope_factor + 0.5 * order_factor)

    if upstream_probs:
        upstream_component = sum(upstream_probs) / len(upstream_probs)
    else:
        upstream_component = 0.0

    combined = own_cloudburst_prob + amplification * upstream_component * (1 - own_cloudburst_prob)
    return float(max(0.0, min(1.0, combined)))


def build_flow_graph(stations: List[dict]) -> Dict[int, List[int]]:
    """
    Toy DEM-derived flow graph: each station's 'upstream' set = other
    stations within a small geographic radius that sit at higher elevation
    (i.e. water would flow from them toward this station). In production
    this is replaced by real D8 flow-direction routing over CartoDEM/SRTM.
    """
    import math

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))

    graph = {s["id"]: [] for s in stations}
    for s in stations:
        for other in stations:
            if other["id"] == s["id"]:
                continue
            dist = haversine_km(s["lat"], s["lon"], other["lat"], other["lon"])
            if dist <= 60 and other["elevation_m"] > s["elevation_m"] + 15:
                graph[s["id"]].append(other["id"])
    return graph
