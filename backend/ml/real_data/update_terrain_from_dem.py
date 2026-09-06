"""
One-off script: samples a real DEM GeoTIFF (SRTM from EarthExplorer, or
CartoDEM from Bhuvan) at every station's coordinates and prints updated
elevation_m / slope_deg / drainage_order values, ready to paste into
seed.py's STATIONS list -- replacing the current hand-typed estimates
with real terrain data.

Run:
    python -m ml.real_data.update_terrain_from_dem /path/to/dem.tif
"""
import sys

from . import dem as dem_mod


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m ml.real_data.update_terrain_from_dem /path/to/dem.tif")
        sys.exit(1)

    dem_path = sys.argv[1]
    import seed

    stations = [
        {"id": i, "name": s[0], "lat": s[3], "lon": s[4]}
        for i, s in enumerate(seed.STATIONS)
    ]
    results = dem_mod.sample_all_stations(dem_path, stations)

    print("\n# Paste these elevation_m/slope_deg/drainage_order values into")
    print("# the matching STATIONS row in seed.py:\n")
    for r in results:
        if r["elevation_m"] is None:
            print(f"# {r['name']}: OUTSIDE this DEM tile's coverage -- download the tile "
                  f"that covers this station's lat/lon and rerun.")
        else:
            print(f"# {r['name']}: elevation_m={r['elevation_m']}, "
                  f"slope_deg={r['slope_deg']}, drainage_order={r['drainage_order']}")


if __name__ == "__main__":
    main()
