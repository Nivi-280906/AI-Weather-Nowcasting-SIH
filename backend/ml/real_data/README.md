# Real-data ingestion

Replaces `ml/data_simulator.py`'s synthetic fields with the four data
sources named in the problem statement: IMDAA, INSAT-3D/3DR, GPM IMERG
(QPE), and DEM (SRTM/CartoDEM). Nothing here needs network access or
government-portal credentials -- it only reads files you've already
downloaded yourself.

## 1. Folder layout

Put your downloaded files in one directory, like this:

```
real_data/
  imdaa/     *.nc      (from rds.ncmrwf.gov.in)
  insat/     *.nc or *.h5   (from mosdac.gov.in)
  qpe/       *.nc4 or *.HDF5   (GPM IMERG, from NASA Earthdata/GES DISC)
  dem.tif                (SRTM from EarthExplorer, or CartoDEM from Bhuvan)
```

## 2. Train on real data

```bash
cd backend
python -m ml.train --real-data-dir /path/to/real_data --epochs 8
```

This builds real training samples the first time it runs (this is the
slow part -- CAPE/CIN is a per-grid-cell parcel calculation), trains
`NowcastNet` on them, and saves weights to `ml/weights/nowcastnet.pt` --
the exact same file the live FastAPI backend already loads. No other
code changes needed; the dashboard picks up the new weights on restart.

Omit `--real-data-dir` to keep training on the synthetic simulator as
before (useful for a quick sanity-check run without real data on hand).

## 3. Update station terrain from a real DEM

```bash
python -m ml.real_data.update_terrain_from_dem /path/to/dem.tif
```

Prints real elevation/slope/drainage-order values sampled at each
station's coordinates -- paste them into `seed.py`'s `STATIONS` list to
replace the current hand-typed estimates. Delete `nowcast.db` and
restart afterward so the seeded values refresh.

## 4. Known simplifications (be upfront about these if asked)

- **IWV**: true Integrated Water Vapour needs a physical retrieval
  algorithm. We use INSAT's WV-channel brightness temperature as a
  documented moisture proxy instead (see `insat.py`). CTT, by contrast,
  IS the TIR-1 brightness temperature directly -- that part is exact,
  not a proxy.
- **Drainage order**: approximated from local DEM slope/neighbour
  elevation rather than full D8/D-infinity flow-accumulation routing.
  A library like `richdem` or `pysheds` would refine this.
- **Labels**: there's no bundled "a thunderstorm happened here" ground
  truth in any of these datasets. Labels are derived from the same
  physical thresholds (CAPE/CIN, IWV surge, CTT collapse, convergence,
  realized QPE) that meteorologists use operationally -- a physically
  grounded proxy, not a verified event catalog. A more rigorous version
  would cross-reference IMD storm reports or lightning-network data.

None of this is hidden inside the code -- every module above documents
its own approximations at the top of the file.
