# AI-Driven Hyper-Local Early Warning System
### Severe Weather Nowcasting — Thunderstorms · Cloudbursts · Flash Floods (2–6h lead time)

A full, runnable prototype: PyTorch multi-task model + FastAPI/WebSocket backend +
SQLite database + a live map dashboard. Everything runs from one folder in VS Code.

---

## 1. How to run (VS Code / any machine)

**Requirements:** Python 3.10+ (no GPU needed — model is small on purpose for real-time inference).

```bash
# from the project root
./run.sh              # Mac/Linux
run_windows.bat        # Windows
```

This will: create a venv, install dependencies, seed the database with 20 flash-flood /
cloudburst-prone stations across India, auto-train a quick warm-start model if no
weights exist yet (~30s, first run only), and start the server.

Then open **http://localhost:8000** — you'll land straight on the dashboard (live
map, alerts, and XAI panel are served from the same FastAPI process, so there's
nothing else to start).

**To train a stronger model** (recommended before a demo/judging round):
```bash
cd backend
source venv/bin/activate
python -m ml.train --epochs 20 --steps-per-epoch 300 --batch-size 32
```
This overwrites `backend/ml/weights/nowcastnet.pt`; restart the server to use it.

If you only want the API (no auto-reload UI), run manually:
```bash
cd backend && uvicorn main:app --reload --port 8000
```

---

## 2. What's inside

```
backend/
  main.py              FastAPI app: REST endpoints + /ws/live real-time stream
  database.py           SQLite/SQLAlchemy setup (swap DATABASE_URL for Postgres/PostGIS)
  models.py              ORM: Station, Prediction, Alert
  schemas.py              Pydantic response models
  seed.py                  20 representative stations (Himalayan foothills, Western
                            Ghats, NE hills, urban flood hotspots) with elevation/
                            slope/drainage-order (would come from CartoDEM/SRTM)
  alert_engine.py         Threshold -> WATCH/WARNING/SEVERE categorized alerts
  ml/
    data_simulator.py     Generates physically-plausible IWV/CAPE/CIN/shear/
                           convergence/CTT/QPE spatiotemporal grids (stand-in for
                           live IMDAA + INSAT-3D/3DR + QPE feeds — same tensor shape,
                           so swapping in real data connectors doesn't touch the model)
    model.py               Shared ConvGRU spatiotemporal backbone + 3 task heads
                            (thunderstorm / cloudburst / flash-flood-seed) with
                            MC-Dropout for uncertainty
    train.py                Training loop
    dem_utils.py             Novelty #1: DEM flow-routing that propagates cloudburst
                              risk downstream through the drainage network
    xai.py                    Novelty #2: real-time occlusion-based explainability
                               ("why did the model raise this alert")
frontend/
  index.html, style.css, app.js   Live map dashboard (Leaflet + WebSocket), per-station
                                   risk breakdown with real predicted event time and
                                   calculation timestamp, uncertainty bands, XAI driver
                                   list (collapsed by default — expand for full detail),
                                   live alert feed, and browser notifications
```

---

## 3. How this maps to the problem statement

| Requirement | Implementation |
|---|---|
| Nowcast severe thunderstorms, cloudbursts, flash floods simultaneously, 2–6h lead time | Single `NowcastNet` with a shared backbone and three output heads, run per-station every 15s (`main.py`, `ml/model.py`) |
| Bypass NWP latency — deep learning, not thermodynamic simulation | Spatiotemporal ConvGRU model, inference in milliseconds per station |
| IWV-based storm nowcasting | `iwv` and `iwv_rate` channels are first-class model inputs; `data_simulator.py` models rapid moisture-pool accumulation explicitly |
| CAPE/CIN instability, convergence & shear (lift), CTT drop rate | All modeled as separate input channels feeding the shared encoder |
| DEM-based flash flood translation | `dem_utils.py` — static slope/elevation/drainage-order per station plus a flow graph that propagates upstream cloudburst risk downstream |
| Multi-modal data fusion (IMDAA + INSAT + QPE + DEM) | Unified `[T, C, H, W]` tensor per station; `data_simulator.py` documents exactly which real API/product would fill each channel |
| Explainable AI module | `ml/xai.py` — occlusion-based attribution, surfaced live in the dashboard's "why this alert" panel |
| Automated categorized alerting via API | `alert_engine.py` + `/api/alerts` + WebSocket push, plus browser push notifications for SEVERE alerts |
| Interactive spatial dashboard for disaster managers | `frontend/` — live map, per-station drill-down (full detail hidden until expanded), predicted event time + calculation timestamp, alert feed |

---

## 4. Novelty beyond a standard nowcasting CNN

1. **Downstream flood propagation, not just point risk.** Most nowcasting demos stop
   at "cloudburst probability at point X." The actual flash flood usually happens
   *downstream*, minutes to an hour later. `dem_utils.propagate_flood_risk()` combines
   a station's own cloudburst probability with its upstream neighbours' probability,
   weighted by local slope and drainage/stream order — closer to how flash floods
   actually reach villages below a cloudburst than a purely point-wise model.
2. **Calibrated uncertainty via MC-Dropout**, shown on the dashboard as a ± band per
   head, so disaster managers can distinguish a confident SEVERE call from a shaky one
   — directly reduces false-alarm fatigue, which the problem statement flags as a
   requirement ("low false-alarm rates").
3. **Real-time occlusion-based XAI** computed per-alert (not offline), so every push
   alert ships with "top 3 physical drivers" (e.g. "IWV rate +18%, CTT drop +11%"),
   giving first responders an actionable meteorological reason, not just a number.

---

## 5. Synthetic demo mode vs. real data

`ml/data_simulator.py` generates physically-plausible synthetic fields (quiet
baseline + injected moisture/instability/convergence "storm onset" events) so the
full pipeline — ingestion → model → DEM propagation → XAI → alert → dashboard —
runs end-to-end with zero external downloads or credentials. This is the default
when you just run `run.sh` / `run_windows.bat`.

**A real-data ingestion pipeline also exists**, in `ml/real_data/` — it reads
actual downloaded IMDAA (rds.ncmrwf.gov.in), INSAT-3D/3DR (mosdac.gov.in), GPM
IMERG QPE, and SRTM/CartoDEM files, and derives the same 9 features (CAPE/CIN via
metpy, real wind shear and convergence, INSAT-derived IWV proxy and cloud-top
temperature, real QPE) in the exact tensor shape the model already expects. See
`ml/real_data/README.md` for the folder layout and:

```bash
python -m ml.train --real-data-dir /path/to/downloaded/data --epochs 8
```

This has been tested end-to-end against synthetic mock files matching the real
schemas (no crashes, no NaNs, physically sensible CAPE/elevation/slope values) —
but not yet against actual downloaded government data, since that requires each
team member's own dataset requests/logins to complete. `ml/real_data/README.md`
documents the honest simplifications made (IWV is a documented brightness-
temperature proxy, not full physical retrieval; drainage order is a local-slope
proxy, not full D8 flow routing; labels are physically-thresholded, not from a
verified event catalog) — say these plainly if asked, rather than overclaiming.

---

## 6. Extending toward production

- Swap `sqlite:///./nowcast.db` for Postgres + PostGIS (schema already supports it).
- Point `ml/train.py --real-data-dir` at real downloaded data once available (see
  `ml/real_data/README.md`) instead of hand-rolling new connectors.
- Replace the toy `build_flow_graph()` neighbour heuristic and the DEM drainage-order
  proxy in `ml/real_data/dem.py` with real D8/D-infinity flow-direction routing
  (e.g. `pysheds`, `richdem`).
- Replace the physically-thresholded labels in `ml/real_data/labels.py` with a
  verified event catalog (IMD storm reports, lightning-network data, confirmed
  flood records) once available.
- Add SMS/IVR push (Twilio or a government SMS gateway) alongside the JSON alert API
  for last-mile alerts to vulnerable communities without smartphones.
