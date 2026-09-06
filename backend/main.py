"""
AI-Driven Hyper-Local Early Warning System — backend API.

Endpoints:
  GET  /api/stations                 -> all monitoring stations
  GET  /api/predict/{station_id}     -> run a fresh nowcast for one station
  GET  /api/predict/all              -> run nowcast for every station (dashboard load)
  GET  /api/alerts?limit=50          -> recent alerts
  GET  /api/history/{station_id}     -> past predictions for trend charts
  WS   /ws/live                      -> streams a fresh full-grid nowcast every N seconds

Run:
    uvicorn main:app --reload --port 8000
"""
import asyncio
import json
import os
from datetime import datetime
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from database import get_db, init_db, SessionLocal
import models
import schemas
import seed
from alert_engine import build_alerts
from ml.model import NowcastNet
from ml import data_simulator as ds
from ml.xai import attribute, top_drivers
from ml.dem_utils import propagate_flood_risk, build_flow_graph

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "ml", "weights", "nowcastnet.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="AI-Driven Hyper-Local Early Warning System")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_model = NowcastNet(in_features=ds.N_FEATURES).to(DEVICE)
_flow_graph = {}
_rng = np.random.default_rng()


def _ensure_weights():
    """If no trained weights exist yet, run a quick warm-start training pass
    (small, fast) so the API is usable immediately after clone. For a
    stronger model, run `python -m ml.train --epochs 20` separately."""
    if os.path.exists(WEIGHTS_PATH):
        _model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print("Loaded trained weights.")
        return
    print("No weights found — running a quick warm-start fit (~30s)...")
    from ml.train import run as train_run
    train_run(epochs=3, steps_per_epoch=60, batch_size=24, out_path=WEIGHTS_PATH)
    _model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))


@app.on_event("startup")
def on_startup():
    init_db()
    seed.run()
    _ensure_weights()
    _model.eval()

    db = SessionLocal()
    try:
        stations = [
            {"id": s.id, "lat": s.lat, "lon": s.lon, "elevation_m": s.elevation_m}
            for s in db.query(models.Station).all()
        ]
        global _flow_graph
        _flow_graph = build_flow_graph(stations)
    finally:
        db.close()


def _run_inference_for_station(db: Session, station: models.Station) -> models.Prediction:
    seq, _ = ds.generate_sequence(_rng)  # stand-in for live IMDAA/INSAT pull for this cell
    x = torch.from_numpy(seq[None, ...]).to(DEVICE)  # [1,T,C,H,W]

    mc_result, _ = _model.mc_predict(x, n_samples=10)
    t_mean, t_std = mc_result["thunder"]
    c_mean, c_std = mc_result["cloudburst"]
    f_mean, f_std = mc_result["flashflood"]

    thunder = float(t_mean.item())
    cloudburst = float(c_mean.item())
    flash_seed = float(f_mean.item())

    attributions = attribute(_model, x)

    # novelty: DEM-based downstream propagation using upstream stations'
    # most recent cloudburst probability (falls back to own value if none yet)
    upstream_ids = _flow_graph.get(station.id, [])
    upstream_probs = []
    for uid in upstream_ids:
        last = (
            db.query(models.Prediction)
            .filter(models.Prediction.station_id == uid)
            .order_by(models.Prediction.issued_at.desc())
            .first()
        )
        if last:
            upstream_probs.append(last.cloudburst_prob)

    propagated = propagate_flood_risk(
        station.id, cloudburst, station.slope_deg, station.drainage_order, upstream_probs
    )
    flash_flood_final = max(flash_seed, propagated)

    pred = models.Prediction(
        station_id=station.id,
        issued_at=datetime.utcnow(),
        valid_from_hr=2,
        valid_to_hr=6,
        thunderstorm_prob=round(thunder, 4),
        cloudburst_prob=round(cloudburst, 4),
        flash_flood_prob=round(flash_flood_final, 4),
        propagated_flood_risk=round(propagated, 4),
        thunderstorm_uncertainty=round(float(t_std.item()), 4),
        cloudburst_uncertainty=round(float(c_std.item()), 4),
        flash_flood_uncertainty=round(float(f_std.item()), 4),
        features={f: float(np.max(seq[-1, i])) for i, f in enumerate(ds.FEATURES)},
        attributions=attributions,
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)

    for a in build_alerts(
        station.name, thunder, cloudburst, flash_flood_final,
        valid_from_hr=pred.valid_from_hr, valid_to_hr=pred.valid_to_hr,
    ):
        db.add(models.Alert(
            station_id=station.id, prediction_id=pred.id,
            created_at=pred.issued_at,
            category=a["category"], severity=a["severity"],
            valid_from_hr=a["valid_from_hr"], valid_to_hr=a["valid_to_hr"],
            message=a["message"],
        ))
    db.commit()

    return pred


@app.get("/api/stations", response_model=List[schemas.StationOut])
def get_stations(db: Session = Depends(get_db)):
    return db.query(models.Station).all()


@app.get("/api/predict/{station_id}", response_model=schemas.PredictionOut)
def predict_station(station_id: int, db: Session = Depends(get_db)):
    station = db.query(models.Station).get(station_id)
    if not station:
        raise HTTPException(404, "station not found")
    pred = _run_inference_for_station(db, station)
    return pred


@app.get("/api/predict/all")
def predict_all(db: Session = Depends(get_db)):
    stations = db.query(models.Station).all()
    out = []
    for s in stations:
        pred = _run_inference_for_station(db, s)
        out.append({
            "station": schemas.StationOut.model_validate(s).model_dump(),
            "prediction": schemas.PredictionOut.model_validate(pred).model_dump(),
            "top_drivers": {
                "thunderstorm": top_drivers(pred.attributions, "thunderstorm"),
                "cloudburst": top_drivers(pred.attributions, "cloudburst"),
                "flash_flood": top_drivers(pred.attributions, "flash_flood"),
            },
        })
    return out


@app.get("/api/alerts", response_model=List[schemas.AlertOut])
def get_alerts(limit: int = 50, db: Session = Depends(get_db)):
    return (
        db.query(models.Alert)
        .order_by(models.Alert.created_at.desc())
        .limit(limit)
        .all()
    )


@app.get("/api/history/{station_id}", response_model=List[schemas.PredictionOut])
def get_history(station_id: int, limit: int = 30, db: Session = Depends(get_db)):
    return (
        db.query(models.Prediction)
        .filter(models.Prediction.station_id == station_id)
        .order_by(models.Prediction.issued_at.desc())
        .limit(limit)
        .all()
    )


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """Streams a full-grid nowcast refresh every 15s so the dashboard map
    and alert feed update in real time without polling."""
    await websocket.accept()
    db = SessionLocal()
    try:
        while True:
            stations = db.query(models.Station).all()
            payload = []
            for s in stations:
                pred = _run_inference_for_station(db, s)
                payload.append({
                    "station_id": s.id,
                    "name": s.name,
                    "lat": s.lat,
                    "lon": s.lon,
                    "thunderstorm_prob": pred.thunderstorm_prob,
                    "cloudburst_prob": pred.cloudburst_prob,
                    "flash_flood_prob": pred.flash_flood_prob,
                    "propagated_flood_risk": pred.propagated_flood_risk,
                    "uncertainty": {
                        "thunderstorm": pred.thunderstorm_uncertainty,
                        "cloudburst": pred.cloudburst_uncertainty,
                        "flash_flood": pred.flash_flood_uncertainty,
                    },
                    "issued_at": pred.issued_at.isoformat(),
                })
            await websocket.send_text(json.dumps({"type": "nowcast_update", "data": payload}))
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        db.close()


# Serve the frontend directly so the whole thing runs from one process.
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
