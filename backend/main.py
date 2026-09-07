"""
AI-Driven Hyper-Local Early Warning System
Backend API for Render deployment.

Endpoints:
    GET  /api/stations
    GET  /api/predict/{station_id}
    GET  /api/predict/all
    GET  /api/alerts
    GET  /api/history/{station_id}

WebSocket:
    WS /ws/live
"""

import asyncio
import json
import os
from datetime import datetime
from typing import List

import numpy as np
import torch

from fastapi import (
    FastAPI,
    Depends,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
)

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


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "weights",
    "nowcastnet.pt",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("======================================")
print("Weather Nowcasting Backend Starting")
print("BASE_DIR:", BASE_DIR)
print("DEVICE:", DEVICE)
print("WEIGHTS:", WEIGHTS_PATH)
print("======================================")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AI-Driven Hyper-Local Early Warning System",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL
# ============================================================

_model = NowcastNet(
    in_features=ds.N_FEATURES
).to(DEVICE)

_model.eval()

_flow_graph = {}

_rng = np.random.default_rng()


# ============================================================
# LOAD MODEL
# ============================================================

def _ensure_weights():

    if os.path.exists(WEIGHTS_PATH):

        print("Loading trained model weights...")

        try:

            state = torch.load(
                WEIGHTS_PATH,
                map_location=DEVICE,
            )

            _model.load_state_dict(state)

            print("Model weights loaded successfully.")

        except Exception as e:

            print("ERROR loading model weights:")
            print(e)

            raise

        return

    print("WARNING: Model weights not found.")
    print("Running warm-start training...")

    from ml.train import run as train_run

    train_run(
        epochs=3,
        steps_per_epoch=60,
        batch_size=24,
        out_path=WEIGHTS_PATH,
    )

    state = torch.load(
        WEIGHTS_PATH,
        map_location=DEVICE,
    )

    _model.load_state_dict(state)

    print("Warm-start model loaded.")


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def on_startup():

    global _flow_graph

    print("Initializing database...")

    init_db()

    print("Seeding database...")

    seed.run()

    print("Preparing model...")

    _ensure_weights()

    _model.eval()

    print("Building flow graph...")

    db = SessionLocal()

    try:

        stations = [
            {
                "id": s.id,
                "lat": s.lat,
                "lon": s.lon,
                "elevation_m": s.elevation_m,
            }
            for s in db.query(models.Station).all()
        ]

        _flow_graph = build_flow_graph(stations)

        print(
            f"Flow graph created for {len(stations)} stations."
        )

    finally:

        db.close()

    print("Application startup complete.")


# ============================================================
# RUN ONE STATION INFERENCE
# ============================================================

def _run_inference_for_station(
    db: Session,
    station: models.Station,
) -> models.Prediction:

    # --------------------------------------------------------
    # Generate weather sequence
    # --------------------------------------------------------

    seq, _ = ds.generate_sequence(_rng)

    x = torch.from_numpy(
        seq[None, ...]
    ).float().to(DEVICE)

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        mc_result, _ = _model.mc_predict(
            x,
            n_samples=5,
        )

    t_mean, t_std = mc_result["thunder"]

    c_mean, c_std = mc_result["cloudburst"]

    f_mean, f_std = mc_result["flashflood"]

    thunder = float(t_mean.item())

    cloudburst = float(c_mean.item())

    flash_seed = float(f_mean.item())

    # --------------------------------------------------------
    # XAI
    # --------------------------------------------------------

    try:

        attributions = attribute(
            _model,
            x,
        )

    except Exception as e:

        print(
            f"XAI warning for station {station.id}:",
            e,
        )

        attributions = {}

    # --------------------------------------------------------
    # Upstream flood propagation
    # --------------------------------------------------------

    upstream_ids = _flow_graph.get(
        station.id,
        [],
    )

    upstream_probs = []

    for uid in upstream_ids:

        last = (
            db.query(models.Prediction)
            .filter(
                models.Prediction.station_id == uid
            )
            .order_by(
                models.Prediction.issued_at.desc()
            )
            .first()
        )

        if last:

            upstream_probs.append(
                last.cloudburst_prob
            )

    propagated = propagate_flood_risk(
        station.id,
        cloudburst,
        station.slope_deg,
        station.drainage_order,
        upstream_probs,
    )

    flash_flood_final = max(
        flash_seed,
        propagated,
    )

    # --------------------------------------------------------
    # Create prediction
    # --------------------------------------------------------

    pred = models.Prediction(

        station_id=station.id,

        issued_at=datetime.utcnow(),

        valid_from_hr=2,

        valid_to_hr=6,

        thunderstorm_prob=round(
            thunder,
            4,
        ),

        cloudburst_prob=round(
            cloudburst,
            4,
        ),

        flash_flood_prob=round(
            flash_flood_final,
            4,
        ),

        propagated_flood_risk=round(
            propagated,
            4,
        ),

        thunderstorm_uncertainty=round(
            float(t_std.item()),
            4,
        ),

        cloudburst_uncertainty=round(
            float(c_std.item()),
            4,
        ),

        flash_flood_uncertainty=round(
            float(f_std.item()),
            4,
        ),

        features={
            f: float(
                np.max(
                    seq[-1, i]
                )
            )
            for i, f in enumerate(ds.FEATURES)
        },

        attributions=attributions,
    )

    db.add(pred)

    db.commit()

    db.refresh(pred)

    # --------------------------------------------------------
    # Generate alerts
    # --------------------------------------------------------

    alerts = build_alerts(
        station.name,
        thunder,
        cloudburst,
        flash_flood_final,
        valid_from_hr=pred.valid_from_hr,
        valid_to_hr=pred.valid_to_hr,
    )

    for a in alerts:

        db.add(
            models.Alert(

                station_id=station.id,

                prediction_id=pred.id,

                created_at=pred.issued_at,

                category=a["category"],

                severity=a["severity"],

                valid_from_hr=a[
                    "valid_from_hr"
                ],

                valid_to_hr=a[
                    "valid_to_hr"
                ],

                message=a["message"],
            )
        )

    db.commit()

    return pred


# ============================================================
# STATIONS
# ============================================================

@app.get(
    "/api/stations",
    response_model=List[schemas.StationOut],
)
def get_stations(
    db: Session = Depends(get_db),
):

    return (
        db.query(models.Station)
        .all()
    )


# ============================================================
# SINGLE STATION PREDICTION
# ============================================================

@app.get(
    "/api/predict/{station_id}",
    response_model=schemas.PredictionOut,
)
def predict_station(
    station_id: int,
    db: Session = Depends(get_db),
):

    station = (
        db.query(models.Station)
        .filter(
            models.Station.id == station_id
        )
        .first()
    )

    if station is None:

        raise HTTPException(
            status_code=404,
            detail="Station not found",
        )

    try:

        prediction = _run_inference_for_station(
            db,
            station,
        )

        return prediction

    except Exception as e:

        db.rollback()

        print(
            f"Prediction error for station {station_id}:",
            e,
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# ALL STATIONS
# ============================================================

@app.get("/api/predict/all")
def predict_all(
    db: Session = Depends(get_db),
):

    stations = (
        db.query(models.Station)
        .all()
    )

    output = []

    for station in stations:

        try:

            pred = _run_inference_for_station(
                db,
                station,
            )

            output.append({

                "station":
                    schemas.StationOut
                    .model_validate(station)
                    .model_dump(),

                "prediction":
                    schemas.PredictionOut
                    .model_validate(pred)
                    .model_dump(),

                "top_drivers": {

                    "thunderstorm":
                        top_drivers(
                            pred.attributions,
                            "thunderstorm",
                        ),

                    "cloudburst":
                        top_drivers(
                            pred.attributions,
                            "cloudburst",
                        ),

                    "flash_flood":
                        top_drivers(
                            pred.attributions,
                            "flash_flood",
                        ),
                },
            })

        except Exception as e:

            print(
                f"Prediction failed for station {station.id}:",
                e,
            )

    return output


# ============================================================
# ALERTS
# ============================================================

@app.get(
    "/api/alerts",
    response_model=List[schemas.AlertOut],
)
def get_alerts(
    limit: int = 50,
    db: Session = Depends(get_db),
):

    limit = min(
        max(limit, 1),
        100,
    )

    return (
        db.query(models.Alert)
        .order_by(
            models.Alert.created_at.desc()
        )
        .limit(limit)
        .all()
    )


# ============================================================
# HISTORY
# ============================================================

@app.get(
    "/api/history/{station_id}",
    response_model=List[schemas.PredictionOut],
)
def get_history(
    station_id: int,
    limit: int = 30,
    db: Session = Depends(get_db),
):

    limit = min(
        max(limit, 1),
        100,
    )

    return (
        db.query(models.Prediction)
        .filter(
            models.Prediction.station_id
            == station_id
        )
        .order_by(
            models.Prediction.issued_at.desc()
        )
        .limit(limit)
        .all()
    )


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws/live")
async def ws_live(
    websocket: WebSocket,
):

    await websocket.accept()

    print("WebSocket connection accepted.")

    try:

        while True:

            # -----------------------------------------------
            # IMPORTANT:
            # Run heavy synchronous ML work in a thread.
            # -----------------------------------------------

            def generate_grid():

                db = SessionLocal()

                try:

                    stations = (
                        db.query(models.Station)
                        .all()
                    )

                    payload = []

                    for station in stations:

                        try:

                            pred = (
                                _run_inference_for_station(
                                    db,
                                    station,
                                )
                            )

                            payload.append({

                                "station_id":
                                    station.id,

                                "name":
                                    station.name,

                                "lat":
                                    station.lat,

                                "lon":
                                    station.lon,

                                "thunderstorm_prob":
                                    pred.thunderstorm_prob,

                                "cloudburst_prob":
                                    pred.cloudburst_prob,

                                "flash_flood_prob":
                                    pred.flash_flood_prob,

                                "propagated_flood_risk":
                                    pred.propagated_flood_risk,

                                "uncertainty": {

                                    "thunderstorm":
                                        pred.thunderstorm_uncertainty,

                                    "cloudburst":
                                        pred.cloudburst_uncertainty,

                                    "flash_flood":
                                        pred.flash_flood_uncertainty,
                                },

                                "issued_at":
                                    pred.issued_at.isoformat(),
                            })

                        except Exception as e:

                            print(
                                f"WebSocket prediction error "
                                f"station {station.id}:",
                                e,
                            )

                    return payload

                finally:

                    db.close()

            payload = await asyncio.to_thread(
                generate_grid
            )

            # -----------------------------------------------
            # Send update
            # -----------------------------------------------

            await websocket.send_text(
                json.dumps({

                    "type":
                        "nowcast_update",

                    "data":
                        payload,
                })
            )

            print(
                f"WebSocket update sent: "
                f"{len(payload)} stations"
            )

            # -----------------------------------------------
            # Wait before next refresh
            # -----------------------------------------------

            await asyncio.sleep(60)

    except WebSocketDisconnect:

        print(
            "WebSocket client disconnected."
        )

    except Exception as e:

        print(
            "WebSocket error:",
            e,
        )

        try:

            await websocket.close()

        except Exception:
            pass


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service":
            "AI-Driven Hyper-Local Early Warning System",
        "device": DEVICE,
    }


# ============================================================
# FRONTEND
# ============================================================

frontend_dir = os.path.join(
    os.path.dirname(BASE_DIR),
    "frontend",
)

print(
    "Frontend directory:",
    frontend_dir,
)

if os.path.isdir(frontend_dir):

    print(
        "Frontend found. Mounting..."
    )

    app.mount(
        "/",
        StaticFiles(
            directory=frontend_dir,
            html=True,
        ),
        name="frontend",
    )

else:

    print(
        "WARNING: Frontend directory not found:"
    )

    print(frontend_dir)