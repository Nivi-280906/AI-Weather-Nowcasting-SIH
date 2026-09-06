from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime


class StationOut(BaseModel):
    id: int
    name: str
    district: str
    state: str
    lat: float
    lon: float
    elevation_m: float
    slope_deg: float
    drainage_order: int

    class Config:
        from_attributes = True


class PredictionOut(BaseModel):
    id: int
    station_id: int
    issued_at: datetime
    valid_from_hr: int
    valid_to_hr: int
    thunderstorm_prob: float
    cloudburst_prob: float
    flash_flood_prob: float
    propagated_flood_risk: float
    thunderstorm_uncertainty: float
    cloudburst_uncertainty: float
    flash_flood_uncertainty: float
    features: Optional[Dict] = None
    attributions: Optional[Dict] = None

    class Config:
        from_attributes = True


class AlertOut(BaseModel):
    id: int
    station_id: int
    prediction_id: int
    created_at: datetime
    category: str
    severity: str
    valid_from_hr: int
    valid_to_hr: int
    message: str

    class Config:
        from_attributes = True

