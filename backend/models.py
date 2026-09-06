"""
ORM models for stations, predictions, and alerts.
"""
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


class Station(Base):
    """A grid cell / monitoring point (represents a hyper-local location)."""
    __tablename__ = "stations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    district = Column(String)
    state = Column(String)
    lat = Column(Float)
    lon = Column(Float)
    elevation_m = Column(Float, default=0.0)
    slope_deg = Column(Float, default=0.0)
    drainage_order = Column(Integer, default=1)  # Strahler-like stream order

    predictions = relationship("Prediction", back_populates="station")


class Prediction(Base):
    """One nowcast issued for a station at a point in time."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(Integer, ForeignKey("stations.id"))
    issued_at = Column(DateTime, default=datetime.utcnow)
    valid_from_hr = Column(Integer, default=2)
    valid_to_hr = Column(Integer, default=6)

    thunderstorm_prob = Column(Float)
    cloudburst_prob = Column(Float)
    flash_flood_prob = Column(Float)

    # Novelty: downstream-propagated flood risk (DEM flow-routed)
    propagated_flood_risk = Column(Float, default=0.0)

    # Uncertainty (MC-Dropout std) per head
    thunderstorm_uncertainty = Column(Float, default=0.0)
    cloudburst_uncertainty = Column(Float, default=0.0)
    flash_flood_uncertainty = Column(Float, default=0.0)

    # Raw input signature snapshot (for XAI / audit trail)
    features = Column(JSON)
    # XAI attribution: {feature_name: contribution_score}
    attributions = Column(JSON)

    station = relationship("Station", back_populates="predictions")


class Alert(Base):
    """Automated alert generated when a threshold is breached."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(Integer, ForeignKey("stations.id"))
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    created_at = Column(DateTime, default=datetime.utcnow)  # when the model computed this
    category = Column(String)       # THUNDERSTORM / CLOUDBURST / FLASH_FLOOD
    severity = Column(String)       # WATCH / WARNING / SEVERE
    valid_from_hr = Column(Integer, default=2)  # event expected to START this many hours from created_at
    valid_to_hr = Column(Integer, default=6)     # event expected to END/pass by this many hours from created_at
    message = Column(String)
