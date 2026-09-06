"""
Converts model probabilities into categorized, human-readable alerts.
Timing (when the event is expected, when it was calculated) is attached
by the caller (main.py) using real server clock time — this module only
decides category/severity text.
"""
from typing import List, Optional

THRESHOLDS = {
    "THUNDERSTORM": {"WATCH": 0.40, "WARNING": 0.60, "SEVERE": 0.80},
    "CLOUDBURST": {"WATCH": 0.35, "WARNING": 0.55, "SEVERE": 0.75},
    "FLASH_FLOOD": {"WATCH": 0.30, "WARNING": 0.50, "SEVERE": 0.70},
}


def _severity(prob: float, thresholds: dict) -> Optional[str]:
    if prob >= thresholds["SEVERE"]:
        return "SEVERE"
    if prob >= thresholds["WARNING"]:
        return "WARNING"
    if prob >= thresholds["WATCH"]:
        return "WATCH"
    return None


def build_alerts(station_name: str, thunder: float, cloudburst: float,
                  flash_flood: float, valid_from_hr: int = 2, valid_to_hr: int = 6) -> List[dict]:
    alerts = []

    for category, prob in (
        ("THUNDERSTORM", thunder),
        ("CLOUDBURST", cloudburst),
        ("FLASH_FLOOD", flash_flood),
    ):
        sev = _severity(prob, THRESHOLDS[category])
        if sev is None:
            continue
        alerts.append({
            "category": category,
            "severity": sev,
            "valid_from_hr": valid_from_hr,
            "valid_to_hr": valid_to_hr,
            "message": (
                f"{sev}: {category.replace('_', ' ').title()} risk "
                f"{prob * 100:.0f}% for {station_name}."
            ),
        })
    return alerts
