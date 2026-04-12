"""
FHVHV (Uber/Lyft) Waiting Time Predictor.

Model: fhvhv_model.pkl
Thresholds: fhvhv_thresholds.pkl
Classes: {0: "Facile", 1: "Medio", 2: "Difficile"}
Feature engineering: sin/cos cyclical encoding.
"""
import logging
import warnings
from datetime import date
from typing import Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd

from .config import (
    FHVHV_MODEL_PATH,
    FHVHV_THRESHOLDS_PATH,
    FHVHV_CLASS_NAMES,
    FHVHV_CLASS_DESCRIPTIONS,
)
from .taxi_predictor import _load_zone_lookup

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

US_HOLIDAYS = [
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4),
    date(2025, 9, 1), date(2025, 10, 13), date(2025, 11, 11),
    date(2025, 11, 27), date(2025, 12, 25),
]

_FHVHV_FEATURES = [
    "PULocationID",
    "hour_sin", "hour_cos",
    "minute_sin", "minute_cos",
    "dow_sin", "dow_cos",
    "month",
    "is_festivo",
]


def _build_fhvhv_features(
    location_id: int,
    hour: int,
    minute: int,
    day_of_week: int,
    month: int,
    is_festivo: bool,
) -> Dict[str, Any]:
    """Compute the features expected by the FHVHV model."""
    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))
    minute_sin = float(np.sin(2 * np.pi * minute / 60))
    minute_cos = float(np.cos(2 * np.pi * minute / 60))
    dow_sin = float(np.sin(2 * np.pi * day_of_week / 7))
    dow_cos = float(np.cos(2 * np.pi * day_of_week / 7))

    return {
        "PULocationID": int(location_id),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "minute_sin": minute_sin,
        "minute_cos": minute_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "month": int(month),
        "is_festivo": int(is_festivo),
    }


def _convert_to_class(waiting_time: float, p33: float, p66: float) -> int:
    """Convert waiting time to class based on thresholds."""
    if waiting_time <= p33:
        return 0
    elif waiting_time <= p66:
        return 1
    else:
        return 2


def _format_waiting_time(minutes: float) -> str:
    """Format waiting time as mm:ss string."""
    total_seconds = int(minutes * 60)
    mins = total_seconds // 60
    secs = total_seconds % 60
    return f"{mins:02d}:{secs:02d}"


class FHvhvPredictor:
    """Singleton wrapper for the FHVHV waiting time model."""

    _instance: Optional["FHvhvPredictor"] = None
    _model = None
    _thresholds = None
    _zone_lookup: Optional[pd.DataFrame] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> None:
        """Load model and thresholds from disk (lazy, idempotent)."""
        if self._model is not None:
            return
        logger.info("[FHvhvPredictor] Loading model from %s ...", FHVHV_MODEL_PATH)
        self._model = joblib.load(FHVHV_MODEL_PATH)
        self._thresholds = joblib.load(FHVHV_THRESHOLDS_PATH)
        self._zone_lookup = _load_zone_lookup()
        logger.info("[FHvhvPredictor] Model loaded.")

    def _get_zone_info(self, location_id: int) -> Dict[str, str]:
        if self._zone_lookup is None:
            self.load()
        row = self._zone_lookup[self._zone_lookup["LocationID"] == location_id]
        if len(row) > 0:
            return {
                "zone": row.iloc[0]["Zone"],
                "borough": row.iloc[0]["Borough"],
            }
        return {"zone": "Unknown", "borough": "Unknown"}

    def predict(
        self,
        location_id: int,
        hour: int,
        minute: int,
        day_of_week: int,
        month: int,
        is_festivo: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict FHVHV waiting time.

        Returns a dict with model_type="fhvhv".
        """
        if self._model is None:
            self.load()

        feats = _build_fhvhv_features(
            location_id=location_id,
            hour=hour,
            minute=minute,
            day_of_week=day_of_week,
            month=month,
            is_festivo=is_festivo,
        )
        X = pd.DataFrame([feats])[_FHVHV_FEATURES]
        X["PULocationID"] = X["PULocationID"].astype("category")

        waiting_time = float(self._model.predict(X)[0])
        waiting_time = max(0.0, waiting_time)

        p33 = self._thresholds.get("p33", 5.0)
        p66 = self._thresholds.get("p66", 10.0)
        predicted_class = _convert_to_class(waiting_time, p33, p66)

        zone_info = self._get_zone_info(location_id)

        return {
            "model_type": "fhvhv",
            "location_id": location_id,
            "location_name": zone_info["zone"],
            "borough": zone_info["borough"],
            "predicted_waiting_time": _format_waiting_time(waiting_time),
            "predicted_class": predicted_class,
            "predicted_class_name": FHVHV_CLASS_NAMES[predicted_class],
            "predicted_class_description": FHVHV_CLASS_DESCRIPTIONS[predicted_class],
        }


_fhvhv_predictor: Optional[FHvhvPredictor] = None


def get_fhvhv_predictor() -> FHvhvPredictor:
    """Return the FHvhvPredictor singleton."""
    global _fhvhv_predictor
    if _fhvhv_predictor is None:
        _fhvhv_predictor = FHvhvPredictor()
    return _fhvhv_predictor