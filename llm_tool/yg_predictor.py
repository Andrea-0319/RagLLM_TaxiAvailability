"""
Yellow/Green Taxi Predictor — wrapper for Riccardo's LightGBM model.

Model: taxi_lgbm_model_production.pkl
Classes: {0: "Bassa", 1: "Media", 2: "Alta"}
Feature engineering: sin/cos cyclical encoding (no StandardScaler).
"""
import logging
import warnings
from typing import Optional, List, Dict, Any

import joblib
import numpy as np
import pandas as pd

from .config import (
    YG_MODEL_PATH,
    YG_CLASS_NAMES,
    YG_CLASS_DESCRIPTIONS,
)
from .taxi_predictor import _load_zone_lookup  # reuse existing zone lookup loader

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Feature order must match training exactly
_YG_FEATURES = [
    "zone", "month", "quarter",
    "hour_sin", "hour_cos",
    "day_sin", "day_cos",
    "vehicle_type", "service_mode",
]
_YG_CATEGORICAL = ["zone", "month", "quarter", "vehicle_type", "service_mode"]

_VEHICLE_ENCODING = {"yellow": 0, "green": 1}
_SERVICE_ENCODING = {"hail": 0, "dispatch": 1}


def _build_yg_features(
    zone: int,
    hour: int,
    minute: int,
    day_of_week: int,
    month: int,
    vehicle_type: str,
    service_mode: str,
) -> Dict[str, Any]:
    """Compute the 9 features expected by the YG model."""
    if vehicle_type not in _VEHICLE_ENCODING:
        raise ValueError(f"vehicle_type non valido: {vehicle_type!r}. Valori accettati: {list(_VEHICLE_ENCODING)}")
    if service_mode not in _SERVICE_ENCODING:
        raise ValueError(f"service_mode non valido: {service_mode!r}. Valori accettati: {list(_SERVICE_ENCODING)}")

    quarter = minute // 15
    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))
    day_sin  = float(np.sin(2 * np.pi * day_of_week / 7))
    day_cos  = float(np.cos(2 * np.pi * day_of_week / 7))

    return {
        "zone":         int(zone),
        "month":        int(month),
        "quarter":      int(quarter),
        "hour_sin":     hour_sin,
        "hour_cos":     hour_cos,
        "day_sin":      day_sin,
        "day_cos":      day_cos,
        "vehicle_type": int(_VEHICLE_ENCODING[vehicle_type]),
        "service_mode": int(_SERVICE_ENCODING[service_mode]),
    }


def _cast_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in _YG_CATEGORICAL:
        df[col] = df[col].astype("category")
    return df


class YGPredictor:
    """Singleton wrapper for the Yellow/Green LightGBM model (Riccardo)."""

    _instance: Optional["YGPredictor"] = None
    _model = None
    _zone_lookup: Optional[pd.DataFrame] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> None:
        """Load model from disk (lazy, idempotent)."""
        if self._model is not None:
            return
        logger.info("[YGPredictor] Loading model from %s ...", YG_MODEL_PATH)
        self._model = joblib.load(YG_MODEL_PATH)
        self._zone_lookup = _load_zone_lookup()
        logger.info("[YGPredictor] Model loaded.")

    def _get_zone_info(self, location_id: int) -> Dict[str, str]:
        if self._zone_lookup is None:
            self.load()
        row = self._zone_lookup[self._zone_lookup["LocationID"] == location_id]
        if len(row) > 0:
            return {
                "zone":    row.iloc[0]["Zone"],
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
        vehicle_type: str,
        service_mode: str,
    ) -> Dict[str, Any]:
        """
        Predict taxi availability for one vehicle type / service mode.

        Returns a normalised dict with model_type="yg".
        """
        if self._model is None:
            self.load()

        feats = _build_yg_features(
            zone=location_id, hour=hour, minute=minute,
            day_of_week=day_of_week, month=month,
            vehicle_type=vehicle_type, service_mode=service_mode,
        )
        X = pd.DataFrame([feats])[_YG_FEATURES]
        X = _cast_categorical(X)

        raw_class = int(self._model.predict(X)[0])

        # Fallback for out-of-range classes
        if raw_class not in YG_CLASS_NAMES:
            logger.warning(
                "[YGPredictor] Model returned unexpected class %d — falling back to 1 (Media).",
                raw_class,
            )
            raw_class = 1

        zone_info = self._get_zone_info(location_id)

        return {
            "model_type":             "yg",
            "location_id":            location_id,
            "location_name":          zone_info["zone"],
            "borough":                zone_info["borough"],
            "vehicle_type":           vehicle_type,
            "service_mode":           service_mode,
            "predicted_class":        raw_class,
            "predicted_class_name":   YG_CLASS_NAMES[raw_class],
            "availability_description": YG_CLASS_DESCRIPTIONS[raw_class],
        }

    def predict_all(
        self,
        location_id: int,
        hour: int,
        minute: int,
        day_of_week: int,
        month: int,
    ) -> List[Dict[str, Any]]:
        """
        Predict for all three YG configurations:
          1. Yellow Taxi  (hail)
          2. Green Taxi   (hail)
          3. Green Taxi   (dispatch)
        """
        return [
            self.predict(location_id, hour, minute, day_of_week, month, "yellow", "hail"),
            self.predict(location_id, hour, minute, day_of_week, month, "green",  "hail"),
            self.predict(location_id, hour, minute, day_of_week, month, "green",  "dispatch"),
        ]


_yg_predictor: Optional[YGPredictor] = None


def get_yg_predictor() -> YGPredictor:
    """Return the YGPredictor singleton."""
    global _yg_predictor
    if _yg_predictor is None:
        _yg_predictor = YGPredictor()
    return _yg_predictor
