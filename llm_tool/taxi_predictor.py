"""
NYC Taxi Demand Predictor - LangChain Tool.

Wraps the trained LightGBM model as a LangChain @tool for use by the LLM agent.
Uses lazy loading (singleton) to avoid loading the model until first prediction.
"""

import json
import warnings
import logging
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import joblib
import shap

from langchain_core.tools import tool
from langchain_core.tools.base import ToolException

from .config import (
    MODEL_ARTIFACTS_PATH, ZONE_DEFAULTS_PATH, OUTPUT_DIR, FEATURE_COLS,
    CLASS_NAMES, ZONE_LOOKUP_URL,
    hour_minute_to_half_bucket, ZONE_ALIASES,
)

logger = logging.getLogger(__name__)

# Local cache paths
ZONE_LOOKUP_LOCAL = OUTPUT_DIR / "zone_lookup.csv"
ZONE_STATS_PATH = OUTPUT_DIR / "zone_stats_summary.csv"

warnings.filterwarnings("ignore")


def _load_zone_lookup() -> pd.DataFrame:
    """Load the NYC taxi zone lookup table."""
    if ZONE_LOOKUP_LOCAL.exists():
        df = pd.read_csv(ZONE_LOOKUP_LOCAL)
    else:
        print(f"[TaxiPredictor] Downloading zone lookup from {ZONE_LOOKUP_URL}...")
        df = pd.read_csv(ZONE_LOOKUP_URL, storage_options={"timeout": 10})
        df.to_csv(ZONE_LOOKUP_LOCAL, index=False)
        print(f"[TaxiPredictor] Saved zone lookup to {ZONE_LOOKUP_LOCAL}")

    df.columns = ["LocationID", "Borough", "Zone", "service_zone"]
    return df


class TaxiPredictorModel:
    """Singleton wrapper for the trained LightGBM model and artifacts."""

    _instance: Optional["TaxiPredictorModel"] = None
    _model = None
    _scaler = None
    _explainer = None
    _zone_defaults = None
    _zone_lookup = None
    _zone_stats = None
    _feature_cols = None
    _class_names = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> None:
        """Load model and artifacts from disk (lazy, idempotent)."""
        if self._model is not None:
            return

        print("[TaxiPredictor] Loading model artifacts...")

        artifacts = joblib.load(MODEL_ARTIFACTS_PATH)
        self._model = artifacts["model"]
        self._scaler = artifacts["scaler"]
        self._feature_cols = artifacts["feature_cols"]
        self._class_names = artifacts.get("class_names", CLASS_NAMES)

        # Initialize SHAP explainer
        self._explainer = shap.TreeExplainer(self._model)

        self._zone_defaults = pd.read_csv(ZONE_DEFAULTS_PATH)
        self._zone_defaults = self._zone_defaults.set_index("PULocationID")

        self._zone_lookup = _load_zone_lookup()

        if ZONE_STATS_PATH.exists():
            self._zone_stats = pd.read_csv(ZONE_STATS_PATH)

        print("[TaxiPredictor] Model loaded successfully!")

    def get_zone_info(self, location_id: int) -> Dict[str, str]:
        """Get zone information (borough, zone name) for a location ID."""
        if self._zone_lookup is None:
            self.load()

        zone_info = self._zone_lookup[self._zone_lookup["LocationID"] == location_id]
        if len(zone_info) > 0:
            return {
                "borough": zone_info.iloc[0]["Borough"],
                "zone": zone_info.iloc[0]["Zone"],
                "service_zone": zone_info.iloc[0]["service_zone"],
            }
        return {"borough": "Unknown", "zone": "Unknown", "service_zone": "Unknown"}

    def get_zone_defaults(self, location_id: int) -> Dict[str, Any]:
        """Get default values for features not provided by user."""
        if self._zone_defaults is None:
            self.load()

        if location_id in self._zone_defaults.index:
            return self._zone_defaults.loc[location_id].to_dict()

        return {
            "unique_taxi_types": 3,
            "avg_trip_duration_min": 17.8,
            "borough_encoded": 6,
            "service_zone_encoded": 3,
        }

    def predict(
        self,
        location_id: int,
        half_hour_bucket: int,
        day_of_week: int,
        month: int,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Predict taxi availability for a given location and time.
        """
        if self._model is None:
            self.load()

        zone_defaults = self.get_zone_defaults(location_id)
        hour = half_hour_bucket // 2

        features = {
            "PULocationID": location_id,
            "half_hour_bucket": half_hour_bucket,
            "day_of_week": day_of_week,
            "month": month,
            "unique_taxi_types": zone_defaults["unique_taxi_types"],
            "avg_trip_duration_min": zone_defaults["avg_trip_duration_min"],
            "is_weekend": 1 if day_of_week >= 5 else 0,
            "is_rush_hour": 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0,
            "is_night": 1 if (hour >= 23) or (hour <= 5) else 0,
            "borough_encoded": zone_defaults["borough_encoded"],
            "service_zone_encoded": zone_defaults["service_zone_encoded"],
        }

        X = pd.DataFrame([features])[self._feature_cols]
        X_scaled = self._scaler.transform(X)

        # 1. Prediction
        pred_class = int(self._model.predict(X_scaled)[0])
        pred_proba = self._model.predict_proba(X_scaled)[0]
        confidence = float(pred_proba[pred_class])
        
        print(f"      [ML] Predizione: Classe {pred_class} ({self._class_names[pred_class]}) - Confidenza: {confidence:.2%}")

        # 2. SHAP Explanation
        shap_values = self._explainer.shap_values(X_scaled)
        # In multi-class, shap_values is a list of arrays (one per class)
        if isinstance(shap_values, list):
            class_shap = shap_values[pred_class][0]
        else:
            class_shap = shap_values[0][:, pred_class] if len(shap_values.shape) > 2 else shap_values[0]

        # Map SHAP values to feature names and sort by importance
        impacts = []
        for i, val in enumerate(class_shap):
            impacts.append({
                "feature": self._feature_cols[i],
                "impact": float(val),
                "direction": "positivo" if val > 0 else "negativo"
            })
        
        # Top 3 impacts by absolute value
        top_impacts = sorted(impacts, key=lambda x: abs(x["impact"]), reverse=True)[:3]

        zone_info = self.get_zone_info(location_id)
        class_names = CLASS_NAMES

        return {
            "success": True,
            "location_id": location_id,
            "location_name": zone_info["zone"],
            "borough": zone_info["borough"],
            "day_of_week": day_of_week,
            "month": month,
            "time_bucket": half_hour_bucket,
            "predicted_class": pred_class,
            "predicted_class_name": class_names[pred_class],
            "confidence": confidence,
            "top_insights": top_impacts,
            "probabilities": {
                class_names[i]: round(float(pred_proba[i]), 4) for i in range(5)
            },
            "language": language,
        }

    def get_historical_trends(self, location_id: int, day_of_week: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query historical averages for a zone."""
        if self._zone_stats is None:
            self.load()
            if self._zone_stats is None:
                return []

        mask = (self._zone_stats["PULocationID"] == location_id)
        if day_of_week is not None:
            mask &= (self._zone_stats["day_of_week"] == day_of_week)

        res = self._zone_stats[mask].copy()
        # Sort by day and hour bucket
        res = res.sort_values(["day_of_week", "half_hour_bucket"])

        return res.to_dict(orient="records")


_predictor = None


def get_predictor() -> TaxiPredictorModel:
    """Get or create the predictor singleton."""
    global _predictor
    if _predictor is None:
        _predictor = TaxiPredictorModel()
    return _predictor


def resolve_zone_id(zone_input: str, return_all: bool = False) -> Any:
    """
    Resolve a zone name or ID string to a LocationID or list of candidates.

    Args:
        zone_input: Zone name, alias, or numeric ID string
        return_all: If True, return list of {id, name} if multiple matches found.

    Returns:
        int (LocationID) or list of dicts if return_all=True and ambiguous, or None.
    """
    zone_input = zone_input.strip()

    # 1. Try as numeric ID
    try:
        loc_id = int(zone_input)
        if 1 <= loc_id <= 265:
            return loc_id
    except ValueError:
        pass

    zone_lower = zone_input.lower()
    
    # 2. Try exact alias match
    if zone_lower in ZONE_ALIASES:
        return ZONE_ALIASES[zone_lower]

    # 3. Try partial alias matches
    candidates = []
    seen_ids = set()
    for alias, loc_id in ZONE_ALIASES.items():
        if alias == zone_lower:
            return loc_id
        if zone_lower in alias or alias in zone_lower:
            if loc_id not in seen_ids:
                candidates.append({"id": loc_id, "name": alias.title()})
                seen_ids.add(loc_id)

    # 4. Try zone lookup CSV
    try:
        zl = _load_zone_lookup()
        # Case-insensitive substring match
        matches = zl[zl["Zone"].str.lower().str.contains(zone_lower, na=False)]
        for _, row in matches.iterrows():
            lid, name = int(row["LocationID"]), row["Zone"]
            if lid not in seen_ids:
                candidates.append({"id": lid, "name": name})
                seen_ids.add(lid)
    except Exception:
        pass

    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]["id"]
    
    return candidates if return_all else candidates[0]["id"]


@tool
def predict_taxi_availability(
    location_id: int,
    half_hour_bucket: int,
    day_of_week: int,
    month: int,
    language: str = "en",
) -> str:
    """
    [DEPRECATED — use predict_yellow_green_availability instead]
    Predict taxi availability for a NYC zone at a specific time.
    Includes insights (SHAP) explaining WHY the availability is what it is.
    """
    # ─── Parameter validation ────────────────────────────────────────────
    errors = []
    if location_id is None or (not isinstance(location_id, int)) or not (1 <= location_id <= 265):
        errors.append(f"location_id must be an integer between 1 and 265")
    if half_hour_bucket is None or (not isinstance(half_hour_bucket, int)) or not (0 <= half_hour_bucket <= 47):
        errors.append(f"half_hour_bucket must be an integer between 0 and 47")
    if day_of_week is None or (not isinstance(day_of_week, int)) or not (0 <= day_of_week <= 6):
        errors.append(f"day_of_week must be an integer between 0 and 6")
    if month is None or (not isinstance(month, int)) or not (1 <= month <= 12):
        errors.append(f"month must be an integer between 1 and 12")
    
    if errors:
        raise ToolException("; ".join(errors))

    try:
        predictor = get_predictor()
        result = predictor.predict(
            location_id=location_id,
            half_hour_bucket=half_hour_bucket,
            day_of_week=day_of_week,
            month=month,
            language=language,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise ToolException(f"Errore: {e}")


@tool
def get_historical_trends(
    location_id: int,
    day_of_week: Optional[int] = None,
) -> str:
    """
    Retrieves historical taxi availability trends for a specific NYC zone.
    Use this to answer questions about 'usual' patterns or general borough availability.
    
    Args:
        location_id: NYC taxi zone ID (1-265).
        day_of_week: Optional day of week (0-6). If omitted, returns weekly average.
    """
    try:
        predictor = get_predictor()
        trends = predictor.get_historical_trends(location_id, day_of_week)
        
        if not trends:
            return "Nessun dato storico trovato per questa zona."
            
        # Aggregate by hour for brevity in the LLM prompt
        df = pd.DataFrame(trends)
        df["hour"] = df["half_hour_bucket"] // 2
        hourly = df.groupby("hour")["availability_index"].mean().reset_index()
        
        zone_info = predictor.get_zone_info(location_id)
        
        summary = {
            "location_id": location_id,
            "location_name": zone_info.get("zone", "?"),
            "borough": zone_info.get("borough", ""),
            "day": day_of_week,
            "hourly_avg_availability": hourly.to_dict(orient="records"),
            "note": "Availability Index: 0.0 (Empty) -> 1.0 (Full)"
        }
        
        return json.dumps(summary, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Trend error: {e}", exc_info=True)
        raise ToolException(f"Errore trend: {e}")


@tool
def predict_yellow_green_availability(
    location_id: int,
    hour: int,
    minute: int,
    day_of_week: int,
    month: int,
    vehicle_type: str = "all",
    language: str = "it",
) -> str:
    """
    Predict taxi availability for Yellow and/or Green taxis in a NYC zone.

    Args:
        location_id:  NYC taxi zone ID (1-265).
        hour:         Hour of day (0-23).
        minute:       Minute of hour (0-59).
        day_of_week:  0=Monday ... 6=Sunday.
        month:        Month (1-12).
        vehicle_type: "yellow" | "green" | "all" (default: "all" — returns all three types).
        language:     Response language ("it" or "en").
    """
    import json as _json
    from langchain_core.tools.base import ToolException as _TE

    errors = []
    if not isinstance(location_id, int) or not (1 <= location_id <= 265):
        errors.append("location_id must be an integer between 1 and 265")
    if not isinstance(hour, int) or not (0 <= hour <= 23):
        errors.append("hour must be between 0 and 23")
    if not isinstance(minute, int) or not (0 <= minute <= 59):
        errors.append("minute must be between 0 and 59")
    if not isinstance(day_of_week, int) or not (0 <= day_of_week <= 6):
        errors.append("day_of_week must be between 0 and 6")
    if not isinstance(month, int) or not (1 <= month <= 12):
        errors.append("month must be between 1 and 12")
    if errors:
        raise _TE("; ".join(errors))

    try:
        from .yg_predictor import get_yg_predictor
        yg = get_yg_predictor()

        if vehicle_type == "all" or vehicle_type is None:
            results = yg.predict_all(location_id, hour, minute, day_of_week, month)
        elif vehicle_type == "yellow":
            results = [yg.predict(location_id, hour, minute, day_of_week, month, "yellow", "hail")]
        elif vehicle_type == "green":
            results = [
                yg.predict(location_id, hour, minute, day_of_week, month, "green", "hail"),
                yg.predict(location_id, hour, minute, day_of_week, month, "green", "dispatch"),
            ]
        else:
            # Unknown type — fall back to all
            results = yg.predict_all(location_id, hour, minute, day_of_week, month)

        return _json.dumps({"model": "yg", "results": results}, ensure_ascii=False)
    except Exception as e:
        logger.error("[YG tool] %s", e, exc_info=True)
        raise _TE(f"Errore YG: {e}")


@tool
def predict_fhvhv_availability(
    location_id: int,
    hour: int,
    minute: int,
    day_of_week: int,
    month: int,
    language: str = "it",
) -> str:
    """
    Predict taxi availability for FHVHV vehicles (Uber, Lyft, rideshare) in a NYC zone.
    NOTE: This model is not yet available — returns a coming-soon notice.

    Args:
        location_id:  NYC taxi zone ID (1-265).
        hour:         Hour of day (0-23).
        minute:       Minute of hour (0-59).
        day_of_week:  0=Monday ... 6=Sunday.
        month:        Month (1-12).
        language:     Response language ("it" or "en").
    """
    import json as _json
    from .i18n import get_msg
    msg = get_msg(language, "fhvhv_coming_soon")
    return _json.dumps({"model": "fhvhv", "coming_soon": True, "message": msg}, ensure_ascii=False)

