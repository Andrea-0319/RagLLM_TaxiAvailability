"""
Diagnostic script: Test YG model predictions across different hours.
"""
import sys
import os
from pathlib import Path

# Project root is parent of scripts/
project_root = Path(__file__).parent.parent

# Add global Python site-packages for lightgbm (append at end, after anaconda packages)
python311_path = Path(r"C:\Users\andre\AppData\Local\Programs\Python\Python311\Lib\site-packages")
if str(python311_path) not in sys.path:
    sys.path.append(str(python311_path))

import numpy as np
import pandas as pd
import joblib

# Load config directly without importing llm_tool package (to avoid shap/ sklearn issues)
def load_config_directly():
    """Load only YG config values without triggering package imports."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('config', project_root / 'llm_tool' / 'config.py')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

config = load_config_directly()
YG_MODEL_PATH = config.YG_MODEL_PATH
YG_CLASS_NAMES = config.YG_CLASS_NAMES
YG_CLASS_DESCRIPTIONS = config.YG_CLASS_DESCRIPTIONS

# Also load zone lookup loader from taxi_predictor without importing package
def load_zone_lookup():
    """Load zone lookup CSV."""
    import urllib.request
    zone_lookup_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    cache_path = project_root / "output" / "taxi_zone_lookup.csv"
    
    if not cache_path.exists():
        print(f"Downloading zone lookup from {zone_lookup_url} ...")
        urllib.request.urlretrieve(zone_lookup_url, cache_path)
    
    return pd.read_csv(cache_path)

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
):
    """Compute the 9 features expected by the YG model."""
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

    _instance = None
    _model = None
    _zone_lookup = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self):
        """Load model from disk (lazy, idempotent)."""
        if self._model is not None:
            return
        print(f"[YGPredictor] Loading model from {YG_MODEL_PATH} ...")
        self._model = joblib.load(YG_MODEL_PATH)
        self._zone_lookup = load_zone_lookup()
        print("[YGPredictor] Model loaded.")

    def _get_zone_info(self, location_id: int):
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
    ):
        """Predict taxi availability for one vehicle type / service mode."""
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


def diagnose_hours():
    print("=" * 60)
    print("DIAGNOSTIC: YG Model - Hour Range Predictions")
    print("=" * 60)
    
    predictor = YGPredictor()
    predictor.load()
    
    # Test zone: Manhattan (LocationID 100 is typical)
    test_location = 100
    day_of_week = 1  # Monday
    month = 1
    
    print(f"\nTest Location: {test_location}")
    print(f"Day of Week: {day_of_week} (Monday)")
    print(f"Month: {month}")
    print("-" * 60)
    
    # Test different hours
    hours_to_test = [6, 8, 10, 12, 14, 16, 18, 20, 22]
    results = []
    
    for hour in hours_to_test:
        result = predictor.predict(
            location_id=test_location,
            hour=hour,
            minute=0,
            day_of_week=day_of_week,
            month=month,
            vehicle_type="yellow",
            service_mode="hail",
        )
        results.append({
            "hour": hour,
            "class": result["predicted_class"],
            "class_name": result["predicted_class_name"],
        })
        print(f"Hour {hour:02d}:00 -> Class {result['predicted_class']} ({result['predicted_class_name']})")
    
    print("-" * 60)
    
    # Check if all predictions are the same
    classes = [r["class"] for r in results]
    unique_classes = set(classes)
    
    print(f"\nUnique classes predicted: {unique_classes}")
    print(f"All same: {len(unique_classes) == 1}")
    
    # Check hour_sin/hour_cos values for different hours
    print("\n" + "=" * 60)
    print("Hour Cyclical Encoding Values:")
    print("=" * 60)
    
    for hour in hours_to_test:
        feats = _build_yg_features(
            zone=test_location, hour=hour, minute=0,
            day_of_week=day_of_week, month=month,
            vehicle_type="yellow", service_mode="hail",
        )
        print(f"Hour {hour:02d}: hour_sin={feats['hour_sin']:.4f}, hour_cos={feats['hour_cos']:.4f}")
    
    return len(unique_classes) > 1

if __name__ == "__main__":
    success = diagnose_hours()
    print("\n" + "=" * 60)
    if success:
        print("RESULT: Model produces DIFFERENT predictions across hours")
    else:
        print("RESULT: Model produces SAME predictions across hours - ISSUE DETECTED!")
    print("=" * 60)