import joblib
import pandas as pd
import numpy as np
import os

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "taxi_lgbm_model_production.pkl")
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance_production.csv")

FEATURES = [
    "zone",
    "month",
    "quarter",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "vehicle_type",
    "service_mode",
]

CATEGORICAL_FEATURES = [
    "zone",
    "month",
    "quarter",
    "vehicle_type",
    "service_mode",
]

model = joblib.load(MODEL_PATH)

try:
    feature_importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
except FileNotFoundError:
    feature_importance_df = pd.DataFrame(columns=["feature", "importance"])


def _encode_vehicle_type(vehicle_type: str) -> int:
    mapping = {
        "yellow": 0,
        "green": 1
    }
    if vehicle_type not in mapping:
        raise ValueError(f"vehicle_type non valido: {vehicle_type}")
    return mapping[vehicle_type]


def _encode_service_mode(service_mode: str) -> int:
    mapping = {
        "hail": 0,
        "dispatch": 1
    }
    if service_mode not in mapping:
        raise ValueError(f"service_mode non valido: {service_mode}")
    return mapping[service_mode]


def _build_time_features(dt: pd.Timestamp) -> dict:
    hour = dt.hour
    day_of_week = dt.dayofweek
    month = dt.month
    quarter = dt.minute // 15

    return {
        "month": month,
        "quarter": quarter,
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "day_sin": float(np.sin(2 * np.pi * day_of_week / 7)),
        "day_cos": float(np.cos(2 * np.pi * day_of_week / 7)),
        "hour": hour,
        "day_of_week": day_of_week,
    }


def _cast_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df


def _get_top_model_features(top_n: int = 5) -> list[dict]:
    if feature_importance_df.empty:
        return []

    top_df = (
        feature_importance_df
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return top_df.to_dict(orient="records")


def predict_taxi_availability(
    zone: int,
    datetime_str: str,
    vehicle_type: str,
    service_mode: str = "hail"
) -> dict:
    dt = pd.Timestamp(datetime_str)

    tf = _build_time_features(dt)

    vt = _encode_vehicle_type(vehicle_type)
    sm = _encode_service_mode(service_mode)

    X = pd.DataFrame([{
        "zone": int(zone),
        "month": int(tf["month"]),
        "quarter": int(tf["quarter"]),
        "hour_sin": tf["hour_sin"],
        "hour_cos": tf["hour_cos"],
        "day_sin": tf["day_sin"],
        "day_cos": tf["day_cos"],
        "vehicle_type": int(vt),
        "service_mode": int(sm),
    }])

    X = X[FEATURES]
    X = _cast_categorical(X)

    pred = int(model.predict(X)[0])

    return {
        "zone": int(zone),
        "datetime": datetime_str,
        "vehicle_type": vehicle_type,
        "service_mode": service_mode,
        "availability_class": pred,
        "features_used": {
            "zone": int(zone),
            "month": int(tf["month"]),
            "quarter": int(tf["quarter"]),
            "hour": int(tf["hour"]),
            "day_of_week": int(tf["day_of_week"]),
            "hour_sin": tf["hour_sin"],
            "hour_cos": tf["hour_cos"],
            "day_sin": tf["day_sin"],
            "day_cos": tf["day_cos"],
            "vehicle_type_encoded": int(vt),
            "service_mode_encoded": int(sm),
        },
        "top_model_features": _get_top_model_features(top_n=5)
    }
