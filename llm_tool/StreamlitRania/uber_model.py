import joblib
import pandas as pd
import numpy as np
import os

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(MODEL_DIR, "waiting_time_lgbm.pkl")
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    from roberto import waiting_time_lgbm as _model_module
    model = _model_module.model


def is_festivo(day_of_week):
    return 1 if day_of_week >= 5 else 0


def predict_uber_waiting_time(zone: int, datetime_str: str):

    dt = pd.Timestamp(datetime_str).floor('15min')

    dow = dt.dayofweek

    X = pd.DataFrame([{
        "PULocationID": pd.Categorical(
            [zone],
            categories=list(range(1, 264))
        )[0],

        "hour_sin": np.sin(2 * np.pi * dt.hour / 24),
        "hour_cos": np.cos(2 * np.pi * dt.hour / 24),

        "minute_sin": np.sin(2 * np.pi * dt.minute / 60),
        "minute_cos": np.cos(2 * np.pi * dt.minute / 60),

        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),

        "month": dt.month,
        "is_festivo": is_festivo(dow)
    }])

    # 🔥 IMPORTANTISSIMO
    X["PULocationID"] = X["PULocationID"].astype("category")

    pred = model.predict(X)[0]

    return {
        "zone": zone,
        "datetime": datetime_str,
        "waiting_time": float(pred)
    }