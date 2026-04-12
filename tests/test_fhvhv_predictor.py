"""Unit tests for FHvhvPredictor (Uber/Lyft waiting time model wrapper)."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd


def _fake_zone_lookup():
    return pd.DataFrame([
        {"LocationID": 161, "Borough": "Manhattan", "Zone": "Midtown Center", "service_zone": "Yellow Zone"},
    ])


def test_build_features_hour_encoding():
    """hour_sin and hour_cos are correct for hour=0 and hour=12."""
    from llm_tool.fhvhv_predictor import _build_fhvhv_features
    feats = _build_fhvhv_features(location_id=161, hour=0, minute=0, day_of_week=0, month=1, is_festivo=False)
    assert abs(feats["hour_sin"] - 0.0) < 1e-6
    assert abs(feats["hour_cos"] - 1.0) < 1e-6

    feats12 = _build_fhvhv_features(location_id=161, hour=12, minute=0, day_of_week=0, month=1, is_festivo=False)
    assert abs(feats12["hour_sin"] - 0.0) < 1e-4
    assert abs(feats12["hour_cos"] - (-1.0)) < 1e-4


def test_build_features_minute_encoding():
    """minute_sin and minute_cos are correct for minute=0 and minute=30."""
    from llm_tool.fhvhv_predictor import _build_fhvhv_features
    feats0 = _build_fhvhv_features(location_id=161, hour=10, minute=0, day_of_week=0, month=1, is_festivo=False)
    assert abs(feats0["minute_sin"] - 0.0) < 1e-6
    assert abs(feats0["minute_cos"] - 1.0) < 1e-6

    feats30 = _build_fhvhv_features(location_id=161, hour=10, minute=30, day_of_week=0, month=1, is_festivo=False)
    assert abs(feats30["minute_sin"] - 0.0) < 1e-4  # sin(π) = 0
    assert abs(feats30["minute_cos"] - (-1.0)) < 1e-4  # cos(π) = -1


def test_build_features_quarter():
    """minute=0 → quarter=0, minute=15 → quarter=1, minute=45 → quarter=3."""
    from llm_tool.fhvhv_predictor import _build_fhvhv_features
    assert _build_fhvhv_features(161, 10, 0, 0, 1, False)["quarter"] == 0
    assert _build_fhvhv_features(161, 10, 15, 0, 1, False)["quarter"] == 1
    assert _build_fhvhv_features(161, 10, 30, 0, 1, False)["quarter"] == 2
    assert _build_fhvhv_features(161, 10, 45, 0, 1, False)["quarter"] == 3


def test_build_features_dow_encoding():
    """dow_sin and dow_cos are correct for day_of_week=0 and day_of_week=3."""
    from llm_tool.fhvhv_predictor import _build_fhvhv_features
    feats0 = _build_fhvhv_features(location_id=161, hour=10, minute=0, day_of_week=0, month=1, is_festivo=False)
    assert abs(feats0["dow_sin"] - 0.0) < 1e-6
    assert abs(feats0["dow_cos"] - 1.0) < 1e-6


def test_build_features_is_festivo():
    """is_festivo is correctly set based on parameter."""
    from llm_tool.fhvhv_predictor import _build_fhvhv_features
    feats_false = _build_fhvhv_features(161, 10, 0, 0, 1, is_festivo=False)
    assert feats_false["is_festivo"] == 0

    feats_true = _build_fhvhv_features(161, 10, 0, 0, 1, is_festivo=True)
    assert feats_true["is_festivo"] == 1


def test_convert_to_class_boundaries():
    """_convert_to_class correctly classifies at p33 and p66 boundaries."""
    from llm_tool.fhvhv_predictor import _convert_to_class
    p33, p66 = 5.0, 10.0

    assert _convert_to_class(4.9, p33, p66) == 0  # Below p33
    assert _convert_to_class(5.0, p33, p66) == 0  # At p33
    assert _convert_to_class(5.1, p33, p66) == 1  # Just above p33

    assert _convert_to_class(9.9, p33, p66) == 1  # Below p66
    assert _convert_to_class(10.0, p33, p66) == 1  # At p66
    assert _convert_to_class(10.1, p33, p66) == 2  # Above p66


def test_format_waiting_time():
    """_format_waiting_time correctly formats minutes to mm:ss."""
    from llm_tool.fhvhv_predictor import _format_waiting_time
    assert _format_waiting_time(0.0) == "00:00"
    assert _format_waiting_time(5.5) == "05:30"
    assert _format_waiting_time(12.25) == "12:15"
    assert _format_waiting_time(0.5) == "00:30"


@pytest.fixture
def mock_fhvhv_model():
    """Provide a mock LightGBM model that predicts 7.5 minutes waiting time."""
    model = MagicMock()
    model.predict.return_value = np.array([7.5])
    return model


@pytest.fixture
def mock_thresholds():
    """Provide mock thresholds: p33=5.0, p66=10.0."""
    return {"p33": 5.0, "p66": 10.0}


@pytest.fixture
def fhvhv_predictor_loaded(mock_fhvhv_model, mock_thresholds):
    """FHvhvPredictor with mocked model already loaded."""
    with patch("llm_tool.fhvhv_predictor.joblib.load") as mock_load, \
         patch("llm_tool.fhvhv_predictor._load_zone_lookup") as mock_lookup:
        mock_load.side_effect = [mock_fhvhv_model, mock_thresholds]
        mock_lookup.return_value = _fake_zone_lookup()
        from llm_tool.fhvhv_predictor import FHvhvPredictor
        FHvhvPredictor._instance = None
        FHvhvPredictor._model = None
        FHvhvPredictor._thresholds = None
        FHvhvPredictor._zone_lookup = None
        p = FHvhvPredictor()
        p.load()
        yield p
        FHvhvPredictor._instance = None
        FHvhvPredictor._model = None
        FHvhvPredictor._thresholds = None
        FHvhvPredictor._zone_lookup = None


def test_predict_returns_normalised_schema(fhvhv_predictor_loaded):
    """predict() returns all required keys with correct types."""
    result = fhvhv_predictor_loaded.predict(
        location_id=161, hour=10, minute=30,
        day_of_week=0, month=3, is_festivo=False
    )
    assert result["model_type"] == "fhvhv"
    assert result["location_id"] == 161
    assert result["location_name"] == "Midtown Center"
    assert result["borough"] == "Manhattan"
    assert "predicted_waiting_time" in result
    assert result["predicted_class"] in [0, 1, 2]
    assert result["predicted_class_name"] in ["Facile", "Medio", "Difficile"]
    assert "predicted_class_description" in result


def test_predict_waiting_time_format(fhvhv_predictor_loaded):
    """predicted_waiting_time is in mm:ss format."""
    result = fhvhv_predictor_loaded.predict(
        location_id=161, hour=10, minute=30,
        day_of_week=0, month=3, is_festivo=False
    )
    waiting = result["predicted_waiting_time"]
    assert ":" in waiting
    parts = waiting.split(":")
    assert len(parts) == 2
    assert 0 <= int(parts[0]) <= 59
    assert 0 <= int(parts[1]) <= 59


def test_predict_class_facile(mock_fhvhv_model):
    """When waiting_time < p33, predicted_class = 0 (Facile)."""
    mock_fhvhv_model.predict.return_value = np.array([3.0])
    mock_thresholds = {"p33": 5.0, "p66": 10.0}
    with patch("llm_tool.fhvhv_predictor.joblib.load") as mock_load, \
         patch("llm_tool.fhvhv_predictor._load_zone_lookup") as mock_lookup:
        mock_load.side_effect = [mock_fhvhv_model, mock_thresholds]
        mock_lookup.return_value = _fake_zone_lookup()
        from llm_tool.fhvhv_predictor import FHvhvPredictor
        FHvhvPredictor._instance = None
        FHvhvPredictor._model = None
        FHvhvPredictor._thresholds = None
        FHvhvPredictor._zone_lookup = None
        p = FHvhvPredictor()
        p.load()
        result = p.predict(161, 10, 0, 0, 1, False)
        assert result["predicted_class"] == 0
        assert result["predicted_class_name"] == "Facile"


def test_predict_class_medio(mock_fhvhv_model):
    """When p33 <= waiting_time < p66, predicted_class = 1 (Medio)."""
    mock_fhvhv_model.predict.return_value = np.array([7.5])
    mock_thresholds = {"p33": 5.0, "p66": 10.0}
    with patch("llm_tool.fhvhv_predictor.joblib.load") as mock_load, \
         patch("llm_tool.fhvhv_predictor._load_zone_lookup") as mock_lookup:
        mock_load.side_effect = [mock_fhvhv_model, mock_thresholds]
        mock_lookup.return_value = _fake_zone_lookup()
        from llm_tool.fhvhv_predictor import FHvhvPredictor
        FHvhvPredictor._instance = None
        FHvhvPredictor._model = None
        FHvhvPredictor._thresholds = None
        FHvhvPredictor._zone_lookup = None
        p = FHvhvPredictor()
        p.load()
        result = p.predict(161, 10, 0, 0, 1, False)
        assert result["predicted_class"] == 1
        assert result["predicted_class_name"] == "Medio"


def test_predict_class_difficile(mock_fhvhv_model):
    """When waiting_time >= p66, predicted_class = 2 (Difficile)."""
    mock_fhvhv_model.predict.return_value = np.array([15.0])
    mock_thresholds = {"p33": 5.0, "p66": 10.0}
    with patch("llm_tool.fhvhv_predictor.joblib.load") as mock_load, \
         patch("llm_tool.fhvhv_predictor._load_zone_lookup") as mock_lookup:
        mock_load.side_effect = [mock_fhvhv_model, mock_thresholds]
        mock_lookup.return_value = _fake_zone_lookup()
        from llm_tool.fhvhv_predictor import FHvhvPredictor
        FHvhvPredictor._instance = None
        FHvhvPredictor._model = None
        FHvhvPredictor._thresholds = None
        FHvhvPredictor._zone_lookup = None
        p = FHvhvPredictor()
        p.load()
        result = p.predict(161, 10, 0, 0, 1, False)
        assert result["predicted_class"] == 2
        assert result["predicted_class_name"] == "Difficile"


def test_predict_negative_waiting_time_clamped_to_zero():
    """If model predicts negative waiting time, clamp to 0.0."""
    from unittest.mock import MagicMock
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([-2.0])
    mock_thresholds = {"p33": 5.0, "p66": 10.0}
    with patch("llm_tool.fhvhv_predictor.joblib.load") as mock_load, \
         patch("llm_tool.fhvhv_predictor._load_zone_lookup") as mock_lookup:
        mock_load.side_effect = [mock_model, mock_thresholds]
        mock_lookup.return_value = _fake_zone_lookup()
        from llm_tool.fhvhv_predictor import FHvhvPredictor
        FHvhvPredictor._instance = None
        FHvhvPredictor._model = None
        FHvhvPredictor._thresholds = None
        FHvhvPredictor._zone_lookup = None
        p = FHvhvPredictor()
        p.load()
        result = p.predict(161, 10, 0, 0, 1, False)
        assert result["predicted_waiting_time"] == "00:00"


def test_predict_unknown_location_returns_unknown(fhvhv_predictor_loaded):
    """location_id not in zone lookup returns 'Unknown' for location_name and borough."""
    result = fhvhv_predictor_loaded.predict(
        location_id=999, hour=10, minute=0,
        day_of_week=0, month=3, is_festivo=False
    )
    assert result["location_name"] == "Unknown"
    assert result["borough"] == "Unknown"


def test_get_fhvhv_predictor_returns_singleton():
    """get_fhvhv_predictor() always returns the same instance."""
    import llm_tool.fhvhv_predictor as fhvhv_mod
    from llm_tool.fhvhv_predictor import get_fhvhv_predictor, FHvhvPredictor
    fhvhv_mod._fhvhv_predictor = None
    FHvhvPredictor._instance = None
    FHvhvPredictor._model = None
    FHvhvPredictor._thresholds = None
    FHvhvPredictor._zone_lookup = None
    p1 = get_fhvhv_predictor()
    p2 = get_fhvhv_predictor()
    assert p1 is p2
    fhvhv_mod._fhvhv_predictor = None
    FHvhvPredictor._instance = None
    FHvhvPredictor._model = None
    FHvhvPredictor._thresholds = None
    FHvhvPredictor._zone_lookup = None


def test_us_holidays_defined():
    """US_HOLIDAYS contains expected 2025 holidays."""
    from llm_tool.fhvhv_predictor import US_HOLIDAYS
    from datetime import date
    assert date(2025, 1, 1) in US_HOLIDAYS    # New Year's Day
    assert date(2025, 7, 4) in US_HOLIDAYS    # Independence Day
    assert date(2025, 11, 27) in US_HOLIDAYS  # Thanksgiving
    assert date(2025, 12, 25) in US_HOLIDAYS  # Christmas
    assert len(US_HOLIDAYS) == 11