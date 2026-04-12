"""Unit tests for YGPredictor (Yellow/Green taxi model wrapper)."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ─── Feature engineering helpers ─────────────────────────────────────────────

def test_build_features_hour_encoding():
    """hour_sin and hour_cos are correct for hour=0 and hour=12."""
    from llm_tool.yg_predictor import _build_yg_features
    feats = _build_yg_features(zone=161, hour=0, minute=0, day_of_week=0, month=1,
                                vehicle_type="yellow", service_mode="hail")
    assert abs(feats["hour_sin"] - 0.0) < 1e-6
    assert abs(feats["hour_cos"] - 1.0) < 1e-6

    feats12 = _build_yg_features(zone=161, hour=12, minute=0, day_of_week=0, month=1,
                                  vehicle_type="yellow", service_mode="hail")
    assert abs(feats12["hour_sin"] - 0.0) < 1e-4
    assert abs(feats12["hour_cos"] - (-1.0)) < 1e-4


def test_build_features_quarter():
    """minute=0 → quarter=0, minute=15 → quarter=1, minute=45 → quarter=3."""
    from llm_tool.yg_predictor import _build_yg_features
    assert _build_yg_features(161, 10, 0,  0, 1, "yellow", "hail")["quarter"] == 0
    assert _build_yg_features(161, 10, 15, 0, 1, "yellow", "hail")["quarter"] == 1
    assert _build_yg_features(161, 10, 30, 0, 1, "yellow", "hail")["quarter"] == 2
    assert _build_yg_features(161, 10, 45, 0, 1, "yellow", "hail")["quarter"] == 3


def test_build_features_vehicle_encoding():
    """yellow → 0, green → 1; hail → 0, dispatch → 1."""
    from llm_tool.yg_predictor import _build_yg_features
    yellow = _build_yg_features(161, 10, 0, 0, 1, "yellow", "hail")
    green  = _build_yg_features(161, 10, 0, 0, 1, "green",  "dispatch")
    assert yellow["vehicle_type"] == 0
    assert yellow["service_mode"] == 0
    assert green["vehicle_type"]  == 1
    assert green["service_mode"]  == 1


def test_build_features_invalid_vehicle():
    """Unknown vehicle_type raises ValueError."""
    from llm_tool.yg_predictor import _build_yg_features
    with pytest.raises(ValueError, match="vehicle_type non valido"):
        _build_yg_features(161, 10, 0, 0, 1, "fhvhv", "hail")


def test_build_features_invalid_service():
    """Unknown service_mode raises ValueError."""
    from llm_tool.yg_predictor import _build_yg_features
    with pytest.raises(ValueError, match="service_mode non valido"):
        _build_yg_features(161, 10, 0, 0, 1, "yellow", "unknown")


# ─── YGPredictor singleton ────────────────────────────────────────────────────

@pytest.fixture
def mock_yg_model():
    """Provide a mock LightGBM model that predicts class 2 (Alta)."""
    model = MagicMock()
    model.predict.return_value = np.array([2])
    return model


@pytest.fixture
def yg_predictor_loaded(mock_yg_model):
    """YGPredictor with mocked model already loaded."""
    with patch("llm_tool.yg_predictor.joblib.load", return_value=mock_yg_model), \
         patch("llm_tool.yg_predictor._load_zone_lookup") as mock_lookup:
        mock_lookup.return_value = _fake_zone_lookup()
        from llm_tool.yg_predictor import YGPredictor
        # Reset singleton state for test isolation
        YGPredictor._instance = None
        p = YGPredictor()
        p.load()
        yield p
        YGPredictor._instance = None


def _fake_zone_lookup():
    import pandas as pd
    return pd.DataFrame([
        {"LocationID": 161, "Borough": "Manhattan", "Zone": "Midtown Center", "service_zone": "Yellow Zone"},
    ])


def test_predict_returns_normalised_schema(yg_predictor_loaded):
    """predict() returns all required keys with correct types."""
    result = yg_predictor_loaded.predict(
        location_id=161, hour=10, minute=30,
        day_of_week=0, month=3,
        vehicle_type="yellow", service_mode="hail"
    )
    assert result["model_type"] == "yg"
    assert result["location_id"] == 161
    assert result["location_name"] == "Midtown Center"
    assert result["borough"] == "Manhattan"
    assert result["vehicle_type"] == "yellow"
    assert result["service_mode"] == "hail"
    assert result["predicted_class"] == 2
    assert result["predicted_class_name"] == "Alta"
    assert "availability_description" in result
    # No confidence or probabilities — model doesn't provide them
    assert "confidence" not in result
    assert "probabilities" not in result


def test_predict_all_returns_three_results(yg_predictor_loaded):
    """predict_all() returns exactly 3 results: yellow-hail, green-hail, green-dispatch."""
    results = yg_predictor_loaded.predict_all(
        location_id=161, hour=10, minute=30, day_of_week=0, month=3
    )
    assert len(results) == 3
    types = [(r["vehicle_type"], r["service_mode"]) for r in results]
    assert ("yellow", "hail")    in types
    assert ("green",  "hail")    in types
    assert ("green",  "dispatch") in types


def test_predict_class_out_of_range_falls_back(mock_yg_model):
    """If model returns class outside [0-2], fallback to class 1 (Media) with warning."""
    mock_yg_model.predict.return_value = np.array([5])  # Invalid
    with patch("llm_tool.yg_predictor.joblib.load", return_value=mock_yg_model), \
         patch("llm_tool.yg_predictor._load_zone_lookup") as mock_lookup:
        mock_lookup.return_value = _fake_zone_lookup()
        from llm_tool.yg_predictor import YGPredictor
        YGPredictor._instance = None
        p = YGPredictor()
        p.load()
        result = p.predict(161, 10, 0, 0, 1, "yellow", "hail")
        assert result["predicted_class"] == 1
        assert result["predicted_class_name"] == "Media"
        YGPredictor._instance = None


def test_get_yg_predictor_returns_singleton():
    """get_yg_predictor() always returns the same instance."""
    from llm_tool.yg_predictor import get_yg_predictor, YGPredictor
    YGPredictor._instance = None
    p1 = get_yg_predictor()
    p2 = get_yg_predictor()
    assert p1 is p2
    YGPredictor._instance = None
