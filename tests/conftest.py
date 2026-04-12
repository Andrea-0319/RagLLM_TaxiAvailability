"""
Pytest configuration and shared fixtures for NYC Taxi Bot tests.
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_params() -> Dict[str, Any]:
    """Sample parameters for testing."""
    return {
        "location_id": 161,
        "month": 3,
        "day_of_week": 0,
        "hour": 8,
        "minute": 0,
    }


@pytest.fixture
def chat_history_empty() -> List:
    """Empty chat history."""
    return []


@pytest.fixture
def chat_history_single() -> List:
    """Single turn chat history."""
    return [
        HumanMessage(content="Quanti taxi a Midtown?"),
        AIMessage(content="Mi serve sapere anche l'orario. Quale ora ti interessa?"),
    ]


@pytest.fixture
def chat_history_multi_turn() -> List:
    """Multi-turn chat history with params."""
    return [
        HumanMessage(content="Quanti taxi a JFK?"),
        AIMessage(content="JFK - Sabato - Notte: 🔴 Molto Difficile (confidenza: 82.3%)"),
        HumanMessage(content="e lunedi alle 8 di mattina?"),
    ]


@pytest.fixture
def mock_llm():
    """Mock LLM that returns predetermined responses."""
    def create_mock_response(response_text: str):
        mock_msg = MagicMock()
        mock_msg.content = response_text
        return mock_msg

    with patch("llm_tool.llm_factory.get_llm") as mock:
        mock_instance = MagicMock()
        mock_instance.invoke = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_llm_with_response(mock_llm, request):
    """Mock LLM with specific response - use parametrize in tests."""
    def set_response(response_text: str):
        mock_msg = MagicMock()
        mock_msg.content = response_text
        mock_llm.invoke.return_value = mock_msg
    return set_response


@pytest.fixture
def mock_llm_extract_success():
    """Mock LLM for extraction that returns valid JSON."""
    with patch("llm_tool.input_validator.InputValidator._get_llm") as mock:
        mock_instance = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = '{"zone": "midtown", "month": 3, "day_of_week": 0, "hour": 8, "minute": null}'
        mock_instance.invoke.return_value = mock_resp
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_llm_extract_failure():
    """Mock LLM for extraction that fails."""
    with patch("llm_tool.input_validator.InputValidator._get_llm") as mock:
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = Exception("LLM API error")
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_predictor():
    """Mock TaxiPredictor."""
    with patch("llm_tool.agent.get_predictor") as mock:
        predictor = MagicMock()
        predictor.predict.return_value = {
            "success": True,
            "location_id": 161,
            "location_name": "Midtown Center",
            "borough": "Manhattan",
            "day_of_week": 0,
            "month": 3,
            "time_bucket": 16,
            "predicted_class": 2,
            "predicted_class_name": "Medio",
            "confidence": 0.75,
            "top_insights": [
                {"feature": "is_rush_hour", "impact": 0.5, "direction": "positivo"},
                {"feature": "hour", "impact": -0.3, "direction": "negativo"},
            ],
            "probabilities": {
                "Molto Difficile": 0.1,
                "Difficile": 0.2,
                "Medio": 0.4,
                "Facile": 0.2,
                "Molto Facile": 0.1,
            },
            "language": "it",
        }
        mock.return_value = predictor
        yield predictor


@pytest.fixture
def mock_historical_trends():
    """Mock historical trends."""
    with patch("llm_tool.agent.get_historical_trends") as mock:
        mock.invoke.return_value = '{"location_id": 161, "hourly_avg_availability": [{"hour": 8, "avg": 0.3}, {"hour": 9, "avg": 0.5}]}'
        yield mock


@pytest.fixture
def mock_yg_predictor():
    """Mock YGPredictor that returns deterministic YG results."""
    with patch("llm_tool.agent.get_yg_predictor") as mock:
        yg = MagicMock()
        _yg_result = {
            "model_type": "yg",
            "location_id": 161,
            "location_name": "Midtown Center",
            "borough": "Manhattan",
            "vehicle_type": "yellow",
            "service_mode": "hail",
            "predicted_class": 2,
            "predicted_class_name": "Alta",
            "availability_description": "trovare un taxi è generalmente facile in questa zona e fascia oraria",
        }
        yg.predict.return_value = _yg_result
        yg.predict_all.return_value = [
            _yg_result,
            {**_yg_result, "vehicle_type": "green", "service_mode": "hail", "predicted_class": 0, "predicted_class_name": "Bassa"},
            {**_yg_result, "vehicle_type": "green", "service_mode": "dispatch", "predicted_class": 1, "predicted_class_name": "Media"},
        ]
        
        def predict_with_vehicle(location_id, hour, minute, day_of_week, month, vehicle_type, service_mode):
            result = {**_yg_result, "vehicle_type": vehicle_type, "service_mode": service_mode}
            if vehicle_type == "green":
                if service_mode == "hail":
                    result["predicted_class"] = 0
                    result["predicted_class_name"] = "Bassa"
                else:
                    result["predicted_class"] = 1
                    result["predicted_class_name"] = "Media"
            return result
        
        yg.predict.side_effect = predict_with_vehicle
        mock.return_value = yg
        yield yg


@pytest.fixture
def mock_fhvhv_predictor():
    """Mock FHvhvPredictor that returns deterministic FHVHV results."""
    with patch("llm_tool.agent.get_fhvhv_predictor") as mock:
        fhvhv = MagicMock()
        _fhvhv_result = {
            "model_type": "fhvhv",
            "location_id": 161,
            "location_name": "Midtown Center",
            "borough": "Manhattan",
            "predicted_waiting_time": "05:30",
            "predicted_class": 1,
            "predicted_class_name": "Medio",
            "predicted_class_description": "la disponibilità degli FHVHV è intermedia in questa zona e fascia oraria",
        }
        fhvhv.predict.return_value = _fhvhv_result
        mock.return_value = fhvhv
        yield fhvhv


@pytest.fixture
def validator():
    """Get InputValidator instance."""
    from llm_tool import get_validator
    return get_validator()


@pytest.fixture
def agent():
    """Get TaxiAgent instance."""
    from llm_tool import get_agent
    return get_agent()


@pytest.fixture
def predictor():
    """Get TaxiPredictor instance (lazy loaded)."""
    from llm_tool import get_predictor
    return get_predictor()


# ─── Utility Functions ─────────────────────────────────────────────────────────

def create_state(
    messages: List = None,
    intent: str = "predict",
    params: Dict = None,
    results: List = None,
) -> Dict[str, Any]:
    """Create a test state for agent nodes."""
    return {
        "messages": messages or [],
        "intent": intent,
        "current_params": params or {},
        "results": results or [],
        "validation_errors": [],
        "hour_range": [],
        "next_step": "",
        "language": "it",
    }


# ─── Custom Assertions ─────────────────────────────────────────────────────────

def assert_valid_prediction(result: Dict[str, Any]) -> None:
    """Assert that a prediction result has valid structure."""
    assert "success" in result, "FAIL: Missing 'success' field"
    assert "predicted_class" in result, "FAIL: Missing 'predicted_class'"
    assert 0 <= result["predicted_class"] <= 4, f"FAIL: predicted_class={result['predicted_class']} outside [0-4]"
    assert 0 <= result["confidence"] <= 1, f"FAIL: confidence={result['confidence']} outside [0,1]"
    assert "probabilities" in result, "FAIL: Missing 'probabilities'"
    assert isinstance(result["probabilities"], dict), "FAIL: probabilities not a dict"


def assert_valid_extraction(extracted: Dict[str, Any]) -> None:
    """Assert that extracted parameters have valid structure."""
    required_keys = ["zone", "month", "day_of_week", "hour", "minute"]
    for key in required_keys:
        assert key in extracted, f"FAIL: Missing key '{key}' in extraction"


def assert_has_failure_message(response: Dict[str, Any]) -> None:
    """Assert response contains error/failure information."""
    text = response.get("text", "")
    has_failure = any(
        marker in text.lower() 
        for marker in ["errore", "error", "problema", "failed", "nessun"]
    )
    assert has_failure, f"FAIL: Expected failure message but got: {text[:100]}"