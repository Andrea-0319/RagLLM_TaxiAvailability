"""
Comprehensive unit tests for LangGraph agent nodes.

Tests cover:
- intent_classifier_node: Intent detection (predict/trend/oos), fast-path, multi-turn
- extractor_node: Parameter extraction, hour range detection, multi-turn merge
- guardrail_node: Validation logic, disambiguation, errors
- predictor_node: Single prediction, trends, hour range
- formatter_node: Response generation, OOS, errors, disambiguation
- _build_template: Template generation logic
"""
import sys
sys.path.insert(0, r'C:\Users\andre\Desktop\Progetto_Accenture')
import json
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from llm_tool.agent import (
    intent_classifier_node, extractor_node, guardrail_node, 
    predictor_node, formatter_node, _build_template, AgentState
)
from llm_tool.config import CLASS_NAMES, CLASS_EMOJIS


@pytest.fixture
def mock_llm_predict():
    """Mock LLM for intent prediction."""
    with patch("llm_tool.agent.get_llm") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        mock_instance.invoke = MagicMock()
        yield mock_instance


def mock_llm_with_intent_response(mock_llm_predict, intent: str):
    """Helper to set intent response."""
    msg = MagicMock()
    msg.content = f'{{"intent": "{intent}"}}'
    mock_llm_predict.invoke.return_value = msg


@pytest.fixture
def mock_llm_oos():
    """Mock LLM for OOS responses."""
    with patch("llm_tool.agent.get_llm") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        
        def set_response(response_content: str):
            msg = MagicMock()
            msg.content = response_content
            mock_instance.invoke.return_value = msg
        
        mock_instance.set_response = set_response
        yield mock_instance


@pytest.fixture
def mock_llm_insight():
    """Mock LLM for insight generation."""
    with patch("llm_tool.agent.get_llm") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        
        def set_response(response_content: str):
            msg = MagicMock()
            msg.content = response_content
            mock_instance.invoke.return_value = msg
        
        mock_instance.set_response = set_response
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
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = json.dumps({
            "location_id": 161,
            "hourly_avg_availability": [
                {"hour": 8, "avg": 0.3},
                {"hour": 9, "avg": 0.5},
                {"hour": 10, "avg": 0.6},
            ]
        })
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def base_state():
    """Base state for testing."""
    return {
        "messages": [],
        "intent": "predict",
        "current_params": {},
        "results": [],
        "validation_errors": [],
        "hour_range": [],
        "next_step": "",
        "language": "it",
    }


def run_test(name: str, state_in: dict, expected: dict, actual: dict):
    """Print test results in standard format."""
    msg_content = state_in.get("messages", [{}])[-1].content if state_in.get("messages") else "N/A"
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"Input message: {msg_content[:100]}...")
    print(f"Expected: {expected}")
    print(f"Actual:   {actual}")
    
    for key, exp_val in expected.items():
        assert actual.get(key) == exp_val, f"MISMATCH {key}: expected={exp_val}, actual={actual.get(key)}"
    print(f"\nPASS: {name}")


# ─── INTENT CLASSIFIER NODE TESTS ───────────────────────────────────────────────────────────────

def test_intent_predict_simple(mock_llm_predict, base_state):
    """
    Test: Simple prediction query maps to 'predict' intent.
    Justification: Basic taxi availability query should trigger prediction workflow.
    """
    msg = MagicMock()
    msg.content = '{"intent": "predict"}'
    mock_llm_predict.invoke.return_value = msg
    
    state = base_state.copy()
    state["messages"] = [HumanMessage(content="Quanti taxi a Midtown?")]
    
    actual = intent_classifier_node(state)
    expected = {"intent": "predict"}
    
    return run_test("intent_predict_simple", state, expected, actual)
    

def test_intent_trend_question(mock_llm_predict, base_state):
    """
    Test: Trend query maps to 'trend' intent.
    Justification: Historical trend queries should use trend analysis workflow.
    """
    msg = MagicMock()
    msg.content = '{"intent": "trend"}'
    mock_llm_predict.invoke.return_value = msg
    
    state = base_state.copy()
    state["messages"] = [HumanMessage(content="Come sono di solito i trend a JFK?")]
    
    actual = intent_classifier_node(state)
    expected = {"intent": "trend"}
    
    return run_test("intent_trend_question", state, expected, actual)


def test_intent_oos_greeting(mock_llm_predict, base_state):
    """
    Test: Greeting maps to 'oos' intent.
    Justification: Non-taxi related queries should be handled as out of scope.
    """
    msg = MagicMock()
    msg.content = '{"intent": "oos"}'
    mock_llm_predict.invoke.return_value = msg
    
    state = base_state.copy()
    state["messages"] = [HumanMessage(content="Ciao come stai?")]
    
    actual = intent_classifier_node(state)
    expected = {"intent": "oos"}
    
    return run_test("intent_oos_greeting", state, expected, actual)


def test_intent_oos_weather(mock_llm_predict, base_state):
    """
    Test: Weather query maps to 'oos' intent.
    Justification: Weather queries are not part of taxi prediction domain.
    """
    msg = MagicMock()
    msg.content = '{"intent": "oos"}'
    mock_llm_predict.invoke.return_value = msg
    
    state = base_state.copy()
    state["messages"] = [HumanMessage(content="Che tempo fa oggi?")]
    
    actual = intent_classifier_node(state)
    expected = {"intent": "oos"}
    
    return run_test("intent_oos_weather", state, expected, actual)


def test_intent_fastpath_zone_id(base_state):
    """
    Test: Zone ID fast-path bypasses LLM and returns 'predict' directly.
    Justification: Internal commands (zona id N) should skip intent classification.
    """
    state = base_state.copy()
    state["messages"] = [HumanMessage(content="zona id 161")]
    
    actual = intent_classifier_node(state)
    expected = {"intent": "predict"}
    
    return run_test("intent_fastpath_zone_id", state, expected, actual)


def test_intent_followup_detection(mock_llm_predict, base_state):
    """
    Test: Follow-up query after prediction uses context to detect intent.
    Justification: Multi-turn context helps classify ambiguous queries like "e alle 17:30?".
    """
    msg = MagicMock()
    msg.content = '{"intent": "predict"}'
    mock_llm_predict.invoke.return_value = msg
    
    state = base_state.copy()
    state["messages"] = [
        HumanMessage(content="Quanti taxi a Midtown?"),
        AIMessage(content="Midtown - 8:00 - Lunedi - Marzo"),
        HumanMessage(content="e alle 17:30?"),
    ]
    
    actual = intent_classifier_node(state)
    expected = {"intent": "predict"}
    
    return run_test("intent_followup_detection", state, expected, actual)


# ─── EXTRACTOR NODE TESTS ──────────────────────────────────────────────────────────────

def test_extractor_basic_params(base_state):
    """
    Test: Basic parameter extraction works.
    Justification: Core functionality: extract location, time, date from user input.
    """
    with patch("llm_tool.agent.get_validator") as mock_val:
        validator = MagicMock()
        validator.extract.return_value = {"zone": "161", "month": 3, "day_of_week": 0, "hour": 8, "minute": 0}
        validator.validate_and_resolve.return_value = {
            "zone": "161", "month": 3, "day_of_week": 0, "hour": 8, "minute": 0,
            "location_id": 161, "candidates": []
        }
        mock_val.return_value = validator
        
        state = base_state.copy()
        state["messages"] = [HumanMessage(content="Quanti taxi a Midtown domani alle 8?")]
        state["intent"] = "predict"
        
        actual = extractor_node(state)
        
        expected_keys = ["current_params", "next_step"]
        assert all(k in actual for k in expected_keys), f"Missing keys in extractor output"
        assert actual["next_step"] in ["guardrail", "format"], f"Invalid next_step: {actual['next_step']}"
        
        print(f"\nPASS: extractor_basic_params")
        return True


def test_extractor_hour_range_pomeriggio(base_state):
    """
    Test: "pomeriggio" keyword maps to hour_range [14-18].
    Justification: Italian time-of-day keywords should map to hours.
    """
    with patch("llm_tool.agent.get_validator") as mock_val:
        validator = MagicMock()
        validator.extract.return_value = {"zone": "161", "month": None, "day_of_week": None, "hour": None, "minute": None}
        validator.validate_and_resolve.return_value = {
            "zone": "161", "month": None, "day_of_week": None, "hour": None, "minute": None,
            "location_id": 161, "candidates": []
        }
        mock_val.return_value = validator
        
        state = base_state.copy()
        state["messages"] = [HumanMessage(content="Quanti taxi a Midtown nel pomeriggio?")]
        state["intent"] = "predict"
        
        actual = extractor_node(state)
        
        expected = {"hour_range": [14, 15, 16, 17, 18]}
        
        passed = actual.get("hour_range") == expected["hour_range"]
        print(f"\n{'PASS' if passed else 'FAIL'}: extractor_hour_range_pomeriggio")
        print(f"  Expected hour_range: {expected['hour_range']}")
        print(f"  Actual hour_range: {actual.get('hour_range')}")
        return passed


def test_extractor_hour_range_mattina(base_state):
    """
    Test: "mattina" keyword maps to hour_range [7-11].
    Justification: Italian morning keyword maps correctly.
    """
    with patch("llm_tool.agent.get_validator") as mock_val:
        validator = MagicMock()
        validator.extract.return_value = {"zone": "161", "month": None, "day_of_week": None, "hour": None, "minute": None}
        validator.validate_and_resolve.return_value = {
            "zone": "161", "month": None, "day_of_week": None, "hour": None, "minute": None,
            "location_id": 161, "candidates": []
        }
        mock_val.return_value = validator
        
        state = base_state.copy()
        state["messages"] = [HumanMessage(content="Quanti taxi a Midtown di mattina?")]
        state["intent"] = "predict"
        
        actual = extractor_node(state)
        
        expected = {"hour_range": [7, 8, 9, 10, 11]}
        
        passed = actual.get("hour_range") == expected["hour_range"]
        print(f"\n{'PASS' if passed else 'FAIL'}: extractor_hour_range_mattina")
        print(f"  Expected hour_range: {expected['hour_range']}")
        print(f"  Actual hour_range: {actual.get('hour_range')}")
        return passed


def test_extractor_hour_range_sera(base_state):
    """
    Test: "sera" keyword maps to hour_range [19-22].
    Justification: Italian evening keyword maps correctly.
    """
    with patch("llm_tool.agent.get_validator") as mock_val:
        validator = MagicMock()
        validator.extract.return_value = {"zone": "161", "month": None, "day_of_week": None, "hour": None, "minute": None}
        validator.validate_and_resolve.return_value = {
            "zone": "161", "month": None, "day_of_week": None, "hour": None, "minute": None,
            "location_id": 161, "candidates": []
        }
        mock_val.return_value = validator
        
        state = base_state.copy()
        state["messages"] = [HumanMessage(content="Quanti taxi a Midtown di sera?")]
        state["intent"] = "predict"
        
        actual = extractor_node(state)
        
        expected = {"hour_range": [19, 20, 21, 22]}
        
        passed = actual.get("hour_range") == expected["hour_range"]
        print(f"\n{'PASS' if passed else 'FAIL'}: extractor_hour_range_sera")
        print(f"  Expected hour_range: {expected['hour_range']}")
        print(f"  Actual hour_range: {actual.get('hour_range')}")
        return passed


def test_extractor_hour_range_notte(base_state):
    """
    Test: "notte" keyword maps to hour_range [23, 0, 1, 2, 3].
    Justification: Italian night keyword spans midnight.
    """
    with patch("llm_tool.agent.get_validator") as mock_val:
        validator = MagicMock()
        validator.extract.return_value = {"zone": "161", "month": None, "day_of_week": None, "hour": None, "minute": None}
        validator.validate_and_resolve.return_value = {
            "zone": "161", "month": None, "day_of_week": None, "hour": None, "minute": None,
            "location_id": 161, "candidates": []
        }
        mock_val.return_value = validator
        
        state = base_state.copy()
        state["messages"] = [HumanMessage(content="Quanti taxi a Midtown di notte?")]
        state["intent"] = "predict"
        
        actual = extractor_node(state)
        
        expected = {"hour_range": [23, 0, 1, 2, 3]}
        
        passed = actual.get("hour_range") == expected["hour_range"]
        print(f"\n{'PASS' if passed else 'FAIL'}: extractor_hour_range_notte")
        print(f"  Expected hour_range: {expected['hour_range']}")
        print(f"  Actual hour_range: {actual.get('hour_range')}")
        return passed


def test_extractor_oos_skips_extraction(base_state):
    """
    Test: OOS intent skips extraction and goes directly to format.
    Justification: Conversational responses skip parameter extraction.
    """
    with patch("llm_tool.agent.get_validator") as mock_val:
        validator = MagicMock()
        mock_val.return_value = validator
        
        state = base_state.copy()
        state["messages"] = [HumanMessage(content="Ciao come stai?")]
        state["intent"] = "oos"
        
        actual = extractor_node(state)
        expected = {"next_step": "format"}
        
        passed = actual.get("next_step") == expected["next_step"]
        print(f"\n{'PASS' if passed else 'FAIL'}: extractor_oos_skips_extraction")
        print(f"  Expected next_step: {expected['next_step']}")
        print(f"  Actual next_step: {actual.get('next_step')}")
        return passed


def test_extractor_multi_turn_merge(base_state):
    """
    Test: New params override previous session params.
    Justification: Multi-turn conversation should carry over and update params.
    """
    with patch("llm_tool.agent.get_validator") as mock_val:
        validator = MagicMock()
        validator.extract.return_value = {"zone": "100", "month": 5, "day_of_week": None, "hour": 12, "minute": 0}
        validator.validate_and_resolve.return_value = {
            "zone": "100", "month": 5, "day_of_week": 2, "hour": 12, "minute": 0,
            "location_id": 100, "candidates": []
        }
        mock_val.return_value = validator
        
        state = base_state.copy()
        state["messages"] = [HumanMessage(content="zona 100, maggio, ore 12")]
        state["intent"] = "predict"
        state["current_params"] = {"location_id": 161, "month": 3, "day_of_week": 0, "hour": 8, "minute": 0}
        
        actual = extractor_node(state)
        
        passed = actual.get("current_params", {}).get("location_id") == 100
        print(f"\n{'PASS' if passed else 'FAIL'}: extractor_multi_turn_merge")
        print(f"  Expected location_id: 100")
        print(f"  Actual location_id: {actual.get('current_params', {}).get('location_id')}")
        return passed


# ─── GUARDRAIL NODE TESTS ────────────────────────────────────────────────────────

def test_guardrail_valid_params(base_state):
    """
    Test: Valid params route to predictor.
    Justification: Core validation: all required params valid.
    """
    state = base_state.copy()
    state["current_params"] = {"location_id": 161, "month": 3, "day_of_week": 0, "hour": 8, "minute": 0}
    state["candidates"] = []
    
    actual = guardrail_node(state)
    expected = {"next_step": "predict"}
    
    passed = actual.get("next_step") == expected["next_step"]
    print(f"\n{'PASS' if passed else 'FAIL'}: guardrail_valid_params")
    print(f"  Expected next_step: {expected['next_step']}")
    print(f"  Actual next_step: {actual.get('next_step')}")
    return passed


def test_guardrail_missing_zone_no_candidates(base_state):
    """
    Test: Missing zone without candidates asks user for zone.
    Justification: Disambiguate when zone cannot be determined.
    """
    state = base_state.copy()
    state["current_params"] = {"month": 3, "day_of_week": 0, "hour": 8, "minute": 0}
    state["candidates"] = []
    
    actual = guardrail_node(state)
    expected = {"next_step": "ask_zone"}
    
    passed = actual.get("next_step") == expected["next_step"]
    print(f"\n{'PASS' if passed else 'FAIL'}: guardrail_missing_zone_no_candidates")
    print(f"  Expected next_step: {expected['next_step']}")
    print(f"  Actual next_step: {actual.get('next_step')}")
    return passed


def test_guardrail_missing_zone_with_candidates(base_state):
    """
    Test: Missing zone with candidates triggers disambiguation.
    Justification: When multiple zone matches exist, ask user to choose.
    """
    state = base_state.copy()
    state["current_params"] = {"month": 3, "day_of_week": 0, "hour": 8, "minute": 0}
    state["candidates"] = [
        {"location_id": 100, "location_name": "Midtown", "borough": "Manhattan"},
        {"location_id": 161, "location_name": "Midtown East", "borough": "Manhattan"},
    ]
    
    actual = guardrail_node(state)
    expected = {"next_step": "disambiguate"}
    
    passed = actual.get("next_step") == expected["next_step"]
    print(f"\n{'PASS' if passed else 'FAIL'}: guardrail_missing_zone_with_candidates")
    print(f"  Expected next_step: {expected['next_step']}")
    print(f"  Actual next_step: {actual.get('next_step')}")
    return passed


def test_guardrail_invalid_zone_id(base_state):
    """
    Test: Invalid zone ID (>265) generates validation error.
    Justification: Zone IDs must be in valid range 1-265.
    """
    state = base_state.copy()
    state["current_params"] = {"location_id": 999, "month": 3, "day_of_week": 0, "hour": 8, "minute": 0}
    state["candidates"] = []
    
    actual = guardrail_node(state)
    
    passed = len(actual.get("validation_errors", [])) > 0 and actual.get("next_step") == "format"
    print(f"\n{'PASS' if passed else 'FAIL'}: guardrail_invalid_zone_id")
    print(f"  Expected validation_errors > 0")
    print(f"  Actual validation_errors: {actual.get('validation_errors')}")
    return passed


def test_guardrail_invalid_hour(base_state):
    """
    Test: Invalid hour (>23) generates validation error.
    Justification: Hour must be 0-23.
    """
    state = base_state.copy()
    state["current_params"] = {"location_id": 161, "month": 3, "day_of_week": 0, "hour": 25, "minute": 0}
    state["candidates"] = []
    
    actual = guardrail_node(state)
    
    passed = len(actual.get("validation_errors", [])) > 0 and actual.get("next_step") == "format"
    print(f"\n{'PASS' if passed else 'FAIL'}: guardrail_invalid_hour")
    print(f"  Expected validation_errors > 0")
    print(f"  Actual validation_errors: {actual.get('validation_errors')}")
    return passed


def test_guardrail_invalid_month(base_state):
    """
    Test: Invalid month (>12) generates validation error.
    Justification: Month must be 1-12.
    """
    state = base_state.copy()
    state["current_params"] = {"location_id": 161, "month": 15, "day_of_week": 0, "hour": 8, "minute": 0}
    state["candidates"] = []
    
    actual = guardrail_node(state)
    
    passed = len(actual.get("validation_errors", [])) > 0 and actual.get("next_step") == "format"
    print(f"\n{'PASS' if passed else 'FAIL'}: guardrail_invalid_month")
    print(f"  Expected validation_errors > 0")
    print(f"  Actual validation_errors: {actual.get('validation_errors')}")
    return passed


# ─── PREDICTOR NODE TESTS ────────────────────────────────────────────────────

def test_predictor_valid_call(mock_predictor, base_state):
    """
    Test: Valid predictor call returns results.
    Justification: Core functionality: prediction with valid params.
    """
    state = base_state.copy()
    state["current_params"] = {"location_id": 161, "month": 3, "day_of_week": 0, "hour": 8, "minute": 0}
    state["intent"] = "predict"
    state["hour_range"] = []
    
    actual = predictor_node(state)
    
    passed = len(actual.get("results", [])) > 0 and actual.get("next_step") == "format"
    print(f"\n{'PASS' if passed else 'FAIL'}: predictor_valid_call")
    print(f"  Expected results > 0")
    print(f"  Actual results count: {len(actual.get('results', []))}")
    return passed


def test_predictor_trend_intent(mock_historical_trends, base_state):
    """
    Test: Trend intent calls historical trends.
    Justification: Trend queries should fetch historical data.
    """
    state = base_state.copy()
    state["current_params"] = {"location_id": 161, "month": None, "day_of_week": 0}
    state["intent"] = "trend"
    state["hour_range"] = []
    
    actual = predictor_node(state)
    
    passed = len(actual.get("results", [])) > 0 and actual.get("next_step") == "format"
    print(f"\n{'PASS' if passed else 'FAIL'}: predictor_trend_intent")
    print(f"  Expected results > 0")
    print(f"  Actual results count: {len(actual.get('results', []))}")
    return passed


def test_predictor_hour_range(mock_predictor, base_state):
    """
    Test: Hour range triggers multiple predictions.
    Justification: Range queries should predict for each hour in range.
    """
    state = base_state.copy()
    state["current_params"] = {"location_id": 161, "month": 3, "day_of_week": 0}
    state["intent"] = "predict"
    state["hour_range"] = [14, 15, 16, 17, 18]
    
    actual = predictor_node(state)
    
    passed = len(actual.get("results", [])) == 5 and actual.get("next_step") == "format"
    print(f"\n{'PASS' if passed else 'FAIL'}: predictor_hour_range")
    print(f"  Expected results count: 5")
    print(f"  Actual results count: {len(actual.get('results', []))}")
    return passed


# ─── FORMATTER NODE TESTS ───────────────────────────────────────────────────

def test_formatter_oos_response(mock_llm_oos, base_state):
    """
    Test: OOS intent generates conversational response.
    Justification: Out of scope queries should get friendly response.
    """
    mock_llm_oos.set_response("Ciao! Posso aiutarti con informazioni sui taxi NYC.")
    
    state = base_state.copy()
    state["intent"] = "oos"
    state["messages"] = [HumanMessage(content="Ciao come stai?"), AIMessage(content="Ciao! Posso aiutarti con informazioni sui taxi NYC.")]
    
    actual = formatter_node(state)
    
    passed = len(actual.get("messages", [])) > 0
    print(f"\n{'PASS' if passed else 'FAIL'}: formatter_oos_response")
    print(f"  Expected messages > 0")
    print(f"  Actual messages count: {len(actual.get('messages', []))}")
    return passed


def test_formatter_validation_error(base_state):
    """
    Test: Validation errors generate error message.
    Justification: Invalid params should show helpful error.
    """
    state = base_state.copy()
    state["validation_errors"] = ["Invalid zone ID", "Invalid hour"]
    state["next_step"] = "format"
    
    actual = formatter_node(state)
    
    passed = len(actual.get("messages", [])) > 0
    print(f"\n{'PASS' if passed else 'FAIL'}: formatter_validation_error")
    print(f"  Expected messages > 0")
    print(f"  Actual messages count: {len(actual.get('messages', []))}")
    return passed


def test_formatter_ask_zone(base_state):
    """
    Test: Missing zone generates ask_zone message.
    Justification: User must provide zone for prediction.
    """
    state = base_state.copy()
    state["next_step"] = "ask_zone"
    
    actual = formatter_node(state)
    
    passed = len(actual.get("messages", [])) > 0
    print(f"\n{'PASS' if passed else 'FAIL'}: formatter_ask_zone")
    print(f"  Expected messages > 0")
    print(f"  Actual messages count: {len(actual.get('messages', []))}")
    return passed


def test_formatter_disambiguate(base_state):
    """
    Test: Ambiguous zone generates disambiguation message.
    Justification: Multiple matches need user choice.
    """
    state = base_state.copy()
    state["next_step"] = "disambiguate"
    state["candidates"] = [
        {"location_id": 100, "location_name": "Midtown"},
        {"location_id": 161, "location_name": "Midtown East"},
    ]
    
    actual = formatter_node(state)
    
    passed = len(actual.get("messages", [])) > 0
    print(f"\n{'PASS' if passed else 'FAIL'}: formatter_disambiguate")
    print(f"  Expected messages > 0")
    print(f"  Actual messages count: {len(actual.get('messages', []))}")
    return passed


def test_formatter_no_results(base_state):
    """
    Test: No results generates no_data message.
    Justification: Empty results should inform user.
    """
    state = base_state.copy()
    state["intent"] = "predict"
    state["results"] = []
    state["current_params"] = {"location_id": 161}
    
    actual = formatter_node(state)
    
    passed = len(actual.get("messages", [])) > 0
    print(f"\n{'PASS' if passed else 'FAIL'}: formatter_no_results")
    print(f"  Expected messages > 0")
    print(f"  Actual messages count: {len(actual.get('messages', []))}")
    return passed


def test_formatter_with_template(mock_llm_insight, base_state):
    """
    Test: Template generated for valid results.
    Justification: Deterministic template carries key data.
    """
    mock_llm_insight.set_response("Il sabato sera la disponibilita e generalmente bassa.")
    
    state = base_state.copy()
    state["intent"] = "predict"
    state["results"] = [{
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
        "probabilities": {
            "Molto Difficile": 0.1,
            "Difficile": 0.2,
            "Medio": 0.4,
            "Facile": 0.2,
            "Molto Facile": 0.1,
        },
    }]
    state["current_params"] = {"location_id": 161, "hour": 8}
    state["messages"] = [HumanMessage(content="Quanti taxi a Midtown?")]
    
    actual = formatter_node(state)
    
    passed = len(actual.get("messages", [])) > 0
    print(f"\n{'PASS' if passed else 'FAIL'}: formatter_with_template")
    print(f"  Expected messages > 0")
    print(f"  Actual messages count: {len(actual.get('messages', []))}")
    return passed


def test_formatter_with_llm_insight(mock_llm_insight, base_state):
    """
    Test: LLM insight added to template.
    Justification: AI insight provides additional context.
    """
    mock_llm_insight.set_response("Il sabato sera la disponibilita e generalmente bassa.")
    
    state = base_state.copy()
    state["intent"] = "predict"
    state["results"] = [{
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
        "probabilities": {
            "Molto Difficile": 0.1,
            "Difficile": 0.2,
            "Medio": 0.4,
            "Facile": 0.2,
            "Molto Facile": 0.1,
        },
    }]
    state["current_params"] = {"location_id": 161}
    state["messages"] = [HumanMessage(content="Quanti taxi a Midtown?")]
    
    actual = formatter_node(state)
    
    passed = len(actual.get("messages", [])) > 0 and "Insight" in actual["messages"][-1].content
    print(f"\n{'PASS' if passed else 'FAIL'}: formatter_with_llm_insight")
    print(f"  Expected 'Insight' in message")
    print(f"  Actual has 'Insight': {'Insight' in actual['messages'][-1].content if actual.get('messages') else False}")
    return passed


def test_formatter_llm_failure_graceful(base_state):
    """
    Test: LLM failure falls back to template only.
    Justification: Graceful degradation when LLM fails.
    """
    with patch("llm_tool.agent.get_llm") as mock:
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = Exception("LLM error")
        mock.return_value = mock_instance
        
        state = base_state.copy()
        state["intent"] = "predict"
        state["results"] = [{
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
            "probabilities": {
                "Molto Difficile": 0.1,
                "Difficile": 0.2,
                "Medio": 0.4,
                "Facile": 0.2,
                "Molto Facile": 0.1,
            },
        }]
        state["current_params"] = {"location_id": 161}
        state["messages"] = [HumanMessage(content="Quanti taxi a Midtown?")]
        
        actual = formatter_node(state)
        
        passed = len(actual.get("messages", [])) > 0
        print(f"\n{'PASS' if passed else 'FAIL'}: formatter_llm_failure_graceful")
        print(f"  Expected messages > 0 (template fallback)")
        print(f"  Actual messages count: {len(actual.get('messages', []))}")
        return passed


# ─── _BUILD_TEMPLATE TESTS ───────────────────────────────────────────────────

def test_build_template_single_result():
    """
    Test: Template generated for single prediction.
    Justification: Single time predictions need complete template.
    """
    results = [{
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
        "probabilities": {
            "Molto Difficile": 0.1,
            "Difficile": 0.2,
            "Medio": 0.4,
            "Facile": 0.2,
            "Molto Facile": 0.1,
        },
    }]
    params = {"location_id": 161, "hour": 8, "minute": 0, "day_of_week": 0, "month": 3}
    
    actual = _build_template(results, params)
    
    passed = "Midtown Center" in actual and "Medio" in actual
    print(f"\n{'PASS' if passed else 'FAIL'}: build_template_single_result")
    print(f"  Expected zone name and class in template")
    print(f"  Actual contains zone: {'Midtown Center' in actual}, class: {'Medio' in actual}")
    return passed


def test_build_template_hour_range():
    """
    Test: Template generated for hour range predictions.
    Justification: Range queries show hourly breakdown.
    """
    results = [
        {"time_bucket": 28, "predicted_class": 0, "predicted_class_name": "Molto Difficile", "confidence": 0.8},
        {"time_bucket": 30, "predicted_class": 1, "predicted_class_name": "Difficile", "confidence": 0.7},
        {"time_bucket": 32, "predicted_class": 2, "predicted_class_name": "Medio", "confidence": 0.6},
    ]
    params = {"location_id": 161, "day_of_week": 0, "month": 3}
    
    actual = _build_template(results, params)
    
    passed = "14" in actual or "15" in actual or "16" in actual
    print(f"\n{'PASS' if passed else 'FAIL'}: build_template_hour_range")
    print(f"  Expected hour breakdown in template")
    print(f"  Actual contains hourly data: {passed}")
    return passed


def test_build_template_trend():
    """
    Test: Template generated for historical trends.
    Justification: Trend data shows hourly averages.
    """
    results = [{
        "location_id": 161,
        "location_name": "JFK",
        "borough": "Queens",
        "hourly_avg_availability": [
            {"hour": 8, "avg": 0.3},
            {"hour": 9, "avg": 0.5},
        ]
    }]
    params = {"location_id": 161}
    
    actual = _build_template(results, params)
    
    passed = "Trend Storico" in actual or "hourly" in actual.lower()
    print(f"\n{'PASS' if passed else 'FAIL'}: build_template_trend")
    print(f"  Expected trend content in template")
    print(f"  Actual contains trend: {passed}")
    return passed


def test_build_template_empty_results():
    """
    Test: Empty results generates warning.
    Justification: No data should inform user.
    """
    results = []
    params = {}
    
    actual = _build_template(results, params)
    
    passed = "Nessun risultato" in actual or "⚠" in actual
    print(f"\n{'PASS' if passed else 'FAIL'}: build_template_empty_results")
    print(f"  Expected 'Nessun risultato' or warning in template")
    print(f"  Actual: {actual}")
    return passed


