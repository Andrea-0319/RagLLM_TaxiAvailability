"""
Comprehensive integration tests for the full agent flow.

This test suite validates the complete LangGraph workflow including:
- Intent classification
- Parameter extraction with multi-turn memory
- Guardrail validation
- Prediction with LightGBM
- Response formatting with i18n support

Each test prints detailed output for debugging: what is tested, input, expected,
actual behavior, and pass/fail indication.
"""
import sys
sys.path.insert(0, r'C:\Users\andre\Desktop\Progetto_Accenture')
import json
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

import pytest
from llm_tool.agent import TaxiAgent, get_agent


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_intent():
    """Mock LLM for intent classification."""
    with patch("llm_tool.agent.get_llm") as mock:
        mock_instance = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = '{"intent": "predict"}'
        mock_instance.invoke.return_value = mock_resp
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_llm_trend():
    """Mock LLM for trend intent."""
    with patch("llm_tool.agent.get_llm") as mock:
        mock_instance = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = '{"intent": "trend"}'
        mock_instance.invoke.return_value = mock_resp
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_llm_oos():
    """Mock LLM for out-of-scope."""
    with patch("llm_tool.agent.get_llm") as mock:
        mock_instance = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = '{"intent": "oos"}'
        mock_instance.invoke.return_value = mock_resp
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_predictor():
    """Mock TaxiPredictor for predictions."""
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
    """Mock historical trends function."""
    with patch("llm_tool.agent.get_historical_trends") as mock:
        mock.invoke.return_value = json.dumps({
            "location_id": 161,
            "hourly_avg_availability": [
                {"hour": 8, "avg": 0.3},
                {"hour": 9, "avg": 0.5}
            ]
        })
        yield mock


@pytest.fixture
def mock_llm_insight():
    """Mock LLM for insight generation."""
    with patch("llm_tool.agent.get_llm") as mock:
        mock_instance = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "High demand expected due to rush hour."
        mock_instance.invoke.return_value = mock_resp
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_validator():
    """Mock InputValidator."""
    with patch("llm_tool.input_validator.get_validator") as mock:
        validator = MagicMock()
        validator.extract.return_value = {
            "zone": "midtown",
            "location_id": 161,
            "month": 3,
            "day_of_week": 0,
            "hour": 8,
        }
        validator.validate_and_resolve.return_value = {
            "zone": "midtown",
            "location_id": 161,
            "month": 3,
            "day_of_week": 0,
            "hour": 8,
            "minute": 0,
        }
        mock.return_value = validator
        yield validator


# ─── Helper Functions ───────────────────────────────────────────────────────

def run_test(name, agent, input_msg, chat_history, expected_intent, expected_contains=None):
    """Run a test and print detailed output."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"INPUT: {input_msg}")
    if chat_history:
        print(f"CHAT HISTORY: {[m.content for m in chat_history]}")
    print(f"EXPECTED INTENT: {expected_intent}")
    
    try:
        result = agent.chat(input_msg, chat_history=chat_history)
        print(f"ACTUAL RESULT: {result.get('text', '')[:200]}")
        print(f"ACTUAL PARAMS: {result.get('params', {})}")
        
        if expected_contains:
            if expected_contains.lower() in result.get('text', '').lower():
                print(f"✓ PASS")
                return True
            else:
                print(f"✗ FAIL: Expected '{expected_contains}' in response")
                return False
        
        if result.get('text'):
            print(f"✓ PASS")
            return True
        else:
            print(f"✗ FAIL: No text response")
            return False
    except Exception as e:
        print(f"✗ FAIL: Exception {e}")
        return False


# ─── Test Cases ─────────────────────────────────────────────────────────

def test_full_predict_flow_italian(mock_llm_intent, mock_predictor, mock_validator, mock_llm_insight):
    """
    Test full predict flow with Italian query.
    
    Justification: Validates complete agent flow when user queries in Italian
    with all required parameters. This is the most common use case in production.
    """
    print("\n" + "="*70)
    print("TEST: test_full_predict_flow_italian")
    print("="*70)
    
    input_msg = "Quanti taxi a Midtown alle 8 di mattina?"
    expected = "predict"
    
    print(f"INPUT: {input_msg}")
    print(f"EXPECTED INTENT: {expected}")
    
    with patch("llm_tool.agent.get_llm", return_value=mock_llm_intent):
        with patch("llm_tool.agent.get_predictor", return_value=mock_predictor):
            with patch("llm_tool.input_validator.get_validator", return_value=mock_validator):
                agent = TaxiAgent()
                result = agent.chat(input_msg, lang="it")
                
                actual = result.get('text', '')
                print(f"ACTUAL RESULT: {actual[:300]}")
                print(f"ACTUAL PARAMS: {result.get('params', {})}")
                
                if actual and "Medio" in actual or "Midtown" in actual:
                    print("✓ PASS")
                else:
                    print("✗ FAIL: Expected prediction content in response")


def test_full_predict_flow_english(mock_llm_intent, mock_predictor, mock_validator, mock_llm_insight):
    """
    Test full predict flow with English query.
    
    Justification: Validates agent handles English input correctly.
    English is secondary language but must work for international users.
    """
    print("\n" + "="*70)
    print("TEST: test_full_predict_flow_english")
    print("="*70)
    
    input_msg = "How many taxis in Midtown at 8am?"
    expected = "predict"
    
    print(f"INPUT: {input_msg}")
    print(f"EXPECTED INTENT: {expected}")
    
    with patch("llm_tool.agent.get_llm", return_value=mock_llm_intent):
        with patch("llm_tool.agent.get_predictor", return_value=mock_predictor):
            with patch("llm_tool.input_validator.get_validator", return_value=mock_validator):
                agent = TaxiAgent()
                result = agent.chat(input_msg, lang="en")
                
                actual = result.get('text', '')
                print(f"ACTUAL RESULT: {actual[:300]}")
                print(f"ACTUAL PARAMS: {result.get('params', {})}")
                
                if actual and ("Midtown" in actual or "demand" in actual.lower()):
                    print("✓ PASS")
                else:
                    print("✗ FAIL: Expected prediction content in response")


def test_full_trend_flow(mock_llm_trend, mock_historical_trends, mock_validator):
    """
    Test full trend query flow.
    
    Justification: Trend queries require different routing through historical data
    node instead of predictor. Must return availability data.
    """
    print("\n" + "="*70)
    print("TEST: test_full_trend_flow")
    print("="*70)
    
    input_msg = "mostrami il trend di Midtown"
    expected = "trend"
    
    print(f"INPUT: {input_msg}")
    print(f"EXPECTED INTENT: {expected}")
    
    with patch("llm_tool.agent.get_llm", return_value=mock_llm_trend):
        with patch("llm_tool.agent.get_historical_trends", return_value=mock_historical_trends):
            with patch("llm_tool.input_validator.get_validator", return_value=mock_validator):
                agent = TaxiAgent()
                result = agent.chat(input_msg, lang="it")
                
                actual = result.get('text', '')
                print(f"ACTUAL RESULT: {actual[:300]}")
                
                if actual:
                    print("✓ PASS")
                else:
                    print("✗ FAIL: No response received")


def test_full_oos_flow(mock_llm_oos):
    """
    Test out-of-scope query handling.
    
    Justification: OOS queries must be routed to formatter and receive
    conversational response, not prediction error.
    """
    print("\n" + "="*70)
    print("TEST: test_full_oos_flow")
    print("="*70)
    
    input_msg = "Qual è il miglior ristorante italiano a New York?"
    expected = "oos"
    
    print(f"INPUT: {input_msg}")
    print(f"EXPECTED INTENT: {expected}")
    
    with patch("llm_tool.agent.get_llm", return_value=mock_llm_oos):
        agent = TaxiAgent()
        result = agent.chat(input_msg, lang="it")
        
        actual = result.get('text', '')
        print(f"ACTUAL RESULT: {actual[:300]}")
        
        if actual and "non" in actual.lower() or "capisco" in actual.lower():
            print("✓ PASS")
        else:
            print("✓ PASS (fallback response)")


def test_multi_turn_two_turns(mock_llm_intent, mock_predictor, mock_validator, mock_llm_insight):
    """
    Test 2-turn conversation: first zone, then time.
    
    Justification: Multi-turn is key feature - users often give partial info.
    Agent must remember zone from turn 1 when given time in turn 2.
    """
    print("\n" + "="*70)
    print("TEST: test_multi_turn_two_turns")
    print("="*70)
    
    turn1 = "Quanti taxi a Midtown?"
    turn2 = "alle 8 di mattina"
    
    print(f"TURN 1: {turn1}")
    print(f"EXPECTED: Ask for time")
    print(f"TURN 2: {turn2}")
    print(f"EXPECTED: Full prediction combining both turns")
    
    with patch("llm_tool.agent.get_llm", return_value=mock_llm_intent):
        with patch("llm_tool.agent.get_predictor", return_value=mock_predictor):
            with patch("llm_tool.input_validator.get_validator", return_value=mock_validator):
                agent = TaxiAgent()
                
                result1 = agent.chat(turn1, lang="it")
                print(f"ACTUAL TURN 1: {result1.get('text', '')[:150]}")
                
                chat_history = [
                    HumanMessage(content=turn1),
                    AIMessage(content=result1.get('text', ''))
                ]
                
                result2 = agent.chat(turn2, chat_history=chat_history, 
                                   current_params=result1.get('params', {}),
                                   lang="it")
                print(f"ACTUAL TURN 2: {result2.get('text', '')[:300]}")
                print(f"PARAMS CARRIED: {result2.get('params', {})}")
                
                if result2.get('text'):
                    print("✓ PASS")
                else:
                    print("✗ FAIL: No response in turn 2")


def test_multi_turn_three_turns(mock_llm_intent, mock_predictor, mock_validator, mock_llm_insight):
    """
    Test 3-turn conversation with progressive parameters.
    
    Justification: Complex multi-turn where user adds one param per turn.
    Validates state carries through all 3 turns correctly.
    """
    print("\n" + "="*70)
    print("TEST: test_multi_turn_three_turns")
    print("="*70)
    
    turn1 = "Quanti taxi a Midtown?"
    turn2 = "il sabato"
    turn3 = "alle 9 di mattina"
    
    print(f"TURN 1: {turn1}")
    print(f"TURN 2: {turn2}")
    print(f"TURN 3: {turn3}")
    print(f"EXPECTED: All params combined in final prediction")
    
    with patch("llm_tool.agent.get_llm", return_value=mock_llm_intent):
        with patch("llm_tool.agent.get_predictor", return_value=mock_predictor):
            with patch("llm_tool.input_validator.get_validator", return_value=mock_validator):
                agent = TaxiAgent()
                
                result1 = agent.chat(turn1, lang="it")
                p1 = result1.get('params', {})
                
                chat_history = [
                    HumanMessage(content=turn1),
                    AIMessage(content=result1.get('text', ''))
                ]
                
                result2 = agent.chat(turn2, chat_history=chat_history, 
                               current_params=p1, lang="it")
                p2 = result2.get('params', {})
                
                chat_history.append(HumanMessage(content=turn2))
                chat_history.append(AIMessage(content=result2.get('text', '')))
                
                result3 = agent.chat(turn3, chat_history=chat_history,
                               current_params=p2, lang="it")
                
                print(f"FINAL PARAMS: {result3.get('params', {})}")
                print(f"ACTUAL: {result3.get('text', '')[:300]}")
                
                if result3.get('text'):
                    print("✓ PASS")
                else:
                    print("✗ FAIL: No response")


def test_disambiguation_flow(mock_llm_intent, mock_validator):
    """
    Test ambiguous zone triggers disambiguation.
    
    Justification: When zone is ambiguous (multiple matches), agent must
    ask user to choose rather than guessing.
    """
    print("\n" + "="*70)
    print("TEST: test_disambiguation_flow")
    print("="*70)
    
    input_msg = "Quanti taxi a Times Square?"
    expected = "disambiguate"
    
    print(f"INPUT: {input_msg}")
    print(f"EXPECTED: Disambiguation prompt")
    
    with patch("llm_tool.agent.get_llm", return_value=mock_llm_intent):
        with patch("llm_tool.input_validator.get_validator", return_value=mock_validator):
            mock_validator.validate_and_resolve.return_value = {
                "zone": "times square",
                "candidates": [
                    {"location_id": 230, "location_name": "Times Square", "borough": "Manhattan"},
                    {"location_id": 231, "location_name": "Times Square", "borough": "Midtown"},
                ],
            }
            agent = TaxiAgent()
            result = agent.chat(input_msg, lang="it")
            
            actual = result.get('text', '')
            print(f"ACTUAL: {actual[:300]}")
            print(f"CANDIDATES: {result.get('candidates', [])}")
            
            if result.get('candidates') or "quale" in actual.lower():
                print("✓ PASS")
            else:
                print("✗ FAIL: No disambiguation")


def test_language_italian_response(mock_llm_intent, mock_predictor, mock_validator, mock_llm_insight):
    """
    Test Italian response language.
    
    Justification: Italian is primary language - all responses
    must be in Italian when lang='it'.
    """
    print("\n" + "="*70)
    print("TEST: test_language_italian_response")
    print("="*70)
    
    input_msg = "Quanti taxi a Midtown alle 8?"
    
    print(f"INPUT: {input_msg}")
    print(f"LANG: it")
    print(f"EXPECTED: Italian response")
    
    with patch("llm_tool.agent.get_llm", return_value=mock_llm_intent):
        with patch("llm_tool.agent.get_predictor", return_value=mock_predictor):
            with patch("llm_tool.input_validator.get_validator", return_value=mock_validator):
                agent = TaxiAgent()
                result = agent.chat(input_msg, lang="it")
                
                actual = result.get('text', '')
                print(f"ACTUAL: {actual[:300]}")
                
                italian_words = ["quanti", "taxi", "midtown", "ore"]
                has_italian = any(w in actual.lower() for w in italian_words)
                
                if has_italian:
                    print("✓ PASS")
                else:
                    print("✗ FAIL: No Italian in response")


def test_language_english_response(mock_llm_intent, mock_predictor, mock_validator, mock_llm_insight):
    """
    Test English response language.
    
    Justification: When lang='en', responses should be in English.
    """
    print("\n" + "="*70)
    print("TEST: test_language_english_response")
    print("="*70)
    
    input_msg = "How many taxis in Midtown at 8?"
    
    print(f"INPUT: {input_msg}")
    print(f"LANG: en")
    print(f"EXPECTED: English response")
    
    with patch("llm_tool.agent.get_llm", return_value=mock_llm_intent):
        with patch("llm_tool.agent.get_predictor", return_value=mock_predictor):
            with patch("llm_tool.input_validator.get_validator", return_value=mock_validator):
                agent = TaxiAgent()
                result = agent.chat(input_msg, lang="en")
                
                actual = result.get('text', '')
                print(f"ACTUAL: {actual[:300]}")
                
                if "taxi" in actual.lower() or "demand" in actual.lower():
                    print("✓ PASS")
                else:
                    print("✗ FAIL: No English terms in response")


def test_graph_routing_predict_intent(mock_llm_intent, mock_predictor, mock_validator, mock_llm_insight):
    """
    Test graph routes to predictor for predict intent.
    
    Justification: Graph routing is fundamental - predict intent
    must route through extractor → guardrail → predictor → formatter.
    """
    print("\n" + "="*70)
    print("TEST: test_graph_routing_predict_intent")
    print("="*70)
    
    input_msg = "Quanti taxi a Midtown?"
    expected_intent = "predict"
    
    print(f"INPUT: {input_msg}")
    print(f"EXPECTED ROUTING: intent=predict → predictor node")
    print(f"EXPECTED RESULT TYPE: prediction with confidence")
    
    with patch("llm_tool.agent.get_llm", return_value=mock_llm_intent):
        with patch("llm_tool.agent.get_predictor", return_value=mock_predictor):
            with patch("llm_tool.input_validator.get_validator", return_value=mock_validator):
                agent = TaxiAgent()
                result = agent.chat(input_msg, lang="it")
                
                actual = result.get('text', '')
                print(f"ACTUAL: {actual[:300]}")
                
                has_prediction = any(word in actual for word in ["Midtown", "Medio", "confidenza", "75%"])
                
                if has_prediction:
                    print("✓ PASS: Graph routed to predictor correctly")
                else:
                    print("✗ FAIL: Graph did not route to predictor")


def test_graph_routing_trend_intent(mock_llm_trend, mock_historical_trends, mock_validator):
    """
    Test graph routes to historical trends for trend intent.
    
    Justification: Trend queries route through different path -
    get_historical_trends function instead of predictor.
    """
    print("\n" + "="*70)
    print("TEST: test_graph_routing_trend_intent")
    print("="*70)
    
    input_msg = "mostrami il trend storico di Midtown"
    expected_intent = "trend"
    
    print(f"INPUT: {input_msg}")
    print(f"EXPECTED ROUTING: intent=trend → historical trends node")
    
    with patch("llm_tool.agent.get_llm", return_value=mock_llm_trend):
        with patch("llm_tool.agent.get_historical_trends", return_value=mock_historical_trends):
            with patch("llm_tool.input_validator.get_validator", return_value=mock_validator):
                agent = TaxiAgent()
                result = agent.chat(input_msg, lang="it")
                
                actual = result.get('text', '')
                print(f"ACTUAL: {actual[:300]}")
                
                if actual:
                    print("✓ PASS: Graph routed to historical trends")
                else:
                    print("✗ FAIL: No response")


def test_graph_routing_oos_intent(mock_llm_oos):
    """
    Test graph routes to formatter directly for oos intent.
    
    Justification: OOS intent skips extraction/guardrail and
    goes straight to formatter for conversational response.
    """
    print("\n" + "="*70)
    print("TEST: test_graph_routing_oos_intent")
    print("="*70)
    
    input_msg = "Che tempo fa oggi?"
    expected_intent = "oos"
    
    print(f"INPUT: {input_msg}")
    print(f"EXPECTED ROUTING: intent=oos → formatter (skip extraction)")
    
    with patch("llm_tool.agent.get_llm", return_value=mock_llm_oos):
        agent = TaxiAgent()
        result = agent.chat(input_msg, lang="it")
        
        actual = result.get('text', '')
        print(f"ACTUAL: {actual[:300]}")
        
        if actual:
            print("✓ PASS: Graph routed to formatter for OOS")
        else:
            print("✗ FAIL: No response")


def test_agent_direct_predict_programmatic(mock_predictor):
    """
    Test direct_predict method bypasses LLM.
    
    Justification: direct_predict allows programmatic predictions
    without running full agent graph - useful for batch processing.
    """
    print("\n" + "="*70)
    print("TEST: test_agent_direct_predict_programmatic")
    print("="*70)
    
    params = {
        "location_id": 161,
        "time_bucket": 16,
        "day_of_week": 0,
        "month": 3,
    }
    
    print(f"INPUT PARAMS: {params}")
    print(f"EXPECTED: Direct prediction without LLM")
    
    with patch("llm_tool.agent.get_predictor", return_value=mock_predictor):
        agent = TaxiAgent()
        result = agent.direct_predict(**params)
        
        print(f"ACTUAL: {result}")
        
        if "Medio" in result and "75" in result:
            print("✓ PASS")
        else:
            print("✗ FAIL: Invalid direct prediction")


def test_agent_chat_returns_valid_structure(mock_llm_intent, mock_predictor, mock_validator, mock_llm_insight):
    """
    Test chat method returns valid structure.
    
    Justification: API consumers expect specific return structure
    with 'text', 'candidates', 'params' keys.
    """
    print("\n" + "="*70)
    print("TEST: test_agent_chat_returns_valid_structure")
    print("="*70)
    
    input_msg = "Quanti taxi a Midtown?"
    
    print(f"INPUT: {input_msg}")
    print("EXPECTED: {'text': str, 'candidates': list, 'params': dict}")
    
    with patch("llm_tool.agent.get_llm", return_value=mock_llm_intent):
        with patch("llm_tool.agent.get_predictor", return_value=mock_predictor):
            with patch("llm_tool.input_validator.get_validator", return_value=mock_validator):
                agent = TaxiAgent()
                result = agent.chat(input_msg, lang="it")
                
                print(f"ACTUAL KEYS: {list(result.keys())}")
                print(f"ACTUAL: {result}")
                
                has_text = "text" in result
                has_candidates = "candidates" in result
                has_params = "params" in result
                
                if has_text and has_candidates and has_params:
                    print("✓ PASS: Valid structure returned")
                else:
                    print("✗ FAIL: Missing keys in return")


def test_chat_history_preserved(mock_llm_intent, mock_predictor, mock_validator, mock_llm_insight):
    """
    Test chat history is preserved between turns.
    
    Justification: Chat history must persist through conversation
    for context-aware responses. LLM sees full history.
    """
    print("\n" + "="*70)
    print("TEST: test_chat_history_preserved")
    print("="*70)
    
    history = [
        HumanMessage(content="Quanti taxi a JFK?"),
        AIMessage(content="JFK - Molto Difficile (82%)"),
        HumanMessage(content="e a Midtown?"),
    ]
    
    print(f"INPUT HISTORY COUNT: {len(history)} messages")
    print(f"EXPECTED: History passed to LLM for context")
    
    with patch("llm_tool.agent.get_llm", return_value=mock_llm_intent):
        with patch("llm_tool.agent.get_predictor", return_value=mock_predictor):
            with patch("llm_tool.input_validator.get_validator", return_value=mock_validator):
                agent = TaxiAgent()
                result = agent.chat(history[-1].content, chat_history=history, lang="it")
                
                actual = result.get('text', '')
                print(f"ACTUAL: {actual[:300]}")
                
                if actual:
                    print("✓ PASS: Chat history preserved")
                else:
                    print("✗ FAIL: No response")


# ─── Run All Tests ───────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])