"""
Test file that intentionally has failures to demonstrate failure visibility.
"""
import sys
sys.path.insert(0, r'C:\Users\andre\Desktop\Progetto_Accenture')

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage
from llm_tool.agent import intent_classifier_node


class TestFailureVisibility:
    """Tests that intentionally fail to demonstrate error visibility."""

    def test_intent_classifier_fail_on_purpose(self):
        """
        GIUSTIFICAZIONE: Test che fallisce intenzionalmente per mostrare 
        come i fallimenti vengono visualizzati con messaggi chiari.
        
        Questo test VERIFICA che il classifier riconosca 'predict' ma 
        simulates un fallimento per mostrare l'output.
        """
        print("\n" + "="*60)
        print("TEST: test_intent_classifier_fail_on_purpose")
        print("FUNZIONALITA': Intent classifier - risposta attesa 'predict'")
        print("="*60)
        
        with patch("llm_tool.agent.get_llm") as mock_llm:
            mock_instance = MagicMock()
            mock_msg = MagicMock()
            mock_msg.content = '{"intent": "predict"}'
            mock_instance.invoke.return_value = mock_msg
            mock_llm.return_value = mock_instance
            
            state = {
                "messages": [HumanMessage(content="Quanti taxi a Midtown?")],
                "intent": "",
                "current_params": {},
                "results": [],
                "validation_errors": [],
                "hour_range": [],
                "next_step": "",
                "language": "it",
            }
            
            result = intent_classifier_node(state)
            intent = result.get("intent", "")
            
            print(f"   INPUT: 'Quanti taxi a Midtown?'")
            print(f"   ATTESO: 'predict'")
            print(f"   ATTUALE: '{intent}'")
            
            # Intentional failure - this should show as FAIL in output
            assert intent == "trend", f"FAIL: Expected 'trend' but got '{intent}' - questa funzionalità ha fallito!"
            
            print("   ESITO: PASS (il test è passato accidentalmente)")

    def test_zone_resolution_fail_boundary(self):
        """
        GIUSTIFICAZIONE: Test che fallisce alla boundary per mostrare 
        come i fallimenti vengono visualizzati.
        """
        from llm_tool.taxi_predictor import resolve_zone_id
        
        print("\n" + "="*60)
        print("TEST: test_zone_resolution_fail_boundary")
        print("FUNZIONALITA': Zone resolution - ID 266 dovrebbe essere invalido")
        print("="*60)
        
        result = resolve_zone_id("266")
        
        print(f"   INPUT: '266'")
        print(f"   ATTESO: None (invalido)")
        print(f"   ATTUALE: {result}")
        
        # Intentional failure - expecting None but getting None actually passes
        # Let's make it fail by expecting wrong value
        assert result == 999, f"FAIL: Zone ID 266 incorrectly resolved to {result} - funzionalità zone resolution ha fallito!"
        
        print("   ESITO: PASS")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])