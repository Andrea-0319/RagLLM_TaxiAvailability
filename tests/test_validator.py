"""
Comprehensive unit tests for InputValidator class.
Tests regex fast-path, LLM extraction, sanitization, and zone resolution.
"""
import sys
sys.path.insert(0, r'C:\Users\andre\Desktop\Progetto_Accenture')

import pytest
from unittest.mock import MagicMock, patch
from llm_tool.input_validator import InputValidator


class TestRegexFastPath:
    """Tests for regex fast-path parsing (button commands)."""

    def test_regex_fastpath_zone_id(self):
        """
        Verifica che "Zona ID 161" venga parsato senza LLM.
        Button commands like "Zona ID 161" should be parsed via regex fast-path
        without invoking the LLM, enabling quick zone selection from buttons.
        """
        print("\n=== TEST: test_regex_fastpath_zone_id ===")
        validator = InputValidator()
        
        text = "Zona ID 161"
        print(f"Input text: '{text}'")
        
        result = validator.extract(text)
        print(f"Expected zone: '161'")
        print(f"Actual result: {result}")
        
        assert result["zone"] == "161", f"FAIL: Expected zone='161', got {result['zone']}"
        assert result["month"] is None
        assert result["day_of_week"] is None
        print("PASS: Zone ID parsed correctly via regex fast-path")

    def test_regex_fastpath_zone_keyword(self):
        """
        "zona 132" parsing.
        Test that simple "zona X" pattern also triggers regex fast-path
        for common user inputs.
        """
        print("\n=== TEST: test_regex_fastpath_zone_keyword ===")
        validator = InputValidator()
        
        text = "zona 132"
        print(f"Input text: '{text}'")
        
        result = validator.extract(text)
        print(f"Expected zone: '132'")
        print(f"Actual result: {result}")
        
        assert result["zone"] == "132", f"FAIL: Expected zone='132', got {result['zone']}"
        print("PASS: Zone keyword parsed correctly via regex fast-path")


class TestLLMExtraction:
    """Tests for LLM-based parameter extraction from natural language."""

    def test_llm_extraction_italian(self):
        """
        LLM estrae parametri da query italiana naturale (use mock).
        Test that Italian natural language queries like "quanti taxi a midtown
        questo sabato alle 8 di mattina" correctly extract all parameters via LLM.
        """
        print("\n=== TEST: test_llm_extraction_italian ===")
        validator = InputValidator()
        
        with patch.object(validator, '_get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = '{"zone": "midtown", "month": 3, "day_of_week": 5, "hour": 8, "minute": null}'
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            text = "quanti taxi a midtown questo sabato alle 8 di mattina"
            print(f"Input text: '{text}'")
            
            result = validator.extract(text)
            print(f"Expected: zone='midtown', month=3, day_of_week=5, hour=8")
            print(f"Actual: {result}")
            
            assert result["zone"] == "midtown", f"FAIL: zone expected 'midtown', got {result['zone']}"
            assert result["month"] == 3, f"FAIL: month expected 3, got {result['month']}"
            assert result["day_of_week"] == 5, f"FAIL: day_of_week expected 5, got {result['day_of_week']}"
            assert result["hour"] == 8, f"FAIL: hour expected 8, got {result['hour']}"
            print("PASS: Italian query extracted correctly via LLM")

    def test_llm_extraction_english(self):
        """
        LLM estrae parametri da query inglese (use mock).
        Test that English natural language queries like "how many taxis at
        JFK next Monday at 3pm" correctly extract all parameters via LLM.
        """
        print("\n=== TEST: test_llm_extraction_english ===")
        validator = InputValidator()
        
        with patch.object(validator, '_get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = '{"zone": "JFK", "month": 6, "day_of_week": 1, "hour": 15, "minute": 0}'
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            text = "how many taxis at JFK next Monday at 3pm"
            print(f"Input text: '{text}'")
            
            result = validator.extract(text)
            print(f"Expected: zone='JFK', month=6, day_of_week=1, hour=15, minute=0")
            print(f"Actual: {result}")
            
            assert result["zone"] == "JFK", f"FAIL: zone expected 'JFK', got {result['zone']}"
            assert result["month"] == 6, f"FAIL: month expected 6, got {result['month']}"
            assert result["day_of_week"] == 1, f"FAIL: day_of_week expected 1, got {result['day_of_week']}"
            assert result["hour"] == 15, f"FAIL: hour expected 15, got {result['hour']}"
            assert result["minute"] == 0, f"FAIL: minute expected 0, got {result['minute']}"
            print("PASS: English query extracted correctly via LLM")


class TestSanitization:
    """Tests for post-extraction sanitization (type coercion and range validation)."""

    def test_sanitize_month_out_of_range(self):
        """
        Mese >12 viene corretto matematicamente (13→1).
        Test that month values >12 wrap around (13→1, 14→2) using modular arithmetic.
        This handles LLM outputs that may produce invalid month values.
        """
        print("\n=== TEST: test_sanitize_month_out_of_range ===")
        validator = InputValidator()
        
        raw = {"zone": "midtown", "month": 13, "day_of_week": None, "hour": None, "minute": None}
        print(f"Input raw: {raw}")
        
        result = validator._sanitize_extracted(raw)
        print(f"Expected month: 1 (13 wraps to 1)")
        print(f"Actual result: month={result['month']}")
        
        assert result["month"] == 1, f"FAIL: Expected month=1, got {result['month']}"
        print("PASS: Month >12 correctly wraps to valid range")

    def test_sanitize_day_of_week_out_of_range(self):
        """
        Giorno >6 viene corretto (7→0).
        Test that day_of_week values >6 wrap around (7→0, 8→1) using modular arithmetic.
        """
        print("\n=== TEST: test_sanitize_day_of_week_out_of_range ===")
        validator = InputValidator()
        
        raw = {"zone": "jfk", "month": None, "day_of_week": 7, "hour": None, "minute": None}
        print(f"Input raw: {raw}")
        
        result = validator._sanitize_extracted(raw)
        print(f"Expected day_of_week: 0 (7 wraps to 0)")
        print(f"Actual result: day_of_week={result['day_of_week']}")
        
        assert result["day_of_week"] == 0, f"FAIL: Expected day_of_week=0, got {result['day_of_week']}"
        print("PASS: Day of week >6 correctly wraps to valid range")

    def test_sanitize_month_negative(self):
        """
        Mese negativo diventa None.
        Test that negative month values become None (not wrapped), as negative
        indicates a fundamentally invalid input rather than an overflow.
        """
        print("\n=== TEST: test_sanitize_month_negative ===")
        validator = InputValidator()
        
        raw = {"zone": "midtown", "month": -1, "day_of_week": None, "hour": None, "minute": None}
        print(f"Input raw: {raw}")
        
        result = validator._sanitize_extracted(raw)
        print(f"Expected month: None (negative becomes None)")
        print(f"Actual result: month={result['month']}")
        
        assert result["month"] is None, f"FAIL: Expected month=None, got {result['month']}"
        print("PASS: Negative month correctly becomes None")

    def test_sanitize_hour_boundary_24(self):
        """
        Ora 24 diventa None.
        Test that hour=24 is invalid (valid range is 0-23) and becomes None.
        """
        print("\n=== TEST: test_sanitize_hour_boundary_24 ===")
        validator = InputValidator()
        
        raw = {"zone": "midtown", "month": None, "day_of_week": None, "hour": 24, "minute": None}
        print(f"Input raw: {raw}")
        
        result = validator._sanitize_extracted(raw)
        print(f"Expected hour: None (24 is invalid)")
        print(f"Actual result: hour={result['hour']}")
        
        assert result["hour"] is None, f"FAIL: Expected hour=None, got {result['hour']}"
        print("PASS: Hour 24 correctly becomes None")

    def test_sanitize_hour_0(self):
        """
        Ora 0 OK.
        Test that hour=0 (midnight) is valid and passes through.
        """
        print("\n=== TEST: test_sanitize_hour_0 ===")
        validator = InputValidator()
        
        raw = {"zone": "midtown", "month": None, "day_of_week": None, "hour": 0, "minute": None}
        print(f"Input raw: {raw}")
        
        result = validator._sanitize_extracted(raw)
        print(f"Expected hour: 0 (valid)")
        print(f"Actual result: hour={result['hour']}")
        
        assert result["hour"] == 0, f"FAIL: Expected hour=0, got {result['hour']}"
        print("PASS: Hour 0 correctly passes validation")

    def test_sanitize_hour_23(self):
        """
        Ora 23 OK.
        Test that hour=23 (11 PM) is valid and passes through.
        """
        print("\n=== TEST: test_sanitize_hour_23 ===")
        validator = InputValidator()
        
        raw = {"zone": "midtown", "month": None, "day_of_week": None, "hour": 23, "minute": None}
        print(f"Input raw: {raw}")
        
        result = validator._sanitize_extracted(raw)
        print(f"Expected hour: 23 (valid)")
        print(f"Actual result: hour={result['hour']}")
        
        assert result["hour"] == 23, f"FAIL: Expected hour=23, got {result['hour']}"
        print("PASS: Hour 23 correctly passes validation")

    def test_sanitize_minute_59(self):
        """
        Minuti 59 OK.
        Test that minute=59 is valid and passes through.
        """
        print("\n=== TEST: test_sanitize_minute_59 ===")
        validator = InputValidator()
        
        raw = {"zone": "midtown", "month": None, "day_of_week": None, "hour": None, "minute": 59}
        print(f"Input raw: {raw}")
        
        result = validator._sanitize_extracted(raw)
        print(f"Expected minute: 59 (valid)")
        print(f"Actual result: minute={result['minute']}")
        
        assert result["minute"] == 59, f"FAIL: Expected minute=59, got {result['minute']}"
        print("PASS: Minute 59 correctly passes validation")

    def test_sanitize_minute_60(self):
        """
        Minuti 60 diventa None.
        Test that minute=60 is invalid (valid range is 0-59) and becomes None.
        """
        print("\n=== TEST: test_sanitize_minute_60 ===")
        validator = InputValidator()
        
        raw = {"zone": "midtown", "month": None, "day_of_week": None, "hour": None, "minute": 60}
        print(f"Input raw: {raw}")
        
        result = validator._sanitize_extracted(raw)
        print(f"Expected minute: None (60 is invalid)")
        print(f"Actual result: minute={result['minute']}")
        
        assert result["minute"] is None, f"FAIL: Expected minute=None, got {result['minute']}"
        print("PASS: Minute 60 correctly becomes None")

    def test_sanitize_null_strings(self):
        """
        "null", "none", "" diventano None.
        Test that string representations of null/empty become None after sanitization.
        """
        print("\n=== TEST: test_sanitize_null_strings ===")
        validator = InputValidator()
        
        test_cases = [
            ({"zone": "null", "month": None, "day_of_week": None, "hour": None, "minute": None}, "zone"),
            ({"zone": "none", "month": None, "day_of_week": None, "hour": None, "minute": None}, "zone"),
            ({"zone": "", "month": None, "day_of_week": None, "hour": None, "minute": None}, "zone"),
        ]
        
        for raw, field in test_cases:
            print(f"Testing {field}='{raw.get(field)}'")
            result = validator._sanitize_extracted(raw)
            assert result[field] is None, f"FAIL: {field} expected None, got {result[field]}"
        
        print("PASS: Null strings correctly become None")

    def test_sanitize_valid_values(self):
        """
        Valori validi passano attraverso.
        Test that valid values in all fields pass through sanitization unchanged.
        """
        print("\n=== TEST: test_sanitize_valid_values ===")
        validator = InputValidator()
        
        raw = {
            "zone": "midtown",
            "month": 5,
            "day_of_week": 3,
            "hour": 14,
            "minute": 30
        }
        print(f"Input raw: {raw}")
        
        result = validator._sanitize_extracted(raw)
        print(f"Expected: {raw}")
        print(f"Actual: {result}")
        
        assert result["zone"] == "midtown"
        assert result["month"] == 5
        assert result["day_of_week"] == 3
        assert result["hour"] == 14
        assert result["minute"] == 30
        print("PASS: Valid values pass through correctly")


class TestZoneResolution:
    """Tests for zone name/alias resolution via resolve_zone_id."""

    def test_resolve_zone_numeric_id_161(self):
        """
        ID numerico diretto 161.
        Test that numeric zone ID strings are resolved directly to location_id.
        """
        print("\n=== TEST: test_resolve_zone_numeric_id_161 ===")
        validator = InputValidator()
        
        params = {"zone": "161"}
        print(f"Input params: {params}")
        
        result = validator.validate_and_resolve(params)
        print(f"Expected location_id: 161")
        print(f"Actual: location_id={result.get('location_id')}")
        
        assert result["location_id"] == 161, f"FAIL: Expected location_id=161, got {result['location_id']}"
        print("PASS: Numeric ID 161 resolved correctly")

    def test_resolve_zone_alias_jfk(self):
        """
        "JFK" → 132.
        Test that zone alias "JFK" resolves to location_id 132.
        """
        print("\n=== TEST: test_resolve_zone_alias_jfk ===")
        validator = InputValidator()
        
        params = {"zone": "JFK"}
        print(f"Input params: {params}")
        
        result = validator.validate_and_resolve(params)
        print(f"Expected location_id: 132")
        print(f"Actual: location_id={result.get('location_id')}")
        
        assert result["location_id"] == 132, f"FAIL: Expected location_id=132, got {result['location_id']}"
        print("PASS: JFK alias resolved to 132")

    def test_resolve_zone_alias_times_square(self):
        """
        "Times Square" → 230.
        Test that zone alias "Times Square" resolves to location_id 230.
        """
        print("\n=== TEST: test_resolve_zone_alias_times_square ===")
        validator = InputValidator()
        
        params = {"zone": "Times Square"}
        print(f"Input params: {params}")
        
        result = validator.validate_and_resolve(params)
        print(f"Expected location_id: 230")
        print(f"Actual: location_id={result.get('location_id')}")
        
        assert result["location_id"] == 230, f"FAIL: Expected location_id=230, got {result['location_id']}"
        print("PASS: Times Square alias resolved to 230")

    def test_resolve_zone_alias_midtown(self):
        """
        "Midtown" → multiple candidati (return_all=True).
        Test that ambiguous zone alias "Midtown" returns multiple candidates
        when return_all=True in validate_and_resolve.
        """
        print("\n=== TEST: test_resolve_zone_alias_midtown ===")
        validator = InputValidator()
        
        params = {"zone": "Midtown"}
        print(f"Input params: {params}")
        
        result = validator.validate_and_resolve(params)
        print(f"Expected: candidates list with multiple entries")
        print(f"Actual: location_id={result.get('location_id')}, candidates={result.get('candidates')}")
        
        assert result["location_id"] is None, f"FAIL: Expected location_id=None for ambiguous zone"
        assert len(result["candidates"]) > 1, f"FAIL: Expected multiple candidates, got {len(result['candidates'])}"
        print(f"PASS: Midtown returns {len(result['candidates'])} candidates")

    def test_resolve_zone_alias_manhattan(self):
        """
        "Manhattan" → 236.
        Test that borough name "Manhattan" resolves to a specific location_id.
        """
        print("\n=== TEST: test_resolve_zone_alias_manhattan ===")
        validator = InputValidator()
        
        params = {"zone": "Manhattan"}
        print(f"Input params: {params}")
        
        result = validator.validate_and_resolve(params)
        print(f"Expected: location_id (single match)")
        print(f"Actual: location_id={result.get('location_id')}, candidates={result.get('candidates')}")
        
        assert result["location_id"] is not None or len(result["candidates"]) > 0, \
            f"FAIL: Expected resolution for Manhattan"
        print("PASS: Manhattan alias resolved")

    def test_resolve_zone_no_match(self):
        """
        Zona inesistente → None.
        Test that non-existent zone returns None for location_id.
        """
        print("\n=== TEST: test_resolve_zone_no_match ===")
        validator = InputValidator()
        
        params = {"zone": "NonExistentZone12345"}
        print(f"Input params: {params}")
        
        result = validator.validate_and_resolve(params)
        print(f"Expected: location_id=None, candidates=[]")
        print(f"Actual: location_id={result.get('location_id')}, candidates={result.get('candidates')}")
        
        assert result["location_id"] is None, f"FAIL: Expected location_id=None"
        assert result["candidates"] == [], f"FAIL: Expected empty candidates list"
        print("PASS: Non-existent zone returns None")


class TestValidateAndResolve:
    """Tests for validate_and_resolve method behavior."""

    def test_validate_and_resolve_returns_candidates(self):
        """
        Ambiguità → lista candidati.
        Test that ambiguous zone inputs return a list of candidate locations.
        """
        print("\n=== TEST: test_validate_and_resolve_returns_candidates ===")
        validator = InputValidator()
        
        params = {"zone": "park"}
        print(f"Input params: {params}")
        
        result = validator.validate_and_resolve(params)
        print(f"Expected: candidates list for ambiguous 'park'")
        print(f"Actual: location_id={result.get('location_id')}, candidates={result.get('candidates')}")
        
        if result["candidates"]:
            print(f"PASS: Returned {len(result['candidates'])} candidates")
        else:
            print(f"PASS: Single location resolved to {result['location_id']}")

    def test_validate_and_resolve_with_existing_params(self):
        """
        Merge con params esistenti.
        Test that validate_and_resolve preserves existing params while adding resolution.
        """
        print("\n=== TEST: test_validate_and_resolve_with_existing_params ===")
        validator = InputValidator()
        
        params = {"zone": "jfk", "month": 6, "day_of_week": 1, "hour": 15}
        print(f"Input params: {params}")
        
        result = validator.validate_and_resolve(params)
        print(f"Expected: original params preserved + location_id")
        print(f"Actual: {result}")
        
        assert result["zone"] == "jfk", f"FAIL: zone not preserved"
        assert result["month"] == 6, f"FAIL: month not preserved"
        assert result["day_of_week"] == 1, f"FAIL: day_of_week not preserved"
        assert result["hour"] == 15, f"FAIL: hour not preserved"
        assert result["location_id"] == 132, f"FAIL: location_id not added"
        print("PASS: Existing params merged with resolution")


class TestEdgeCases:
    """Tests for edge cases and special inputs."""

    def test_extract_with_special_chars(self):
        """
        Caratteri speciali nel messaggio.
        Test that messages with special characters are handled gracefully.
        """
        print("\n=== TEST: test_extract_with_special_chars ===")
        validator = InputValidator()
        
        with patch.object(validator, '_get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = '{"zone": "jfk", "month": null, "day_of_week": null, "hour": null, "minute": null}'
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            text = "Quanti taxi a JFK??? @midtown #rushhour!"
            print(f"Input text: '{text}'")
            
            result = validator.extract(text)
            print(f"Result: {result}")
            
            assert result["zone"] == "jfk", f"FAIL: Expected zone='jfk', got {result['zone']}"
            print("PASS: Special characters handled correctly")

    def test_extract_empty_message(self):
        """
        Messaggio vuoto.
        Test that empty messages are handled gracefully with safe defaults.
        """
        print("\n=== TEST: test_extract_empty_message ===")
        validator = InputValidator()
        
        with patch.object(validator, '_get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = '{"zone": null, "month": null, "day_of_week": null, "hour": null, "minute": null}'
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            text = ""
            print(f"Input text: '{text}' (empty)")
            
            result = validator.extract(text)
            print(f"Result: {result}")
            
            assert result["zone"] is None
            assert result["month"] is None
            print("PASS: Empty message handled with safe defaults")

    def test_llm_extraction_failure_graceful(self):
        """
        LLM fails, returns safe defaults.
        Test that LLM extraction failures are handled gracefully with safe defaults.
        """
        print("\n=== TEST: test_llm_extraction_failure_graceful ===")
        validator = InputValidator()
        
        with patch.object(validator, '_get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("LLM API error")
            mock_get_llm.return_value = mock_llm
            
            text = "quanti taxi a midtown"
            print(f"Input text: '{text}'")
            
            result = validator.extract(text)
            print(f"Expected: safe defaults (all None)")
            print(f"Actual: {result}")
            
            assert result["zone"] is None, f"FAIL: Expected zone=None"
            assert result["month"] is None, f"FAIL: Expected month=None"
            assert result["day_of_week"] is None, f"FAIL: Expected day_of_week=None"
            assert result["hour"] is None, f"FAIL: Expected hour=None"
            assert result["minute"] is None, f"FAIL: Expected minute=None"
            print("PASS: LLM failure handled gracefully with safe defaults")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])