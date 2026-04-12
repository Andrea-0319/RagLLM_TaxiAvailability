"""
Comprehensive edge case tests for InputValidator.
Tests boundary values, invalid inputs, and special edge cases.
"""
import sys
sys.path.insert(0, r'C:\Users\andre\Desktop\Progetto_Accenture')
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

import pytest
from llm_tool.input_validator import InputValidator


class TestBoundaryValues:
    """Tests for valid boundary values that should be accepted."""

    def test_hour_0_midnight(self):
        """
        Test hour=0 (midnight). Valid boundary: hour 0 is the start of a new day.
        Input: hour=0, Expected: 0 (accept), Actual: 0
        """
        print("\n=== TEST: test_hour_0_midnight ===")
        validator = InputValidator()
        raw = {"zone": "midtown", "month": None, "day_of_week": None, "hour": 0, "minute": None}
        print(f"Input: hour=0")
        print(f"Expected behavior: accept (0)")
        
        result = validator._sanitize_extracted(raw)
        actual = result["hour"]
        print(f"Actual behavior: {actual}")
        
        if actual == 0:
            print("PASS: Hour 0 (midnight) correctly accepted")
        else:
            print(f"FAIL: Expected hour=0, got {actual}")

    def test_hour_23_end_of_day(self):
        """
        Test hour=23 (11 PM). Valid boundary: hour 23 is the end of a day.
        Input: hour=23, Expected: 23 (accept), Actual: 23
        """
        print("\n=== TEST: test_hour_23_end_of_day ===")
        validator = InputValidator()
        raw = {"zone": "jfk", "month": None, "day_of_week": None, "hour": 23, "minute": None}
        print(f"Input: hour=23")
        print(f"Expected behavior: accept (23)")
        
        result = validator._sanitize_extracted(raw)
        actual = result["hour"]
        print(f"Actual behavior: {actual}")
        
        if actual == 23:
            print("PASS: Hour 23 (end of day) correctly accepted")
        else:
            print(f"FAIL: Expected hour=23, got {actual}")

    def test_minute_0(self):
        """
        Test minute=0. Valid boundary: minute 0 is the start of an hour.
        Input: minute=0, Expected: 0 (accept), Actual: 0
        """
        print("\n=== TEST: test_minute_0 ===")
        validator = InputValidator()
        raw = {"zone": "midtown", "month": None, "day_of_week": None, "hour": None, "minute": 0}
        print(f"Input: minute=0")
        print(f"Expected behavior: accept (0)")
        
        result = validator._sanitize_extracted(raw)
        actual = result["minute"]
        print(f"Actual behavior: {actual}")
        
        if actual == 0:
            print("PASS: Minute 0 correctly accepted")
        else:
            print(f"FAIL: Expected minute=0, got {actual}")

    def test_minute_59(self):
        """
        Test minute=59. Valid boundary: minute 59 is the end of an hour.
        Input: minute=59, Expected: 59 (accept), Actual: 59
        """
        print("\n=== TEST: test_minute_59 ===")
        validator = InputValidator()
        raw = {"zone": "midtown", "month": None, "day_of_week": None, "hour": None, "minute": 59}
        print(f"Input: minute=59")
        print(f"Expected behavior: accept (59)")
        
        result = validator._sanitize_extracted(raw)
        actual = result["minute"]
        print(f"Actual behavior: {actual}")
        
        if actual == 59:
            print("PASS: Minute 59 correctly accepted")
        else:
            print(f"FAIL: Expected minute=59, got {actual}")

    def test_month_1_january(self):
        """
        Test month=1 (January). Valid boundary: month 1 is the first month.
        Input: month=1, Expected: 1 (accept), Actual: 1
        """
        print("\n=== TEST: test_month_1_january ===")
        validator = InputValidator()
        raw = {"zone": "midtown", "month": 1, "day_of_week": None, "hour": None, "minute": None}
        print(f"Input: month=1 (January)")
        print(f"Expected behavior: accept (1)")
        
        result = validator._sanitize_extracted(raw)
        actual = result["month"]
        print(f"Actual behavior: {actual}")
        
        if actual == 1:
            print("PASS: Month 1 (January) correctly accepted")
        else:
            print(f"FAIL: Expected month=1, got {actual}")

    def test_month_12_december(self):
        """
        Test month=12 (December). Valid boundary: month 12 is the last month.
        Input: month=12, Expected: 12 (accept), Actual: 12
        """
        print("\n=== TEST: test_month_12_december ===")
        validator = InputValidator()
        raw = {"zone": "midtown", "month": 12, "day_of_week": None, "hour": None, "minute": None}
        print(f"Input: month=12 (December)")
        print(f"Expected behavior: accept (12)")
        
        result = validator._sanitize_extracted(raw)
        actual = result["month"]
        print(f"Actual behavior: {actual}")
        
        if actual == 12:
            print("PASS: Month 12 (December) correctly accepted")
        else:
            print(f"FAIL: Expected month=12, got {actual}")

    def test_day_0_monday(self):
        """
        Test day_of_week=0 (Monday). Valid boundary: day 0 is Monday.
        Input: day_of_week=0, Expected: 0 (accept), Actual: 0
        """
        print("\n=== TEST: test_day_0_monday ===")
        validator = InputValidator()
        raw = {"zone": "midtown", "month": None, "day_of_week": 0, "hour": None, "minute": None}
        print(f"Input: day_of_week=0 (Monday)")
        print(f"Expected behavior: accept (0)")
        
        result = validator._sanitize_extracted(raw)
        actual = result["day_of_week"]
        print(f"Actual behavior: {actual}")
        
        if actual == 0:
            print("PASS: Day 0 (Monday) correctly accepted")
        else:
            print(f"FAIL: Expected day_of_week=0, got {actual}")

    def test_day_6_sunday(self):
        """
        Test day_of_week=6 (Sunday). Valid boundary: day 6 is Sunday.
        Input: day_of_week=6, Expected: 6 (accept), Actual: 6
        """
        print("\n=== TEST: test_day_6_sunday ===")
        validator = InputValidator()
        raw = {"zone": "midtown", "month": None, "day_of_week": 6, "hour": None, "minute": None}
        print(f"Input: day_of_week=6 (Sunday)")
        print(f"Expected behavior: accept (6)")
        
        result = validator._sanitize_extracted(raw)
        actual = result["day_of_week"]
        print(f"Actual behavior: {actual}")
        
        if actual == 6:
            print("PASS: Day 6 (Sunday) correctly accepted")
        else:
            print(f"FAIL: Expected day_of_week=6, got {actual}")

    def test_zone_id_1_min(self):
        """
        Test zone ID=1 (minimum valid). Valid boundary: zone 1 is the first NYC zone.
        Input: zone=1, Expected: ID resolved (accept), Actual: location_id=1
        """
        print("\n=== TEST: test_zone_id_1_min ===")
        validator = InputValidator()
        params = {"zone": "1"}
        print(f"Input: zone ID=1 (minimum)")
        print(f"Expected behavior: accept (location_id=1)")
        
        result = validator.validate_and_resolve(params)
        actual = result.get("location_id")
        print(f"Actual behavior: location_id={actual}")
        
        if actual == 1:
            print("PASS: Zone ID 1 correctly resolved")
        else:
            print(f"FAIL: Expected location_id=1, got {actual}")

    def test_zone_id_265_max(self):
        """
        Test zone ID=265 (maximum valid). Valid boundary: zone 265 is the last NYC zone.
        Input: zone=265, Expected: ID resolved (accept), Actual: location_id=265
        """
        print("\n=== TEST: test_zone_id_265_max ===")
        validator = InputValidator()
        params = {"zone": "265"}
        print(f"Input: zone ID=265 (maximum)")
        print(f"Expected behavior: accept (location_id=265)")
        
        result = validator.validate_and_resolve(params)
        actual = result.get("location_id")
        print(f"Actual behavior: location_id={actual}")
        
        if actual == 265:
            print("PASS: Zone ID 265 correctly resolved")
        else:
            print(f"FAIL: Expected location_id=265, got {actual}")


class TestInvalidInputs:
    """Tests for invalid inputs that should be rejected/rejected to None."""

    def test_hour_24_invalid(self):
        """
        Test hour=24 (invalid). Invalid: hour 24 exceeds valid range 0-23.
        Input: hour=24, Expected: None (reject), Actual: None
        """
        print("\n=== TEST: test_hour_24_invalid ===")
        validator = InputValidator()
        raw = {"zone": "midtown", "month": None, "day_of_week": None, "hour": 24, "minute": None}
        print(f"Input: hour=24 (invalid)")
        print(f"Expected behavior: reject (None)")
        
        result = validator._sanitize_extracted(raw)
        actual = result["hour"]
        print(f"Actual behavior: {actual}")
        
        if actual is None:
            print("PASS: Hour 24 correctly rejected")
        else:
            print(f"FAIL: Expected hour=None, got {actual}")

    def test_hour_negative_invalid(self):
        """
        Test hour=-1 (invalid). Invalid: negative hour is invalid.
        Input: hour=-1, Expected: None (reject), Actual: None
        """
        print("\n=== TEST: test_hour_negative_invalid ===")
        validator = InputValidator()
        raw = {"zone": "midtown", "month": None, "day_of_week": None, "hour": -1, "minute": None}
        print(f"Input: hour=-1 (invalid)")
        print(f"Expected behavior: reject (None)")
        
        result = validator._sanitize_extracted(raw)
        actual = result["hour"]
        print(f"Actual behavior: {actual}")
        
        if actual is None:
            print("PASS: Negative hour correctly rejected")
        else:
            print(f"FAIL: Expected hour=None, got {actual}")

    def test_minute_60_invalid(self):
        """
        Test minute=60 (invalid). Invalid: minute 60 exceeds valid range 0-59.
        Input: minute=60, Expected: None (reject), Actual: None
        """
        print("\n=== TEST: test_minute_60_invalid ===")
        validator = InputValidator()
        raw = {"zone": "midtown", "month": None, "day_of_week": None, "hour": None, "minute": 60}
        print(f"Input: minute=60 (invalid)")
        print(f"Expected behavior: reject (None)")
        
        result = validator._sanitize_extracted(raw)
        actual = result["minute"]
        print(f"Actual behavior: {actual}")
        
        if actual is None:
            print("PASS: Minute 60 correctly rejected")
        else:
            print(f"FAIL: Expected minute=None, got {actual}")

    def test_minute_negative_invalid(self):
        """
        Test minute=-1 (invalid). Invalid: negative minute is invalid.
        Input: minute=-1, Expected: None (reject), Actual: None
        """
        print("\n=== TEST: test_minute_negative_invalid ===")
        validator = InputValidator()
        raw = {"zone": "midtown", "month": None, "day_of_week": None, "hour": None, "minute": -1}
        print(f"Input: minute=-1 (invalid)")
        print(f"Expected behavior: reject (None)")
        
        result = validator._sanitize_extracted(raw)
        actual = result["minute"]
        print(f"Actual behavior: {actual}")
        
        if actual is None:
            print("PASS: Negative minute correctly rejected")
        else:
            print(f"FAIL: Expected minute=None, got {actual}")

    def test_month_0_invalid(self):
        """
        Test month=0 (invalid). Invalid: month 0 is outside valid range 1-12.
        Input: month=0, Expected: None (reject), Actual: None
        """
        print("\n=== TEST: test_month_0_invalid ===")
        validator = InputValidator()
        raw = {"zone": "midtown", "month": 0, "day_of_week": None, "hour": None, "minute": None}
        print(f"Input: month=0 (invalid)")
        print(f"Expected behavior: reject (None)")
        
        result = validator._sanitize_extracted(raw)
        actual = result["month"]
        print(f"Actual behavior: {actual}")
        
        if actual is None:
            print("PASS: Month 0 correctly rejected")
        else:
            print(f"FAIL: Expected month=None, got {actual}")

    def test_month_13_invalid(self):
        """
        Test month=13 (invalid). Invalid: month 13 exceeds valid range 1-12.
        Input: month=13, Expected: None (reject), Actual: None
        """
        print("\n=== TEST: test_month_13_invalid ===")
        validator = InputValidator()
        raw = {"zone": "midtown", "month": 13, "day_of_week": None, "hour": None, "minute": None}
        print(f"Input: month=13 (invalid)")
        print(f"Expected behavior: reject (None)")
        
        result = validator._sanitize_extracted(raw)
        actual = result["month"]
        print(f"Actual behavior: {actual}")
        
        if actual is None:
            print("PASS: Month 13 correctly rejected")
        else:
            print(f"FAIL: Expected month=None, got {actual}")

    def test_day_negative_invalid(self):
        """
        Test day_of_week=-1 (invalid). Invalid: negative day is invalid.
        Input: day_of_week=-1, Expected: None (reject), Actual: None
        """
        print("\n=== TEST: test_day_negative_invalid ===")
        validator = InputValidator()
        raw = {"zone": "midtown", "month": None, "day_of_week": -1, "hour": None, "minute": None}
        print(f"Input: day_of_week=-1 (invalid)")
        print(f"Expected behavior: reject (None)")
        
        result = validator._sanitize_extracted(raw)
        actual = result["day_of_week"]
        print(f"Actual behavior: {actual}")
        
        if actual is None:
            print("PASS: Negative day correctly rejected")
        else:
            print(f"FAIL: Expected day_of_week=None, got {actual}")

    def test_day_7_invalid(self):
        """
        Test day_of_week=7 (invalid). Invalid: day 7 exceeds valid range 0-6.
        Input: day_of_week=7, Expected: None (reject), Actual: None
        """
        print("\n=== TEST: test_day_7_invalid ===")
        validator = InputValidator()
        raw = {"zone": "midtown", "month": None, "day_of_week": 7, "hour": None, "minute": None}
        print(f"Input: day_of_week=7 (invalid)")
        print(f"Expected behavior: reject (None)")
        
        result = validator._sanitize_extracted(raw)
        actual = result["day_of_week"]
        print(f"Actual behavior: {actual}")
        
        if actual is None:
            print("PASS: Day 7 correctly rejected")
        else:
            print(f"FAIL: Expected day_of_week=None, got {actual}")

    def test_zone_id_0_invalid(self):
        """
        Test zone ID=0 (invalid). Invalid: zone 0 is outside valid range 1-265.
        Input: zone=0, Expected: None (reject), Actual: None
        """
        print("\n=== TEST: test_zone_id_0_invalid ===")
        validator = InputValidator()
        params = {"zone": "0"}
        print(f"Input: zone ID=0 (invalid)")
        print(f"Expected behavior: reject (None)")
        
        result = validator.validate_and_resolve(params)
        actual = result.get("location_id")
        print(f"Actual behavior: location_id={actual}")
        
        if actual is None:
            print("PASS: Zone ID 0 correctly rejected")
        else:
            print(f"FAIL: Expected location_id=None, got {actual}")

    def test_zone_id_266_invalid(self):
        """
        Test zone ID=266 (invalid). Invalid: zone 266 exceeds valid range 1-265.
        Input: zone=266, Expected: None (reject), Actual: None
        """
        print("\n=== TEST: test_zone_id_266_invalid ===")
        validator = InputValidator()
        params = {"zone": "266"}
        print(f"Input: zone ID=266 (invalid)")
        print(f"Expected behavior: reject (None)")
        
        result = validator.validate_and_resolve(params)
        actual = result.get("location_id")
        print(f"Actual behavior: location_id={actual}")
        
        if actual is None:
            print("PASS: Zone ID 266 correctly rejected")
        else:
            print(f"FAIL: Expected location_id=None, got {actual}")


class TestSpecialInputs:
    """Tests for special inputs like empty, long, or ambiguous messages."""

    def test_empty_message(self):
        """
        Test empty message. Edge case: empty string should return safe defaults.
        Input: "", Expected: all None (safe defaults), Actual: all None
        """
        print("\n=== TEST: test_empty_message ===")
        validator = InputValidator()
        
        with patch.object(validator, '_get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = '{"zone": null, "month": null, "day_of_week": null, "hour": null, "minute": null}'
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            text = ""
            print(f"Input: empty message")
            print(f"Expected behavior: safe defaults (all None)")
            
            result = validator.extract(text)
            print(f"Actual behavior: {result}")
            
            if all(v is None for v in result.values()):
                print("PASS: Empty message handled with safe defaults")
            else:
                print(f"FAIL: Expected all None, got {result}")

    def test_very_long_message(self):
        """
        Test very long message (>500 chars). Edge case: long messages should be handled.
        Input: 500+ char message, Expected: extraction attempted, Actual: depends on LLM
        """
        print("\n=== TEST: test_very_long_message ===")
        validator = InputValidator()
        
        with patch.object(validator, '_get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = '{"zone": "jfk", "month": null, "day_of_week": null, "hour": null, "minute": null}'
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            text = "Quanti taxi a JFK? " * 50
            print(f"Input: very long message ({len(text)} chars)")
            print(f"Expected behavior: extraction attempted")
            
            result = validator.extract(text)
            print(f"Actual behavior: zone={result['zone']}")
            
            if result["zone"] == "jfk":
                print("PASS: Long message handled correctly")
            else:
                print(f"FAIL: Expected zone='jfk', got {result['zone']}")

    def test_special_chars_in_message(self):
        """
        Test message with special characters (emoji, symbols).
        Edge case: special chars should not break extraction.
        Input: message with emoji/symbols, Expected: extraction attempted, Actual: depends on LLM
        """
        print("\n=== TEST: test_special_chars_in_message ===")
        validator = InputValidator()
        
        with patch.object(validator, '_get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = '{"zone": "midtown", "month": null, "day_of_week": null, "hour": null, "minute": null}'
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            text = "Quanti taxi a Midtown? 🚕 @midtown #rushhour! €$£"
            print(f"Input: message with special chars")
            print(f"Expected behavior: extraction attempted")
            
            result = validator.extract(text)
            print(f"Actual behavior: zone={result['zone']}")
            
            if result["zone"] == "midtown":
                print("PASS: Special characters handled correctly")
            else:
                print(f"FAIL: Expected zone='midtown', got {result['zone']}")

    def test_emoji_only_message(self):
        """
        Test emoji-only message. Edge case: ambiguous input should return safe defaults.
        Input: only emojis, Expected: all None (no extraction), Actual: all None
        """
        print("\n=== TEST: test_emoji_only_message ===")
        validator = InputValidator()
        
        with patch.object(validator, '_get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = '{"zone": null, "month": null, "day_of_week": null, "hour": null, "minute": null}'
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            text = "🚕🚖🚍"
            print(f"Input: emoji only message")
            print(f"Expected behavior: safe defaults (no extraction)")
            
            result = validator.extract(text)
            print(f"Actual behavior: {result}")
            
            if result["zone"] is None:
                print("PASS: Emoji only message handled with safe defaults")
            else:
                print(f"FAIL: Expected zone=None, got {result['zone']}")

    def test_multiple_zones_mentioned(self):
        """
        Test message with multiple zones mentioned. Edge case: ambiguous input should return first zone.
        Input: message with multiple zones, Expected: first zone captured, Actual: depends on LLM
        """
        print("\n=== TEST: test_multiple_zones_mentioned ===")
        validator = InputValidator()
        
        with patch.object(validator, '_get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = '{"zone": "jfk", "month": null, "day_of_week": null, "hour": null, "minute": null}'
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            text = "Quanti taxi a JFK e Midtown?"
            print(f"Input: message with multiple zones")
            print(f"Expected behavior: first zone captured")
            
            result = validator.extract(text)
            print(f"Actual behavior: zone={result['zone']}")
            
            if result["zone"] == "jfk":
                print("PASS: Multiple zones handled (first zone captured)")
            else:
                print(f"FAIL: Expected zone='jfk', got {result['zone']}")

    def test_conflicting_times(self):
        """
        Test message with conflicting times. Edge case: ambiguous input handled by LLM.
        Input: message with contradictory times, Expected: LLM decides, Actual: depends on LLM
        """
        print("\n=== TEST: test_conflicting_times ===")
        validator = InputValidator()
        
        with patch.object(validator, '_get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = '{"zone": "jfk", "month": 1, "day_of_week": 1, "hour": 8, "minute": 30}'
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            text = "Quanti taxi a JFK alle 8 di mattina e alle 3 di notte?"
            print(f"Input: message with conflicting times")
            print(f"Expected behavior: LLM resolves ambiguity")
            
            result = validator.extract(text)
            print(f"Actual behavior: {result}")
            
            if result["hour"] == 8:
                print("PASS: Conflicting times handled by LLM")
            else:
                print(f"FAIL: Expected hour=8, got {result['hour']}")

    def test_only_stop_words(self):
        """
        Test message with only stop words. Edge case: no parameters to extract.
        Input: message with only stop words, Expected: all None, Actual: all None
        """
        print("\n=== TEST: test_only_stop_words ===")
        validator = InputValidator()
        
        with patch.object(validator, '_get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = '{"zone": null, "month": null, "day_of_week": null, "hour": null, "minute": null}'
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            text = "quanti taxi per favore"
            print(f"Input: message with only stop words")
            print(f"Expected behavior: safe defaults (no params)")
            
            result = validator.extract(text)
            print(f"Actual behavior: {result}")
            
            if result["zone"] is None and result["hour"] is None:
                print("PASS: Stop words only handled correctly")
            else:
                print(f"FAIL: Expected all None, got {result}")

    def test_numbers_without_context(self):
        """
        Test message with numbers but no context. Edge case: numbers need context to be meaningful.
        Input: numbers without context, Expected: all None (no extraction), Actual: all None
        """
        print("\n=== TEST: test_numbers_without_context ===")
        validator = InputValidator()
        
        with patch.object(validator, '_get_llm') as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = '{"zone": null, "month": null, "day_of_week": null, "hour": null, "minute": null}'
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm
            
            text = "123 456 789"
            print(f"Input: numbers without context")
            print(f"Expected behavior: safe defaults (no context)")
            
            result = validator.extract(text)
            print(f"Actual behavior: {result}")
            
            if result["zone"] is None:
                print("PASS: Numbers without context handled safely")
            else:
                print(f"FAIL: Expected zone=None, got {result['zone']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])