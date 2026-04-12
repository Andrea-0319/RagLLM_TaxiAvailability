"""
Unit tests for TaxiPredictorModel and related utility functions.

These tests verify the core functionality of the taxi predictor module without
requiring actual ML model loading. We mock the model artifacts and test
utility functions, zone resolution, and data structures.
"""

import sys
sys.path.insert(0, r'C:\Users\andre\Desktop\Progetto_Accenture')

import pytest
from unittest.mock import MagicMock, patch
from llm_tool.taxi_predictor import TaxiPredictorModel, resolve_zone_id, get_predictor
from llm_tool.config import (
    hour_minute_to_half_bucket,
    half_bucket_to_time,
    CLASS_NAMES,
    CLASS_EMOJIS,
    BOROUGH_ENCODING,
    ZONE_ALIASES,
)


class TestResolveZoneId:
    """Tests for the resolve_zone_id function."""

    def test_resolve_zone_id_numeric_valid(self):
        """
        Test: Valid numeric ID within range 1-265.
        Justification: Verify that numeric zone IDs within the valid range are correctly resolved.
        Input: "161"
        Expected: 161
        """
        print("\n=== TEST: resolve_zone_id - Numeric Valid ===")
        print(f"Input: '161'")
        print(f"Expected: 161")
        result = resolve_zone_id("161")
        print(f"Actual: {result}")
        assert result == 161, f"FAIL: Expected 161, got {result}"
        print("PASS: Valid numeric ID resolved correctly")

    def test_resolve_zone_id_numeric_invalid_too_high(self):
        """
        Test: Numeric ID greater than 265 should return None.
        Justification: Verify that out-of-range numeric IDs are rejected.
        Input: "300"
        Expected: None
        """
        print("\n=== TEST: resolve_zone_id - Numeric Invalid Too High ===")
        print(f"Input: '300'")
        print(f"Expected: None")
        result = resolve_zone_id("300")
        print(f"Actual: {result}")
        assert result is None, f"FAIL: Expected None, got {result}"
        print("PASS: Out-of-range numeric ID correctly rejected")

    def test_resolve_zone_id_numeric_invalid_negative(self):
        """
        Test: Negative numeric ID should return None.
        Justification: Verify that negative numeric IDs are rejected.
        Input: "-5"
        Expected: None
        """
        print("\n=== TEST: resolve_zone_id - Numeric Invalid Negative ===")
        print(f"Input: '-5'")
        print(f"Expected: None")
        result = resolve_zone_id("-5")
        print(f"Actual: {result}")
        assert result is None, f"FAIL: Expected None, got {result}"
        print("PASS: Negative numeric ID correctly rejected")

    def test_resolve_zone_id_exact_alias_jfk(self):
        """
        Test: Exact alias match for JFK airport.
        Justification: Verify that known airport aliases resolve to correct LocationID.
        Input: "JFK"
        Expected: 132
        """
        print("\n=== TEST: resolve_zone_id - Exact Alias JFK ===")
        print(f"Input: 'JFK'")
        print(f"Expected: 132")
        result = resolve_zone_id("JFK")
        print(f"Actual: {result}")
        assert result == 132, f"FAIL: Expected 132, got {result}"
        print("PASS: JFK alias resolved to correct LocationID")

    def test_resolve_zone_id_exact_alias_lga(self):
        """
        Test: Exact alias match for LaGuardia airport.
        Justification: Verify that LGA alias resolves to correct LocationID.
        Input: "LGA"
        Expected: 138
        """
        print("\n=== TEST: resolve_zone_id - Exact Alias LGA ===")
        print(f"Input: 'LGA'")
        print(f"Expected: 138")
        result = resolve_zone_id("LGA")
        print(f"Actual: {result}")
        assert result == 138, f"FAIL: Expected 138, got {result}"
        print("PASS: LGA alias resolved to correct LocationID")

    def test_resolve_zone_id_exact_alias_manhattan(self):
        """
        Test: Exact alias match for Manhattan borough.
        Justification: Verify that Manhattan borough alias resolves to correct default zone.
        Input: "Manhattan"
        Expected: 236
        """
        print("\n=== TEST: resolve_zone_id - Exact Alias Manhattan ===")
        print(f"Input: 'Manhattan'")
        print(f"Expected: 236")
        result = resolve_zone_id("Manhattan")
        print(f"Actual: {result}")
        assert result == 236, f"FAIL: Expected 236, got {result}"
        print("PASS: Manhattan alias resolved to correct LocationID")

    def test_resolve_zone_id_partial_match_midtown(self):
        """
        Test: Partial substring match should return candidates.
        Justification: Verify that partial matches return candidate zones.
        Input: "midtown"
        Expected: List of matching zones or single ID
        """
        print("\n=== TEST: resolve_zone_id - Partial Match Midtown ===")
        print(f"Input: 'midtown'")
        result = resolve_zone_id("midtown")
        print(f"Actual: {result}")
        assert result is not None, f"FAIL: Expected candidates, got None"
        print(f"PASS: Partial match returned result")

    def test_resolve_zone_id_no_match(self):
        """
        Test: Non-existent zone should return None.
        Justification: Verify that completely unknown zones return None.
        Input: "xyznonexistentzone123"
        Expected: None
        """
        print("\n=== TEST: resolve_zone_id - No Match ===")
        print(f"Input: 'xyznonexistentzone123'")
        print(f"Expected: None")
        result = resolve_zone_id("xyznonexistentzone123")
        print(f"Actual: {result}")
        assert result is None, f"FAIL: Expected None, got {result}"
        print("PASS: Non-existent zone correctly returns None")

    def test_resolve_zone_id_case_insensitive(self):
        """
        Test: Case insensitivity for zone resolution.
        Justification: Verify that zone names are resolved regardless of case.
        Input: "jFk" (mixed case)
        Expected: 132 (same as "JFK")
        """
        print("\n=== TEST: resolve_zone_id - Case Insensitive ===")
        print(f"Input: 'jFk'")
        print(f"Expected: 132")
        result = resolve_zone_id("jFk")
        print(f"Actual: {result}")
        assert result == 132, f"FAIL: Expected 132, got {result}"
        print("PASS: Case insensitive matching works correctly")

    def test_resolve_zone_id_with_spaces(self):
        """
        Test: Zone name with extra spaces should still match.
        Justification: Verify that whitespace is properly stripped during resolution.
        Input: "jfk airport"
        Expected: 132
        """
        print("\n=== TEST: resolve_zone_id - With Spaces ===")
        print(f"Input: 'jfk airport'")
        print(f"Expected: 132")
        result = resolve_zone_id("jfk airport")
        print(f"Actual: {result}")
        assert result == 132, f"FAIL: Expected 132, got {result}"
        print("PASS: Zone with spaces resolved correctly")

    def test_resolve_zone_id_return_all_false(self):
        """
        Test: Single match with return_all=False returns int.
        Justification: Verify that single matches return the integer ID directly.
        Input: "jfk", return_all=False
        Expected: 132 (int)
        """
        print("\n=== TEST: resolve_zone_id - Return All False (Single Match) ===")
        print(f"Input: 'jfk', return_all=False")
        print(f"Expected: 132 (int)")
        result = resolve_zone_id("jfk", return_all=False)
        print(f"Actual: {result}, Type: {type(result)}")
        assert result == 132, f"FAIL: Expected 132, got {result}"
        assert isinstance(result, int), f"FAIL: Expected int, got {type(result)}"
        print("PASS: Single match returns int when return_all=False")

    def test_resolve_zone_id_return_all_true(self):
        """
        Test: Multiple matches with return_all=True returns list.
        Justification: Verify that ambiguous matches return list when return_all=True.
        Input: "midtown", return_all=True
        Expected: list of candidate dicts
        """
        print("\n=== TEST: resolve_zone_id - Return All True (Multiple Matches) ===")
        print(f"Input: 'midtown', return_all=True")
        result = resolve_zone_id("midtown", return_all=True)
        print(f"Actual: {result}, Type: {type(result)}")
        if isinstance(result, list):
            print(f"Number of candidates: {len(result)}")
            for cand in result[:3]:
                print(f"  - {cand}")
        assert isinstance(result, list), f"FAIL: Expected list, got {type(result)}"
        print("PASS: Multiple matches returns list when return_all=True")


class TestTimeConversion:
    """Tests for time conversion utilities."""

    def test_hour_minute_to_half_bucket(self):
        """
        Test: Convert hour and minute to half-hour bucket.
        Justification: Verify correct bucket calculation for various times.
        Input: hour=8, minute=15
        Expected: 16 (8*2 + 0)
        """
        print("\n=== TEST: hour_minute_to_half_bucket ===")
        print(f"Input: hour=8, minute=15")
        print(f"Expected: 16")
        result = hour_minute_to_half_bucket(8, 15)
        print(f"Actual: {result}")
        assert result == 16, f"FAIL: Expected 16, got {result}"
        print("PASS: hour_minute_to_half_bucket works correctly")

    def test_half_bucket_to_time_0(self):
        """
        Test: Bucket 0 represents first half-hour (00:00-00:30).
        Justification: Verify that bucket 0 converts to correct time range.
        Input: bucket=0
        Expected: (0, 0, 30)
        """
        print("\n=== TEST: half_bucket_to_time - Bucket 0 ===")
        print(f"Input: bucket=0")
        print(f"Expected: (0, 0, 30)")
        result = half_bucket_to_time(0)
        print(f"Actual: {result}")
        assert result == (0, 0, 30), f"FAIL: Expected (0, 0, 30), got {result}"
        print("PASS: Bucket 0 converts to 00:00-00:30")

    def test_half_bucket_to_time_1(self):
        """
        Test: Bucket 1 represents second half-hour (00:30-00:59).
        Justification: Verify that bucket 1 converts to correct time range.
        Input: bucket=1
        Expected: (0, 30, 59)
        """
        print("\n=== TEST: half_bucket_to_time - Bucket 1 ===")
        print(f"Input: bucket=1")
        print(f"Expected: (0, 30, 59)")
        result = half_bucket_to_time(1)
        print(f"Actual: {result}")
        assert result == (0, 30, 59), f"FAIL: Expected (0, 30, 59), got {result}"
        print("PASS: Bucket 1 converts to 00:30-00:59")

    def test_half_bucket_to_time_47(self):
        """
        Test: Bucket 47 represents last half-hour of day (23:30-23:59).
        Justification: Verify final bucket of the day converts correctly.
        Input: bucket=47
        Expected: (23, 30, 59)
        """
        print("\n=== TEST: half_bucket_to_time - Bucket 47 ===")
        print(f"Input: bucket=47")
        print(f"Expected: (23, 30, 59)")
        result = half_bucket_to_time(47)
        print(f"Actual: {result}")
        assert result == (23, 30, 59), f"FAIL: Expected (23, 30, 59), got {result}"
        print("PASS: Bucket 47 converts to 23:30-23:59")


class TestClassStructures:
    """Tests for CLASS_NAMES and CLASS_EMOJIS data structures."""

    def test_class_names_structure(self):
        """
        Test: CLASS_NAMES contains all 5 availability classes.
        Justification: Verify the class mapping has all required classes (0-4).
        Expected: Keys 0,1,2,3,4 all present
        """
        print("\n=== TEST: CLASS_NAMES Structure ===")
        print(f"Expected: 5 classes (0-4)")
        print(f"Actual keys: {list(CLASS_NAMES.keys())}")
        assert len(CLASS_NAMES) == 5, f"FAIL: Expected 5 classes, got {len(CLASS_NAMES)}"
        for i in range(5):
            assert i in CLASS_NAMES, f"FAIL: Missing class {i} in CLASS_NAMES"
            print(f"  Class {i}: {CLASS_NAMES[i]}")
        print("PASS: CLASS_NAMES has all 5 classes")

    def test_class_emojis_structure(self):
        """
        Test: CLASS_EMOJIS maps correctly to all 5 classes.
        Justification: Verify emoji mapping matches class names.
        Expected: All 5 classes have emoji mapping
        """
        print("\n=== TEST: CLASS_EMOJIS Structure ===")
        print(f"Expected: 5 emoji mappings")
        print(f"Actual keys: {list(CLASS_EMOJIS.keys())}")
        assert len(CLASS_EMOJIS) == 5, f"FAIL: Expected 5 emojis, got {len(CLASS_EMOJIS)}"
        for i in range(5):
            assert i in CLASS_EMOJIS, f"FAIL: Missing emoji for class {i}"
        print(f"  All 5 classes have emoji mappings [OK]")
        
        assert CLASS_EMOJIS[0] is not None, f"FAIL: Class 0 emoji is None"
        assert CLASS_EMOJIS[1] is not None, f"FAIL: Class 1 emoji is None"
        assert CLASS_EMOJIS[2] is not None, f"FAIL: Class 2 emoji is None"
        assert CLASS_EMOJIS[3] is not None, f"FAIL: Class 3 emoji is None"
        assert CLASS_EMOJIS[4] is not None, f"FAIL: Class 4 emoji is None"
        print("PASS: CLASS_EMOJIS mapping is correct")


class TestBoroughEncoding:
    """Tests for borough encoding values."""

    def test_borough_encoding(self):
        """
        Test: Borough encoding values are correctly defined.
        Justification: Verify all NYC boroughs have encoding values.
        Expected: 7 boroughs (including Unknown and EWR)
        """
        print("\n=== TEST: Borough Encoding ===")
        print(f"Expected: 7 boroughs (Bronx, Brooklyn, EWR, Manhattan, Queens, Staten Island, Unknown)")
        print(f"Actual: {BOROUGH_ENCODING}")
        
        expected_boroughs = {
            "Bronx": 0,
            "Brooklyn": 1,
            "EWR": 2,
            "Manhattan": 3,
            "Queens": 4,
            "Staten Island": 5,
            "Unknown": 6,
        }
        
        assert len(BOROUGH_ENCODING) == 7, f"FAIL: Expected 7 boroughs, got {len(BOROUGH_ENCODING)}"
        
        for borough, expected_code in expected_boroughs.items():
            assert borough in BOROUGH_ENCODING, f"FAIL: Missing borough {borough}"
            assert BOROUGH_ENCODING[borough] == expected_code, f"FAIL: {borough} encoding is {BOROUGH_ENCODING[borough]}, expected {expected_code}"
            print(f"  {borough}: {BOROUGH_ENCODING[borough]} [OK]")
        
        print("PASS: Borough encoding values are correct")


class TestZoneAliases:
    """Tests for zone aliases."""

    def test_zone_aliases_count(self):
        """
        Test: Verify that a substantial number of zone aliases are defined.
        Justification: Ensure the alias system has adequate coverage.
        Expected: At least 50 aliases defined
        """
        print("\n=== TEST: Zone Aliases Count ===")
        print(f"Expected: At least 50 aliases")
        alias_count = len(ZONE_ALIASES)
        print(f"Actual: {alias_count} aliases")
        
        assert alias_count >= 50, f"FAIL: Expected at least 50 aliases, got {alias_count}"
        
        key_aliases = ["jfk", "lga", "manhattan", "midtown manhattan", "times square", "central park"]
        for alias in key_aliases:
            if alias in ZONE_ALIASES:
                print(f"  '{alias}': {ZONE_ALIASES[alias]} [OK]")
            else:
                print(f"  '{alias}': NOT FOUND")
        
        print(f"PASS: Zone aliases count is {alias_count} (>= 50)")


class TestTaxiPredictorModel:
    """Tests for the TaxiPredictorModel class."""

    @patch('llm_tool.taxi_predictor._load_zone_lookup')
    def test_get_zone_info(self, mock_load_lookup):
        """
        Test: get_zone_info returns correct zone information.
        Justification: Verify zone info retrieval works correctly.
        """
        print("\n=== TEST: TaxiPredictorModel.get_zone_info ===")
        
        mock_df = MagicMock()
        mock_df.__getitem__ = MagicMock(return_value=MagicMock(
            iloc=MagicMock(return_value=MagicMock(
                __getitem__=MagicMock(side_effect=lambda k: {"Borough": "Manhattan", "Zone": "Midtown", "service_zone": "Yellow Zone"}[k])
            ))
        ))
        mock_load_lookup.return_value = mock_df
        
        predictor = TaxiPredictorModel()
        predictor._zone_lookup = mock_df
        
        result = predictor.get_zone_info(161)
        print(f"Result: {result}")
        
        assert "borough" in result, "FAIL: Missing borough in result"
        assert "zone" in result, "FAIL: Missing zone in result"
        print("PASS: get_zone_info returns correct structure")

    def test_predictor_singleton(self):
        """
        Test: TaxiPredictorModel follows singleton pattern.
        Justification: Verify that multiple instantiations return same instance.
        """
        print("\n=== TEST: TaxiPredictorModel Singleton ===")
        
        p1 = TaxiPredictorModel()
        p2 = TaxiPredictorModel()
        
        print(f"p1 id: {id(p1)}")
        print(f"p2 id: {id(p2)}")
        
        assert p1 is p2, "FAIL: TaxiPredictorModel is not a singleton"
        print("PASS: TaxiPredictorModel follows singleton pattern")

    @patch('llm_tool.taxi_predictor._load_zone_lookup')
    def test_get_zone_defaults(self, mock_load_lookup):
        """
        Test: get_zone_defaults returns valid defaults.
        Justification: Verify default values retrieval.
        """
        print("\n=== TEST: TaxiPredictorModel.get_zone_defaults ===")
        
        class MockSeries:
            def __init__(self):
                self._data = {
                    "unique_taxi_types": 3,
                    "avg_trip_duration_min": 17.8,
                    "borough_encoded": 6,
                    "service_zone_encoded": 3,
                }
            
            def to_dict(self):
                return self._data
        
        class MockLoc:
            def __getitem__(self, key):
                if key == 161:
                    return MockSeries()
                raise KeyError(key)
        
        class MockDataFrame:
            def __init__(self):
                self.loc = MockLoc()
            
            def __contains__(self, key):
                return True
            
            @property
            def index(self):
                return [161, 162, 163]
        
        predictor = TaxiPredictorModel()
        predictor._zone_defaults = MockDataFrame()
        
        result = predictor.get_zone_defaults(161)
        print(f"Result: {result}")
        
        assert "unique_taxi_types" in result, "FAIL: Missing unique_taxi_types"
        assert "avg_trip_duration_min" in result, "FAIL: Missing avg_trip_duration_min"
        print("PASS: get_zone_defaults returns valid defaults")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])