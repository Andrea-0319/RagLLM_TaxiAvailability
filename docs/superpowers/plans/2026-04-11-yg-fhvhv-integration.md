# YG+FHVHV Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Riccardo's Yellow/Green taxi LightGBM model as a new `@tool` in the chatbot, add a FHVHV "coming soon" stub tool, update the agent pipeline to route by `vehicle_type`, and keep the legacy generic tool deprecated-but-intact.

**Architecture:** New `llm_tool/yg_predictor.py` wraps the Riccardo model (3-class, no scaler, sin/cos feature engineering) and exposes a normalised output dict. Two new `@tool` functions are added to `taxi_predictor.py`. The agent's extractor and predictor nodes gain `vehicle_type` awareness. The formatter is updated to render multi-type YG results and the FHVHV coming-soon stub. The old generic tool remains in code but is hidden from the LLM.

**Tech Stack:** Python 3.11+, LightGBM, joblib, pandas, numpy, LangChain `@tool`, LangGraph, pytest, unittest.mock

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| **Create** | `llm_tool/yg_predictor.py` | Singleton wrapper for Riccardo's model; feature engineering, predict/predict_all, normalised output |
| **Modify** | `llm_tool/config.py` | Add `YG_MODEL_PATH`, `YG_CLASS_NAMES`, `YG_CLASS_EMOJIS`, `YG_CLASS_DESCRIPTIONS`, `VEHICLE_TYPE_DISPLAY` |
| **Modify** | `llm_tool/taxi_predictor.py` | Mark old tool deprecated, add `predict_yellow_green_availability` and `predict_fhvhv_availability` tools |
| **Modify** | `llm_tool/prompts.py` | Add `vehicle_type` extraction rule to `_EXTRACTION_SYSTEM_PROMPT` |
| **Modify** | `llm_tool/input_validator.py` | Extract & sanitize `vehicle_type` from LLM output |
| **Modify** | `llm_tool/agent.py` | Propagate `vehicle_type` in extractor, route in predictor_node, render in formatter |
| **Modify** | `llm_tool/i18n.py` | Add `fhvhv_coming_soon` key in IT+EN |
| **Create** | `output/yg_model_production.pkl` | Copy of `riccardo/taxi_lgbm_model_production.pkl` |
| **Create** | `tests/test_yg_predictor.py` | Unit tests for YGPredictor (all methods, mocked model) |
| **Modify** | `tests/conftest.py` | Add `mock_yg_predictor` fixture |
| **Modify** | `tests/test_agent_nodes.py` | Tests for `vehicle_type` propagation in extractor + predictor routing |
| **Modify** | `tests/test_integration.py` | End-to-end tests for YG and FHVHV paths |

---

## Task 1: Copy model artifact + add config constants

**Files:**
- Modify: `llm_tool/config.py`
- Create: `output/yg_model_production.pkl` (shell copy)

- [ ] **Step 1: Copy the model file**

```bash
cp riccardo/taxi_lgbm_model_production.pkl output/yg_model_production.pkl
```

Expected: file exists at `output/yg_model_production.pkl`, ~10 MB.

- [ ] **Step 2: Add constants to config.py**

In `llm_tool/config.py`, after the `ZONE_DEFAULTS_PATH` line (line 22), add:

```python
YG_MODEL_PATH = OUTPUT_DIR / "yg_model_production.pkl"
```

After the `CLASS_EMOJIS` block (after line 42), add:

```python
# ─── Yellow/Green Model (Riccardo) ──────────────────────────────────────
YG_CLASS_NAMES = {0: "Bassa", 1: "Media", 2: "Alta"}
YG_CLASS_EMOJIS = {0: "🔴", 1: "🟡", 2: "🟢"}
YG_CLASS_DESCRIPTIONS = {
    0: "trovare un taxi è difficile in questa zona e fascia oraria",
    1: "la disponibilità dei taxi è intermedia in questa zona e fascia oraria",
    2: "trovare un taxi è generalmente facile in questa zona e fascia oraria",
}
VEHICLE_TYPE_DISPLAY = {
    "yellow":         "Taxi Giallo 🟡",
    "green_hail":     "Taxi Verde (Hail) 🟢",
    "green_dispatch": "Taxi Verde (Dispatch) 🟢",
    "fhvhv":          "FHVHV (Uber/Lyft) 🚗",
}
```

- [ ] **Step 3: Verify config imports cleanly**

```bash
cd C:\Users\andre\Desktop\Progetto_Accenture
python -c "from llm_tool.config import YG_MODEL_PATH, YG_CLASS_NAMES, YG_CLASS_EMOJIS, YG_CLASS_DESCRIPTIONS, VEHICLE_TYPE_DISPLAY; print('OK', YG_MODEL_PATH)"
```

Expected output: `OK .../output/yg_model_production.pkl`

- [ ] **Step 4: Commit**

```bash
git add output/yg_model_production.pkl llm_tool/config.py
git commit -m "feat: copy YG model artifact and add config constants"
```

---

## Task 2: Create YGPredictor — write tests first (TDD)

**Files:**
- Create: `tests/test_yg_predictor.py`
- Create: `llm_tool/yg_predictor.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_yg_predictor.py`:

```python
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
```

- [ ] **Step 2: Run to confirm ALL fail (module not yet created)**

```bash
cd C:\Users\andre\Desktop\Progetto_Accenture
python -m pytest tests/test_yg_predictor.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'llm_tool.yg_predictor'` or similar.

- [ ] **Step 3: Implement `llm_tool/yg_predictor.py`**

Create `llm_tool/yg_predictor.py`:

```python
"""
Yellow/Green Taxi Predictor — wrapper for Riccardo's LightGBM model.

Model: taxi_lgbm_model_production.pkl
Classes: {0: "Bassa", 1: "Media", 2: "Alta"}
Feature engineering: sin/cos cyclical encoding (no StandardScaler).
"""
import logging
import warnings
from typing import Optional, List, Dict, Any

import joblib
import numpy as np
import pandas as pd

from .config import (
    YG_MODEL_PATH,
    YG_CLASS_NAMES,
    YG_CLASS_DESCRIPTIONS,
)
from .taxi_predictor import _load_zone_lookup  # reuse existing zone lookup loader

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Feature order must match training exactly
_YG_FEATURES = [
    "zone", "month", "quarter",
    "hour_sin", "hour_cos",
    "day_sin", "day_cos",
    "vehicle_type", "service_mode",
]
_YG_CATEGORICAL = ["zone", "month", "quarter", "vehicle_type", "service_mode"]

_VEHICLE_ENCODING = {"yellow": 0, "green": 1}
_SERVICE_ENCODING = {"hail": 0, "dispatch": 1}


def _build_yg_features(
    zone: int,
    hour: int,
    minute: int,
    day_of_week: int,
    month: int,
    vehicle_type: str,
    service_mode: str,
) -> Dict[str, Any]:
    """Compute the 9 features expected by the YG model."""
    if vehicle_type not in _VEHICLE_ENCODING:
        raise ValueError(f"vehicle_type non valido: {vehicle_type!r}. Valori accettati: {list(_VEHICLE_ENCODING)}")
    if service_mode not in _SERVICE_ENCODING:
        raise ValueError(f"service_mode non valido: {service_mode!r}. Valori accettati: {list(_SERVICE_ENCODING)}")

    quarter = minute // 15
    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))
    day_sin  = float(np.sin(2 * np.pi * day_of_week / 7))
    day_cos  = float(np.cos(2 * np.pi * day_of_week / 7))

    return {
        "zone":         int(zone),
        "month":        int(month),
        "quarter":      int(quarter),
        "hour_sin":     hour_sin,
        "hour_cos":     hour_cos,
        "day_sin":      day_sin,
        "day_cos":      day_cos,
        "vehicle_type": int(_VEHICLE_ENCODING[vehicle_type]),
        "service_mode": int(_SERVICE_ENCODING[service_mode]),
    }


def _cast_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in _YG_CATEGORICAL:
        df[col] = df[col].astype("category")
    return df


class YGPredictor:
    """Singleton wrapper for the Yellow/Green LightGBM model (Riccardo)."""

    _instance: Optional["YGPredictor"] = None
    _model = None
    _zone_lookup: Optional[pd.DataFrame] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> None:
        """Load model from disk (lazy, idempotent)."""
        if self._model is not None:
            return
        logger.info("[YGPredictor] Loading model from %s ...", YG_MODEL_PATH)
        self._model = joblib.load(YG_MODEL_PATH)
        self._zone_lookup = _load_zone_lookup()
        logger.info("[YGPredictor] Model loaded.")

    def _get_zone_info(self, location_id: int) -> Dict[str, str]:
        if self._zone_lookup is None:
            self.load()
        row = self._zone_lookup[self._zone_lookup["LocationID"] == location_id]
        if len(row) > 0:
            return {
                "zone":    row.iloc[0]["Zone"],
                "borough": row.iloc[0]["Borough"],
            }
        return {"zone": "Unknown", "borough": "Unknown"}

    def predict(
        self,
        location_id: int,
        hour: int,
        minute: int,
        day_of_week: int,
        month: int,
        vehicle_type: str,
        service_mode: str,
    ) -> Dict[str, Any]:
        """
        Predict taxi availability for one vehicle type / service mode.

        Returns a normalised dict with model_type="yg".
        """
        if self._model is None:
            self.load()

        feats = _build_yg_features(
            zone=location_id, hour=hour, minute=minute,
            day_of_week=day_of_week, month=month,
            vehicle_type=vehicle_type, service_mode=service_mode,
        )
        X = pd.DataFrame([feats])[_YG_FEATURES]
        X = _cast_categorical(X)

        raw_class = int(self._model.predict(X)[0])

        # Fallback for out-of-range classes
        if raw_class not in YG_CLASS_NAMES:
            logger.warning(
                "[YGPredictor] Model returned unexpected class %d — falling back to 1 (Media).",
                raw_class,
            )
            raw_class = 1

        zone_info = self._get_zone_info(location_id)

        return {
            "model_type":             "yg",
            "location_id":            location_id,
            "location_name":          zone_info["zone"],
            "borough":                zone_info["borough"],
            "vehicle_type":           vehicle_type,
            "service_mode":           service_mode,
            "predicted_class":        raw_class,
            "predicted_class_name":   YG_CLASS_NAMES[raw_class],
            "availability_description": YG_CLASS_DESCRIPTIONS[raw_class],
        }

    def predict_all(
        self,
        location_id: int,
        hour: int,
        minute: int,
        day_of_week: int,
        month: int,
    ) -> List[Dict[str, Any]]:
        """
        Predict for all three YG configurations:
          1. Yellow Taxi  (hail)
          2. Green Taxi   (hail)
          3. Green Taxi   (dispatch)
        """
        return [
            self.predict(location_id, hour, minute, day_of_week, month, "yellow", "hail"),
            self.predict(location_id, hour, minute, day_of_week, month, "green",  "hail"),
            self.predict(location_id, hour, minute, day_of_week, month, "green",  "dispatch"),
        ]


_yg_predictor: Optional[YGPredictor] = None


def get_yg_predictor() -> YGPredictor:
    """Return the YGPredictor singleton."""
    global _yg_predictor
    if _yg_predictor is None:
        _yg_predictor = YGPredictor()
    return _yg_predictor
```

- [ ] **Step 4: Run tests — all should pass**

```bash
cd C:\Users\andre\Desktop\Progetto_Accenture
python -m pytest tests/test_yg_predictor.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add llm_tool/yg_predictor.py tests/test_yg_predictor.py
git commit -m "feat: add YGPredictor wrapper for Riccardo's YG model (TDD)"
```

---

## Task 3: Add new `@tool` functions + FHVHV stub + deprecate old tool

**Files:**
- Modify: `llm_tool/taxi_predictor.py`
- Modify: `llm_tool/i18n.py`

- [ ] **Step 1: Add FHVHV coming-soon message to i18n.py**

In `llm_tool/i18n.py`, add `"fhvhv_coming_soon"` to both language dicts:

```python
# In "it":
"fhvhv_coming_soon": "🚗 Il modello per i taxi FHVHV (Uber, Lyft, NCC) è in arrivo!\n\nPer ora posso predire la disponibilità di taxi gialli e verdi. Vuoi che controlli per te?",

# In "en":
"fhvhv_coming_soon": "🚗 The FHVHV model (Uber, Lyft, rideshare) is coming soon!\n\nFor now I can predict availability for yellow and green taxis. Want me to check those for you?",
```

- [ ] **Step 2: Mark old tool as deprecated in taxi_predictor.py**

In `llm_tool/taxi_predictor.py`, find the `@tool` decorator just before `def predict_taxi_availability(` (line ~294) and update the docstring:

```python
@tool
def predict_taxi_availability(
    location_id: int,
    half_hour_bucket: int,
    day_of_week: int,
    month: int,
    language: str = "en",
) -> str:
    """
    [DEPRECATED — use predict_yellow_green_availability instead]
    Predict taxi availability for a NYC zone at a specific time.
    Includes insights (SHAP) explaining WHY the availability is what it is.
    """
```

- [ ] **Step 3: Add two new tools at the end of taxi_predictor.py**

Append to `llm_tool/taxi_predictor.py` (after the `get_historical_trends` tool, line ~374):

```python


@tool
def predict_yellow_green_availability(
    location_id: int,
    hour: int,
    minute: int,
    day_of_week: int,
    month: int,
    vehicle_type: str = "all",
    language: str = "it",
) -> str:
    """
    Predict taxi availability for Yellow and/or Green taxis in a NYC zone.

    Args:
        location_id:  NYC taxi zone ID (1-265).
        hour:         Hour of day (0-23).
        minute:       Minute of hour (0-59).
        day_of_week:  0=Monday ... 6=Sunday.
        month:        Month (1-12).
        vehicle_type: "yellow" | "green" | "all" (default: "all" — returns all three types).
        language:     Response language ("it" or "en").
    """
    import json as _json
    from langchain_core.tools.base import ToolException as _TE

    errors = []
    if not isinstance(location_id, int) or not (1 <= location_id <= 265):
        errors.append("location_id must be an integer between 1 and 265")
    if not isinstance(hour, int) or not (0 <= hour <= 23):
        errors.append("hour must be between 0 and 23")
    if not isinstance(minute, int) or not (0 <= minute <= 59):
        errors.append("minute must be between 0 and 59")
    if not isinstance(day_of_week, int) or not (0 <= day_of_week <= 6):
        errors.append("day_of_week must be between 0 and 6")
    if not isinstance(month, int) or not (1 <= month <= 12):
        errors.append("month must be between 1 and 12")
    if errors:
        raise _TE("; ".join(errors))

    try:
        from .yg_predictor import get_yg_predictor
        yg = get_yg_predictor()

        if vehicle_type == "all" or vehicle_type is None:
            results = yg.predict_all(location_id, hour, minute, day_of_week, month)
        elif vehicle_type == "yellow":
            results = [yg.predict(location_id, hour, minute, day_of_week, month, "yellow", "hail")]
        elif vehicle_type == "green":
            results = [
                yg.predict(location_id, hour, minute, day_of_week, month, "green", "hail"),
                yg.predict(location_id, hour, minute, day_of_week, month, "green", "dispatch"),
            ]
        else:
            # Unknown type — fall back to all
            results = yg.predict_all(location_id, hour, minute, day_of_week, month)

        return _json.dumps({"model": "yg", "results": results}, ensure_ascii=False)
    except Exception as e:
        logger.error("[YG tool] %s", e, exc_info=True)
        raise _TE(f"Errore YG: {e}")


@tool
def predict_fhvhv_availability(
    location_id: int,
    hour: int,
    minute: int,
    day_of_week: int,
    month: int,
    language: str = "it",
) -> str:
    """
    Predict taxi availability for FHVHV vehicles (Uber, Lyft, rideshare) in a NYC zone.
    NOTE: This model is not yet available — returns a coming-soon notice.

    Args:
        location_id:  NYC taxi zone ID (1-265).
        hour:         Hour of day (0-23).
        minute:       Minute of hour (0-59).
        day_of_week:  0=Monday ... 6=Sunday.
        month:        Month (1-12).
        language:     Response language ("it" or "en").
    """
    import json as _json
    from .i18n import get_msg
    msg = get_msg(language, "fhvhv_coming_soon")
    return _json.dumps({"model": "fhvhv", "coming_soon": True, "message": msg}, ensure_ascii=False)
```

- [ ] **Step 4: Verify the new tools import correctly**

```bash
cd C:\Users\andre\Desktop\Progetto_Accenture
python -c "from llm_tool.taxi_predictor import predict_yellow_green_availability, predict_fhvhv_availability; print('Tools OK:', predict_yellow_green_availability.name, predict_fhvhv_availability.name)"
```

Expected: `Tools OK: predict_yellow_green_availability predict_fhvhv_availability`

- [ ] **Step 5: Verify FHVHV tool returns coming_soon**

```bash
python -c "
import json
from llm_tool.taxi_predictor import predict_fhvhv_availability
result = json.loads(predict_fhvhv_availability.invoke({'location_id': 161, 'hour': 10, 'minute': 0, 'day_of_week': 0, 'month': 3}))
print(result)
assert result['coming_soon'] is True
print('FHVHV stub OK')
"
```

Expected: `{'model': 'fhvhv', 'coming_soon': True, 'message': '🚗 Il modello ...'}`

- [ ] **Step 6: Commit**

```bash
git add llm_tool/taxi_predictor.py llm_tool/i18n.py
git commit -m "feat: add predict_yellow_green and predict_fhvhv tools, deprecate old tool"
```

---

## Task 4: Extend extraction prompt to capture `vehicle_type`

**Files:**
- Modify: `llm_tool/prompts.py`
- Modify: `llm_tool/input_validator.py`

- [ ] **Step 1: Update `_EXTRACTION_SYSTEM_PROMPT` in prompts.py**

Replace the current `RULES` block (lines 53-61) in `llm_tool/prompts.py` with:

```python
_EXTRACTION_SYSTEM_PROMPT = """\
You are a NYC Taxi Data Extractor. Extract parameters from the user's message.

CONTEXT:
- Today: {today} (day_of_week={dow}, 0=Monday…6=Sunday)
- Current month: {month}
- Current time: {time}

RULES:
1. "zone": the NYC neighborhood/place mentioned (string or null).
2. "month": 1-12. Compute relative references ("next month", "mese prossimo").
3. "day_of_week": 0-6. Compute "tomorrow"/"domani" relative to current day.
4. "hour": 0-23. Compute relative ("in two hours"). Use null if not mentioned.
5. "minute": 0-59. Default null if not mentioned.
6. "vehicle_type": identify taxi type if mentioned:
   - "yellow" for taxi giallo, yellow cab, yellow taxi
   - "green"  for taxi verde, green cab, green taxi
   - "fhvhv"  for Uber, Lyft, NCC, rideshare, FHVHV, app cab
   - "all"    if not mentioned or user wants all types
   Default: "all"

Return ONLY valid JSON, no explanation:
{{"zone": "<str|null>", "month": <int|null>, "day_of_week": <int|null>, "hour": <int|null>, "minute": <int|null>, "vehicle_type": "<yellow|green|fhvhv|all>"}}
"""
```

- [ ] **Step 2: Update `_sanitize_extracted` in input_validator.py**

In `llm_tool/input_validator.py`, at the end of `_sanitize_extracted` (after the `minute` block, before `return sanitized`), add:

```python
        # vehicle_type — accept known values, default "all"
        vt = raw.get("vehicle_type")
        if vt and str(vt).strip().lower() in ("yellow", "green", "fhvhv", "all"):
            sanitized["vehicle_type"] = str(vt).strip().lower()
        else:
            sanitized["vehicle_type"] = "all"
```

Also update the default dict at the top of `_sanitize_extracted` (line ~124):

```python
        sanitized: Dict[str, Any] = {
            "zone": None, "month": None, "day_of_week": None,
            "hour": None, "minute": None, "vehicle_type": "all",
        }
```

- [ ] **Step 3: Update `extract()` fast-path in input_validator.py**

The fast-path (zone ID regex match) returns early without a `vehicle_type` key. Add it:

```python
        if id_match:
            print(f"   [Validator] Fast-path: ID {id_match.group(1)}")
            return {"zone": id_match.group(1), "month": None, "day_of_week": None,
                    "hour": None, "minute": None, "vehicle_type": "all"}
```

- [ ] **Step 4: Verify sanitize handles vehicle_type correctly**

```bash
python -c "
from llm_tool.input_validator import get_validator
v = get_validator()
# Test sanitize directly
result = v._sanitize_extracted({'zone': 'midtown', 'month': 3, 'day_of_week': 0, 'hour': 10, 'minute': None, 'vehicle_type': 'yellow'})
print(result)
assert result['vehicle_type'] == 'yellow'

result2 = v._sanitize_extracted({'zone': 'jfk', 'month': None, 'day_of_week': None, 'hour': None, 'minute': None, 'vehicle_type': 'invalid'})
assert result2['vehicle_type'] == 'all'
print('vehicle_type sanitization OK')
"
```

Expected: both assertions pass.

- [ ] **Step 5: Commit**

```bash
git add llm_tool/prompts.py llm_tool/input_validator.py
git commit -m "feat: add vehicle_type extraction to prompt and sanitizer"
```

---

## Task 5: Update `extractor_node` to propagate `vehicle_type`

**Files:**
- Modify: `llm_tool/agent.py`

- [ ] **Step 1: Update extractor_node in agent.py**

In `agent.py`, find the `extractor_node` function. The merge loop currently iterates over:

```python
for key in ("location_id", "month", "day_of_week", "hour", "minute"):
```

Change it to:

```python
for key in ("location_id", "month", "day_of_week", "hour", "minute", "vehicle_type"):
```

This ensures `vehicle_type` is carried in `current_params` for routing in the predictor node.

- [ ] **Step 2: Verify extractor carries vehicle_type**

```bash
python -c "
from llm_tool.agent import extractor_node
from langchain_core.messages import HumanMessage

# Simulate a state where LLM extraction already resolved vehicle_type
# We'll test by patching the validator
from unittest.mock import patch, MagicMock

mock_raw = {'zone': 'midtown', 'month': 3, 'day_of_week': 0, 'hour': 10, 'minute': None, 'vehicle_type': 'yellow'}
mock_resolved = {'location_id': 161, 'month': 3, 'day_of_week': 0, 'hour': 10, 'minute': None, 'vehicle_type': 'yellow', 'candidates': []}

with patch('llm_tool.agent.get_validator') as mock_v:
    v = MagicMock()
    v.extract.return_value = mock_raw
    v.validate_and_resolve.return_value = mock_resolved
    mock_v.return_value = v

    state = {
        'messages': [HumanMessage(content='taxi giallo a midtown alle 10 lunedi marzo')],
        'intent': 'predict',
        'current_params': {},
        'candidates': [],
        'hour_range': [],
        'next_step': '',
        'language': 'it',
        'results': [],
        'validation_errors': [],
    }
    out = extractor_node(state)
    assert out['current_params']['vehicle_type'] == 'yellow', f'Got: {out}'
    print('extractor_node vehicle_type propagation OK')
"
```

Expected: `extractor_node vehicle_type propagation OK`

- [ ] **Step 3: Commit**

```bash
git add llm_tool/agent.py
git commit -m "feat: propagate vehicle_type through extractor_node"
```

---

## Task 6: Update `predictor_node` to route by `vehicle_type`

**Files:**
- Modify: `llm_tool/agent.py`
- Modify: `llm_tool/agent.py` (same file — imports)

- [ ] **Step 1: Add import at the top of agent.py**

Find the imports block in `agent.py` and add:

```python
from .yg_predictor import get_yg_predictor
```

Place it after the existing `from .taxi_predictor import get_historical_trends, get_predictor` line.

- [ ] **Step 2: Replace `predictor_node` function body**

Replace the current `predictor_node` function (lines ~256-294 in agent.py) with this full implementation:

```python
def predictor_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 5: Route to the correct predictor based on vehicle_type.

    Routing:
      vehicle_type = "fhvhv"              → FHVHV coming-soon stub
      vehicle_type = "all" | None         → YGPredictor.predict_all (3 results)
      vehicle_type = "yellow"             → YGPredictor.predict yellow-hail (1 result)
      vehicle_type = "green"              → YGPredictor.predict green-hail + green-dispatch (2 results)
    """
    print(f"--- [5. PREDICTOR] ---")
    p          = state["current_params"]
    intent     = state["intent"]
    lang       = state.get("language", "it")
    results    = []

    try:
        if intent == "trend":
            trend_json = get_historical_trends.invoke({
                "location_id": p["location_id"],
                "day_of_week": p.get("day_of_week"),
            })
            results = [json.loads(trend_json)]
            return {"results": results, "next_step": "format"}

        now       = datetime.now(ZoneInfo("America/New_York"))
        eval_hour   = p.get("hour")   if p.get("hour")   is not None else now.hour
        eval_minute = p.get("minute") if p.get("minute") is not None else 0
        eval_dow    = p.get("day_of_week") if p.get("day_of_week") is not None else now.weekday()
        eval_month  = p.get("month") if p.get("month") is not None else now.month
        location_id = p["location_id"]

        vehicle_type = p.get("vehicle_type", "all") or "all"
        hour_range   = state.get("hour_range", [])

        # ── FHVHV stub ────────────────────────────────────────────────────────
        if vehicle_type == "fhvhv":
            from .i18n import get_msg
            results = [{
                "model_type":   "fhvhv",
                "coming_soon":  True,
                "message":      get_msg(lang, "fhvhv_coming_soon"),
                "location_id":  location_id,
            }]
            return {"results": results, "next_step": "format"}

        # ── Yellow/Green model ────────────────────────────────────────────────
        yg = get_yg_predictor()

        if hour_range:
            # Range prediction: iterate hours, use yellow for brevity (avoid 3x output per slot)
            for h in hour_range:
                results.append(yg.predict(location_id, h, 0, eval_dow, eval_month, "yellow", "hail"))
        elif vehicle_type == "all" or vehicle_type is None:
            results = yg.predict_all(location_id, eval_hour, eval_minute, eval_dow, eval_month)
        elif vehicle_type == "yellow":
            results = [yg.predict(location_id, eval_hour, eval_minute, eval_dow, eval_month, "yellow", "hail")]
        elif vehicle_type == "green":
            results = [
                yg.predict(location_id, eval_hour, eval_minute, eval_dow, eval_month, "green", "hail"),
                yg.predict(location_id, eval_hour, eval_minute, eval_dow, eval_month, "green", "dispatch"),
            ]
        else:
            # Unknown vehicle_type: fall back to all
            results = yg.predict_all(location_id, eval_hour, eval_minute, eval_dow, eval_month)

        return {"results": results, "next_step": "format"}

    except Exception as e:
        logger.error(f"[Predictor] {e}", exc_info=True)
        return {"validation_errors": [str(e)], "next_step": "format"}
```

- [ ] **Step 3: Verify predictor_node imports and runs without crash**

```bash
python -c "
from llm_tool.agent import predictor_node
print('predictor_node import OK')
"
```

Expected: no ImportError.

- [ ] **Step 4: Commit**

```bash
git add llm_tool/agent.py
git commit -m "feat: route predictor_node by vehicle_type (YG + FHVHV stub)"
```

---

## Task 7: Update `formatter_node` to render YG and FHVHV results

**Files:**
- Modify: `llm_tool/agent.py`

The formatter's `_build_template` currently expects `predicted_class_name`, `confidence`, `probabilities` (old 5-class format). The YG model produces `predicted_class_name` (same key ✓), `availability_description` (new), but **no** `confidence` or `probabilities`. FHVHV produces `coming_soon: True`.

- [ ] **Step 1: Update config imports in agent.py**

The agent imports `CLASS_NAMES, CLASS_EMOJIS` from config. Add the YG versions:

```python
from .config import (
    CLASS_NAMES, CLASS_EMOJIS, DAY_NAMES_IT, MONTH_NAMES_IT,
    YG_CLASS_NAMES, YG_CLASS_EMOJIS,
    VEHICLE_TYPE_DISPLAY,
)
```

- [ ] **Step 2: Replace `_build_template` function in agent.py**

Replace the current `_build_template` function (lines ~86-133 in agent.py) with:

```python
def _build_template(results: List[Dict], params: Dict) -> str:
    """Build the structured, emoji-rich part of the response deterministically."""
    if not results:
        return "⚠️ Nessun risultato disponibile."

    now        = datetime.now(ZoneInfo("America/New_York"))
    eval_hour   = params.get("hour")   if params.get("hour")   is not None else now.hour
    eval_minute = params.get("minute") if params.get("minute") is not None else 0
    eval_dow    = params.get("day_of_week") if params.get("day_of_week") is not None else now.weekday()
    eval_month  = params.get("month") if params.get("month") is not None else now.month

    r0        = results[0]
    zone_name = r0.get("location_name", "?")
    borough   = r0.get("borough", "")
    day_name  = DAY_NAMES_IT.get(eval_dow, "?")
    month_name = MONTH_NAMES_IT.get(eval_month, "?")

    lines = [f"🚕 *{zone_name}* ({borough})", f"📅 {day_name} — {month_name}", ""]

    # ── FHVHV coming soon ─────────────────────────────────────────────────────
    if r0.get("coming_soon"):
        lines.append(r0.get("message", "🚗 FHVHV model coming soon."))
        return "\n".join(lines)

    # ── Historical trend ──────────────────────────────────────────────────────
    if "hourly_avg_availability" in r0:
        lines.append("📊 *Trend Storico:*")
        lines.append("I dati orari indicano l'indice di disponibilità medio da 0 (Bassa) a 1 (Alta).")
        return "\n".join(lines)

    model_type = r0.get("model_type", "yg")

    # ── YG model results ──────────────────────────────────────────────────────
    if model_type == "yg":
        time_str = f"{eval_hour:02d}:{eval_minute:02d}"
        lines.append(f"🕐 Ore {time_str}")
        lines.append("")

        if len(results) == 1:
            r    = results[0]
            cls  = r["predicted_class"]
            emoji = YG_CLASS_EMOJIS.get(cls, "❓")
            vtype = VEHICLE_TYPE_DISPLAY.get(r["vehicle_type"], r["vehicle_type"])
            lines.append(f"{emoji} *{vtype}*: {r['predicted_class_name']}")
            lines.append(f"   _{r['availability_description']}_")
        else:
            lines.append("📊 *Disponibilità per tipo di taxi:*")
            for r in results:
                cls   = r["predicted_class"]
                emoji  = YG_CLASS_EMOJIS.get(cls, "❓")
                sm     = r.get("service_mode", "")
                key    = f"{r['vehicle_type']}_{sm}" if sm else r["vehicle_type"]
                vtype  = VEHICLE_TYPE_DISPLAY.get(key, r["vehicle_type"])
                lines.append(f"  {emoji} *{vtype}*: {r['predicted_class_name']}")
                lines.append(f"     _{r['availability_description']}_")

        return "\n".join(lines)

    # ── Legacy generic model (5-class, with confidence) ───────────────────────
    name_to_emoji = {v: CLASS_EMOJIS[k] for k, v in CLASS_NAMES.items()}

    if len(results) == 1:
        cls      = r0["predicted_class"]
        time_str = f"{eval_hour:02d}:{eval_minute:02d}"
        lines += [
            f"🕐 Ore {time_str}",
            f"{CLASS_EMOJIS.get(cls, '❓')} *{r0['predicted_class_name']}*"
            f"  —  Confidenza: {r0['confidence']*100:.1f}%",
            "",
            "📊 *Distribuzione probabilità:*",
        ]
        for cls_name, prob in r0.get("probabilities", {}).items():
            e = name_to_emoji.get(cls_name, "•")
            lines.append(f"  {e} {cls_name}: {prob*100:.1f}%")
    else:
        lines.append("📊 *Disponibilità per fascia oraria:*")
        for r in results:
            h    = r.get("time_bucket", 0) // 2
            e    = CLASS_EMOJIS.get(r["predicted_class"], "❓")
            conf = r["confidence"] * 100
            lines.append(f"  {e} *{h:02d}:00* → {r['predicted_class_name']} ({conf:.0f}%)")

    return "\n".join(lines)
```

- [ ] **Step 3: Verify formatter doesn't crash with YG result**

```bash
python -c "
from llm_tool.agent import _build_template

# Simulate YG result (3 types)
results = [
    {'model_type': 'yg', 'location_id': 161, 'location_name': 'Midtown Center', 'borough': 'Manhattan',
     'vehicle_type': 'yellow', 'service_mode': 'hail', 'predicted_class': 2, 'predicted_class_name': 'Alta',
     'availability_description': 'trovare un taxi è generalmente facile'},
    {'model_type': 'yg', 'location_id': 161, 'location_name': 'Midtown Center', 'borough': 'Manhattan',
     'vehicle_type': 'green', 'service_mode': 'hail', 'predicted_class': 0, 'predicted_class_name': 'Bassa',
     'availability_description': 'trovare un taxi è difficile'},
    {'model_type': 'yg', 'location_id': 161, 'location_name': 'Midtown Center', 'borough': 'Manhattan',
     'vehicle_type': 'green', 'service_mode': 'dispatch', 'predicted_class': 1, 'predicted_class_name': 'Media',
     'availability_description': 'la disponibilità è intermedia'},
]
params = {'hour': 10, 'minute': 30, 'day_of_week': 0, 'month': 3}
out = _build_template(results, params)
print(out)
assert 'Taxi Giallo' in out
assert 'Alta' in out
assert 'Bassa' in out
print('formatter YG OK')
"
```

Expected: formatted string with emoji, zone name, all 3 types shown.

- [ ] **Step 4: Verify formatter handles FHVHV stub**

```bash
python -c "
from llm_tool.agent import _build_template
results = [{'model_type': 'fhvhv', 'coming_soon': True, 'message': '🚗 FHVHV coming soon', 'location_id': 161, 'location_name': 'Midtown', 'borough': 'Manhattan'}]
out = _build_template(results, {'hour': 10, 'minute': 0, 'day_of_week': 0, 'month': 3})
print(out)
assert 'coming soon' in out or 'FHVHV' in out
print('formatter FHVHV OK')
"
```

- [ ] **Step 5: Commit**

```bash
git add llm_tool/agent.py
git commit -m "feat: update _build_template and formatter for YG/FHVHV output schema"
```

---

## Task 8: Add `mock_yg_predictor` fixture + run all tests

**Files:**
- Modify: `tests/conftest.py`
- Run: full test suite

- [ ] **Step 1: Add `mock_yg_predictor` fixture to conftest.py**

Append to `tests/conftest.py`:

```python

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
            {**_yg_result, "vehicle_type": "green", "service_mode": "hail",     "predicted_class": 0, "predicted_class_name": "Bassa"},
            {**_yg_result, "vehicle_type": "green", "service_mode": "dispatch", "predicted_class": 1, "predicted_class_name": "Media"},
        ]
        mock.return_value = yg
        yield yg
```

- [ ] **Step 2: Run the full test suite**

```bash
cd C:\Users\andre\Desktop\Progetto_Accenture
python -m pytest tests/ -v --tb=short 2>&1 | tail -40
```

Expected: all tests in `test_yg_predictor.py` PASS; existing tests in `test_agent_nodes.py`, `test_validator.py`, `test_i18n.py` PASS (they do not touch the new code paths yet).

- [ ] **Step 3: Fix any failures**

If any existing test fails, diagnose and fix. Common causes:
- `test_agent_nodes.py` — the `create_state` helper in conftest doesn't include `candidates`. Fix: check if the `AgentState` definition still matches.
- `test_predictor.py` — tests that call the old generic predictor with a real model file; these will skip/fail if `output/ml_model_artifacts_all_months.pkl` is absent. That's OK — mark as `xfail` with `pytest.mark.xfail(reason="large model not in CI")` if needed.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add mock_yg_predictor fixture and verify full suite"
```

---

## Task 9: Integration tests for the new paths

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Add YG and FHVHV integration tests**

Append to `tests/test_integration.py`:

```python
# ─── YG + FHVHV integration tests ────────────────────────────────────────────

class TestYGIntegration:
    """End-to-end tests through the full agent graph using mocked YGPredictor."""

    def test_yellow_taxi_prediction(self, mock_yg_predictor):
        """'taxi giallo a midtown alle 10' → predictor_node returns 1 YG result for yellow."""
        from llm_tool.agent import predictor_node
        from tests.conftest import create_state
        from langchain_core.messages import HumanMessage

        state = create_state(
            messages=[HumanMessage(content="taxi giallo a midtown alle 10")],
            intent="predict",
            params={"location_id": 161, "hour": 10, "minute": 0,
                    "day_of_week": 0, "month": 3, "vehicle_type": "yellow"},
        )
        out = predictor_node(state)
        assert out["next_step"] == "format"
        assert len(out["results"]) == 1
        assert out["results"][0]["vehicle_type"] == "yellow"
        assert out["results"][0]["model_type"] == "yg"

    def test_all_types_prediction(self, mock_yg_predictor):
        """No vehicle_type specified → predictor_node calls predict_all → 3 results."""
        from llm_tool.agent import predictor_node
        from tests.conftest import create_state
        from langchain_core.messages import HumanMessage

        state = create_state(
            messages=[HumanMessage(content="taxi a midtown alle 10")],
            intent="predict",
            params={"location_id": 161, "hour": 10, "minute": 0,
                    "day_of_week": 0, "month": 3, "vehicle_type": "all"},
        )
        out = predictor_node(state)
        assert len(out["results"]) == 3
        mock_yg_predictor.predict_all.assert_called_once_with(161, 10, 0, 0, 3)

    def test_green_taxi_returns_two_modes(self, mock_yg_predictor):
        """vehicle_type='green' → predictor_node returns hail + dispatch."""
        from llm_tool.agent import predictor_node
        from tests.conftest import create_state
        from langchain_core.messages import HumanMessage

        state = create_state(
            messages=[HumanMessage(content="taxi verde a midtown")],
            intent="predict",
            params={"location_id": 161, "hour": 10, "minute": 0,
                    "day_of_week": 0, "month": 3, "vehicle_type": "green"},
        )
        out = predictor_node(state)
        assert len(out["results"]) == 2
        service_modes = [r.get("service_mode") for r in out["results"]]
        assert "hail" in service_modes
        assert "dispatch" in service_modes

    def test_fhvhv_returns_coming_soon(self):
        """vehicle_type='fhvhv' → results contain coming_soon=True, no model call."""
        from llm_tool.agent import predictor_node
        from tests.conftest import create_state
        from langchain_core.messages import HumanMessage

        state = create_state(
            messages=[HumanMessage(content="uber a midtown")],
            intent="predict",
            params={"location_id": 161, "hour": 10, "minute": 0,
                    "day_of_week": 0, "month": 3, "vehicle_type": "fhvhv"},
        )
        out = predictor_node(state)
        assert len(out["results"]) == 1
        assert out["results"][0]["coming_soon"] is True
        assert out["results"][0]["model_type"] == "fhvhv"

    def test_formatter_renders_yg_multi(self, mock_yg_predictor):
        """_build_template handles 3 YG results without raising."""
        from llm_tool.agent import _build_template
        results = [
            {"model_type": "yg", "location_id": 161, "location_name": "Midtown Center",
             "borough": "Manhattan", "vehicle_type": "yellow", "service_mode": "hail",
             "predicted_class": 2, "predicted_class_name": "Alta",
             "availability_description": "facile"},
            {"model_type": "yg", "location_id": 161, "location_name": "Midtown Center",
             "borough": "Manhattan", "vehicle_type": "green", "service_mode": "hail",
             "predicted_class": 0, "predicted_class_name": "Bassa",
             "availability_description": "difficile"},
            {"model_type": "yg", "location_id": 161, "location_name": "Midtown Center",
             "borough": "Manhattan", "vehicle_type": "green", "service_mode": "dispatch",
             "predicted_class": 1, "predicted_class_name": "Media",
             "availability_description": "intermedia"},
        ]
        out = _build_template(results, {"hour": 10, "minute": 0, "day_of_week": 0, "month": 3})
        assert "Midtown Center" in out
        assert "Alta" in out
        assert "Bassa" in out
        assert "Media" in out

    def test_formatter_renders_fhvhv_coming_soon(self):
        """_build_template renders FHVHV coming-soon message."""
        from llm_tool.agent import _build_template
        results = [{"model_type": "fhvhv", "coming_soon": True,
                    "message": "🚗 FHVHV coming soon", "location_id": 161,
                    "location_name": "Midtown", "borough": "Manhattan"}]
        out = _build_template(results, {"hour": 10, "minute": 0, "day_of_week": 0, "month": 3})
        assert "FHVHV" in out or "coming soon" in out


class TestVehicleTypeSanitization:
    """Test vehicle_type extraction and sanitization."""

    def test_sanitize_yellow(self):
        from llm_tool.input_validator import get_validator
        v = get_validator()
        r = v._sanitize_extracted({"zone": "midtown", "month": None, "day_of_week": None,
                                   "hour": None, "minute": None, "vehicle_type": "yellow"})
        assert r["vehicle_type"] == "yellow"

    def test_sanitize_fhvhv(self):
        from llm_tool.input_validator import get_validator
        v = get_validator()
        r = v._sanitize_extracted({"zone": None, "month": None, "day_of_week": None,
                                   "hour": None, "minute": None, "vehicle_type": "fhvhv"})
        assert r["vehicle_type"] == "fhvhv"

    def test_sanitize_unknown_defaults_to_all(self):
        from llm_tool.input_validator import get_validator
        v = get_validator()
        r = v._sanitize_extracted({"zone": None, "month": None, "day_of_week": None,
                                   "hour": None, "minute": None, "vehicle_type": "bus"})
        assert r["vehicle_type"] == "all"

    def test_sanitize_missing_defaults_to_all(self):
        from llm_tool.input_validator import get_validator
        v = get_validator()
        r = v._sanitize_extracted({"zone": None, "month": None, "day_of_week": None,
                                   "hour": None, "minute": None})
        assert r["vehicle_type"] == "all"

    def test_fast_path_zone_id_sets_all(self):
        """Regex fast-path (zona id 161) should include vehicle_type='all'."""
        from llm_tool.input_validator import get_validator
        v = get_validator()
        r = v.extract("zona id 161")
        assert r["vehicle_type"] == "all"
```

- [ ] **Step 2: Run only the new integration tests**

```bash
cd C:\Users\andre\Desktop\Progetto_Accenture
python -m pytest tests/test_integration.py::TestYGIntegration tests/test_integration.py::TestVehicleTypeSanitization -v
```

Expected: all 11 tests PASS.

- [ ] **Step 3: Run the full suite one more time**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -50
```

Expected: no regressions. Collect the pass/fail summary.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration tests for YG all-types, green two-modes, FHVHV coming-soon"
```

---

## Task 10: Final commit and summary

- [ ] **Step 1: Run full test suite one last time and confirm**

```bash
cd C:\Users\andre\Desktop\Progetto_Accenture
python -m pytest tests/ --tb=short -q 2>&1 | tail -20
```

Expected: all tests PASS (except any pre-existing `xfail` for large model files).

- [ ] **Step 2: Verify the feature branch is ready**

```bash
git log --oneline feature/yg-fhvhv-integration | head -10
```

Expected: 9 commits visible (tasks 1-9).

- [ ] **Step 3: Final commit with summary**

```bash
git add -A
git status --short  # Should be empty (everything committed)
```

If any untracked or modified files remain, add and commit them:

```bash
git commit -m "chore: finalize YG+FHVHV integration on feature/yg-fhvhv-integration"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Model artifact copy (Task 1) ✓ | YGPredictor wrapper (Task 2) ✓ | New tools (Task 3) ✓ | FHVHV stub (Task 3) ✓ | vehicle_type extraction (Task 4) ✓ | extractor routing (Task 5) ✓ | predictor routing (Task 6) ✓ | formatter update (Task 7) ✓ | old tool deprecated (Task 3) ✓ | i18n FHVHV key (Task 3) ✓ | tests (Tasks 2, 8, 9) ✓
- [x] **No TBD/TODO:** All code blocks are complete
- [x] **Type consistency:** `_build_yg_features` defined in Task 2, used by `YGPredictor.predict` (same task). `get_yg_predictor` defined in Task 2 step 3, imported in Task 6. `YG_CLASS_NAMES/EMOJIS` defined in Task 1, imported in Task 2 and Task 7. `VEHICLE_TYPE_DISPLAY` keys match the `vehicle_type + service_mode` combination strings used in `_build_template`.
- [x] **Spec decision: green → hail+dispatch:** Covered in predictor_node (Task 6) and integration test (Task 9).
- [x] **Spec decision: fhvhv → coming_soon JSON:** Covered in Task 3 + Task 6.
- [x] **Spec decision: no confidence/probabilities:** _build_template YG branch does not reference `confidence` or `probabilities`.
- [x] **Legacy tool hidden from LLM:** Deprecated docstring added (Task 3) — the tool is not registered in the LangGraph graph, so it won't be invoked by the agent. The graph uses `predictor_node` directly (not via LangChain tool binding), so no further graph change is needed.
