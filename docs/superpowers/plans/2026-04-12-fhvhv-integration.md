# FHvhv Integration Plan

## Overview

Integrate Roberto's FHVHV (Uber/Lyft) waiting time prediction model into the NYC Taxi chatbot, following the same patterns used for the Yellow/Green model.

## Source Files

| File | Description | Action |
|------|-------------|--------|
| `roberto/waiting_time_lgbm.pkl` | Trained LightGBM model | Copy to `output/fhvhv_model.pkl` |
| `roberto/zone_target_map.pkl` | Thresholds (p33, p66) for 3-class conversion | Copy to `output/fhvhv_thresholds.pkl` |

---

## Phase 1: Copy Model Files

```bash
copy roberto\waiting_time_lgbm.pkl output\fhvhv_model.pkl
copy roberto\zone_target_map.pkl output\fhvhv_thresholds.pkl
```

---

## Phase 2: Config Updates

File: `llm_tool/config.py`

Add:
```python
FHVHV_MODEL_PATH = OUTPUT_DIR / "fhvhv_model.pkl"
FHVHV_THRESHOLDS_PATH = OUTPUT_DIR / "fhvhv_thresholds.pkl"

FHVHV_CLASS_NAMES = {0: "Facile", 1: "Medio", 2: "Difficile"}
FHVHV_CLASS_EMOJIS = {0: "🟢", 1: "🟡", 2: "🔴"}
FHVHV_CLASS_DESCRIPTIONS = {
    0: "trovare un FHVHV è generalmente facile in questa zona e fascia oraria",
    1: "la disponibilità degli FHVHV è intermedia in questa zona e fascia oraria",
    2: "trovare un FHVUV è difficile in questa zona e fascia oraria",
}
```

---

## Phase 3: Create FHvhvPredictor Class

File: `llm_tool/fhvhv_predictor.py` (NEW)

### Features (must match training order!)
```python
_FHVHV_FEATURES = [
    "PULocationID", "month", "quarter",
    "hour_sin", "hour_cos",
    "minute_sin", "minute_cos",
    "dow_sin", "dow_cos",
    "is_festivo",
]
```

### Holiday List (same as Roberto)
```python
US_HOLIDAYS = [
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4),
    date(2025, 9, 1), date(2025, 10, 13), date(2025, 11, 11),
    date(2025, 11, 27), date(2025, 12, 25),
]
```

### Feature Engineering
```python
def _build_fhvhv_features(location_id, hour, minute, day_of_week, month):
    quarter = minute // 15
    hour_sin = sin(2π * hour / 24)
    hour_cos = cos(2π * hour / 24)
    minute_sin = sin(2π * minute / 60)
    minute_cos = cos(2π * minute / 60)
    dow_sin = sin(2π * day_of_week / 7)
    dow_cos = cos(2π * day_of_week / 7)
    is_festivo = 1 if (day_of_week == 6 or date in US_HOLIDAYS) else 0
    
    return {...}
```

### Class Conversion (using thresholds)
```python
def _convert_to_class(waiting_time, p33, p66):
    if waiting_time < p33:
        return 0  # Facile
    elif waiting_time < p66:
        return 1  # Medio
    else:
        return 2  # Difficile
```

### Predictor Methods
- `load()` — lazy load model + thresholds
- `predict()` — run inference, return dict with:
  - `model_type`: "fhvhv"
  - `location_id`, `location_name`, `borough`
  - `predicted_waiting_time`: "X:Y" (mm:ss format)
  - `predicted_class`: 0-2
  - `predicted_class_name`: "Facile" | "Medio" | "Difficile"
  - `predicted_class_description`: text

---

## Phase 4: Update Tool

File: `llm_tool/taxi_predictor.py`

Replace current stub `predict_fhvhv_availability` (lines 440-464) with real implementation:

```python
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
    Predict FHVHV (Uber/Lyft) availability in a NYC zone.
    ...
    """
    try:
        from .fhvhv_predictor import get_fhvhv_predictor
        fhvhv = get_fhvhv_predictor()
        result = fhvhv.predict(location_id, hour, minute, day_of_week, month)
        return json.dumps({"model": "fhvhv", **result}, ensure_ascii=False)
    except Exception as e:
        raise ToolException(f"Errore FHVHV: {e}")
```

---

## Phase 5: Tests

### Unit Tests
File: `tests/test_fhvhv_predictor.py` (NEW)
- Feature engineering correctness
- Holiday detection (Sunday + US holidays)
- Threshold classification (edge cases at p33, p66 boundaries)
- Missing zone handling

### Integration Tests
File: `tests/test_integration.py`
- Add FHVHV tool call tests

---

## Deliverables Summary

| File | Action |
|------|--------|
| `output/fhvhv_model.pkl` | Copy from roberto/ |
| `output/fhvhv_thresholds.pkl` | Copy from roberto/ |
| `llm_tool/config.py` | Add paths + class definitions |
| `llm_tool/fhvhv_predictor.py` | NEW — predictor class |
| `llm_tool/taxi_predictor.py` | Update tool |
| `tests/test_fhvhv_predictor.py` | NEW — unit tests |
| `tests/test_integration.py` | Add integration tests |

---

## Notes

- Holiday list uses Roberto's exact dates for consistency
- Output includes predicted waiting time in "mm:ss" format per user request
- Model validation: LocationID must be 1-265
- This is a demonstrative project — no retraining planned