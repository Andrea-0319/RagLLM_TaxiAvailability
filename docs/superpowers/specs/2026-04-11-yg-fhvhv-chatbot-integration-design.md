# Design: Integrazione Modelli YG + FHVHV nel Chatbot NYC Taxi

**Data:** 2026-04-11  
**Autore:** Claude (brainstorming + design)  
**Approccio scelto:** B — Unified Schema  
**Branch di implementazione:** `feature/yg-fhvhv-integration`

---

## 1. Contesto

Il chatbot NYC Taxi ha attualmente un solo tool di predizione (`predict_taxi_availability`) che usa un modello generico multi-taxi con **5 classi** (Molto Difficile → Molto Facile) e SHAP per le spiegazioni.

Il collega Riccardo ha sviluppato un modello LightGBM specializzato **solo per taxi gialli e verdi** con **3 classi** (Bassa, Media, Alta), senza scaler e con feature engineering diverso (sin/cos ciclici vs. half_hour_bucket). Prossimamente arriverà un secondo modello per i **taxi FHVHV** (Uber, Lyft, ecc.).

L'obiettivo è integrare il modello di Riccardo come tool specializzato e predisporre l'infrastruttura per il futuro modello FHVHV, senza rompere la funzionalità esistente e senza inconsistenze nell'architettura.

---

## 2. Decisioni chiave

| Decisione | Scelta |
|-----------|--------|
| Classi modello YG | 3 (0=Bassa, 1=Media, 2=Alta) |
| Vecchia tool generica | Mantenuta nel codice ma nascosta all'agente (deprecata gradualmente) |
| Comportamento FHVHV senza modello | Messaggio "coming soon" esplicito |
| Default quando tipo non specificato | Mostra tutte e 3 le previsioni (yellow-hail, green-hail, green-dispatch) |
| Posizione file modello YG | `output/yg_model_production.pkl` (copia da `riccardo/`) |

---

## 3. Architettura

```
PRIMA:
  user → agent → extractor → guardrail → predictor_node
                                              └─ predict_taxi_availability (1 modello, 5 classi)

DOPO:
  user → agent → extractor (+ vehicle_type) → guardrail → predictor_node
                                                               ├─ vehicle_type = yellow/green/all
                                                               │       └─ YGPredictor (Riccardo, 3 classi)
                                                               └─ vehicle_type = fhvhv
                                                                       └─ FHVHVStub ("coming soon")
  [old tool: codice mantenuto, non esposto all'agente LLM]
```

---

## 4. Componenti

### 4.1 File nuovi

#### `llm_tool/yg_predictor.py`
Wrapper singleton per il modello di Riccardo.

**Classe:** `YellowGreenPredictor` (singleton, lazy loading)

**Metodi:**
- `load()` — carica `output/yg_model_production.pkl` con joblib
- `predict(location_id, hour, minute, day_of_week, month, vehicle_type, service_mode) → dict` — predice per un singolo tipo
- `predict_all(location_id, hour, minute, day_of_week, month) → list[dict]` — predice per tutti e 3 i tipi (yellow-hail, green-hail, green-dispatch)

**Feature engineering interno (da datetime → features modello):**
```python
# quarter = minute // 15  (0-3)
# hour_sin = sin(2π × hour / 24)
# hour_cos = cos(2π × hour / 24)
# day_sin  = sin(2π × day_of_week / 7)
# day_cos  = cos(2π × day_of_week / 7)
# vehicle_type: {"yellow": 0, "green": 1}
# service_mode: {"hail": 0, "dispatch": 1}
```

**Output normalizzato (schema condiviso):**
```python
{
    "model_type": "yg",
    "location_id": int,
    "location_name": str,      # da zone_lookup.csv
    "borough": str,
    "vehicle_type": str,       # "yellow" | "green"
    "service_mode": str,       # "hail" | "dispatch"
    "predicted_class": int,    # 0, 1, 2
    "predicted_class_name": str,  # "Bassa" | "Media" | "Alta"
    "availability_description": str,
    # NB: nessun confidence/probabilities (il modello non li espone)
}
```

#### `docs/superpowers/specs/2026-04-11-yg-fhvhv-chatbot-integration-design.md`
Questo file.

---

### 4.2 File modificati

#### `llm_tool/config.py`
Aggiungere:
```python
# Modello Yellow/Green (Riccardo)
YG_MODEL_PATH = OUTPUT_DIR / "yg_model_production.pkl"

YG_CLASS_NAMES = {0: "Bassa", 1: "Media", 2: "Alta"}
YG_CLASS_EMOJIS = {0: "🔴", 1: "🟡", 2: "🟢"}
YG_CLASS_DESCRIPTIONS = {
    0: "trovare un taxi è difficile in questa zona e fascia oraria",
    1: "la disponibilità dei taxi è intermedia in questa zona e fascia oraria",
    2: "trovare un taxi è generalmente facile in questa zona e fascia oraria",
}

VEHICLE_TYPE_DISPLAY = {
    "yellow": "Taxi Giallo 🟡",
    "green_hail": "Taxi Verde (Hail) 🟢",
    "green_dispatch": "Taxi Verde (Dispatch) 🟢",
    "fhvhv": "FHVHV (Uber/Lyft) 🚗",
}
```

#### `llm_tool/taxi_predictor.py`
1. Vecchia `@tool predict_taxi_availability` → aggiungere docstring `# DEPRECATED: non più esposta all'agente`; il `@tool` rimane per backward compat
2. Aggiungere `@tool predict_yellow_green_availability(location_id, half_hour_bucket, day_of_week, month, vehicle_type="all", language="it") → str`
3. Aggiungere `@tool predict_fhvhv_availability(location_id, half_hour_bucket, day_of_week, month, language="it") → str` → restituisce JSON con flag `coming_soon: True` + messaggio i18n

#### `llm_tool/input_validator.py`
Aggiornare `_EXTRACTION_SYSTEM_PROMPT` (o `prompts.py`) per includere:
```
6. "vehicle_type": identifica il tipo di taxi se menzionato:
   - "yellow" per taxi giallo/yellow cab
   - "green" per taxi verde/green cab
   - "fhvhv" per Uber, Lyft, rideshare, NCC, FHVHV
   - "all" se non specificato o se l'utente vuole tutti i tipi
   Default: "all"
```

Aggiornare `_sanitize_extracted` per gestire `vehicle_type`.

Aggiornare `extractor_node` in `agent.py` per propagare `vehicle_type` in `current_params`.

#### `llm_tool/agent.py`

**`AgentState`** — aggiungere nessun nuovo campo di top-level; `vehicle_type` viene portato dentro `current_params` (come già accade per `hour`, `month`, ecc.)

**`extractor_node`** — estrarre `vehicle_type` dai params validati e aggiungerlo a `merged`:
```python
vehicle_type = resolved.get("vehicle_type", "all")
merged["vehicle_type"] = vehicle_type
```

**`predictor_node`** — routing per tipo:
```python
vehicle_type = p.get("vehicle_type", "all")

if vehicle_type == "fhvhv":
    # Stub: coming soon
    results = [{"model_type": "fhvhv", "coming_soon": True, ...}]
elif vehicle_type == "all" or vehicle_type is None:
    # Mostra tutte e 3 le predizioni: yellow-hail, green-hail, green-dispatch
    yg = get_yg_predictor()
    results = yg.predict_all(location_id, hour, minute, dow, month)
elif vehicle_type == "yellow":
    yg = get_yg_predictor()
    results = [yg.predict(location_id, hour, minute, dow, month, "yellow", "hail")]
elif vehicle_type == "green":
    # Per i taxi verdi mostra SEMPRE entrambe le modalità (hail + dispatch)
    yg = get_yg_predictor()
    results = [
        yg.predict(location_id, hour, minute, dow, month, "green", "hail"),
        yg.predict(location_id, hour, minute, dow, month, "green", "dispatch"),
    ]
```

**Nota sul routing green:** i taxi verdi esistono in due modalità (hail = fermato per strada, dispatch = prenotato). Quando l'utente specifica "green" senza modalità, mostriamo entrambe. Questo replica il comportamento di `Usable.py::format_multi_vehicle_response()`.

**`_build_template`** — gestire i nuovi formati:
- `model_type == "yg"`: mostra `predicted_class_name`, `vehicle_type`, `service_mode`, no confidence
- `coming_soon == True`: messaggio speciale FHVHV
- Quando ci sono più risultati YG (tutti e 3 i tipi): lista per tipo

#### `llm_tool/i18n.py`
Aggiungere chiavi:
- `fhvhv_coming_soon`: messaggio "coming soon" bilingue

---

## 5. Data flow dettagliato

```
User: "disponibilità taxi giallo zona 161 alle 10:30 lunedì marzo"

1. IntentClassifier → "predict"
2. Extractor LLM → {zone: "161", hour: 10, minute: 30, day_of_week: 0, month: 3, vehicle_type: "yellow"}
3. InputValidator.validate_and_resolve → {location_id: 161, ...}
4. Guardrail → valid → "predict"
5. Predictor:
   - vehicle_type = "yellow"
   - YGPredictor.predict(161, 10, 30, 0, 3, "yellow", "hail")
   - Feature engineering: quarter=2, hour_sin=0.766, ..., vehicle_type=0, service_mode=0
   - LightGBM.predict() → class 2 (Alta)
   - Risultato: {model_type:"yg", vehicle_type:"yellow", predicted_class:2, predicted_class_name:"Alta", ...}
6. Formatter → template + LLM insight
```

---

## 6. Error handling

| Scenario | Comportamento |
|----------|---------------|
| Modello YG non trovato (`output/yg_model_production.pkl` mancante) | `ToolException` con messaggio chiaro |
| `vehicle_type` non riconosciuto | Default a "all" |
| FHVHV richiesto | JSON `coming_soon: True` + messaggio i18n |
| `availability_class` fuori range (non 0-2) | Fallback a classe 1 ("Media") con warning log |

---

## 7. Testing

- **Unit**: `tests/test_yg_predictor.py` — mock del modello `.pkl`, testa feature engineering, encoding, predict/predict_all
- **Unit**: `tests/test_fhvhv_stub.py` — verifica che il tool restituisca `coming_soon: True`
- **Integration**: `tests/test_integration.py` (estendere) — messaggio "taxi giallo a midtown alle 10" → YGPredictor chiamato
- **Regression**: i test esistenti devono continuare a passare (vecchia tool non rimossa)

---

## 8. Vincoli e note

- La vecchia `predict_taxi_availability` rimane nel codice ma non viene più registrata come tool attivo nel grafo
- Il file `riccardo/` viene mantenuto nel repo per riferimento storico; il `.pkl` viene copiato in `output/`
- Il modello FHVHV, quando arriverà, dovrà usare la stessa interfaccia di `YellowGreenPredictor` → creare `fhvhv_predictor.py` con stessa struttura
- Il formatter NON deve rompere se `confidence` o `probabilities` sono assenti (il modello YG non li produce)
- I18n: tutti i nuovi messaggi devono essere bilingue (IT + EN)
