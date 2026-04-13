# NYC Taxi Availability Prediction Bot

Un chatbot Telegram che prevede la disponibilità dei taxi a New York City, integrando modelli ML di Roberto e un modello Riccardo per le previsioni Yellow/Green taxi.

---

## Panoramica del Progetto

| Componente | Descrizione |
|------------|-------------|
| **Interfaccia** | Bot Telegram con supporto multi-lingua (IT/EN) |
| **Orchestrazione** | LangGraph StateGraph (Intent → Context → Extractor → Guardrail → Predictor → Formatter) |
| **ML Engine** | LightGBM con modelli per taxi tradizionali (Roberto) e rideshare Uber/Lyft |
| **RAG** | FAISS-based retriever per insight contestuali |

---

## Architettura del Chatbot

### Flusso di Conversazione

```
Utente → Intent Classifier (LLM)
       → Context Refiner (multi-turn memory)
       → Parameter Extractor (semantico + regex)
       → Guardrail (validazione + smart defaults)
       → Predictor (LightGBM)
       → Formatter (template + LLM insight)
```

### Nodi Principali

| Nodo | Funzione |
|------|----------|
| **Intent Classifier** | Classifica l'intente utente (`predict`, `trend`, `oos`) |
| **Context Refiner** | Mantiene memoria dei parametri tra messaggi |
| **Parameter Extractor** | Estrae zona, ora, giorno, mese dal linguaggio naturale |
| **Guardrail** | Valida i parametri, applica defaults, gestisce disambiguazione |
| **Predictor** | Esegue predizioni LightGBM per zona/orario specificato |
| **Formatter** | Genera risposta strutturata + insight LLM contestuale |

### Gestione Tipi di Veicolo

- **Yellow/Green Taxi**: usa il modello di Roberto (`yg_predictor.py`)
- **Uber/Lyft (FHVHV)**: usa il modello di Roberto per rideshare (`fhvhv_predictor.py`)
- **Default**: mostra tutti i tipi disponibili

---

## Modelli ML

### Modello Roberto (Yellow/Green Taxi)

**File**: `roberto/waiting_time_lgbm.pkl`

Prevede la disponibilità taxi nelle 263 zone NYC basandosi su:
- `zone` (PULocationID)
- `month`, `quarter`
- `hour_sin`, `hour_cos` (encoding ciclico)
- `day_sin`, `day_cos`
- `vehicle_type` (yellow/green)
- `service_mode` (hail/dispatch)

**Feature engineering**:
- Encoding trigonometrico per ore e giorni (evita discontinuità)
- Categorie native LightGBM per zone e tipi veicolo

### Modello Roberto (FHVHV - Uber/Lyft)

**File**: `output/fhvhv_model.pkl` + `output/fhvhv_thresholds.pkl`

Prevede il tempo di attesa stimato per Uber/Lyft (3 classi):
- **Facile** (verde): attesa < 3 min
- **Medio** (giallo): attesa 3-7 min
- **Difficile** (rosso): attesa > 7 min

**Feature engineering**:
- `hour_sin`, `hour_cos`, `minute_sin`, `minute_cos`
- `dow_sin`, `dow_cos`
- `month`, `is_festivo` (giorni festivi USA)

### Modello Riccardo

**File**: `riccardo/taxi_lgbm_model_production.pkl`

Modello legacy per confronti. Utilizza feature simili a Roberto:
- `zone`, `month`, `quarter`
- `hour_sin`, `hour_cos`
- `day_sin`, `day_cos`
- `vehicle_type`, `service_mode`

**Nota**: Il modello Riccardo è integrato nella Streamlit app di Rania (`llm_tool/StreamlitRania/app.py`) per retrocompatibilità.

---

## File Structure

```
Progetto_Accenture/
├── llm_tool/
│   ├── agent.py              # LangGraph StateGraph orchestrator
│   ├── taxi_predictor.py     # Yellow/Green LightGBM wrapper
│   ├── yg_predictor.py       # Yellow/Green multi-output predictor
│   ├── fhvhv_predictor.py    # Uber/Lyft predictor (Roberto)
│   ├── input_validator.py    # Parameter extraction + validation
│   ├── prompts.py            # LLM prompts (intent, oos, insight)
│   ├── rag_retriever.py      # FAISS RAG per insight contestuali
│   ├── rag_documents.py      # 55 fatti hardcoded su disponibilità taxi NYC
│   ├── i18n.py               # Internazionalizzazione (IT/EN)
│   ├── llm_factory.py        # Groq + Ollama fallback
│   ├── config.py             # Costanti, mappings, configurazioni
│   └── StreamlitRania/
│       ├── app.py            # Interfaccia Streamlit (usa TaxiAgent)
│       ├── uber_model.py     # Modello Uber placeholder
│       └── NYC_Taxi_Zones.geojson
├── roberto/
│   ├── waiting_time_lgbm.pkl    # Modello LightGBM rideshare
│   ├── zone_target_map.pkl      # Mapping zone → target
│   └── TAXY_TLC_NY_MODEL.ipynb  # Notebook training
├── riccardo/
│   ├── taxi_lgbm_model_production.pkl
│   ├── feature_importance_production.csv
│   ├── Prediction_model_taxi.py
│   └── Usable.py
├── start_bot.py              # Launch script per Telegram bot
├── start_streamlit.py        # Launch script per Streamlit
├── TaxiBot.bat               # Batch launcher (Windows)
└── requirements.txt          # Dipendenze Python
```

---

## Quick Start

### Avviare il Bot Telegram

```bash
# Attiva l'ambiente Python corretto
C:\Users\andre\AppData\Local\Programs\Python\Python311\python.exe start_bot.py
```

Oppure usa il batch:
```cmd
TaxiBot.bat
```

### Avviare Streamlit

```bash
start_streamlit.bat
# oppure
streamlit run llm_tool/StreamlitRania/app.py
```

---

## Comandi Supportati

| Tipo | Esempi |
|------|--------|
| **Predizione singola** | "JFK lunedì alle 17:30", "Times Square, 9:00, martedì" |
| **Fascia oraria** | "Midtown pomeriggio", "Bronx sera di sabato" |
| **Trend storici** | "Come varia la disponibilità a Manhattan?" |
| **Uber/Lyft** | "Quanto tempo aspetto per Uber a JFK?" |
| **Disambiguazione** | Risponde con bottoni inline se la zona è ambigua |

---

## Dipendenze

Vedi `requirements.txt`. Principali:
- `langgraph`, `langchain-core`, `langchain-groq`
- `lightgbm`, `shap`, `scikit-learn`
- `python-telegram-bot`
- `faiss-cpu` (per RAG, opzionale)

---

## Note

- **Multi-turn**: il bot ricorda i parametri tra messaggi (es. "JFK" → "e domani?" mantiene JFK)
- **Rate limiting**: 10 msg/min per utente (anti-spam)
- **Fallback LLM**: Groq Cloud → Ollama locale se non disponibile
- **Graceful degradation**: RAG opzionale, il bot funziona anche senza dipendenze extra

---

## Crediti

- **Chatbot**: Andrea (LangGraph, orchestrazione, integrazione)
- **Modello Yellow/Green**: Roberto (LightGBM training, feature engineering)
- **Modello FHVHV**: Roberto (Uber/Lyft availability prediction)
- **Modello legacy**: Riccardo (confronto e retrocompatibilità)
- **Streamlit**: Rania (interfaccia web, mappa PyDeck)