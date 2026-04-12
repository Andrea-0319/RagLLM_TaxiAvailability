"""
Configuration constants for the NYC Taxi Demand Prediction LLM Tool.

Contains:
- Model paths and artifact locations
- Class definitions and color mappings
- Borough and service zone encodings
- Zone lookup and aliases
- Prompt templates
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
OUTPUT_DIR = PROJECT_ROOT / "output"

MODEL_ARTIFACTS_PATH = OUTPUT_DIR / "ml_model_artifacts_all_months.pkl"
ZONE_DEFAULTS_PATH = OUTPUT_DIR / "zone_defaults.csv"
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

# ─── Feature columns (must match training exactly) ───────────────────────
FEATURE_COLS = [
    "PULocationID", "half_hour_bucket", "day_of_week", "month",
    "unique_taxi_types", "avg_trip_duration_min",
    "is_weekend", "is_rush_hour", "is_night",
    "borough_encoded", "service_zone_encoded",
]

# ─── Availability Classes ────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Molto Difficile",
    1: "Difficile",
    2: "Medio",
    3: "Facile",
    4: "Molto Facile",
}

CLASS_EMOJIS = {0: "🔴", 1: "🟠", 2: "🟡", 3: "🟢", 4: "🔵"}

# ─── Borough Encoding (from training data) ───────────────────────────────
BOROUGH_ENCODING = {
    "Bronx": 0,
    "Brooklyn": 1,
    "EWR": 2,
    "Manhattan": 3,
    "Queens": 4,
    "Staten Island": 5,
    "Unknown": 6,
}

# ─── Service Zone Encoding (from training data) ──────────────────────────
SERVICE_ZONE_ENCODING = {
    "Airports": 0,
    "Boro Zone": 1,
    "EWR": 2,
    "Yellow Zone": 4,
}

# ─── Day of Week ─────────────────────────────────────────────────────────
DAY_NAMES_IT = {
    0: "Lunedi", 1: "Martedi", 2: "Mercoledi",
    3: "Giovedi", 4: "Venerdi", 5: "Sabato", 6: "Domenica",
}

# ─── Month ───────────────────────────────────────────────────────────────
MONTH_NAMES_IT = {
    1: "Gennaio", 2: "Febbraio", 3: "Marzo", 4: "Aprile",
    5: "Maggio", 6: "Giugno", 7: "Luglio", 8: "Agosto",
    9: "Settembre", 10: "Ottobre", 11: "Novembre", 12: "Dicembre",
}

# ─── LLM Config ──────────────────────────────────────────────────────────
# Provider: "groq+ollama" → Groq cloud primario, fallback Ollama locale
#           "ollama"      → solo Ollama locale (nessuna chiave API richiesta)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq+ollama")

# Groq Cloud (primario) — modello large, qualità superiore per structured output
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"   # modello production-ready su GroqCloud

# Ollama Locale (fallback) — small model, nessuna connessione internet richiesta
LLM_MODEL    = "llama3.2:3b"
LLM_BASE_URL = "http://localhost:11434"

LLM_TEMPERATURE = 0.3

# ─── Telegram Config ─────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")

# ─── Zone Aliases (common names -> LocationID) ───────────────────────────
ZONE_ALIASES = {
    # Manhattan
    "midtown manhattan": 161,
    "midtown center": 161,
    "midtown east": 162,
    "midtown north": 163,
    "midtown south": 164,
    "times square": 230,
    "timesquare": 230,
    "theatre district": 230,
    "theater district": 230,
    "wall street": 162,
    "financial district": 162,
    "broadway": 230,
    "central park": 263,
    "harlem": 224,
    "upper east side": 239,
    "upper west side": 237,
    "soho": 232,
    "so-ho": 232,
    "tribeca": 239,
    "chelsea": 229,
    "east village": 233,
    "west village": 234,
    "greenwich village": 234,
    "chinatown": 233,
    "little italy": 233,
    "flatiron": 232,
    "gramercy": 233,
    "murray hill": 236,
    "hell kitchen": 230,
    "hells kitchen": 230,
    "garment district": 236,
    "penn station": 186,
    "grand central": 236,
    "union square": 233,
    "washington heights": 224,
    "inwood": 224,
    "east harlem": 224,
    "spanish harlem": 224,
    # Airports
    "jfk": 132,
    "jfk airport": 132,
    "john f kennedy": 132,
    "lga": 138,
    "laguardia": 138,
    "la guardia": 138,
    "la guard": 138,
    "ewr": 1,
    "newark airport": 1,
    "newark": 1,
    # Boroughs (default zones)
    "bronx": 3,
    "brooklyn": 71,
    "queens": 7,
    "staten island": 50,
    "manhattan": 236,
    # Brooklyn areas
    "williamsburg": 71,
    "downtown brooklyn": 71,
    "dumbo": 71,
    "park slope": 71,
    "brooklyn heights": 71,
    # Queens areas
    "astoria": 7,
    "jamaica": 130,
    "long island city": 71,
    "lic": 71,
    "flushing": 7,
    # Bronx areas
    "fordham": 3,
    "concourse": 3,
    "riverdale": 3,
}


def hour_minute_to_half_bucket(hour: int, minute: int) -> int:
    """Convert hour and minute to half-hour bucket (0-47)."""
    return hour * 2 + (1 if minute >= 30 else 0)


def half_bucket_to_time(bucket: int) -> tuple:
    """Convert half-hour bucket to (hour, minute_start, minute_end)."""
    hour = bucket // 2
    minute_start = 0 if bucket % 2 == 0 else 30
    minute_end = 30 if bucket % 2 == 0 else 59
    return hour, minute_start, minute_end
