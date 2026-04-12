"""
Test end-to-end dell'agent e del validator — senza avviare il bot Telegram.

Verifica:
1. Import corretto di tutti i moduli llm_tool
2. InputValidator: estrazione parametri regex
3. TaxiAgent.direct_predict: predizioni programmatiche
4. TaxiAgent.chat: risposta a query naturali via LLM
"""

import sys
import json
sys.path.insert(0, r'C:\Users\andre\Desktop\Progetto_Accenture')

from llm_tool import get_agent, get_validator, get_predictor

OUTPUT_FILE = r'C:\Users\andre\Desktop\Progetto_Accenture\output\bot_test_results.json'

print("=" * 60)
print("TEST END-TO-END: LLM Tool")
print("=" * 60)

# ── Test 1: Import check ──────────────────────────────────────────────────────
print("\n[1] Import check...")
from llm_tool.config import ZONE_ALIASES, CLASS_NAMES
from llm_tool.taxi_predictor import resolve_zone_id
print(f"    ✅ {len(ZONE_ALIASES)} zone alias caricate, {len(CLASS_NAMES)} classi")

# ── Test 2: Input Validator (regex) ───────────────────────────────────────────
print("\n[2] Input Validator - estrazione regex:")
validator = get_validator()

test_queries_regex = [
    ("Quanti taxi a Midtown lunedì alle 8 a marzo?", "it"),
    ("Taxi at JFK Saturday at 10pm in July", "en"),
    ("Times Square sabato sera", "it"),
]

for query, lang in test_queries_regex:
    params = validator.extract(query)
    resolved = validator.validate_and_resolve(params)
    print(f"    Query: {query!r}")
    print(f"    → zone={params['zone']}, month={params['month']}, "
          f"dow={params['day_of_week']}, hour={params['hour']}")
    print(f"    → location_id={resolved.get('location_id')}, missing={resolved.get('missing')}")
    print()

# ── Test 3: Direct predict (bypassa LLM) ─────────────────────────────────────
print("[3] Direct predict (no LLM):")
agent = get_agent()

direct_scenarios = [
    ("midtown",    3, 0, 8,  0,  "it", "Midtown, Lunedì 8:00 Marzo"),
    ("jfk",        7, 5, 22, 0,  "en", "JFK, Saturday 10pm July"),
    ("brooklyn",   1, 4, 17, 30, "it", "Brooklyn, Venerdì 17:30 Gen"),
]

direct_results = []
for zone, month, dow, hour, minute, lang, desc in direct_scenarios:
    response = agent.direct_predict(zone, month, dow, hour, minute, lang)
    direct_results.append({"description": desc, "response": response})
    print(f"    📍 {desc}")
    # Show first 2 lines of response
    first_lines = response.split('\n')[:3]
    for line in first_lines:
        print(f"       {line}")
    print()

# ── Test 4: Agent chat (richiede Ollama) ──────────────────────────────────────
print("[4] Agent chat (Ollama + Gemma-4-E2B):")
print("    Nota: questa parte richiede Ollama in esecuzione\n")

chat_queries = [
    "Quanti taxi ci sono a Midtown il lunedì alle 8 di mattina a marzo?",
    "How easy to find a taxi at Times Square Saturday 10pm in July?",
    "Brooklyn venerdì sera a gennaio, difficile trovare un taxi?",
]

chat_results = []
for query in chat_queries:
    print(f"    User: {query!r}")
    try:
        response = agent.chat(query)
        chat_results.append({"query": query, "response": response})
        # Show first 150 chars
        preview = response[:150].replace('\n', ' ')
        print(f"    Bot:  {preview}…")
    except Exception as e:
        print(f"    ❌ Errore: {e}")
        chat_results.append({"query": query, "error": str(e)})
    print()

# ── Salva risultati ───────────────────────────────────────────────────────────
all_results = {
    "direct_predict": direct_results,
    "agent_chat": chat_results,
}
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print("=" * 60)
print(f"✅ Risultati salvati in: {OUTPUT_FILE}")
print("=" * 60)