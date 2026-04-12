from groq import Groq
import json
from datetime import datetime
import re
import os
from dotenv import load_dotenv
from utils import zone_lookup

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))


# -------------------------
# 🔍 FILTRO SERVIZIO
# -------------------------
def is_nyc_zone(user_input: str):  #filtre per zone di new york se chiedi di los angeles ti dice che non funziona

    text = user_input.lower()

    zones = zone_lookup["Zone"].dropna().astype(str).str.lower().tolist()

    return any(zone in text for zone in zones)
def is_transport_request(user_input: str):  #filtro per il trasporto
    keywords = ["taxi", "cab", "uber"]
    text = user_input.lower()
    return any(k in text for k in keywords)


# -------------------------
# 🌍 FILTRO CITTÀ
# -------------------------
def detect_city_llm(user_input: str): #usa llm per capire la zona

    prompt = f"""
Classifica la città della richiesta.

Regole:
- Se è New York o zone NYC → "nyc"
- Se è un'altra città → "other"
- Se non è chiaro → "unknown"

Rispondi SOLO JSON:

{{
  "city": "nyc | other | unknown"
}}

Frase:
"{user_input}"
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Classificatore città."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        text = response.choices[0].message.content
        match = re.search(r'\{.*\}', text, re.DOTALL)

        if not match:
            return "unknown"

        data = json.loads(match.group())
        return data.get("city", "unknown")

    except:
        return "unknown"


# -------------------------
# 🧠 FIX GIORNI
# -------------------------
def fix_relative_days(data, user_input):

    if data.get("error"):
        return data

    text = user_input.lower()
    today = datetime.now().weekday()

    if "oggi" in text:
        data["day_of_week"] = today
    elif "domani" in text:
        data["day_of_week"] = (today + 1) % 7
    elif "dopodomani" in text:
        data["day_of_week"] = (today + 2) % 7

    return data


# -------------------------
# ⏱️ FIX ORARI
# -------------------------
def fix_time_expressions(data, user_input):

    text = user_input.lower()

    if "mattina" in text:
        data["hour"] = 9 #orari predefniti per mattina pomeriggio sera
    elif "pomeriggio" in text:
        data["hour"] = 15
    elif "sera" in text:
        data["hour"] = 20
    elif "notte" in text or "stanotte" in text:
        data["hour"] = 2

    return data


# -------------------------
# 📅 FIX MESE
# -------------------------
def extract_month(text, llm_month):

    text = text.lower()

    months = {
        "gennaio": 1, "febbraio": 2, "marzo": 3,
        "aprile": 4, "maggio": 5, "giugno": 6,
        "luglio": 7, "agosto": 8, "settembre": 9,
        "ottobre": 10, "novembre": 11, "dicembre": 12
    }

    for m, val in months.items():
        if m in text:
            return val

    return llm_month


# -------------------------
# 🔒 VALIDAZIONE
# -------------------------
def validate_output(data, today):

    if data.get("error"):
        return data

    # 🔥 gestione sicura dei valori
    hour = data.get("hour")
    if hour is None:
        hour = 9

    month = data.get("month")
    if month is None:
        month = 3

    day = data.get("day_of_week")
    if day is None:
        day = today

    return {
        "zone": data.get("zone", "manhattan"),
        "day_of_week": int(day),
        "hour": max(0, min(23, int(hour))),
        "month": max(1, min(12, int(month)))
    }


# -------------------------
# 🛟 FALLBACK
# -------------------------
def fallback_parser(text):

    text = text.lower()

    zone = "manhattan"
    if "bronx" in text:
        zone = "bronx"
    elif "brooklyn" in text:
        zone = "brooklyn"
    elif "queens" in text:
        zone = "queens"

    hour = 9
    match = re.search(r'(\d{1,2})', text)
    if match:
        hour = int(match.group(1))

    return {
        "zone": zone,
        "day_of_week": datetime.now().weekday(),
        "hour": hour,
        "month": datetime.now().month
    }


# -------------------------
# 🚀 PARSER PRINCIPALE
# -------------------------
def parse_with_llm(user_input: str):

    #  filtro servizio (taxi + uber)
    if not is_transport_request(user_input):
        return {"error": "not_transport"}

    #  filtro città

    if is_nyc_zone(user_input):
        city = "nyc"
    else:
        city = detect_city_llm(user_input)

        if city == "other":
            return {"error": "wrong_city"}

        if city == "unknown":
            return {"error": "unknown_city"}

    today = datetime.now().weekday()

    prompt = f"""
Oggi è il giorno {today} (0=lunedi, 6=domenica).

Regole:
- Estrai SEMPRE tutti i campi
- NON aggiungere testo
- SOLO JSON

Formato:
{{
  "zone": "bronx|brooklyn|manhattan|midtown|queens|staten island|unknown",
  "day_of_week": 0-6,
  "hour": 0-23,
  "month": 1-12
}}

Frase:
"{user_input}"
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Parser JSON rigoroso."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        text = response.choices[0].message.content
        match = re.search(r'\{.*\}', text, re.DOTALL)

        if not match:
            return fallback_parser(user_input)

        data = json.loads(match.group())

    except Exception as e:
        print("Errore parsing:", e)
        return fallback_parser(user_input)

    # 🔥 POST-PROCESSING (CORE)
    data = validate_output(data, today)
    data["month"] = extract_month(user_input, data.get("month", today))
    data = fix_relative_days(data, user_input)
    data = fix_time_expressions(data, user_input)   #questo mi rida i valori di default se non ho messo qualcosa nella richiesta

    print("PARSED FINAL:", data)

    return data