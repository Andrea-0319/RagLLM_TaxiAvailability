from langchain_ollama import ChatOllama
from utils import translate_feature, translate_day

llm = ChatOllama(model="llama3", temperature=0)


def generate_response(user_input, prediction, context):

    text = user_input.lower()

    # 🔍 cosa è esplicito nella domanda
    has_explicit_day = any(x in text for x in [
        "lunedì", "lunedi", "martedì", "martedi", "mercoledì", "mercoledi",
        "giovedì", "giovedi", "venerdì", "venerdi", "sabato", "domenica",
        "oggi", "domani", "dopodomani"
    ])

    has_explicit_month = any(x in text for x in [
        "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
        "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"
    ])

    has_explicit_hour = any(x in text for x in [
        "mattina", "pomeriggio", "sera", "notte"
    ]) or any(ch.isdigit() for ch in text)

    # 📊 features sicure
    if prediction["type"] == "all":
        features = prediction["results"]["yellow"]["features"]
    else:
        features = prediction.get("features", {})

    hour = features.get("hour", None)
    day = features.get("day_of_week", None)
    month = features.get("month", None)

    # 🧠 info temporali SOLO se richieste
    time_parts = []
    if has_explicit_hour and hour is not None:
        time_parts.append(f"ora: {hour}")
    if has_explicit_day and day is not None:
        time_parts.append(f"giorno: {translate_day(day)}")
    if has_explicit_month and month is not None:
        time_parts.append(f"mese: {month}")

    temporal_info = "\n".join(time_parts) if time_parts else "non specificato"

    # =========================================================
    # 🔥 CASO 0 — INSIGHT
    # =========================================================
    if prediction["type"] == "insight":

        details = prediction.get("details", [])

        prompt = f"""
    Sei un assistente AI per i taxi a New York.

    INSIGHT PRINCIPALE:
    {prediction['description']}

    DETTAGLI BASATI SUI DATI:
    {"; ".join(details)}

    CONTESTO:
    {context}

    DOMANDA:
    {user_input}

    REGOLE:
    - Rispondi in italiano
    - NON inventare dati
    - Usa le informazioni sopra (insight + dettagli + contesto)
    - Non aggiungere giorno/mese se non richiesto
    - Evita frasi generiche
    - NON usare codici o ID di zona (es: 237, 161)
    - Usa nomi naturali delle aree (es: Manhattan, zone centrali)

    OBIETTIVO:
    - Spiegare chiaramente quando/dove conviene trovare taxi
    - Dare una risposta naturale e utile

    OUTPUT:
    Risposta in 2-3 frasi, fluida e concreta.
    """

    # =========================================================
    # 🚕 CASO 1 — VEICOLO SINGOLO
    # =========================================================
    elif prediction["type"] in ["yellow", "green"]:

        prompt = f"""
Sei un assistente AI per la disponibilità dei taxi a New York.

DATI:
Disponibilità: {prediction['availability']}

CONTESTO (basato su dati reali):
{context}

DETTAGLI TEMPORALI (usa solo se presenti):
{temporal_info}

ISTRUZIONI:
- Usa esattamente il valore di disponibilità fornito
- Non inventare giorno, mese o orario
- Usa il contesto per spiegare il comportamento (es: pomeriggio → maggiore disponibilità)
- Evita frasi generiche ("in generale", "di solito")
- Spiega in modo concreto e realistico

DOMANDA:
{user_input}

OUTPUT:
Disponibilità: {prediction['availability']}

Spiegazione: breve, basata su orario/zona usando il contesto.

Consiglio: pratico e coerente con la disponibilità.

Massimo 2 frasi.
"""

    # =========================================================
    # 🚗 CASO UBER
    # =========================================================
    elif prediction["type"] == "uber":

        prompt = f"""
Sei un assistente AI per Uber a New York.

DATI:
Tempo di attesa stimato: {prediction['waiting_time']} minuti

CONTESTO:
{context}

DOMANDA:
{user_input}

REGOLE:
- Usa SOLO il tempo fornito
- Non inventare valori
- Usa il contesto solo per spiegare

OUTPUT:

Tempo di attesa: {prediction['waiting_time']} minuti

Spiegazione: breve e concreta.

Consiglio: se conviene aspettare o cambiare opzione.

Massimo 2 frasi.
"""

    # =========================================================
    # 🔥 CASO 2 — CONFRONTO 
    # =========================================================
    else:

        yellow = prediction["results"]["yellow"]
        green = prediction["results"]["green"]
        top_features = prediction.get("top_features", [])

        feature_list = ", ".join([
            translate_feature(f["feature"]) for f in top_features
        ]) if top_features else "orario, zona, tipo di taxi"

        prompt = f"""
Sei un assistente AI per la disponibilità dei taxi a New York.

DATI:
Taxi gialli: {yellow['availability']}
Taxi verdi: {green['availability']}

CONTESTO (derivato da dati reali):
{context}

FATTORI RILEVANTI:
{feature_list}

DOMANDA:
{user_input}

ISTRUZIONI:
- Usa SOLO i valori: bassa, media, alta
- NON usare sinonimi
- NON inventare informazioni
- NON aggiungere giorno/mese se non richiesto
- Usa il contesto per spiegare (es: pomeriggio → alta disponibilità, notte → bassa)
- Collega SEMPRE la spiegazione a orario o zona
- Evita frasi generiche

OBIETTIVO:
- Spiegare il risultato usando pattern reali
- Dare un consiglio utile basato sui dati

OUTPUT:

Taxi gialli: {yellow['availability']}
Taxi verdi: {green['availability']}

Spiegazione: collega orario e zona usando il contesto.

Consiglio: indica chiaramente quale taxi conviene e perché.

Massimo 2 frasi, italiano naturale.
"""

    response = llm.invoke(prompt)
    return response.content.strip()