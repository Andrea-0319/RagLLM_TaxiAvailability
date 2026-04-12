import pandas as pd
from rapidfuzz import process


def translate_feature(f):
    mapping = {
        "hour_sin": "orario della giornata",
        "hour_cos": "orario della giornata",
        "day_sin": "giorno della settimana",
        "day_cos": "giorno della settimana",
        "zone": "zona della città",
        "month": "periodo dell'anno",
        "quarter": "momento preciso nell'ora",
        "vehicle_type": "tipo di taxi"
    }
    return mapping.get(f, f)

def translate_day(d):
    giorni = [
        "lunedì", "martedì", "mercoledì",
        "giovedì", "venerdì", "sabato", "domenica"
    ]
    return giorni[d]

# 📍 carica zone NYC
zone_lookup = pd.read_csv(
    "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
)
zone_lookup.columns = ['LocationID', 'Borough', 'Zone', 'service_zone']


# 🔥 ZONE SMART (FUZZY)
def resolve_zone_id_smart(user_text: str):

    text = user_text.lower()
    choices = zone_lookup["Zone"].str.lower().tolist()

    match, score, idx = process.extractOne(text, choices)

    if score > 70:
        return int(zone_lookup.iloc[idx]["LocationID"])

    for _, row in zone_lookup.iterrows():
        if isinstance(row["Borough"], str) and row["Borough"].lower() in text:
            return int(row["LocationID"])

    return 161