"""
Test del TaxiPredictorModel — verifica che il modello LightGBM si carichi
correttamente e produca predizioni valide per 5 scenari reali.
"""

import sys
sys.path.insert(0, r'C:\Users\andre\Desktop\Progetto_Accenture')

from llm_tool.taxi_predictor import get_predictor

print("=" * 50)
print("TEST: TaxiPredictorModel")
print("=" * 50)

print("\n[1] Caricamento modello...")
predictor = get_predictor()
predictor.load()
print("    ✅ Modello caricato")

# Scenari di test (zone, bucket, giorno, mese)
scenarios = [
    (236, 16, 0, 3,  "Midtown, Lunedì 8:00, Marzo"),
    (230, 40, 5, 7,  "Times Square, Sabato 20:00, Luglio"),
    (132,  8, 3, 5,  "JFK Airport, Giovedì 4:00, Maggio"),
    ( 71, 33, 4, 1,  "Brooklyn, Venerdì 16:30, Gennaio"),
    (  3, 20, 6, 9,  "Bronx, Domenica 10:00, Settembre"),
]

print("\n[2] Predizioni su 5 scenari:\n")
all_passed = True

for loc_id, bucket, dow, month, description in scenarios:
    result = predictor.predict(
        location_id=loc_id,
        half_hour_bucket=bucket,
        day_of_week=dow,
        month=month,
        language="it",
    )

    # Validazione risultato
    assert result["success"], f"Predizione fallita: {result}"
    assert 0 <= result["predicted_class"] <= 4, "Classe non valida"
    assert 0.0 <= result["confidence"] <= 1.0, "Confidenza fuori range"
    assert len(result["probabilities"]) == 5, "Distribuzione incompleta"

    cls = result["predicted_class"]
    name = result["predicted_class_name"]
    conf = result["confidence"] * 100
    zone = result["location_name"]
    emojis = {0: "🔴", 1: "🟠", 2: "🟡", 3: "🟢", 4: "🔵"}

    print(f"  📍 {description}")
    print(f"     Zona ufficiale: {zone}")
    print(f"     {emojis[cls]} {name} — confidenza {conf:.1f}%")
    print()

print("=" * 50)
print("✅ Tutti i test superati!")
print("=" * 50)