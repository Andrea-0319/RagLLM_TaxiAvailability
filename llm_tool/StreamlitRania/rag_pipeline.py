from llm_parser import parse_with_llm
from Prediction_model_taxi import predict_taxi_availability
from Usable import mapping
from utils import resolve_zone_id_smart, zone_lookup
from rag_retriever import retrieve_context
from llm_response import generate_response
from uber_model import predict_uber_waiting_time


# -------------------------
# 🔍 DETECTION
# -------------------------
def detect_vehicle_type(user_input: str):
    text = user_input.lower()

    if "green" in text or "verde" in text:
        return "green"

    if "yellow" in text or "giallo" in text:
        return "yellow"

    return "all"


def detect_service_type(user_input: str):
    text = user_input.lower()

    if "uber" in text:
        return "uber"

    return "taxi"


def detect_intent(user_input):
    text = user_input.lower()

    if "quando" in text or "orario" in text or "momento" in text:
        return "best_time"

    if "zona" in text or "dove" in text:
        return "best_zone"

    return "prediction"


# -------------------------
# 🧩 NODES BASE
# -------------------------
def parse_node(user_input):
    parsed = parse_with_llm(user_input)
    print("NODE parse:", parsed)
    return parsed


def validate_node(parsed):
    if parsed is None:
        return {
            "zone": "midtown",
            "day_of_week": 2,
            "hour": 9,
            "month": 3
        }

    validated = {
        "zone": parsed.get("zone", "midtown"),
        "day_of_week": parsed.get("day_of_week", 2),
        "hour": parsed.get("hour", 9),
        "month": parsed.get("month", 3)
    }

    print("NODE validate:", validated)
    return validated


def zone_node(user_input):
    location_id = resolve_zone_id_smart(user_input)
    print("NODE zone_id:", location_id)
    return location_id


# -------------------------
# 🚕 TAXI MODEL
# -------------------------
def predict_node(validated, location_id, vehicle_type):

    dt = f"2026-{validated['month']:02d}-15 {validated['hour']:02d}:00:00"

    if vehicle_type in ["yellow", "green"]:

        result = predict_taxi_availability(
            zone=location_id,
            datetime_str=dt,
            vehicle_type=vehicle_type
        )

        mapped = mapping(result["availability_class"])

        return {
            "type": vehicle_type,
            "availability": mapped["availability_label"],
            "description": mapped["availability_description"],
            "features": result["features_used"],
            "top_features": result["top_model_features"]
        }

    # MULTI
    results = {}
    top_features = None

    for vt in ["yellow", "green"]:

        res = predict_taxi_availability(
            zone=location_id,
            datetime_str=dt,
            vehicle_type=vt
        )

        mapped = mapping(res["availability_class"])

        results[vt] = {
            "availability": mapped["availability_label"],
            "description": mapped["availability_description"],
            "features": res["features_used"]
        }

        if top_features is None:
            top_features = res["top_model_features"]

    return {
        "type": "all",
        "results": results,
        "top_features": top_features
    }


# -------------------------
# 🚗 UBER MODEL
# -------------------------
def predict_uber_node(validated, location_id):

    dt = f"2026-{validated['month']:02d}-15 {validated['hour']:02d}:00:00"

    result = predict_uber_waiting_time(
        zone=location_id,
        datetime_str=dt
    )

    return {
        "type": "uber",
        "waiting_time": round(result["waiting_time"], 1)
    }


# -------------------------
# 🔥 INSIGHTS
# -------------------------
def best_time_node():
    return {
        "summary": "Il momento migliore per trovare taxi è nel pomeriggio e nel tardo pomeriggio.",
        "details": [
            "Tra le 12 e le 16 la disponibilità è generalmente buona",
            "Tra le 17 e le 19 si raggiunge il picco massimo",
            "Tra le 4 e le 6 la disponibilità è molto bassa"
        ]
    }


def best_zone_node():
    return {
        "summary": "Le zone migliori per trovare taxi sono quelle centrali di Manhattan.",
        "details": [
            "Le aree centrali hanno il volume di corse più alto",
            "Le zone con maggiore attività mantengono più taxi disponibili",
            "Le zone periferiche hanno disponibilità più bassa"
        ]
    }


# -------------------------
# 🚀 PIPELINE
# -------------------------
def pipeline(user_input):

    intent = detect_intent(user_input)
    service = detect_service_type(user_input)

    print("INTENT:", intent)
    print("SERVICE:", service)

    # 🔥 INSIGHT
    if intent == "best_zone":
        insight = best_zone_node()
        context = retrieve_context("zone migliori taxi Manhattan alta disponibilità")

        return generate_response(user_input, {
            "type": "insight",
            "description": insight["summary"],
            "details": insight["details"]
        }, context)

    if intent == "best_time":
        insight = best_time_node()
        context = retrieve_context("orari migliori taxi pomeriggio picco disponibilità")

        return generate_response(user_input, {
            "type": "insight",
            "description": insight["summary"],
            "details": insight["details"]
        }, context)

    # 🔽 PARSE
    parsed = parse_node(user_input)

    if parsed and "error" in parsed:

        if parsed["error"] == "not_taxi":
            return "Posso aiutarti solo con taxi o Uber 🚕"

        if parsed["error"] == "wrong_city":
            return "Supporto solo New York 🚕"

        if parsed["error"] == "unknown_city":
            return "Specifica una zona di New York"

    validated = validate_node(parsed)
    location_id = zone_node(user_input)

    # 🔥 UBER FLOW
    if service == "uber":

        result = predict_uber_node(validated, location_id)
        print("UBER OUTPUT:", result)

        context = retrieve_context("uber waiting time new york")

        return generate_response(user_input, result, context)

    # 🔥 TAXI FLOW
    vehicle_type = detect_vehicle_type(user_input)

    result = predict_node(validated, location_id, vehicle_type)
    print("TAXI OUTPUT:", result)

    # 🔥 RAG QUERY
    if result["type"] == "all":

        y = result["results"]["yellow"]["availability"]
        g = result["results"]["green"]["availability"]

        query_for_rag = f"{user_input} taxi yellow {y} taxi green {g}"

    else:

        query_for_rag = f"{user_input} taxi {result['type']} {result['availability']}"

    print("RAG QUERY:", query_for_rag)

    context = retrieve_context(query_for_rag)

    return generate_response(user_input, result, context)