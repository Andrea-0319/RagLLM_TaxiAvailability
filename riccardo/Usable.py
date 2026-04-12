from Prediction_model_taxi import predict_taxi_availability


def mapping(av_class: int) -> dict:
    CLASS_LABELS = {
        0: "bassa",
        1: "media",
        2: "alta"
    }

    CLASS_DESCRIPTIONS = {
        0: "trovare un taxi è difficile in questa zona e fascia oraria",
        1: "la disponibilità dei taxi è intermedia in questa zona e fascia oraria",
        2: "trovare un taxi è generalmente facile in questa zona e fascia oraria"
    }

    return {
        "availability_class": av_class,
        "availability_label": CLASS_LABELS[av_class],
        "availability_description": CLASS_DESCRIPTIONS[av_class]
    }


def format_single_result(nome_tipo: str, zone: int, datetime_str: str, vehicle_type: str, service_mode: str):
    raw_result = predict_taxi_availability(
        zone=zone,
        datetime_str=datetime_str,
        vehicle_type=vehicle_type,
        service_mode=service_mode
    )

    mapped = mapping(raw_result["availability_class"])

    result = {
        "nome_tipo": nome_tipo,
        "availability_class": mapped["availability_class"],
        "availability_label": mapped["availability_label"],
        "availability_description": mapped["availability_description"]
    }

    return result, raw_result["top_model_features"]


def format_multi_vehicle_response(zone: int, datetime_str: str):
    risultati = []

    result_yellow, top_features = format_single_result(
        nome_tipo="Taxi giallo",
        zone=zone,
        datetime_str=datetime_str,
        vehicle_type="yellow",
        service_mode="hail"
    )
    risultati.append(result_yellow)

    result_green_hail, _ = format_single_result(
        nome_tipo="Taxi verde hail",
        zone=zone,
        datetime_str=datetime_str,
        vehicle_type="green",
        service_mode="hail"
    )
    risultati.append(result_green_hail)

    result_green_dispatch, _ = format_single_result(
        nome_tipo="Taxi verde dispatch",
        zone=zone,
        datetime_str=datetime_str,
        vehicle_type="green",
        service_mode="dispatch"
    )
    risultati.append(result_green_dispatch)

    return risultati, top_features


def stampa_risultati(zone: int, datetime_str: str):
    risultati, top_features = format_multi_vehicle_response(zone, datetime_str)

    for risultato in risultati:
        print(
            risultato["nome_tipo"] + ": "
            + "classe " + str(risultato["availability_class"])
            + " | label " + risultato["availability_label"]
            + " | " + risultato["availability_description"]
        )

    if len(top_features) > 0:
        lista_feature = []
        for elem in top_features:
            lista_feature.append(elem["feature"])

        print("Features più importanti:", ", ".join(lista_feature))
    else:
        print("Features più importanti: non disponibili")


if __name__ == "__main__":
    stampa_risultati(
        zone=161,
        datetime_str="2026-01-15 10:30:00"
    )