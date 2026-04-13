import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import json
import pydeck as pdk
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

from llm_tool.agent import get_agent
from llm_tool.fhvhv_predictor import get_fhvhv_predictor

riccardo_path = os.path.join(PROJECT_ROOT, "riccardo")
if riccardo_path not in sys.path:
    sys.path.insert(0, riccardo_path)
from riccardo.Prediction_model_taxi import predict_taxi_availability
from riccardo.Usable import mapping


# 🗺️ LOAD GEOJSON
@st.cache_data
def load_geojson():
    try:
        with open(os.path.join(os.path.dirname(__file__), "NYC_Taxi_Zones.geojson"), encoding="utf-8") as f:
            data = json.load(f)

        if "features" not in data:
            st.error("GeoJSON non valido ❌")
            return None

        return data

    except Exception as e:
        st.error(f"Errore caricamento GeoJSON: {e}")
        return None


# 🗺️ MAPPA NYC
def show_nyc_map(hour=9, vehicle_type="yellow", month=3, day_of_week=2):

    geojson = load_geojson()

    if geojson is None:
        return None

    for feature in geojson["features"]:

        props = feature["properties"]

        zone_id = int(
            props.get("LocationID") or
            props.get("location_id") or
            0
        )

        if zone_id == 0:
            continue

        try:
            base_date = pd.Timestamp(f"2026-{month:02d}-01")

            while base_date.dayofweek != day_of_week:
                base_date += pd.Timedelta(days=1)

            dt = f"{base_date.date()} {hour:02d}:00:00"

            if vehicle_type == "fhvhv":
                predictor = get_fhvhv_predictor()
                result = predictor.predict(
                    location_id=zone_id,
                    hour=hour,
                    minute=0,
                    day_of_week=day_of_week,
                    month=month,
                    is_festivo=False,
                )
                cls = result["predicted_class"]
                # 0=Facile=verde, 1=Medio=giallo, 2=Difficile=rosso
                fhvhv_colors = {
                    0: [0, 200, 0, 150],
                    1: [255, 200, 0, 150],
                    2: [255, 0, 0, 150],
                }
                color = fhvhv_colors.get(cls, [200, 200, 200, 50])
                label = result["predicted_class_name"]
            else:
                result = predict_taxi_availability(
                    zone=zone_id,
                    datetime_str=dt,
                    vehicle_type=vehicle_type
                )

                mapped = mapping(result["availability_class"])
                label = mapped["availability_label"]

                if label == "alta":
                    color = [0, 200, 0, 150]
                elif label == "media":
                    color = [255, 200, 0, 150]
                else:
                    color = [255, 0, 0, 150]

            props["color"] = color
            props["availability"] = label

        except Exception:
            props["color"] = [200, 200, 200, 50]
            props["availability"] = "N/A"

    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson,
        pickable=True,
        stroked=True,
        filled=True,
        opacity=0.8,
        get_fill_color="properties.color",
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )

    view_state = pdk.ViewState(
        latitude=40.7128,
        longitude=-74.0060,
        zoom=10
    )

    tooltip = {
        "html": f"<b>{{zone}}</b><br/>Taxi: {vehicle_type}<br/>Disponibilità: {{availability}}",
        "style": {"backgroundColor": "black", "color": "white"}
    }

    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v9"
    )


# 🎨 CONFIG
st.set_page_config(
    page_title="Taxi NYC Predictor",
    page_icon="🚕",
    layout="centered"
)

# 🎨 STILE
st.markdown("""
<style>
.big-title {font-size:36px; font-weight:bold; text-align:center;}
.subtitle {text-align:center; color:gray; margin-bottom:30px;}
.card {padding:20px; border-radius:15px; margin-top:20px;}
.green {background-color:#e6f4ea;}
.yellow {background-color:#fff8e1;}
.red {background-color:#fdecea;}
</style>
""", unsafe_allow_html=True)

# 🏷️ TITOLO
st.markdown('<div class="big-title">🚕 Taxi Availability NYC</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Scopri quanto è facile trovare un taxi</div>', unsafe_allow_html=True)


# --- INIT SESSION STATE ---
for key, default in [
    ("chat_history", []),
    ("current_params", {}),
    ("pending_candidates", []),
    ("input_text", ""),
    ("map_deck", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# --- VISUALIZZA CRONOLOGIA CHAT ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --- PROCESS MESSAGE HELPER ---
def _process_message(user_input: str):
    st.session_state.chat_history.append({"role": "human", "content": user_input})

    lc_history = [
        HumanMessage(content=m["content"]) if m["role"] == "human"
        else AIMessage(content=m["content"])
        for m in st.session_state.chat_history[:-1]
    ]

    with st.spinner("Analisi in corso..."):
        agent = get_agent()
        result = agent.chat(
            user_message=user_input,
            chat_history=lc_history,
            current_params=st.session_state.current_params,
            lang="it",
        )

    st.session_state.current_params = result.get("params", {})
    st.session_state.pending_candidates = result.get("candidates", [])
    st.session_state.chat_history.append({"role": "ai", "content": result["text"]})


# --- BOTTONI DISAMBIGUAZIONE (mostrati solo se ci sono candidati) ---
if st.session_state.pending_candidates:
    st.markdown("**Di quale zona intendi?**")
    cols = st.columns(min(len(st.session_state.pending_candidates), 5))
    for i, cand in enumerate(st.session_state.pending_candidates[:5]):
        if cols[i].button(cand["name"], key=f"cand_{i}"):
            st.session_state.pending_candidates = []
            st.session_state.current_params["location_id"] = int(cand["id"])
            _process_message(f"Zona ID {cand['id']}")
            st.rerun()


# --- BOTTONI ESEMPIO ---
examples = [
    "Taxi domani sera a Manhattan",
    "Qual è la zona migliore per trovare taxi?",
    "Quando è più facile trovare taxi?",
    "Taxi alle 14 nel Bronx",
    "Taxi notte a Brooklyn"
]

cols = st.columns(len(examples))

for i, example in enumerate(examples):
    if cols[i].button(example):
        st.session_state.input_text = example


# --- INPUT + INVIO ---
user_input = st.text_input(
    "Scrivi la tua richiesta 👇",
    value=st.session_state.input_text
)

if st.button("🔍 Analizza") and user_input.strip():
    st.session_state.input_text = ""
    _process_message(user_input)
    st.rerun()


# --- COLORE CARD RISPOSTA ---
resp_text = ""
if st.session_state.chat_history:
    last_response = st.session_state.chat_history[-1]
    if last_response.get("role") == "ai":
        resp_text = last_response.get("content", "")

if resp_text:
    resp_lower = resp_text.lower()
    if "alta" in resp_lower:
        color_class = "green"
        emoji = "🟢"
    elif "media" in resp_lower:
        color_class = "yellow"
        emoji = "🟡"
    elif "bassa" in resp_lower or "difficile" in resp_lower:
        color_class = "red"
        emoji = "🔴"
    else:
        color_class = ""
        emoji = ""

    if color_class:
        st.markdown(f'<div class="card {color_class}">', unsafe_allow_html=True)
        st.markdown(f"### {emoji} Risultato")
        st.markdown(resp_text)
        st.markdown("</div>", unsafe_allow_html=True)


# 🗺️ MAPPA
st.markdown("### 🗺️ Mappa disponibilità taxi NYC")

col1, col2, col3, col4 = st.columns(4)

with col1:
    hour = st.selectbox("Ora", list(range(24)), index=12)

with col2:
    vehicle_type = st.selectbox("Tipo taxi", ["yellow", "green", "fhvhv"], index=0)

with col3:
    day = st.selectbox(
        "Giorno",
        ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"],
        index=2
    )

with col4:
    month = st.selectbox("Mese", list(range(1, 13)), index=2, format_func=lambda m: [
        "Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
        "Lug", "Ago", "Set", "Ott", "Nov", "Dic"
    ][m - 1])

day_map = {
    "Lunedì": 0,
    "Martedì": 1,
    "Mercoledì": 2,
    "Giovedì": 3,
    "Venerdì": 4,
    "Sabato": 5,
    "Domenica": 6
}

day_of_week = day_map[day]

if st.button("🗺️ Analizza mappa"):

    with st.spinner("Aggiornamento mappa..."):

        st.session_state.map_deck = show_nyc_map(
            hour=hour,
            vehicle_type=vehicle_type,
            day_of_week=day_of_week,
            month=month,
        )

if st.session_state.map_deck:
    st.pydeck_chart(st.session_state.map_deck)
else:
    st.info("Clicca 'Analizza mappa' per visualizzare i dati")