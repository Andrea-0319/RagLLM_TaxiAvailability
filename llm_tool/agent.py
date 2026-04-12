"""
LangGraph StateGraph Agent for NYC Taxi Demand Prediction (v4).

Graph: Intent (LLM) → Context (multi-turn memory) → Extractor (semantic + regex)
       → Guardrail (validation) → Predictor (LightGBM + SHAP) → Formatter (template + LLM insight)
"""

import json
import logging
import operator
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Optional, Dict, Any, Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from .taxi_predictor import get_historical_trends, get_predictor
from .yg_predictor import get_yg_predictor
from .input_validator import get_validator
from .llm_factory import get_llm
from .i18n import get_msg
from .config import (
    CLASS_NAMES, CLASS_EMOJIS, DAY_NAMES_IT, MONTH_NAMES_IT,
    YG_CLASS_NAMES, YG_CLASS_EMOJIS, VEHICLE_TYPE_DISPLAY,
)
from .prompts import _INTENT_PROMPT, _OOS_PROMPT, _INSIGHT_PROMPT

logger = logging.getLogger(__name__)




# ─── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:          Annotated[List[BaseMessage], operator.add]
    current_params:    Dict[str, Any]        # session-persisted parameters
    intent:            str                   # 'predict' | 'trend' | 'oos'
    candidates:        List[Dict[str, Any]]  # disambiguation options
    results:           List[Dict[str, Any]]  # predictor output
    validation_errors: List[str]             # guardrail feedback
    hour_range:        List[int]             # hours for range predictions
    next_step:         str                   # routing signal
    language:          str                   # language code (e.g. 'en', 'it')


# ─── Helper: deterministic response template ─────────────────────────────────

def _build_template(results: List[Dict], params: Dict) -> str:
    """Build the structured, emoji-rich part of the response deterministically."""
    if not results:
        return "⚠️ Nessun risultato disponibile."

    now        = datetime.now(ZoneInfo("America/New_York"))
    eval_hour   = params.get("hour")   if params.get("hour")   is not None else now.hour
    eval_minute = params.get("minute") if params.get("minute") is not None else 0
    eval_dow    = params.get("day_of_week") if params.get("day_of_week") is not None else now.weekday()
    eval_month  = params.get("month") if params.get("month") is not None else now.month

    r0        = results[0]
    zone_name = r0.get("location_name", "?")
    borough   = r0.get("borough", "")
    day_name  = DAY_NAMES_IT.get(eval_dow, "?")
    month_name = MONTH_NAMES_IT.get(eval_month, "?")

    lines = [f"🚕 *{zone_name}* ({borough})", f"📅 {day_name} — {month_name}", ""]

    if r0.get("coming_soon"):
        lines.append(r0.get("message", "🚗 FHVHV model coming soon."))
        return "\n".join(lines)

    if "hourly_avg_availability" in r0:
        lines.append("📊 *Trend Storico:*")
        lines.append("I dati orari indicano l'indice di disponibilità medio da 0 (Bassa) a 1 (Alta).")
        return "\n".join(lines)

    model_type = r0.get("model_type", "legacy")

    if model_type == "yg":
        time_str = f"{eval_hour:02d}:{eval_minute:02d}"
        lines.append(f"🕐 Ore {time_str}")
        lines.append("")

        if len(results) == 1:
            r    = results[0]
            cls  = r["predicted_class"]
            emoji = YG_CLASS_EMOJIS.get(cls, "❓")
            vtype = VEHICLE_TYPE_DISPLAY.get(r["vehicle_type"], r["vehicle_type"])
            lines.append(f"{emoji} *{vtype}*: {r['predicted_class_name']}")
            lines.append(f"   _{r['availability_description']}_")
        else:
            lines.append("📊 *Disponibilità per tipo di taxi:*")
            for r in results:
                cls   = r["predicted_class"]
                emoji  = YG_CLASS_EMOJIS.get(cls, "❓")
                sm     = r.get("service_mode", "")
                key    = f"{r['vehicle_type']}_{sm}" if sm else r["vehicle_type"]
                vtype  = VEHICLE_TYPE_DISPLAY.get(key, r["vehicle_type"])
                lines.append(f"  {emoji} *{vtype}*: {r['predicted_class_name']}")
                lines.append(f"     _{r['availability_description']}_")

        return "\n".join(lines)

    name_to_emoji = {v: CLASS_EMOJIS[k] for k, v in CLASS_NAMES.items()}

    if len(results) == 1:
        cls      = r0["predicted_class"]
        time_str = f"{eval_hour:02d}:{eval_minute:02d}"
        lines += [
            f"🕐 Ore {time_str}",
            f"{CLASS_EMOJIS.get(cls, '❓')} *{r0['predicted_class_name']}*"
            f"  —  Confidenza: {r0['confidence']*100:.1f}%",
            "",
            "📊 *Distribuzione probabilità:*",
        ]
        for cls_name, prob in r0.get("probabilities", {}).items():
            e = name_to_emoji.get(cls_name, "•")
            lines.append(f"  {e} {cls_name}: {prob*100:.1f}%")
    else:
        lines.append("📊 *Disponibilità per fascia oraria:*")
        for r in results:
            h    = r.get("time_bucket", 0) // 2
            e    = CLASS_EMOJIS.get(r["predicted_class"], "❓")
            conf = r["confidence"] * 100
            lines.append(f"  {e} *{h:02d}:00* → {r['predicted_class_name']} ({conf:.0f}%)")

    return "\n".join(lines)


# ─── Nodes ───────────────────────────────────────────────────────────────────

def intent_classifier_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: LLM-based intent classification (predict / trend / oos).
    Passes recent conversation history so the LLM can detect follow-up messages
    (e.g. "e alle 17:30?" after a prediction → predict, not oos).
    """
    messages  = state["messages"]
    last_msg  = messages[-1].content
    print(f"\n--- [1. INTENT CLASSIFIER] ---")

    if re.match(r'^zona\s*id\s*\d+', last_msg, re.IGNORECASE):
        print("   Intent: PREDICT (Fast-path)")
        return {"intent": "predict"}

    # Build a short context block from the last ~3 turns (6 messages, excluding current)
    recent = messages[max(0, len(messages) - 7):-1]
    context_lines = []
    for m in recent:
        role = "User" if isinstance(m, HumanMessage) else "Bot"
        context_lines.append(f"{role}: {m.content[:200]}")
    context_block = "\n".join(context_lines) if context_lines else "No previous context."

    prompt = _INTENT_PROMPT + f"\n\nRecent conversation:\n{context_block}"

    llm = get_llm(temperature=0.0)
    try:
        resp = llm.invoke([SystemMessage(content=prompt),
                           HumanMessage(content=last_msg)])
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', resp.content.strip(), flags=re.MULTILINE)
        intent = json.loads(raw).get("intent", "predict")
        if intent not in ("predict", "trend", "oos"):
            intent = "predict"
    except Exception as e:
        logger.warning(f"[Intent] LLM failed ({e}) — defaulting to 'predict'")
        intent = "predict"

    print(f"   Intent: {intent.upper()}")
    return {"intent": intent}


def extractor_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 3: Parameter extraction + multi-turn merge.
    New values override previous ones; null means 'keep the old value'.
    """
    last_msg = state["messages"][-1].content
    intent   = state["intent"]
    print(f"--- [3. EXTRACTOR] ---")

    if intent == "oos":
        return {"next_step": "format"}

    validator   = get_validator()
    raw_params  = validator.extract(last_msg)
    resolved    = validator.validate_and_resolve(raw_params, text=last_msg)

    # Multi-turn merge: start from carried params, override only explicit values
    merged = state.get("current_params", {}).copy()
    for key in ("location_id", "month", "day_of_week", "hour", "minute", "vehicle_type"):
        new_val = resolved.get(key)
        if new_val is not None:
            merged[key] = new_val
            print(f"   → {key} = {new_val}")

    # Hour-range detection for slot keywords
    hour_range: List[int] = []
    lmsg = last_msg.lower()
    if any(w in lmsg for w in ("pomeriggio", "afternoon")):
        hour_range = list(range(14, 19))
    elif any(w in lmsg for w in ("mattina", "morning")):
        hour_range = list(range(7, 12))
    elif any(w in lmsg for w in ("sera", "evening")):
        hour_range = list(range(19, 23))
    elif any(w in lmsg for w in ("notte", "night")):
        hour_range = [23, 0, 1, 2, 3]

    if hour_range:
        print(f"   → Hour range: {hour_range}")

    return {
        "current_params": merged,
        "candidates":     resolved.get("candidates", []),
        "hour_range":     hour_range,
        "next_step":      "guardrail",
    }


def guardrail_node(state: AgentState) -> Dict[str, Any]:
    """Node 4: Logical validation of merged parameters."""
    print(f"--- [4. GUARDRAIL] ---")
    p      = state.get("current_params", {})
    errors = []

    lang   = state.get("language", "it")

    # Zone presence
    if not p.get("location_id"):
        if state.get("candidates"):
            print("   → Ambiguous zone → Disambiguator")
            return {"next_step": "disambiguate"}
        print("   → Missing zone → ask_zone")
        return {"next_step": "ask_zone"}

    if not (1 <= p["location_id"] <= 265):
        errors.append(get_msg(lang, "invalid_id"))
    if p.get("hour") is not None and not (0 <= p["hour"] <= 23):
        errors.append(get_msg(lang, "invalid_hour"))
    if p.get("month") is not None and not (1 <= p["month"] <= 12):
        errors.append(get_msg(lang, "invalid_month"))

    if errors:
        print(f"   → Errors: {errors}")
        return {"validation_errors": errors, "next_step": "format"}

    print("   → Valid. → Predictor")
    return {"current_params": p, "next_step": "predict"}


def predictor_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 5: Route to the correct predictor based on vehicle_type.

    Routing:
      vehicle_type = "fhvhv"              → FHVHV coming-soon stub
      vehicle_type = "all" | None         → YGPredictor.predict_all (3 results)
      vehicle_type = "yellow"             → YGPredictor.predict yellow-hail (1 result)
      vehicle_type = "green"              → YGPredictor.predict green-hail + green-dispatch (2 results)
    """
    print(f"--- [5. PREDICTOR] ---")
    p          = state["current_params"]
    intent     = state["intent"]
    lang       = state.get("language", "it")
    results    = []

    try:
        if intent == "trend":
            trend_json = get_historical_trends.invoke({
                "location_id": p["location_id"],
                "day_of_week": p.get("day_of_week"),
            })
            results = [json.loads(trend_json)]
            return {"results": results, "next_step": "format"}

        now       = datetime.now(ZoneInfo("America/New_York"))
        eval_hour   = p.get("hour")   if p.get("hour")   is not None else now.hour
        eval_minute = p.get("minute") if p.get("minute") is not None else 0
        eval_dow    = p.get("day_of_week") if p.get("day_of_week") is not None else now.weekday()
        eval_month  = p.get("month") if p.get("month") is not None else now.month
        location_id = p["location_id"]

        vehicle_type = p.get("vehicle_type", "all") or "all"
        hour_range   = state.get("hour_range", [])

        if vehicle_type == "fhvhv":
            results = [{
                "model_type":   "fhvhv",
                "coming_soon":  True,
                "message":      get_msg(lang, "fhvhv_coming_soon"),
                "location_id":  location_id,
            }]
            return {"results": results, "next_step": "format"}

        yg = get_yg_predictor()

        if hour_range:
            for h in hour_range:
                results.append(yg.predict(location_id, h, 0, eval_dow, eval_month, "yellow", "hail"))
        elif vehicle_type == "all" or vehicle_type is None:
            results = yg.predict_all(location_id, eval_hour, eval_minute, eval_dow, eval_month)
        elif vehicle_type == "yellow":
            results = [yg.predict(location_id, eval_hour, eval_minute, eval_dow, eval_month, "yellow", "hail")]
        elif vehicle_type == "green":
            results = [
                yg.predict(location_id, eval_hour, eval_minute, eval_dow, eval_month, "green", "hail"),
                yg.predict(location_id, eval_hour, eval_minute, eval_dow, eval_month, "green", "dispatch"),
            ]
        else:
            results = yg.predict_all(location_id, eval_hour, eval_minute, eval_dow, eval_month)

        return {"results": results, "next_step": "format"}

    except Exception as e:
        logger.error(f"[Predictor] {e}", exc_info=True)
        return {"validation_errors": [str(e)], "next_step": "format"}


def formatter_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 6: Hybrid formatter.
    Deterministic template carries all data; LLM adds a concise 2-3 sentence insight.
    """
    print(f"--- [6. FORMATTER] ---")
    intent = state.get("intent", "predict")
    lang   = state.get("language", "it")

    # ── Out of scope — risposta conversazionale tramite LLM ─────────────────
    if intent == "oos":
        try:
            llm = get_llm(temperature=0.3)
            # Build LLM message list: system + recent history + current user message
            oos_msgs = [SystemMessage(content=_OOS_PROMPT)]
            for m in state["messages"][:-1][-6:]:   # max 3 turns di context
                oos_msgs.append(m)
            oos_msgs.append(state["messages"][-1])   # messaggio corrente

            resp = llm.invoke(oos_msgs)
            return {"messages": [AIMessage(content=resp.content.strip())]}
        except Exception as e:
            logger.warning(f"[Formatter] OOS LLM failed ({e}) — using fallback")
            msg = get_msg(lang, "oos_fallback")
            return {"messages": [AIMessage(content=msg)]}

    # ── Validation errors ─────────────────────────────────────────────────────
    if state.get("validation_errors"):
        err = ", ".join(state["validation_errors"])
        msg = get_msg(lang, "param_error", err)
        return {"messages": [AIMessage(content=msg)]}

    # ── Missing zone ──────────────────────────────────────────────────────────
    if state.get("next_step") == "ask_zone":
        msg = get_msg(lang, "ask_zone")
        return {"messages": [AIMessage(content=msg)]}

    # ── Disambiguation ────────────────────────────────────────────────────────
    if state.get("next_step") == "disambiguate":
        msg = get_msg(lang, "disambiguate")
        return {"messages": [AIMessage(content=msg)]}

    if not state.get("results"):
        return {"messages": [AIMessage(content=get_msg(lang, "no_data"))]}

    # ── Deterministic template ────────────────────────────────────────────────
    template = _build_template(state["results"], state.get("current_params", {}))

    # ── LLM insight (2-3 sentences only) ─────────────────────────────────────
    try:
        user_msg = state["messages"][-1].content
        llm = get_llm(temperature=0.0)
        insight_input = (f'User asked: "{user_msg}"\n\n'
                         f"Data: {json.dumps(state['results'], ensure_ascii=False)}")
        resp = llm.invoke([SystemMessage(content=_INSIGHT_PROMPT),
                           HumanMessage(content=insight_input)])
        final = f"{template}\n\n💡 *Insight:* {resp.content.strip()}"
    except Exception as e:
        logger.warning(f"[Formatter] LLM insight failed: {e}")
        final = template   # graceful fallback: template-only response

    return {"messages": [AIMessage(content=final)]}


# ─── Graph Construction ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    wf = StateGraph(AgentState)
    wf.add_node("intent",     intent_classifier_node)
    wf.add_node("extractor",  extractor_node)
    wf.add_node("guardrail",  guardrail_node)
    wf.add_node("predictor",  predictor_node)
    wf.add_node("formatter",  formatter_node)

    wf.set_entry_point("intent")
    wf.add_edge("intent",    "extractor")
    wf.add_edge("extractor", "guardrail")
    wf.add_conditional_edges("guardrail", lambda s: s["next_step"], {
        "predict":      "predictor",
        "format":       "formatter",
        "ask_zone":     "formatter",
        "disambiguate": "formatter",
    })
    wf.add_edge("predictor", "formatter")
    wf.add_edge("formatter", END)
    return wf.compile()


# ─── Agent class ─────────────────────────────────────────────────────────────

class TaxiAgent:
    def __init__(self):
        self._graph = build_graph()

    def chat(
        self,
        user_message: str,
        chat_history: Optional[List] = None,
        current_params: Optional[Dict] = None,
        lang: str = "it",
    ) -> Dict[str, Any]:
        """Run the graph for a single user turn. Returns text + metadata."""
        input_state = {
            "messages":          (chat_history or []) + [HumanMessage(content=user_message)],
            "current_params":    current_params or {},
            "results":           [],
            "validation_errors": [],
            "candidates":        [],
            "hour_range":        [],
            "intent":            "",
            "next_step":         "",
            "language":          lang,
        }
        try:
            final = self._graph.invoke(input_state)
            last  = final["messages"][-1]
            return {
                "text":       last.content if hasattr(last, "content") else str(last),
                "candidates": final.get("candidates", []),
                "params":     final.get("current_params", {}),
            }
        except Exception as e:
            logger.error(f"[Agent] Graph error: {e}", exc_info=True)
            return {"text": get_msg(lang, "internal_error", str(e)), "candidates": [], "params": {}}

    def direct_predict(self, **kwargs) -> str:
        """Programmatic prediction — bypasses the LLM graph."""
        res = get_predictor().predict(**kwargs)
        return (f"Previsione per {res['location_name']}: "
                f"{res['predicted_class_name']} ({res['confidence']*100:.1f}%)")


# ─── Singleton ────────────────────────────────────────────────────────────────

_agent: Optional[TaxiAgent] = None


def get_agent() -> TaxiAgent:
    global _agent
    if _agent is None:
        _agent = TaxiAgent()
    return _agent
