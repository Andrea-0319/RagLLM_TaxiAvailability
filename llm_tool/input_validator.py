"""
Input Validator for NYC Taxi Demand Prediction (v4).

Uses LLM-based parameter extraction for natural language.
Regex fast-path is reserved for internal button commands (e.g. "Zona ID 161").
Post-LLM sanitization coerces types and validates ranges before downstream use.
"""

import json
import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_factory import get_llm
from .taxi_predictor import resolve_zone_id
from .prompts import _EXTRACTION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class InputValidator:
    def __init__(self):
        # Lazy init: istanziato al primo utilizzo per evitare import circolari
        self._llm = None  # BaseChatModel

    def _get_llm(self):
        """Lazy init del LLM tramite factory (Groq o Ollama in base a config)."""
        if self._llm is None:
            self._llm = get_llm(temperature=0.0)
        return self._llm

    # ─── Public API ──────────────────────────────────────────────────────────

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract raw params. Regex fast-path per button commands; LLM per NL."""
        # Fast-path: formato interno bottoni "Zona ID 161"
        id_match = re.search(r'\b(?:zona|zone|id)\s*(?:id)?\s*(\d+)\b', text.lower())
        if id_match:
            print(f"   [Validator] Fast-path: ID {id_match.group(1)}")
            return {"zone": id_match.group(1), "month": None, "day_of_week": None,
                    "hour": None, "minute": None, "vehicle_type": "all"}

        print(f"   [Validator] LLM extraction...")
        raw = self._llm_extract(text)
        sanitized = self._sanitize_extracted(raw)
        return sanitized

    def validate_and_resolve(self, params: Dict[str, Any], text: str = "") -> Dict[str, Any]:
        """Resolve zone name/alias → location_id or candidate list."""
        result = dict(params)
        result["candidates"] = []

        if params.get("zone"):
            resolution = resolve_zone_id(str(params["zone"]), return_all=True)
            if isinstance(resolution, int):
                result["location_id"] = resolution
            elif isinstance(resolution, list):
                result["location_id"] = None
                result["candidates"] = resolution[:5]
            else:
                result["location_id"] = None
        else:
            result["location_id"] = None

        return result

    # ─── Private helpers ─────────────────────────────────────────────────────

    def _llm_extract(self, text: str) -> Dict[str, Any]:
        """Invoke LLM for semantic extraction. Returns raw dict (pre-sanitize)."""
        now = datetime.now(ZoneInfo("America/New_York"))
        system = _EXTRACTION_SYSTEM_PROMPT.format(
            today=now.strftime("%A, %B %d, %Y"),
            month=now.month,
            dow=now.weekday(),
            time=now.strftime("%H:%M"),
        )
        try:
            resp = self._get_llm().invoke(
                [SystemMessage(content=system), HumanMessage(content=text)]
            )
            raw = resp.content.strip()
            # Strip markdown code fences if present
            raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"[Validator] LLM extraction failed: {e}")
            return {"zone": None, "month": None, "day_of_week": None, "hour": None, "minute": None}

    def _sanitize_extracted(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-LLM schema enforcement:
        - Tiene solo le chiavi attese.
        - Converte nomi di giorno/mese stringa in int.
        - Clampa i valori nei range validi; None in caso di errore.
        """
        sanitized: Dict[str, Any] = {
            "zone": None, "month": None, "day_of_week": None,
            "hour": None, "minute": None, "vehicle_type": "all",
        }

        # zone — mantieni come stringa, scarta valori null-like
        z = raw.get("zone")
        if z and str(z).strip().lower() not in ("null", "none", ""):
            sanitized["zone"] = str(z).strip()

        # month — int 1-12, correzione matematica (es. 13 -> 1)
        v = raw.get("month")
        if v is not None:
            try:
                m = int(v)
                sanitized["month"] = (m - 1) % 12 + 1 if m > 0 else None
            except (ValueError, TypeError):
                pass

        # day_of_week — int 0-6, correzione matematica (es. calcolo domenica+1=7 -> 0)
        v = raw.get("day_of_week")
        if v is not None:
            try:
                d = int(v)
                sanitized["day_of_week"] = d % 7
            except (ValueError, TypeError):
                pass

        # hour — int 0-23
        v = raw.get("hour")
        if v is not None:
            try:
                h = int(v)
                sanitized["hour"] = h if 0 <= h <= 23 else None
            except (ValueError, TypeError):
                pass

        # minute — int 0-59
        v = raw.get("minute")
        if v is not None:
            try:
                m = int(v)
                sanitized["minute"] = m if 0 <= m <= 59 else None
            except (ValueError, TypeError):
                pass

        # vehicle_type — accept known values, default "all"
        vt = raw.get("vehicle_type")
        if vt and str(vt).strip().lower() in ("yellow", "green", "fhvhv", "all"):
            sanitized["vehicle_type"] = str(vt).strip().lower()
        else:
            sanitized["vehicle_type"] = "all"

        return sanitized


# ─── Singleton ────────────────────────────────────────────────────────────────

_validator: Optional[InputValidator] = None


def get_validator() -> InputValidator:
    global _validator
    if _validator is None:
        _validator = InputValidator()
    return _validator
