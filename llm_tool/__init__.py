"""
LLM Tool Package - NYC Taxi Demand Prediction.

Provides LangChain tools, agents, and validators for predicting taxi
availability in New York City using a trained LightGBM model with
natural language understanding via Gemma-4-E2B (Ollama).
"""

__version__ = "1.1.0"
__author__ = "AI Assistant"

from .taxi_predictor import get_predictor, predict_taxi_availability, resolve_zone_id
from .agent import get_agent, TaxiAgent
from .input_validator import get_validator, InputValidator
from .config import (
    CLASS_NAMES,
    CLASS_EMOJIS,
    DAY_NAMES_IT,
    MONTH_NAMES_IT,
    ZONE_ALIASES,
)

__all__ = [
    # Predictor
    "get_predictor",
    "predict_taxi_availability",
    "resolve_zone_id",
    # Agent
    "get_agent",
    "TaxiAgent",
    # Validator
    "get_validator",
    "InputValidator",
    # Config helpers
    "CLASS_NAMES",
    "CLASS_EMOJIS",
    "DAY_NAMES_IT",
    "MONTH_NAMES_IT",
    "ZONE_ALIASES",
]
