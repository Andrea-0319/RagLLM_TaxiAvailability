"""
LLM Factory — Centralizzazione della scelta del Language Model.

Espone get_llm() che ritorna il modello giusto in base a LLM_PROVIDER (.env):
  - "groq+ollama": tenta ChatGroq; on error logga e usa ChatOllama come fallback
  - "ollama":      usa solo ChatOllama locale, nessuna API key necessaria
"""

import logging
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from .config import (
    LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL,
    LLM_MODEL, LLM_BASE_URL, LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)


def _create_groq(temperature: float) -> BaseChatModel:
    """Istanzia ChatGroq con import lazy (no crash se langchain-groq non installato)."""
    from langchain_groq import ChatGroq  # pip install langchain-groq
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
    )


def _create_ollama(temperature: float) -> BaseChatModel:
    """Istanzia ChatOllama locale."""
    return ChatOllama(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        temperature=temperature,
    )


def get_llm(temperature: float = LLM_TEMPERATURE) -> BaseChatModel:
    """
    Ritorna il ChatModel configurato in base a LLM_PROVIDER.

    Priorità:
      1. Groq (se LLM_PROVIDER == "groq+ollama" e GROQ_API_KEY presente)
      2. Ollama locale (fallback automatico su qualsiasi errore)

    Args:
        temperature: campionamento LLM (0.0 = deterministico, 0.1 = default)

    Returns:
        BaseChatModel pronto per .invoke()
    """
    if LLM_PROVIDER == "groq+ollama" and GROQ_API_KEY:
        try:
            llm = _create_groq(temperature)
            print(f"🧠 [LLM Factory] Usando provider Groq Cloud: {GROQ_MODEL} (temp={temperature:.1f})")
            return llm
        except Exception as e:
            logger.warning(
                "[LLM Factory] Groq non disponibile (%s) → fallback Ollama", e
            )

    print(f"🦙 [LLM Factory] Usando provider locale Ollama: {LLM_MODEL} (temp={temperature:.1f})")
    return _create_ollama(temperature)
