"""
Internationalization (i18n) support for NYC Taxi Bot.
"""

MESSAGES = {
    "it": {
        "welcome": "🚕 *NYC Taxi Availability Bot (v2.2)*\n\nCiao! Sono pronto per la presentazione.\n\n_Cosa posso fare per te:_\n• Predizioni in tempo reale (anche per fasce orarie)\n• Analisi dei trend storici\n• Spiegazione delle cause (AI Spiegabile)\n\nProva a chiedermi: _'Com'è la situazione a JFK?'_",
        "cleaning": "🧹 Contesto ripulito. Come posso aiutarti?",
        "cancel_btn": "❌ Annulla",
        "cancel_msg": "Operazione annullata. Come posso aiutarti?",
        "general_error": "⚠️ Si è verificato un errore. Riprova.",
        "ask_zone": "📍 Quale zona di New York ti interessa?\nEsempi: _JFK, Times Square, Brooklyn, Midtown_",
        "no_data": "⚠️ Nessun dato disponibile. Riprova.",
        "disambiguate": "🔍 Ho trovato più zone simili. Quale intendi?",
        "oos_fallback": "Posso aiutarti solo con i taxi di New York! 🚕\nProva: _'Quanti taxi ci sono a JFK lunedì alle 8?'_",
        "param_error": "⚠️ Problema con i parametri: *{0}*\n\nRiprova con un formato tipo: _'JFK lunedì alle 8 di marzo'_",
        "rate_limit": "⏳ Per favore, attendi un momento. Stai inviando troppi messaggi!",
        "invalid_id": "ID zona non valido (1-265)",
        "invalid_hour": "Ora non valida (0-23)",
        "invalid_month": "Mese non valido (1-12)",
        "internal_error": "⚠️ Si è verificato un errore interno: {0}",
    },
    "en": {
        "welcome": "🚕 *NYC Taxi Availability Bot (v2.2)*\n\nHi! I am ready for the presentation.\n\n_What I can do for you:_\n• Real-time predictions (also for time slots)\n• Historical trends analysis\n• Explanation of causes (Explainable AI)\n\nTry asking: _'How is the situation at JFK?'_",
        "cleaning": "🧹 Context cleared. How can I help you?",
        "cancel_btn": "❌ Cancel",
        "cancel_msg": "Operation cancelled. How can I help you?",
        "general_error": "⚠️ An error occurred. Please try again.",
        "ask_zone": "📍 Which NYC zone are you interested in?\nExamples: _JFK, Times Square, Brooklyn, Midtown_",
        "no_data": "⚠️ No data available. Please try again.",
        "disambiguate": "🔍 I found multiple similar zones. Which one do you mean?",
        "oos_fallback": "I can only help you with New York taxis! 🚕\nTry: _'How many taxis in JFK on Monday at 8am?'_",
        "param_error": "⚠️ Problem with parameters: *{0}*\n\nTry a format like: _'JFK Monday at 8 of March'_",
        "rate_limit": "⏳ Please wait a moment. You are sending messages too fast!",
        "invalid_id": "Invalid Zone ID (1-265)",
        "invalid_hour": "Invalid hour (0-23)",
        "invalid_month": "Invalid month (1-12)",
        "internal_error": "⚠️ An internal error occurred: {0}",
    }
}

def get_msg(lang_code: str, key: str, *args) -> str:
    """Retrieve string based on language code, defaulting to Italian."""
    lang = lang_code[:2].lower() if lang_code else "it" # default to IT since presentation is IT
    if lang not in MESSAGES:
        lang = "it"
    msg = MESSAGES[lang].get(key, MESSAGES["en"].get(key, key))
    if args:
        return msg.format(*args)
    return msg
