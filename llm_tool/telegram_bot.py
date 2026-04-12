"""
Telegram Bot for NYC Taxi Demand Prediction.

Updated to support LangGraph StateGraph with:
- Inline buttons for zone disambiguation.
- Multi-turn context.
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any

from cachetools import TTLCache
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from langchain_core.messages import HumanMessage, AIMessage

from .agent import get_agent
from .taxi_predictor import get_predictor
from .config import PROJECT_ROOT
from .i18n import get_msg

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─── Per-user chat history (in-memory) ───────────────────────────────────────
_chat_histories = TTLCache(maxsize=1000, ttl=3600 * 1)
_user_params = TTLCache(maxsize=1000, ttl=3600 * 1) # Persistent params per session
_rate_limits = TTLCache(maxsize=1000, ttl=60) # Rate limit: 1 minuto
_active_tasks: Dict[int, asyncio.Task] = {} # Pending async tasks per user
MAX_HISTORY = 10

def _cancel_active_task(user_id: int) -> None:
    task = _active_tasks.get(user_id)
    if task and not task.done():
        task.cancel()

async def check_rate_limit(update: Update, user_id: int, lang: str) -> bool:
    """Ritorna True se l'utente supera il limite (spam). Max 10 messaggi al minuto."""
    count = _rate_limits.get(user_id, 0)
    if count >= 10:
        if count == 10: # Avvisa solo alla decima chiamata
            await _safe_reply(update, get_msg(lang, "rate_limit"))
        _rate_limits[user_id] = count + 1
        return True
    _rate_limits[user_id] = count + 1
    return False


def _get_history(user_id: int) -> List:
    return _chat_histories.setdefault(user_id, [])


def _update_history(user_id: int, user_text: str, bot_text: str) -> None:
    history = _get_history(user_id)
    history.append(HumanMessage(content=user_text))
    history.append(AIMessage(content=bot_text))
    if len(history) > MAX_HISTORY * 2:
        _chat_histories[user_id] = history[-(MAX_HISTORY * 2):]


async def _safe_reply(update: Update, text: str, reply_markup=None) -> None:
    """Send reply with Markdown; fallback to plain text on parse error."""
    try:
        if update.message:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
        elif update.callback_query:
            await update.callback_query.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    except Exception as e:
        logger.warning(f"Markdown failed, falling back to plain: {e}")
        plain = text.replace("*", "").replace("_", "").replace("`", "")
        if update.message:
            await update.message.reply_text(plain, reply_markup=reply_markup)
        elif update.callback_query:
            await update.callback_query.message.reply_text(plain, reply_markup=reply_markup)


# ─── Command Handlers ─────────────────────────────────────────────────────────

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    lang = update.effective_user.language_code
    
    if await check_rate_limit(update, user_id, lang): return
    
    _cancel_active_task(user_id)
    # Clear context on start
    _chat_histories.pop(user_id, None)
    _user_params.pop(user_id, None)
    
    welcome = get_msg(lang, "welcome")
    await _safe_reply(update, welcome)


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear user session context."""
    user_id = update.effective_user.id
    lang = update.effective_user.language_code
    
    if await check_rate_limit(update, user_id, lang): return
    
    _cancel_active_task(user_id)
    _chat_histories.pop(user_id, None)
    _user_params.pop(user_id, None)
    await update.message.reply_text(get_msg(lang, "cleaning"))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show available commands and bot capabilities."""
    user_id = update.effective_user.id
    lang = update.effective_user.language_code
    
    if await check_rate_limit(update, user_id, lang): return
    
    help_text = (
        "🚕 *NYC Taxi Predictor Bot*\n\n"
        "Posso aiutarti a prevedere la disponibilità dei taxi a New York e a scoprire i trend storici!\n\n"
        "🗣️ *Cosa puoi chiedermi:*\n"
        "• _\"Com'è la situazione a JFK venerdì alle 18?\"_\n"
        "• _\"Dimmi i trend di Midtown per domani mattina.\"_\n"
        "• _\"Si trovano taxi a Times Square stasera?\"_\n\n"
        "⚙️ *Comandi disponibili:*\n"
        "• /start - Avvia o riavvia il bot\n"
        "• /reset - Pulisce la memoria della conversazione attuale se mi confondo\n"
        "• /help - Mostra questo messaggio"
    )
    await _safe_reply(update, help_text)


async def _run_graph_and_reply(update: Update, user_id: int, user_message: str, lang: str, current_params: dict, action_text: str) -> None:
    """Core logic to run the LLM graph and send the response. Handled as a Cancellable Task."""
    try:
        await update.get_bot().send_chat_action(chat_id=update.effective_chat.id, action="typing")
        agent = get_agent()
        history = _get_history(user_id)
        
        result = await asyncio.to_thread(
            agent.chat, user_message=user_message, chat_history=history, current_params=current_params, lang=lang
        )
        
        response_text = result["text"]
        candidates = result.get("candidates", [])
        
        _update_history(user_id, action_text, response_text)
        _user_params[user_id] = result.get("params", {})

        if candidates:
            keyboard = [
                [InlineKeyboardButton(c["name"], callback_data=f"zone_{c['id']}")]
                for c in candidates
            ]
            keyboard.append([InlineKeyboardButton(get_msg(lang, "cancel_btn"), callback_data="cancel_search")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            await _safe_reply(update, response_text, reply_markup=reply_markup)
        else:
            await _safe_reply(update, response_text)

    except asyncio.CancelledError:
        logger.info(f"[User {user_id}] Task cancelled by newer message.")
    except Exception as e:
        logger.error(f"[User {user_id}] Error: {e}", exc_info=True)
        await _safe_reply(update, get_msg(lang, "general_error"))
    finally:
        if _active_tasks.get(user_id) == asyncio.current_task():
            _active_tasks.pop(user_id, None)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    user_id = update.effective_user.id
    lang = update.effective_user.language_code

    if not user_message:
        return
        
    if await check_rate_limit(update, user_id, lang): return

    logger.info(f"[User {user_id}] → {user_message[:80]!r}")
    _cancel_active_task(user_id)
    
    current_params = _user_params.get(user_id, {})
    task = asyncio.create_task(_run_graph_and_reply(update, user_id, user_message, lang, current_params, user_message))
    _active_tasks[user_id] = task


async def on_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button clicks for zone disambiguation."""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    user_id = update.effective_user.id
    lang = update.effective_user.language_code
    
    if await check_rate_limit(update, user_id, lang): return
    
    if data.startswith("zone_"):
        zone_id = data.split("_")[1]
        
        _cancel_active_task(user_id)
        
        current_params = _user_params.get(user_id, {}).copy()
        current_params["location_id"] = int(zone_id)

        task = asyncio.create_task(_run_graph_and_reply(
            update, user_id, f"Zona ID {zone_id}", lang, current_params, f"(Scelta zona: {zone_id})"
        ))
        _active_tasks[user_id] = task
        
    elif data == "cancel_search":
        _cancel_active_task(user_id)
        _user_params.pop(user_id, None) # Clear current intent
        await query.edit_message_text(get_msg(lang, "cancel_msg"))


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"[TelegramError] {context.error}", exc_info=context.error)


# ─── Application Factory ──────────────────────────────────────────────────────

def create_application() -> Application:
    import os
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_TOKEN mancante nel file .env")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(on_callback_query))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    return app


def run_bot() -> None:
    print("🚀 Avvio NYC Taxi Bot con StateGraph...")
    try:
        predictor = get_predictor()
        predictor.load()
        print("✅ Modello caricato.")
    except Exception as e:
        print(f"❌ Errore: {e}")
        raise

    app = create_application()
    app.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    run_bot()
