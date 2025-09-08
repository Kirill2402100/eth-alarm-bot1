# fa_bot.py
from __future__ import annotations
import os, logging, asyncio
from dataclasses import dataclass
from typing import Dict

from telegram import Update, BotCommand
from telegram.ext import (
    Application, CommandHandler, ContextTypes, MessageHandler, filters
)

# --- rate limiter (опционально, если установлен extra) ---
try:
    from telegram.ext._aioratelimiter import AIORateLimiter  # ptb>=20.8
except Exception:  # без extras тоже ок
    AIORateLimiter = None

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("fund_bot")

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
MASTER_CHAT_ID = int(os.getenv("MASTER_CHAT_ID", "0"))

# LLM переменные просто читаем (используются /digest и т.п.)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_NANO  = os.getenv("LLM_NANO",  "gpt-5-nano")
LLM_MINI  = os.getenv("LLM_MINI",  "gpt-5-mini")
LLM_MAJOR = os.getenv("LLM_MAJOR", "gpt-5")

# Храним в памяти текущую цель и веса (могут жить в Sheets — позже подключим)
@dataclass
class AllocState:
    total: float = 0.0
    weights: Dict[str, float] = None  # keys: JPY/AUD/EUR/GBP

state = AllocState(
    total=0.0,
    weights={"JPY": 40.0, "AUD": 25.0, "EUR": 20.0, "GBP": 15.0}
)

PAIR_BY_CCY = {
    "JPY": "USDJPY",
    "AUD": "AUDUSD",
    "EUR": "EURUSD",
    "GBP": "GBPUSD",
}

def require_master(update: Update) -> bool:
    if MASTER_CHAT_ID == 0:
        return True
    chat_id = update.effective_chat.id if update.effective_chat else None
    return chat_id == MASTER_CHAT_ID

def calc_alloc(total: float, weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(weights.values()) or 1.0
    return {PAIR_BY_CCY[k]: round(total * (v / s), 2) for k, v in weights.items() if k in PAIR_BY_CCY}

async def set_commands(app: Application):
    commands = [
        BotCommand("start", "Запуск и краткая справка"),
        BotCommand("help", "Справка"),
        BotCommand("chatid", "Показать ID текущего чата"),
        BotCommand("ping", "Проверка ответа"),
        BotCommand("setweights", "Задать веса: /setweights jpy=40 aud=25 eur=20 gbp=15"),
        BotCommand("settotal", "Задать общий банк: /settotal 2800"),
        BotCommand("alloc", "Показать целевые/фактические и команды для чатов"),
        BotCommand("digest", "Короткий человеческий дайджест"),
    ]
    await app.bot.set_my_commands(commands)
    log.info("Telegram commands set.")

# ---------------- Handlers ----------------

async def on_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    log.info("START from chat %s", update.effective_chat.id if update.effective_chat else None)
    await update.effective_message.reply_text(
        "Привет! Я фунд-бот.\n\n"
        "Основные команды:\n"
        "• /setweights jpy=40 aud=25 eur=20 gbp=15 — целевые веса\n"
        "• /settotal 2800 — общий банк\n"
        "• /alloc — расчёт сумм по парам и готовые команды для торговых чатов\n"
        "• /digest — утренний дайджест (людским языком)\n"
        "• /chatid — показать id этого чата\n"
        "• /help — справка\n\n"
        f"MASTER_CHAT_ID сейчас: {MASTER_CHAT_ID or 'не задан'}"
    )

async def on_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Справка:\n"
        "1) В мастер-чате задаёшь веса и общий банк:\n"
        "   /setweights jpy=40 aud=25 eur=20 gbp=15\n"
        "   /settotal 2800\n"
        "2) Командой /alloc получаешь суммы по парам и кусочки для копипаста в торговые чаты.\n"
        "3) /digest — сводка и рекомендации по ограничениям.\n"
        "Базовые команды доступны всем, управленческие — только MASTER_CHAT_ID."
    )

async def on_chatid(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if not chat:
        return
    await update.effective_message.reply_text(f"chat_id: <code>{chat.id}</code>", parse_mode="HTML")

async def on_ping(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("pong ✅")

def parse_weights(text: str) -> Dict[str, float] | None:
    # принимает строку после команды: "jpy=40 aud=25 ..."
    parts = text.strip().split()
    out = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip().upper()
        try:
            val = float(v.strip().replace(",", "."))
        except ValueError:
            return None
        if k in {"JPY", "AUD", "EUR", "GBP"}:
            out[k] = val
    return out or None

async def on_setweights(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not require_master(update):
        return await update.effective_message.reply_text("⛔ Эта команда доступна только в мастер-чате.")
    args = (ctx.args or [])
    new = parse_weights(" ".join(args))
    if not new:
        return await update.effective_message.reply_text(
            "Формат: /setweights jpy=40 aud=25 eur=20 gbp=15"
        )
    state.weights.update(new)
    await update.effective_message.reply_text(
        "Целевые веса обновлены: " + " / ".join(f"{k} {int(v)}" for k, v in state.weights.items())
    )

async def on_settotal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not require_master(update):
        return await update.effective_message.reply_text("⛔ Эта команда доступна только в мастер-чате.")
    if not ctx.args:
        return await update.effective_message.reply_text("Формат: /settotal 2800")
    try:
        total = float(ctx.args[0].replace(",", "."))
    except ValueError:
        return await update.effective_message.reply_text("Нужно число. Пример: /settotal 2800")
    state.total = max(0.0, total)
    await update.effective_message.reply_text(f"Общий банк установлен: {state.total:.2f} USDT")

async def on_alloc(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not require_master(update):
        return await update.effective_message.reply_text("⛔ Эта команда доступна только в мастер-чате.")
    if state.total <= 0:
        return await update.effective_message.reply_text("Сначала задай банк: /settotal 2800")
    alloc = calc_alloc(state.total, state.weights)
    lines = []
    copy = []
    for pair, amount in alloc.items():
        lines.append(f"{pair} → <b>{amount:.2f} USDT</b>")
        copy.append(f"{pair}: /setbank {amount:.0f}")
    text = "Распределение по целевым весам:\n" + "\n".join(lines)
    text += "\n\nКоманды для торговых чатов:\n" + "\n".join(copy)
    await update.effective_message.reply_text(text, parse_mode="HTML")

# Здесь можно подключить ваш llm_client; пока делаем заглушку
async def on_digest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not require_master(update):
        return await update.effective_message.reply_text("⛔ Эта команда доступна только в мастер-чате.")
    # Простейший шаблон (LLM можно подключить позже)
    msg = (
        "🧭 Утренний фон\n\n"
        "USDJPY — 🚧 Осторожно, риск интервенций\n"
        "• Окна тишины ±45 мин к «красным» релизам. Смещение — short-bias.\n\n"
        "AUDUSD — ✅ Нейтрально-позитивно\n"
        "• Обычный режим, усреднения по плану.\n\n"
        "EURUSD — ✅ Нейтрально\n"
        "• Волатильность близка к средней.\n\n"
        "GBPUSD — ⚠️ Осторожно\n"
        "• Вола растёт; снизить доборы на 25%."
    )
    await update.effective_message.reply_text(msg)

async def on_unknown(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Не знаю такую команду. Попробуй /help")

async def main():
    log.info("Fund bot is running…")
    builder = Application.builder().token(TELEGRAM_TOKEN)
    if AIORateLimiter:
        builder = builder.rate_limiter(AIORateLimiter())
    app = builder.build()

    # Регистрируем команды при старте
    app.post_init = set_commands

    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("help", on_help))
    app.add_handler(CommandHandler("chatid", on_chatid))
    app.add_handler(CommandHandler("ping", on_ping))

    app.add_handler(CommandHandler("setweights", on_setweights))
    app.add_handler(CommandHandler("settotal", on_settotal))
    app.add_handler(CommandHandler("alloc", on_alloc))
    app.add_handler(CommandHandler("digest", on_digest))

    # лог всех сообщений (для дебага)
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, lambda u, c: log.info("Update: %s", u)))

    # неизвестные команды
    app.add_handler(MessageHandler(filters.COMMAND, on_unknown))

    # гарантируем long polling
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        pass
    await app.run_polling(close_loop=False)

if __name__ == "__main__":
    asyncio.run(main())
