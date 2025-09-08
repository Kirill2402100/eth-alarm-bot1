# fa_bot.py
# ----------
# Фунд-бот (мастер + торговые чаты).
# Команды:
#   /start, /help
#   /settotal <USDT>        — задаёт общий банк (только в мастер-чате)
#   /setweights jpy=40 aud=25 eur=20 gbp=15
#   /weights                — показать текущие целевые веса
#   /alloc                  — расчёт сумм и готовые /setbank для чатов
#   /digest                 — «человеческий» дайджест (по всем парам в мастере, по одной — в профильных чатах)
#   /ping                   — проверка ответа
#   /check_sheets           — диагностика подключения Google Sheets (создаёт лист FA_Signals при необходимости)

import asyncio
import base64
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import gspread
from telegram import Update, BotCommand
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# Rate limiter (если установлен)
try:
    from telegram.ext import AIORateLimiter
except Exception:  # библиотека без экстры
    AIORateLimiter = None  # type: ignore

import llm_client

# -----------------------
# Настройки и дефолты
# -----------------------

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
MASTER_CHAT_ID = int(os.environ.get("MASTER_CHAT_ID", "0") or "0")

# Сопоставление чатов и инструментов: {"USDJPY": <chat_id>, ...}
try:
    SYMBOL_CHAT_MAP: Dict[str, int] = json.loads(os.environ.get("SYMBOL_CHAT_MAP", "{}"))
    SYMBOL_CHAT_MAP = {k.upper(): int(v) for k, v in SYMBOL_CHAT_MAP.items()}
except Exception:
    SYMBOL_CHAT_MAP = {}

# Дефолтные веса (в процентах)
try:
    DEFAULT_WEIGHTS = json.loads(os.environ.get("DEFAULT_WEIGHTS", '{"JPY":40,"AUD":25,"EUR":20,"GBP":15}'))
except Exception:
    DEFAULT_WEIGHTS = {"JPY": 40, "AUD": 25, "EUR": 20, "GBP": 15}

ALL_PAIRS = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]

# Текущее состояние (в памяти процесса)
STATE = {
    "total_bank": 0.0,
    "weights": DEFAULT_WEIGHTS.copy(),  # JPY/AUD/EUR/GBP
}

# -----------------------
# Логирование
# -----------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("fund_bot")


# -----------------------
# Утилиты / Sheets
# -----------------------

def _read_creds_any():
    raw = os.environ.get("GOOGLE_CREDENTIALS") or os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not raw:
        b64 = os.environ.get("GOOGLE_CREDENTIALS_B64")
        if b64:
            try:
                raw = base64.b64decode(b64).decode("utf-8")
            except Exception:
                pass
    if not raw:
        return None, "env GOOGLE_CREDENTIALS(_JSON/_B64) not set"
    try:
        data = json.loads(raw)
        return data, None
    except Exception as e:
        return None, f"creds JSON parse error: {e}"


def _open_sheet():
    sid = os.environ.get("SHEET_ID")
    if not sid:
        raise RuntimeError("SHEET_ID not set")
    data, err = _read_creds_any()
    if err:
        raise RuntimeError(err)
    try:
        gc = gspread.service_account_from_dict(data)
        sh = gc.open_by_key(sid)
        return sh, data.get("client_email")
    except Exception as e:
        raise RuntimeError(f"gspread auth/open error: {e}")


async def ensure_fa_sheet() -> bool:
    """Создаёт лист FA_Signals при отсутствии и заголовки."""
    sh, _ = _open_sheet()
    try:
        ws = sh.worksheet("FA_Signals")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="FA_Signals", rows=100, cols=10)
        ws.append_row(["pair", "risk", "bias", "ttl", "updated_at", "scan_lock_until", "reserve_off", "dca_scale"])
    return True


# -----------------------
# Вспомогательные функции
# -----------------------

def pairs_for_chat(chat_id: int) -> List[str]:
    """Мастер-чат → все пары; профильный чат → своя пара; иначе USDJPY."""
    if MASTER_CHAT_ID and chat_id == MASTER_CHAT_ID:
        return ALL_PAIRS[:]
    # поиск по маппингу
    for sym, cid in SYMBOL_CHAT_MAP.items():
        if int(cid) == int(chat_id):
            return [sym.upper()]
    return ["USDJPY"]


def fmt_weights(w: Dict[str, float]) -> str:
    return f"JPY {w.get('JPY',0)} / AUD {w.get('AUD',0)} / EUR {w.get('EUR',0)} / GBP {w.get('GBP',0)}"


def compute_allocation(total: float, weights: Dict[str, float]) -> Dict[str, float]:
    parts = {
        "USDJPY": total * (weights.get("JPY", 0) / 100.0),
        "AUDUSD": total * (weights.get("AUD", 0) / 100.0),
        "EURUSD": total * (weights.get("EUR", 0) / 100.0),
        "GBPUSD": total * (weights.get("GBP", 0) / 100.0),
    }
    # Округляем до целого USDT для красоты
    return {k: round(v) for k, v in parts.items()}


def _parse_weights(args: List[str]) -> Tuple[bool, Dict[str, float], str]:
    """Парсит 'jpy=40 aud=25 eur=20 gbp=15'."""
    out = {}
    try:
        for token in args:
            if "=" not in token:
                return False, {}, f"Неверный формат: {token}"
            k, v = token.split("=", 1)
            k = k.strip().upper()
            v = float(v.strip())
            if k not in ("JPY", "AUD", "EUR", "GBP"):
                return False, {}, f"Неизвестный ключ: {k}"
            out[k] = v
        s = sum(out.values())
        if abs(s - 100.0) > 1e-6:
            return False, {}, f"Сумма весов должна быть 100, сейчас {s}"
        return True, out, ""
    except Exception as e:
        return False, {}, f"Ошибка парсинга: {e}"


# -----------------------
# Команды
# -----------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    # LLM статус
    llm_ok, llm_msg = llm_client.llm_ready()
    llm_mark = "✅" if llm_ok else f"❌ ({llm_msg})"

    # Sheets статус
    sheet_mark = "❌"
    try:
        _ = os.environ.get("SHEET_ID")
        data, err = _read_creds_any()
        if _ and not err:
            sheet_mark = "✅"
        else:
            sheet_mark = "❌"
    except Exception:
        sheet_mark = "❌"

    txt = (
        f"Фунд-бот запущен { '✅' }  LLM: {llm_mark}  Sheets: {sheet_mark}.\n\n"
        f"Привет! Я фунд-бот.\nТекущий чат id: <code>{chat_id}</code>\n\n"
        f"Команды: /help"
    )
    await context.bot.send_message(chat_id=chat_id, text=txt, parse_mode=ParseMode.HTML)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Что я умею\n"
        "• <code>/settotal 2800</code> — задать общий банк (только в мастер-чате).\n"
        "• <code>/setweights jpy=40 aud=25 eur=20 gbp=15</code> — выставить целевые веса.\n"
        "• <code>/weights</code> — показать целевые веса.\n"
        "• <code>/alloc</code> — расчёт сумм и готовые команды <code>/setbank</code> для торговых чатов.\n"
        "• <code>/digest</code> — короткий «человеческий» дайджест по USDJPY / AUDUSD / EURUSD / GBPUSD.\n"
        "• <code>/ping</code> — проверить связь.\n"
        "• <code>/check_sheets</code> — диагностика подключения к Google Sheets.\n\n"
        "Банк по парам задаётся вручную в торговых чатах; я сверяю распределение и даю советы/дайджест."
    )
    await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)


async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("pong")


async def cmd_weights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(f"Целевые веса: {fmt_weights(STATE['weights'])}")


async def cmd_setweights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok, w, err = _parse_weights(context.args)
    if not ok:
        await update.effective_message.reply_text(f"Ошибка: {err}\nПример: /setweights jpy=40 aud=25 eur=20 gbp=15")
        return
    STATE["weights"] = w
    await update.effective_message.reply_text(f"Новые веса приняты: {fmt_weights(w)}")


async def cmd_settotal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if MASTER_CHAT_ID and chat_id != MASTER_CHAT_ID:
        await update.effective_message.reply_text("Эта команда доступна только в мастер-чате.")
        return
    if not context.args:
        await update.effective_message.reply_text("Пример: /settotal 2800")
        return
    try:
        total = float(context.args[0])
        if total <= 0:
            raise ValueError()
    except Exception:
        await update.effective_message.reply_text("Нужно положительное число, пример: /settotal 2800")
        return
    STATE["total_bank"] = total
    await update.effective_message.reply_text(f"Общий банк установлен: {round(total)} USDT")


async def cmd_alloc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if MASTER_CHAT_ID and chat_id != MASTER_CHAT_ID:
        await update.effective_message.reply_text("Эта команда доступна только в мастер-чате.")
        return
    total = STATE["total_bank"]
    if total <= 0:
        await update.effective_message.reply_text("Сначала задай общий банк: /settotal 2800")
        return

    parts = compute_allocation(total, STATE["weights"])
    lines = [
        f"Целевые веса: {fmt_weights(STATE['weights'])}",
        "Распределение банка:"
    ]
    for sym in ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]:
        usdt = parts[sym]
        cmd = "/setbank " + str(usdt)
        lines.append(f"{sym} → {usdt} USDT   → команда в чат {sym}: {cmd}")
    text = "\n".join(lines)
    await update.effective_message.reply_text(text)


async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    syms = pairs_for_chat(chat_id)

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    try:
        text = await llm_client.make_digest_for_pairs(syms)
        if not text or not text.strip():
            # надёжный фолбэк
            blocks = []
            for s in syms:
                pretty = f"{s[:3]}/{s[3:]}"
                blocks.append(f"<b>{pretty}</b> — фон спокойный; обычный режим.")
            text = "\n\n".join(blocks)
        await context.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML)
    except Exception as e:
        log.exception("digest failed")
        await context.bot.send_message(chat_id=chat_id, text=f"LLM ошибка: {e}")


async def cmd_check_sheets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    try:
        data, err = _read_creds_any()
        sid = os.environ.get("SHEET_ID")
        if err:
            await context.bot.send_message(chat_id=chat_id, text=f"Sheets: ❌ {err}")
            return
        if not sid:
            await context.bot.send_message(chat_id=chat_id, text=f"Sheets: ❌ SHEET_ID not set")
            return
        sh, svc = _open_sheet()
        await ensure_fa_sheet()
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"Sheets: ✅ подключено\n"
                f"Документ: {sh.title}\n"
                f"Сервис-аккаунт: <code>{svc or 'n/a'}</code>\n"
                f"Лист: FA_Signals готов."
            ),
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"Sheets: ❌ {e}")


# -----------------------
# Инициализация и запуск
# -----------------------

def add_handlers(app: Application):
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ping", cmd_ping))

    app.add_handler(CommandHandler("weights", cmd_weights))
    app.add_handler(CommandHandler("setweights", cmd_setweights))
    app.add_handler(CommandHandler("settotal", cmd_settotal))
    app.add_handler(CommandHandler("alloc", cmd_alloc))

    app.add_handler(CommandHandler("digest", cmd_digest))
    app.add_handler(CommandHandler("check_sheets", cmd_check_sheets))


def build_app() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    builder = Application.builder().token(BOT_TOKEN)
    if AIORateLimiter is not None:
        builder = builder.rate_limiter(AIORateLimiter())

    app = builder.build()
    add_handlers(app)

    # Команды в меню бота
    try:
        app.bot.set_my_commands([
            BotCommand("help", "что умею"),
            BotCommand("digest", "дайджест по парам"),
            BotCommand("weights", "показать веса"),
            BotCommand("setweights", "задать веса (только мастер)"),
            BotCommand("settotal", "задать общий банк (только мастер)"),
            BotCommand("alloc", "рассчитать распределение"),
            BotCommand("check_sheets", "проверка Google Sheets"),
            BotCommand("ping", "проверка связи"),
        ])
    except Exception:
        pass
    return app


def main():
    log.info("Fund bot is running…")
    app = build_app()
    # webhooks не включаем, работаем на getUpdates
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
