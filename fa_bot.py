# fa_bot.py
import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

import gspread
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, ContextTypes, filters,
)

# --- rate limiter (optional) ---
try:
    from telegram.ext import AIORateLimiter
    _RATE_LIMITER = AIORateLimiter()
except Exception:
    AIORateLimiter = None
    _RATE_LIMITER = None

# наш тонкий клиент к LLM (см. llm_client.py)
from llm_client import chat as llm_chat


# ------------- ENV -------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
SHEET_ID = os.getenv("SHEET_ID")  # Google Sheet ID
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS")  # JSON строки
ALLOWED_CHAT_IDS = {
    int(x) for x in os.getenv("ALLOWED_CHAT_IDS", "").split(",") if x.strip().lstrip("-").isdigit()
}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("fa_bot")


# ------------- Google Sheets helpers -------------
_HEADERS = [
    "pair", "risk", "bias", "ttl", "updated_at",
    "scan_lock_until", "reserve_off", "dca_scale", "notes"
]

def _get_sheet() -> Optional[gspread.Spreadsheet]:
    if not (SHEET_ID and GOOGLE_CREDENTIALS):
        return None
    try:
        gc = gspread.service_account_from_dict(json.loads(GOOGLE_CREDENTIALS))
        return gc.open_by_key(SHEET_ID)
    except Exception:
        log.exception("Failed to open Google Sheet")
        return None

def _ensure_ws(sh: gspread.Spreadsheet, title: str) -> gspread.Worksheet:
    try:
        ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=2000, cols=max(20, len(_HEADERS)))
        ws.append_row(_HEADERS)
    return ws

def _find_row_by_pair(ws: gspread.Worksheet, symbol: str) -> Optional[int]:
    symbol = (symbol or "").upper().strip()
    try:
        col = ws.col_values(1)  # first column: pair
    except Exception:
        log.exception("col_values failed")
        return None
    for idx, val in enumerate(col, start=1):
        if idx == 1:
            continue  # header
        if (val or "").upper().strip() == symbol:
            return idx
    return None

def upsert_fa_signal(symbol: str, data: dict, notes: str = "") -> Tuple[bool, str]:
    """Возвращает (ok, msg)."""
    sh = _get_sheet()
    if not sh:
        return False, "Google Sheets is not configured."

    ws = _ensure_ws(sh, "FA_Signals")
    symbol = (symbol or "").upper().strip()
    if not symbol:
        return False, "Symbol is empty."

    # нормализация/дефолты
    safe = {
        "pair": symbol,
        "risk": (data.get("risk") or "Green").capitalize(),
        "bias": (data.get("bias") or "neutral").lower(),
        "ttl": int(data.get("ttl") or 60),
        "updated_at": data.get("updated_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "scan_lock_until": str(data.get("scan_lock_until") or ""),
        "reserve_off": bool(data.get("reserve_off") or False),
        "dca_scale": float(data.get("dca_scale") if data.get("dca_scale") is not None else 1.0),
        "notes": (notes or "")[:1000],
    }
    # клипы
    if safe["risk"] not in ("Green", "Amber", "Red"):
        safe["risk"] = "Green"
    if safe["bias"] not in ("neutral", "long-only", "short-only"):
        safe["bias"] = "neutral"
    safe["ttl"] = max(5, min(1440, safe["ttl"]))
    safe["dca_scale"] = max(0.0, min(1.0, safe["dca_scale"]))

    values = [safe.get(h, "") for h in _HEADERS]

    try:
        row = _find_row_by_pair(ws, symbol)
        if row:
            rng = f"A{row}:I{row}"
            ws.update(rng, [values])
            return True, f"Updated row {row}"
        else:
            ws.append_row(values)
            return True, "Appended new row"
    except Exception:
        log.exception("upsert_fa_signal failed")
        return False, "Sheets write failed"


# ------------- LLM policy synthesis -------------
FA_JSON_SCHEMA = """
Верни ТОЛЬКО валидный JSON-объект со следующими ключами:
- risk: одна из ["Green","Amber","Red"]
- bias: одна из ["neutral","long-only","short-only"]
- ttl: целое кол-во минут (5..1440)
- scan_lock_until: ISO дата-время в UTC или "" если не нужно
- reserve_off: boolean
- dca_scale: число 0..1
"""

async def synthesize_policy_from_text(raw_news: str) -> dict:
    system = (
        "Ты риск-офицер FX. На вход — краткие факторы/новости по инструменту. "
        "Сформируй риск-политику для робота-скальпера. Будь консервативен при высокой неопределённости."
    )
    user = f"{FA_JSON_SCHEMA}\n\nНовости и факторы:\n{raw_news}\n"
    out = await llm_chat(system, user, json_mode=True)
    try:
        data = json.loads(out)
    except Exception:
        log.warning("LLM returned non-JSON, fallback to defaults. Raw: %s", out[:300])
        data = {}
    # дефолты/валидация будут ещё раз на этапе upsert
    return data


# ------------- Telegram helpers -------------
def _is_allowed(update: Update) -> bool:
    if not ALLOWED_CHAT_IDS:
        return True
    chat_id = update.effective_chat.id if update.effective_chat else None
    return (chat_id in ALLOWED_CHAT_IDS)

def _format_policy(symbol: str, data: dict) -> str:
    risk = (data.get("risk") or "Green").capitalize()
    bias = (data.get("bias") or "neutral").lower()
    ttl = int(data.get("ttl") or 60)
    scan_lock = data.get("scan_lock_until") or ""
    reserve_off = bool(data.get("reserve_off") or False)
    dca_scale = float(data.get("dca_scale") if data.get("dca_scale") is not None else 1.0)

    lines = [
        f"📊 <b>FA Policy</b> for <code>{symbol}</code>",
        f"• risk: <b>{risk}</b>",
        f"• bias: <b>{bias}</b>",
        f"• ttl: <b>{ttl}m</b>",
        f"• reserve_off: <b>{'on' if reserve_off else 'off'}</b>",
        f"• dca_scale: <b>{dca_scale:.2f}</b>",
    ]
    if scan_lock:
        lines.append(f"• scan_lock_until: <code>{scan_lock}</code>")
    return "\n".join(lines)


# ------------- Handlers -------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update):
        return
    await update.message.reply_html(
        "Привет! Я FA-бот. Команды:\n"
        "<code>/fa &lt;SYMBOL&gt; &lt;текст&gt;</code> — сделать политику из текста и записать в Sheet\n"
        "<code>/fa SYMBOL</code> в ответ на сообщение — возьму текст из реплая\n"
        "<code>/fa_test &lt;текст&gt;</code> — только LLM JSON без записи\n"
        "<code>/status &lt;SYMBOL&gt;</code> — показать текущую политику из листа\n"
        "<code>/ping</code>"
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update):
        return
    await cmd_start(update, context)

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update):
        return
    await update.message.reply_text("pong")

def _extract_symbol_and_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Tuple[Optional[str], str]:
    """Возвращает (SYMBOL, TEXT). TEXT может быть пустым."""
    symbol = None
    text = ""
    if context.args:
        symbol = (context.args[0] or "").upper()
        if len(context.args) > 1:
            text = " ".join(context.args[1:])
    if not text and update.message and update.message.reply_to_message:
        text = (update.message.reply_to_message.text or update.message.reply_to_message.caption or "") or text
    return symbol, (text or "").strip()

async def cmd_fa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update):
        return

    symbol, text = _extract_symbol_and_text(update, context)
    if not symbol:
        await update.message.reply_html("Укажи символ: <code>/fa EURUSD ...</code> (можно ответом на сообщение с текстом)")
        return
    if not text:
        await update.message.reply_html("Добавь текст факторов после символа или ответь на сообщение с новостью.")
        return

    await update.message.chat.send_action("typing")
    policy = await synthesize_policy_from_text(text)

    # проставим updated_at
    policy["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    ok, msg = upsert_fa_signal(symbol, policy, notes=text)

    prefix = "✅" if ok else "⚠️"
    await update.message.reply_html(
        f"{prefix} {_format_policy(symbol, policy)}\n\n<i>{msg}</i>"
    )
    # отладочный JSON блок
    try:
        pretty = json.dumps(policy, ensure_ascii=False, indent=2)
        await update.message.reply_text(f"JSON:\n{pretty}")
    except Exception:
        pass

async def cmd_fa_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update):
        return
    text = " ".join(context.args) if context.args else ""
    if not text and update.message and update.message.reply_to_message:
        text = update.message.reply_to_message.text or update.message.reply_to_message.caption or ""
    if not text:
        await update.message.reply_html("Дай текст после команды или ответь на сообщение с новостью.")
        return
    await update.message.chat.send_action("typing")
    policy = await synthesize_policy_from_text(text)
    try:
        pretty = json.dumps(policy, ensure_ascii=False, indent=2)
    except Exception:
        pretty = str(policy)
    await update.message.reply_text(pretty)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update):
        return
    if not context.args:
        await update.message.reply_html("Использование: <code>/status SYMBOL</code>")
        return
    symbol = (context.args[0] or "").upper()

    sh = _get_sheet()
    if not sh:
        await update.message.reply_text("Sheets не сконфигурирован.")
        return
    ws = _ensure_ws(sh, "FA_Signals")
    row = _find_row_by_pair(ws, symbol)
    if not row:
        await update.message.reply_text("Нет записи для этого символа.")
        return

    try:
        values = ws.row_values(row)
        data = {h: (values[i] if i < len(values) else "") for i, h in enumerate(_HEADERS)}
        # Типизация
        data["ttl"] = int(data.get("ttl") or 0)
        data["reserve_off"] = str(data.get("reserve_off")).strip().lower() in ("1", "true", "yes", "on")
        try:
            data["dca_scale"] = float(data.get("dca_scale") or 1.0)
        except Exception:
            data["dca_scale"] = 1.0
    except Exception:
        log.exception("status read failed")
        await update.message.reply_text("Ошибка чтения листа.")
        return

    await update.message.reply_html(_format_policy(symbol, data))


# ------------- main -------------
def main():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    builder = Application.builder().token(BOT_TOKEN)
    if _RATE_LIMITER is not None:
        builder = builder.rate_limiter(_RATE_LIMITER)

    app = builder.build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler(["fa", "policy"], cmd_fa))
    app.add_handler(CommandHandler("fa_test", cmd_fa_test))
    app.add_handler(CommandHandler("status", cmd_status))

    log.info("FA bot starting…")
    # drop_pending_updates=True, чтобы не перемалывать старую очередь при рестартах
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
