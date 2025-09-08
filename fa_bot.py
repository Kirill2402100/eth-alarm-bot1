from __future__ import annotations

import os, json, logging, asyncio, re
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional

import gspread
from telegram import Update, BotCommand
from telegram.ext import (
    Application, CommandHandler, ContextTypes, AIORateLimiter
)

from llm_client import run_mini_digest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("fund_bot")

# --------------------------
# ENV
# --------------------------
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
MASTER_CHAT_ID = os.environ.get("MASTER_CHAT_ID")  # str / int строкой
SHEET_ID = os.environ.get("SHEET_ID")
GOOGLE_CREDENTIALS_RAW = os.environ.get("GOOGLE_CREDENTIALS")

DEFAULT_WEIGHTS_ENV = os.environ.get("DEFAULT_WEIGHTS", "")
# ожидаем формат: {"JPY":40,"AUD":25,"EUR":20,"GBP":15} или "jpy=40 aud=25 ..."
def _parse_default_weights(s: str) -> Dict[str, float]:
    if not s:
        return {"JPY": 40, "AUD": 25, "EUR": 20, "GBP": 15}
    s = s.strip()
    try:
        if s.startswith("{"):
            d = json.loads(s)
            return {k.upper(): float(v) for k, v in d.items()}
    except Exception:
        pass
    parts = re.findall(r"([a-zA-Z]{3})\s*=\s*([0-9]+(?:\.[0-9]+)?)", s)
    if parts:
        return {k.upper(): float(v) for k, v in parts}
    return {"JPY": 40, "AUD": 25, "EUR": 20, "GBP": 15}

DEFAULT_WEIGHTS = _parse_default_weights(DEFAULT_WEIGHTS_ENV)

# храним в памяти целевые веса и «общий банк», заданные вручную
STATE = {
    "weights": DEFAULT_WEIGHTS.copy(),
    "total_bank": None,  # задаётся /settotal ТОЛЬКО в мастер-чате
}

# --------------------------
# Google Sheets helpers
# --------------------------
FA_HEADERS = [
    "pair","risk","bias","ttl","updated_at",
    "scan_lock_until","reserve_off","dca_scale"
]

def _load_google_creds() -> Tuple[Optional[dict], Optional[str]]:
    """Возвращает (dict_creds, error_text)."""
    if not GOOGLE_CREDENTIALS_RAW:
        return None, "Переменная GOOGLE_CREDENTIALS отсутствует."
    try:
        creds = json.loads(GOOGLE_CREDENTIALS_RAW)
        # легкая валидация
        if not isinstance(creds, dict) or "client_email" not in creds:
            return None, "GOOGLE_CREDENTIALS не похожи на service account JSON."
        return creds, None
    except Exception as e:
        return None, f"GOOGLE_CREDENTIALS: не удалось распарсить JSON ({e})."

def _open_sheet_or_error() -> Tuple[Optional[gspread.Spreadsheet], Optional[str]]:
    if not SHEET_ID:
        return None, "SHEET_ID не задан."
    creds, err = _load_google_creds()
    if err:
        return None, err
    try:
        gc = gspread.service_account_from_dict(creds)
        sh = gc.open_by_key(SHEET_ID)
        return sh, None
    except gspread.SpreadsheetNotFound:
        return None, "Таблица по SHEET_ID не найдена. Проверьте ID."
    except gspread.exceptions.APIError as e:
        return None, f"Google API error: {e}"
    except Exception as e:
        return None, f"Ошибка доступа к Sheets: {e}"

def _ensure_fa_sheet(sh: gspread.Spreadsheet) -> Tuple[bool, Optional[str]]:
    """Создаёт лист FA_Signals с заголовками, если его нет."""
    try:
        try:
            ws = sh.worksheet("FA_Signals")
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title="FA_Signals", rows=100, cols=max(10, len(FA_HEADERS)))
            ws.append_row(FA_HEADERS)
            return True, None
        # проверим заголовки
        row1 = ws.row_values(1)
        if row1 != FA_HEADERS:
            # мягко синхронизируем: не трогаем существующие строки, только row1
            ws.update("1:1", [FA_HEADERS])
        return True, None
    except Exception as e:
        return False, f"Не удалось подготовить лист FA_Signals: {e}"

async def report_sheets_status(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    sh, err = _open_sheet_or_error()
    if err:
        await context.bot.send_message(chat_id, f"Sheets: не настроено — {err}")
        return
    ok, err2 = _ensure_fa_sheet(sh)
    if ok:
        await context.bot.send_message(chat_id, "Sheets: готово ✅ (лист FA_Signals доступен).")
    else:
        await context.bot.send_message(chat_id, f"Sheets: ошибка — {err2}")

# --------------------------
# Utils
# --------------------------
def _is_master(update: Update) -> bool:
    if not MASTER_CHAT_ID:
        return True  # если не задан — не ограничиваем
    try:
        return str(update.effective_chat.id) == str(MASTER_CHAT_ID)
    except Exception:
        return False

def _fmt_weights(d: Dict[str, float]) -> str:
    return " / ".join(f"{k} {int(v)}" for k, v in d.items())

def _parse_weights_arg(text: str) -> Optional[Dict[str, float]]:
    # ожидаем "jpy=40 aud=25 eur=20 gbp=15"
    parts = re.findall(r"([a-zA-Z]{3})\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not parts:
        return None
    out = {}
    for k, v in parts:
        out[k.upper()] = float(v)
    if abs(sum(out.values()) - 100.0) > 1e-6:
        # не заставляем ровно 100, но предупреждаем в ответе
        pass
    return out

def _alloc_from_total(total: float, weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(weights.values()) or 1.0
    return {k: round(total * (w / s), 2) for k, w in weights.items()}

# --------------------------
# Commands
# --------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await context.bot.send_message(
        chat_id,
        "Привет! Я фунд-бот.\n"
        f"Текущий чат id: <code>{chat_id}</code>\n\n"
        "Команды: /help"
        , parse_mode="HTML"
    )
    # короткий статус
    llm_ok = "✅" if os.environ.get("OPENAI_API_KEY") else "❌"
    if GOOGLE_CREDENTIALS_RAW and SHEET_ID:
        await report_sheets_status(context, chat_id)
    else:
        await context.bot.send_message(
            chat_id,
            "Sheets: не настроено (нет SHEET_ID/GOOGLE_CREDENTIALS)."
        )
    await context.bot.set_my_commands([
        BotCommand("help", "Справка"),
        BotCommand("ping", "Проверка связи"),
        BotCommand("weights", "Показать целевые веса"),
        BotCommand("setweights", "Задать веса (напр. jpy=40 aud=25 eur=20 gbp=15)"),
        BotCommand("settotal", "Задать общий банк (только в мастер-чате)"),
        BotCommand("alloc", "Рассчитать суммы/команды для чатов"),
        BotCommand("digest", "Утренний дайджест"),
        BotCommand("diag", "Диагностика окружения"),
    ])
    await context.bot.send_message(chat_id, f"Фунд-бот запущен {'✅' if llm_ok=='✅' else '❌ LLM'}. "
                                            f"Sheets: {'✅' if (GOOGLE_CREDENTIALS_RAW and SHEET_ID) else '❌'}.")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Что я умею\n"
        "• <code>/settotal 2800</code> — задать общий банк (только в мастер-чате).\n"
        "• <code>/setweights jpy=40 aud=25 eur=20 gbp=15</code> — выставить целевые веса.\n"
        "• <code>/weights</code> — показать целевые веса.\n"
        "• <code>/alloc</code> — расчёт сумм и готовые команды <code>/setbank</code> для торговых чатов.\n"
        "• <code>/digest</code> — короткий «человеческий» дайджест по USDJPY / AUDUSD / EURUSD / GBPUSD.\n"
        "• <code>/diag</code> — проверка настроек окружения (Sheets/LLM).\n"
        "Банк по парам задаётся вручную в торговых чатах; я сверяю распределение и даю советы/дайджест."
    )
    await update.message.reply_text(txt, parse_mode="HTML")

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong ✅")

async def cmd_weights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Целевые веса: {_fmt_weights(STATE['weights'])}")

async def cmd_setweights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    args = update.message.text.split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_text("Формат: /setweights jpy=40 aud=25 eur=20 gbp=15")
        return
    w = _parse_weights_arg(args[1])
    if not w:
        await update.message.reply_text("Не понял веса. Пример: /setweights jpy=40 aud=25 eur=20 gbp=15")
        return
    STATE["weights"] = w
    await update.message.reply_text(f"Ок. Целевые веса → {_fmt_weights(w)}")

async def cmd_settotal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_master(update):
        await update.message.reply_text("Эта команда доступна только в мастер-чате.")
        return
    if not update.message or not update.message.text:
        return
    args = update.message.text.split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_text("Формат: /settotal 2800")
        return
    try:
        total = float(args[1])
        if total <= 0:
            raise ValueError
    except Exception:
        await update.message.reply_text("Нужно положительное число. Пример: /settotal 2800")
        return
    STATE["total_bank"] = total
    await update.message.reply_text(f"Ок. Общий банк = {total:.2f} USDT.")

async def cmd_alloc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if STATE["total_bank"] is None:
        await update.message.reply_text("Сначала задайте общий банк: /settotal 2800 (в мастер-чате).")
        return
    alloc = _alloc_from_total(STATE["total_bank"], STATE["weights"])
    # под ваши 4 чата:
    symbol_map = {"USDJPY":"JPY","AUDUSD":"AUD","EURUSD":"EUR","GBPUSD":"GBP"}
    lines = ["Распределение:"]
    for sym, ccy in symbol_map.items():
        v = alloc.get(ccy, 0.0)
        lines.append(f"• {sym} → {v:.2f} USDT  → команда: <code>/setbank {v:.0f}</code>")
    lines.append("")
    lines.append(f"Целевые веса: {_fmt_weights(STATE['weights'])}")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")

async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Русский дайджест; использует LLM mini с fallback."""
    chat_id = update.effective_chat.id
    # простой каркас «флагов» (в реале сюда подставятся ваши сигналы)
    flags = {
        "USDJPY": {"risk": "CAUTION", "bias": "short-bias", "notes": ["Риск интервенций в риторике Минфина."]},
        "AUDUSD": {"risk": "OK", "bias": "both", "notes": ["Сырьё стабильно, спреды низкие."]},
        "EURUSD": {"risk": "OK", "bias": "both", "notes": ["Риторика ЕЦБ без сюрпризов."]},
        "GBPUSD": {"risk": "CAUTION", "bias": "both", "notes": ["BoE — «дольше на высоких ставках»."]},
    }
    try:
        text = await run_mini_digest(flags)
        if not text.strip():
            text = "Сегодня без особых событий. Фон нейтральный по всем парам."
        await context.bot.send_message(chat_id, text, parse_mode="HTML", disable_web_page_preview=True)
    except Exception as e:
        await context.bot.send_message(chat_id, f"LLM ошибка: {e}")

async def cmd_diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает, что видит бот из ENV и может ли открыть таблицу."""
    parts = []
    parts.append(f"LLM_KEY: {'✅' if os.environ.get('OPENAI_API_KEY') else '❌'}")
    parts.append(f"SHEET_ID: {'✅' if SHEET_ID else '❌'}")
    g_ok = "✅" if GOOGLE_CREDENTIALS_RAW else "❌"
    parts.append(f"GOOGLE_CREDENTIALS: {g_ok}")
    sh, err = _open_sheet_or_error()
    if err:
        parts.append(f"Sheets open: ❌ ({err})")
    else:
        ok, err2 = _ensure_fa_sheet(sh)
        parts.append(f"FA_Signals: {'✅' if ok else '❌'}{'' if not err2 else ' ('+err2+')'}")
    await update.message.reply_text("\n".join(parts))

# --------------------------
# main
# --------------------------
async def main():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN (или TELEGRAM_TOKEN) не задан.")
    app = Application.builder().token(BOT_TOKEN).rate_limiter(AIORateLimiter()).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("weights", cmd_weights))
    app.add_handler(CommandHandler("setweights", cmd_setweights))
    app.add_handler(CommandHandler("settotal", cmd_settotal))
    app.add_handler(CommandHandler("alloc", cmd_alloc))
    app.add_handler(CommandHandler("digest", cmd_digest))
    app.add_handler(CommandHandler("diag", cmd_diag))

    log.info("Fund bot is running…")
    await app.initialize()
    await app.start()
    await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    await app.idle()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
