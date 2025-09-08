from __future__ import annotations

import os
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional

import gspread
from telegram import (
    Update,
    BotCommand,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# ====== ЛОГИ ======
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("fund_bot")

# ====== LLM-слой ======
import llm_client


# ---------- ENV / CONFIG ----------
def _env_required(name: str) -> str:
    v = os.environ.get(name, "").strip()
    return v


TELEGRAM_TOKEN = _env_required("TELEGRAM_BOT_TOKEN") or _env_required("TELEGRAM_TOKEN")
MASTER_CHAT_ID = int(os.environ.get("MASTER_CHAT_ID", "0") or "0")

SHEET_ID = _env_required("SHEET_ID")
GOOGLE_CREDENTIALS_RAW = _env_required("GOOGLE_CREDENTIALS")

# Модели (с дефолтами)
LLM_NANO = os.environ.get("LLM_NANO", "gpt-5-nano")
LLM_MINI = os.environ.get("LLM_MINI", "gpt-5-mini")
LLM_MAJOR = os.environ.get("LLM_MAJOR", "gpt-5")

DEFAULT_WEIGHTS_ENV = os.environ.get("DEFAULT_WEIGHTS", "").strip()
if DEFAULT_WEIGHTS_ENV:
    try:
        DEFAULT_WEIGHTS = json.loads(DEFAULT_WEIGHTS_ENV)
    except Exception:
        # допускаем формат a=10 b=20 ...
        DEFAULT_WEIGHTS = {}
        for tok in DEFAULT_WEIGHTS_ENV.replace(",", " ").split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    DEFAULT_WEIGHTS[k.upper()] = float(v)
                except:
                    pass
else:
    DEFAULT_WEIGHTS = {"JPY": 40, "AUD": 25, "EUR": 20, "GBP": 15}

PAIRS = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]  # сводка/дайджест по этим тикерам


# ---------- GOOGLE SHEETS ----------
SHEET_FA_NAME = "FA_Signals"
SHEET_ALLOC_LOG = "FA_Alloc_Log"

FA_HEADERS = [
    "pair", "risk", "bias", "ttl", "updated_at",
    "scan_lock_until", "reserve_off", "dca_scale",
]

def _ensure_sheets() -> Tuple[bool, str, Optional[gspread.Spreadsheet], Optional[str]]:
    """
    Возвращает (ok, msg, sheet, service_email)
    """
    if not SHEET_ID:
        return False, "SHEET_ID не задан", None, None
    if not GOOGLE_CREDENTIALS_RAW:
        return False, "GOOGLE_CREDENTIALS не задан", None, None

    try:
        creds = json.loads(GOOGLE_CREDENTIALS_RAW)
        svc_email = creds.get("client_email", "")
    except Exception as e:
        return False, f"GOOGLE_CREDENTIALS: ошибка парсинга JSON ({e})", None, None

    try:
        gc = gspread.service_account_from_dict(creds)
        sh = gc.open_by_key(SHEET_ID)
    except gspread.exceptions.SpreadsheetNotFound:
        return False, "Таблица с таким SHEET_ID не найдена. Проверьте ID.", None, svc_email
    except gspread.exceptions.APIError as e:
        # Частая причина — нет доступа (не расшарили сервисный аккаунт).
        return False, f"Нет доступа к таблице (расшарьте на {svc_email}). Детали: {e}", None, svc_email
    except Exception as e:
        return False, f"Ошибка при подключении к таблице: {e}", None, svc_email

    # Лист FA_Signals
    try:
        try:
            ws = sh.worksheet(SHEET_FA_NAME)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=SHEET_FA_NAME, rows=1000, cols=20)
            ws.append_row(FA_HEADERS)
        # гарантируем заголовки
        row1 = ws.row_values(1)
        if [c.strip().lower() for c in row1] != [c.lower() for c in FA_HEADERS]:
            # перезапишем как надо, аккуратно
            ws.delete_rows(1)
            ws.insert_row(FA_HEADERS, 1)
    except Exception as e:
        return False, f"Не удалось создать/проверить лист {SHEET_FA_NAME}: {e}", sh, svc_email

    # Лист лога перекладок (опционально)
    try:
        try:
            sh.worksheet(SHEET_ALLOC_LOG)
        except gspread.WorksheetNotFound:
            ws2 = sh.add_worksheet(title=SHEET_ALLOC_LOG, rows=2000, cols=10)
            ws2.append_row(["ts_utc", "weights_json", "total_bank", "note"])
    except Exception as e:
        # не критично
        log.warning(f"Не удалось создать {SHEET_ALLOC_LOG}: {e}")

    return True, "OK", sh, svc_email


async def _alloc_save(sh: gspread.Spreadsheet, weights: Dict[str, float], total: float, note: str = ""):
    try:
        ws = sh.worksheet(SHEET_ALLOC_LOG)
        ws.append_row([
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            json.dumps(weights, ensure_ascii=False),
            float(total),
            note
        ])
    except Exception as e:
        log.warning(f"Alloc log append failed: {e}")


# ---------- ПАРСИНГ КОМАНД ----------
def _parse_weights(text: str) -> Dict[str, float]:
    """
    /setweights jpy=40 aud=25 eur=20 gbp=15
    """
    out: Dict[str, float] = {}
    for tok in text.strip().split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k.strip().upper()] = float(v.strip())
            except:
                pass
    return out


# ---------- ХЭНДЛЕРЫ ----------
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ok, msg, _, svc = _ensure_sheets()
    sheets_line = f"Sheets: {'✅' if ok else f'не настроено ({msg})'}"
    await update.message.reply_html(
        f"Привет! Я фунд-бот.\nТекущий чат id: <code>{chat_id}</code>\n\nКоманды: /help\n\n{sheets_line}"
    )

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Что я умею\n"
        "• <code>/settotal 2800</code> — задать общий банк (только в мастер-чате).\n"
        "• <code>/setweights jpy=40 aud=25 eur=20 gbp=15</code> — выставить целевые веса.\n"
        "• <code>/weights</code> — показать целевые веса.\n"
        "• <code>/alloc</code> — расчёт сумм и готовые команды /setbank для торговых чатов.\n"
        "• <code>/digest</code> — короткий «человеческий» дайджест по USDJPY / AUDUSD / EURUSD / GBPUSD.\n"
        "• <code>/ping</code> — проверить связь.\n\n"
        "<i>Банк по парам задаётся вручную в торговых чатах; я сверяю распределение и даю советы/дайджест.</i>"
    )
    await update.message.reply_html(txt)

async def cmd_ping(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

async def cmd_setweights(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = (update.message.text or "").split(maxsplit=1)
    if len(args) < 2:
        return await update.message.reply_text("Пример: /setweights jpy=40 aud=25 eur=20 gbp=15")
    w = _parse_weights(args[1])
    if not w:
        return await update.message.reply_text("Не удалось распарсить веса.")
    ctx.bot_data["target_weights"] = w
    await update.message.reply_html("Целевые веса обновлены:\n<code>{}</code>".format(json.dumps(w, ensure_ascii=False)))

async def cmd_weights(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    w = ctx.bot_data.get("target_weights") or DEFAULT_WEIGHTS
    await update.message.reply_html("<b>Целевые веса</b>:\n<code>{}</code>".format(json.dumps(w, ensure_ascii=False)))

async def cmd_settotal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if MASTER_CHAT_ID and update.effective_chat.id != MASTER_CHAT_ID:
        return await update.message.reply_text("Эта команда доступна только в мастер-чате.")
    args = (update.message.text or "").split()
    if len(args) < 2:
        return await update.message.reply_text("Пример: /settotal 2800")
    try:
        total = float(args[1])
    except:
        return await update.message.reply_text("Число не распознано.")
    ctx.bot_data["total_bank"] = total
    await update.message.reply_text(f"Общий банк установлен: {total:.2f} USDT")

async def cmd_alloc(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    total = float(ctx.bot_data.get("total_bank") or 0.0)
    if total <= 0:
        return await update.message.reply_text("Сначала задайте общий банк: /settotal <число> (в мастер-чате).")
    weights = ctx.bot_data.get("target_weights") or DEFAULT_WEIGHTS

    # нормируем на 100
    s = sum(weights.values()) or 1.0
    norm = {k: (v * 100.0 / s) for k, v in weights.items()}
    # расчёт
    alloc = {k: total * (pct / 100.0) for k, pct in norm.items()}
    lines = ["<b>Распределение банка</b>:"]
    # Готовые команды (вы пишете их в соответствующий торговый чат)
    for k, v in (("JPY", "USDJPY"), ("AUD", "AUDUSD"), ("EUR", "EURUSD"), ("GBP", "GBPUSD")):
        if k in alloc:
            amt = alloc[k]
            lines.append(f"{v} → <b>{amt:.0f} USDT</b>   → команда в чат {v}: <code>/setbank {amt:.0f}</code>")
    await update.message.reply_html("\n".join(lines))

    # пробуем записать в лог в таблице (если настроено)
    ok, msg, sh, _ = _ensure_sheets()
    if ok and sh:
        try:
            await asyncio.get_running_loop().run_in_executor(None, _alloc_save, sh, norm, total, "manual /alloc")
        except Exception as e:
            log.warning(f"Alloc log write error: {e}")

async def cmd_digest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Генерирует короткую «человеческую» сводку по четырём парам.
    """
    try:
        txt = await llm_client.make_digest_ru(
            pairs=PAIRS,
            model=LLM_MINI,
            nano_model=LLM_NANO,
        )
        await update.message.reply_html(txt, disable_web_page_preview=True)
    except Exception as e:
        log.exception("Digest error")
        await update.message.reply_text(f"LLM ошибка: {e}")

# ---------- СТАРТ ПРОГРАММЫ ----------
async def on_start(app: Application):
    log.info("Fund bot is running…")
    try:
        await app.bot.set_my_commands([
            BotCommand("start", "Поприветствовать"),
            BotCommand("help", "Справка"),
            BotCommand("settotal", "Задать общий банк (мастер-чат)"),
            BotCommand("setweights", "Установить целевые веса"),
            BotCommand("weights", "Показать целевые веса"),
            BotCommand("alloc", "Рассчитать суммы на чаты"),
            BotCommand("digest", "Утренний дайджест"),
            BotCommand("ping", "Проверка связи"),
        ])
    except Exception as e:
        log.warning(f"set_my_commands failed: {e}")

def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Переменная окружения TELEGRAM_BOT_TOKEN не задана.")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("setweights", cmd_setweights))
    app.add_handler(CommandHandler("weights", cmd_weights))
    app.add_handler(CommandHandler("settotal", cmd_settotal))
    app.add_handler(CommandHandler("alloc", cmd_alloc))
    app.add_handler(CommandHandler("digest", cmd_digest))

    app.post_init = on_start
    app.run_polling(allowed_updates=["message", "edited_message"])

if __name__ == "__main__":
    main()
