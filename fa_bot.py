# fa_bot.py
from __future__ import annotations

import os, json, base64, logging, asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from telegram import Update, BotCommand
from telegram.ext import (
    Application, CommandHandler, ContextTypes, AIORateLimiter as _AIORateLimiter
)

# ====== ЛОГИ ======
log = logging.getLogger("fund_bot")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

# ====== КОНСТАНТЫ / НАСТРОЙКИ ======
SYMBOLS: List[str] = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]

# дефолтные веса, если не заданы в ENV DEFAULT_WEIGHTS (JSON) или /setweights
DEFAULT_WEIGHTS_ENV = os.getenv("DEFAULT_WEIGHTS", "")
try:
    DEFAULT_WEIGHTS: Dict[str, float] = json.loads(DEFAULT_WEIGHTS_ENV) if DEFAULT_WEIGHTS_ENV else \
        {"JPY": 40.0, "AUD": 25.0, "EUR": 20.0, "GBP": 15.0}
except Exception:
    DEFAULT_WEIGHTS = {"JPY": 40.0, "AUD": 25.0, "EUR": 20.0, "GBP": 15.0}

MASTER_CHAT_ID = os.getenv("MASTER_CHAT_ID")  # строкой, сравниваем как строку
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

# LLM модели
LLM_NANO = os.getenv("LLM_NANO", "gpt-5-nano")
LLM_MINI = os.getenv("LLM_MINI", "gpt-5-mini")

# Sheets
SHEET_ID = os.getenv("SHEET_ID")
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS")
GOOGLE_CREDENTIALS_B64 = os.getenv("GOOGLE_CREDENTIALS_B64")

# ====== LLM КЛИЕНТ ======
# (ожидается файл llm_client.py рядом с этим файлом)
try:
    from llm_client import analyze_headlines_nano, daily_digest_mini
    LLM_READY = bool(os.getenv("OPENAI_API_KEY"))
except Exception as e:
    log.warning("LLM disabled: %s", e)
    analyze_headlines_nano = daily_digest_mini = None  # type: ignore
    LLM_READY = False

# ====== GOOGLE SHEETS ======
_gs = {"ok": False, "sh": None, "gc": None}

def _load_google_credentials_dict() -> Optional[dict]:
    raw = GOOGLE_CREDENTIALS
    if not raw and GOOGLE_CREDENTIALS_B64:
        try:
            raw = base64.b64decode(GOOGLE_CREDENTIALS_B64).decode("utf-8")
        except Exception:
            raw = None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None

def _ensure_ws(sh, name: str, headers: Optional[List[str]] = None):
    import gspread
    try:
        ws = sh.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=name, rows=2000, cols=max(20, (len(headers) if headers else 10)))
        if headers:
            ws.append_row(headers)
    return ws

def init_sheets() -> bool:
    """Подключение к Google Sheets; создаём нужные листы."""
    if not SHEET_ID:
        log.info("Sheets: SHEET_ID is empty.")
        return False
    creds = _load_google_credentials_dict()
    if not creds:
        log.info("Sheets: GOOGLE_CREDENTIALS(_B64) not provided.")
        return False
    try:
        import gspread
        gc = gspread.service_account_from_dict(creds)
        sh = gc.open_by_key(SHEET_ID)

        # Листы под политику и логи аллокаций
        _ensure_ws(sh, "FA_Signals",
                   ["pair", "risk", "bias", "ttl", "updated_at", "scan_lock_until", "reserve_off", "dca_scale"])
        _ensure_ws(sh, "FA_Alloc_Log", ["ts_utc", "total_bank", "weights_json", "applied"])

        _gs["ok"] = True
        _gs["gc"] = gc
        _gs["sh"] = sh
        log.info("Sheets: connected and ensured worksheets.")
        return True
    except Exception as e:
        log.exception("Sheets init failed: %s", e)
        _gs["ok"] = False
        _gs["sh"] = None
        _gs["gc"] = None
        return False

def append_alloc_log(total: float, weights: Dict[str, float], applied: str = "GENERATED"):
    if not _gs["ok"]:
        return
    try:
        sh = _gs["sh"]
        ws = sh.worksheet("FA_Alloc_Log")
        ws.append_row([
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            float(total),
            json.dumps(weights, ensure_ascii=False),
            applied
        ])
    except Exception:
        log.exception("append_alloc_log failed")

# ====== ВНУТРЕННЕЕ СОСТОЯНИЕ БОТА ======
# Храним в app.bot_data:
# - weights: Dict[str,float] (по ключам 'JPY','AUD','EUR','GBP')
# - total_bank: float
# - last_alloc_id: str (произвольный идентификатор рекомендаций)
def _get_weights(bot_data: dict) -> Dict[str, float]:
    w = bot_data.get("weights")
    if isinstance(w, dict) and all(k in ("JPY", "AUD", "EUR", "GBP") for k in w.keys()):
        return {k: float(v) for k, v in w.items()}
    bot_data["weights"] = dict(DEFAULT_WEIGHTS)
    return bot_data["weights"]

def _set_weights(bot_data: dict, w: Dict[str, float]):
    bot_data["weights"] = {k.upper(): float(v) for k, v in w.items()}

def _get_total(bot_data: dict) -> Optional[float]:
    v = bot_data.get("total_bank")
    try:
        return None if v is None else float(v)
    except Exception:
        return None

def _set_total(bot_data: dict, x: float):
    bot_data["total_bank"] = float(x)

def _is_master(update: Update) -> bool:
    if not MASTER_CHAT_ID:
        return True  # если не настроено — не ограничиваем
    return str(update.effective_chat.id) == str(MASTER_CHAT_ID)

# ====== УТИЛИТЫ ======
def _parse_weights_arg(text: str) -> Dict[str, float]:
    """
    Примеры:
      jpy=40 aud=25 eur=20 gbp=15
      JPY:40, AUD:25, EUR:20, GBP:15
      jpy 40 aud 25 eur 20 gbp 15
    """
    t = text.replace(",", " ").replace(":", "=").strip()
    parts = [p for p in t.split() if p]
    # собираем пары
    out: Dict[str, float] = {}
    i = 0
    while i < len(parts):
        p = parts[i]
        if "=" in p:
            k, v = p.split("=", 1)
            k = k.strip().upper()
            v = v.strip().rstrip("%")
            try:
                out[k] = float(v)
            except Exception:
                pass
            i += 1
        else:
            k = p.strip().upper()
            if i + 1 < len(parts):
                v = parts[i + 1].strip().rstrip("%")
                try:
                    out[k] = float(v)
                except Exception:
                    pass
                i += 2
            else:
                i += 1
    # нормализуем ключи к набору валют
    remap = {"JPY": "JPY", "AUD": "AUD", "EUR": "EUR", "GBP": "GBP"}
    out2: Dict[str, float] = {}
    for k, v in out.items():
        key = remap.get(k)
        if key:
            out2[key] = float(v)
    return out2

def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    """Приводим к сумме ~100 (если не ноль), иначе оставляем как есть."""
    s = sum(max(0.0, float(v)) for v in w.values())
    if s <= 0:
        return w
    return {k: round((max(0.0, float(v)) / s) * 100.0, 4) for k, v in w.items()}

def _alloc(total: float, weights: Dict[str, float]) -> Dict[str, float]:
    """Возвращает суммы по символам согласно весам (вес по JPY → USDJPY и т.д.)."""
    w = _normalize_weights(weights)
    by_pair = {
        "USDJPY": w.get("JPY", 0.0),
        "AUDUSD": w.get("AUD", 0.0),
        "EURUSD": w.get("EUR", 0.0),
        "GBPUSD": w.get("GBP", 0.0),
    }
    out = {}
    for pair, pct in by_pair.items():
        out[pair] = round(total * (pct / 100.0), 2)
    return out

def _format_weights_line(w: Dict[str, float]) -> str:
    return f"JPY {w.get('JPY',0):.0f} / AUD {w.get('AUD',0):.0f} / EUR {w.get('EUR',0):.0f} / GBP {w.get('GBP',0):.0f}"

def _format_alloc_lines(alloc: Dict[str, float]) -> str:
    lines = []
    lines.append(f"USDJPY → <b>{alloc.get('USDJPY',0):.2f} USDT</b> → команда в чат USDJPY: <code>/setbank {alloc.get('USDJPY',0):.0f}</code>")
    lines.append(f"AUDUSD → <b>{alloc.get('AUDUSD',0):.2f} USDT</b> → <code>/setbank {alloc.get('AUDUSD',0):.0f}</code>")
    lines.append(f"EURUSD → <b>{alloc.get('EURUSD',0):.2f} USDT</b> → <code>/setbank {alloc.get('EURUSD',0):.0f}</code>")
    lines.append(f"GBPUSD → <b>{alloc.get('GBPUSD',0):.2f} USDT</b> → <code>/setbank {alloc.get('GBPUSD',0):.0f}</code>")
    return "\n".join(lines)

# ====== КОМАНДЫ ======
HELP_TEXT = (
    "Что я умею\n"
    "• <code>/settotal 2800</code> — задать общий банк (только в мастер-чате).\n"
    "• <code>/setweights jpy=40 aud=25 eur=20 gbp=15</code> — выставить целевые веса.\n"
    "• <code>/weights</code> — показать целевые веса.\n"
    "• <code>/alloc</code> — расчёт сумм и готовые команды /setbank для торговых чатов.\n"
    "• <code>/digest</code> — короткий «человеческий» дайджест по USDJPY / AUDUSD / EURUSD / GBPUSD.\n"
    "• <code>/ping</code> — проверить связь.\n"
    "• <code>/diag</code> — диагностика окружения (env / Sheets / LLM).\n\n"
    "Банк по парам задаётся вручную в торговых чатах; я сверяю распределение и даю советы/дайджест."
)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    msg = [
        "Привет! Я фунд-бот.",
        f"Текущий чат id: <code>{chat_id}</code>",
        "",
        "Команды: /help",
        "",
        f"Sheets: {'✅' if _gs['ok'] else '❌'}",
        f"LLM: {'✅' if LLM_READY else '❌'}",
    ]
    await update.message.reply_html("\n".join(msg))

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_html(HELP_TEXT)

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

async def cmd_diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    flags = {
        "MASTER_CHAT_ID": MASTER_CHAT_ID or "",
        "SHEET_ID": SHEET_ID or "",
        "GOOGLE_CREDENTIALS": "set" if GOOGLE_CREDENTIALS else "",
        "GOOGLE_CREDENTIALS_B64": "set" if GOOGLE_CREDENTIALS_B64 else "",
        "SHEETS_CONNECTED": "yes" if _gs["ok"] else "no",
        "OPENAI_API_KEY": "set" if os.getenv("OPENAI_API_KEY") else "",
        "LLM_MINI": LLM_MINI,
        "LLM_NANO": LLM_NANO,
    }
    lines = [f"{k}: {v}" for k, v in flags.items()]
    await update.message.reply_text("DIAG:\n" + "\n".join(lines))

async def cmd_settotal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_master(update):
        await update.message.reply_text("Команда доступна только в мастер-чате.")
        return
    if not context.args:
        await update.message.reply_text("Пример: /settotal 2800")
        return
    try:
        total = float(context.args[0])
        if total <= 0:
            raise ValueError
    except Exception:
        await update.message.reply_text("Нужно положительное число. Пример: /settotal 2800")
        return
    _set_total(context.application.bot_data, total)
    await update.message.reply_text(f"Общий банк установлен: {total:.2f} USDT")

async def cmd_setweights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_master(update):
        await update.message.reply_text("Команда доступна только в мастер-чате.")
        return
    if not context.args:
        await update.message.reply_text("Пример: /setweights jpy=40 aud=25 eur=20 gbp=15")
        return
    raw = " ".join(context.args)
    w = _parse_weights_arg(raw)
    if not w or not all(k in w for k in ("JPY","AUD","EUR","GBP")):
        await update.message.reply_text("Не распознал веса. Пример: /setweights jpy=40 aud=25 eur=20 gbp=15")
        return
    w_norm = _normalize_weights(w)
    _set_weights(context.application.bot_data, w_norm)
    await update.message.reply_text("Целевые веса: " + _format_weights_line(w_norm))

async def cmd_weights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    w = _get_weights(context.application.bot_data)
    await update.message.reply_text("Целевые веса: " + _format_weights_line(w))

async def cmd_alloc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total = _get_total(context.application.bot_data)
    if not total:
        await update.message.reply_text("Сначала установи общий банк: /settotal 2800")
        return
    w = _get_weights(context.application.bot_data)
    alloc = _alloc(total, w)
    lines = [
        f"Целевые веса: { _format_weights_line(w) }",
        "",
        _format_alloc_lines(alloc)
    ]
    await update.message.reply_html("\n".join(lines))
    append_alloc_log(total, w, applied="GENERATED")

async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not LLM_READY or analyze_headlines_nano is None or daily_digest_mini is None:
        await update.message.reply_text("LLM не настроен (нет OPENAI_API_KEY или модуль недоступен).")
        return
    try:
        # Заглушка источников — вставь свой сбор данных
        headlines = {
            "USDJPY": ["Минфин Японии о «чрезмерной волатильности»", "Доллар растёт на доходностях US"],
            "AUDUSD": ["Сырьевые рынки стабильны", "Сюрпризов в календаре нет"],
            "EURUSD": ["ЕЦБ без неожиданных сигналов"],
            "GBPUSD": ["BoE намекает: «дольше на высоких ставках»"],
        }
        facts = {k: {"calendar_today": [], "notes": v, "market": {}} for k, v in headlines.items()}

        flags = await asyncio.get_event_loop().run_in_executor(
            None, lambda: analyze_headlines_nano(LLM_NANO, headlines)
        )
        text = await asyncio.get_event_loop().run_in_executor(
            None, lambda: daily_digest_mini(LLM_MINI, facts)
        )

        await update.message.reply_html("Флаги:\n<code>" + json.dumps(flags, ensure_ascii=False, indent=2) + "</code>")
        await update.message.reply_text(text)
    except Exception as e:
        log.exception("digest failed")
        await update.message.reply_text(f"LLM ошибка: {e}")

# ====== СТАРТ ПРИЛОЖЕНИЯ ======
def _make_app() -> Application:
    # RateLimiter доступен, если установлен extras пакет; если нет — запускаем без него.
    try:
        rl = _AIORateLimiter()
    except Exception:
        rl = None

    b = Application.builder().token(TELEGRAM_BOT_TOKEN)
    if rl is not None:
        b = b.rate_limiter(rl)
    app = b.build()

    # Команды меню
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("diag", cmd_diag))

    app.add_handler(CommandHandler("settotal", cmd_settotal))
    app.add_handler(CommandHandler("setweights", cmd_setweights))
    app.add_handler(CommandHandler("weights", cmd_weights))
    app.add_handler(CommandHandler("alloc", cmd_alloc))
    app.add_handler(CommandHandler("digest", cmd_digest))

    return app

async def _post_startup(app: Application):
    try:
        await app.bot.set_my_commands([
            BotCommand("help", "справка"),
            BotCommand("ping", "проверка связи"),
            BotCommand("settotal", "задать общий банк (мастер-чат)"),
            BotCommand("setweights", "задать целевые веса (мастер-чат)"),
            BotCommand("weights", "показать целевые веса"),
            BotCommand("alloc", "рассчитать распределение / команды /setbank"),
            BotCommand("digest", "ежедневный дайджест"),
            BotCommand("diag", "диагностика окружения"),
        ])
    except Exception:
        log.exception("set_my_commands failed")

    txt = f"Фунд-бот запущен {'✅' if _gs['ok'] else '❌ Sheets: не настроено (нет SHEET_ID/GOOGLE_CREDENTIALS).'}"
    if MASTER_CHAT_ID:
        try:
            await app.bot.send_message(chat_id=int(MASTER_CHAT_ID), text=txt)
        except Exception:
            log.exception("Cannot notify master chat")

def main():
    log.info("Fund bot is running…")
    # Инициализируем Sheets (без фатала)
    init_sheets()

    app = _make_app()
    app.post_init = _post_startup  # type: ignore

    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
