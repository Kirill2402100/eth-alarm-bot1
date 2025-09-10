# fa_bot.py
import os
import json
import base64
import logging
from math import floor
from time import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional, List

import asyncio

from telegram import Update, BotCommand
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# --- Таймзона Белграда ---
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None

LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Europe/Belgrade")) if ZoneInfo else None
MORNING_HOUR = int(os.getenv("MORNING_HOUR", "9"))
MORNING_MINUTE = int(os.getenv("MORNING_MINUTE", "30"))

# --- Лимитер Telegram (может быть не установлен) ---
try:
    from telegram.ext import AIORateLimiter
    _RATE_LIMITER_AVAILABLE = True
except Exception:  # pragma: no cover
    AIORateLimiter = None
    _RATE_LIMITER_AVAILABLE = False

# --- Google Sheets ---
try:
    import gspread
    from google.oauth2 import service_account
    _GSHEETS_AVAILABLE = True
except Exception:
    gspread = None
    service_account = None
    _GSHEETS_AVAILABLE = False

# --- HTTP для календаря ---
try:
    import requests
    from urllib.parse import quote
    _REQUESTS_AVAILABLE = True
except Exception:
    requests = None
    quote = None
    _REQUESTS_AVAILABLE = False

# --- LLM-клиент ---
try:
    from llm_client import generate_digest, llm_ping
except Exception:
    async def generate_digest(*args, **kwargs) -> str:
        return "⚠️ LLM сейчас недоступен (нет llm_client.py)."

    async def llm_ping() -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

# -------------------- ЛОГИ --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("fund_bot")

# -------------------- ENV --------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or os.getenv("TELEGRAM_TOKEN", "").strip()
MASTER_CHAT_ID = int(os.getenv("MASTER_CHAT_ID", "0") or "0")

SHEET_ID = os.getenv("SHEET_ID", "").strip()                      # только ID (между /d/ и /edit)
SHEET_WS = os.getenv("SHEET_WS", "FUND_BOT").strip() or "FUND_BOT"  # имя листа/вкладки по умолчанию

_DEFAULT_WEIGHTS_RAW = os.getenv("DEFAULT_WEIGHTS", "").strip()
if _DEFAULT_WEIGHTS_RAW:
    try:
        DEFAULT_WEIGHTS: Dict[str, int] = json.loads(_DEFAULT_WEIGHTS_RAW)
    except Exception:
        DEFAULT_WEIGHTS = {"JPY": 40, "AUD": 25, "EUR": 20, "GBP": 15}
else:
    DEFAULT_WEIGHTS = {"JPY": 40, "AUD": 25, "EUR": 20, "GBP": 15}

LLM_MINI = os.getenv("LLM_MINI", "gpt-5-mini").strip()
LLM_NANO = os.getenv("LLM_NANO", "gpt-5-nano").strip()
LLM_MAJOR = os.getenv("LLM_MAJOR", "gpt-5").strip()
LLM_TOKEN_BUDGET_PER_DAY = int(os.getenv("LLM_TOKEN_BUDGET_PER_DAY", "30000") or "30000")

SYMBOLS = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]

# --- Названия листов с логами по парам (переопределяются через ENV) ---
BMR_SHEETS = {
    "USDJPY": os.getenv("BMR_SHEET_USDJPY", "BMR_DCA_USDJPY"),
    "AUDUSD": os.getenv("BMR_SHEET_AUDUSD", "BMR_DCA_AUDUSD"),
    "EURUSD": os.getenv("BMR_SHEET_EURUSD", "BMR_DCA_EURUSD"),
    "GBPUSD": os.getenv("BMR_SHEET_GBPUSD", "BMR_DCA_GBPUSD"),
}

# --- Календарь ---
TE_BASE = os.getenv("TE_BASE", "https://api.tradingeconomics.com").rstrip("/")
TE_CLIENT = os.getenv("TE_CLIENT", "guest").strip()
TE_KEY = os.getenv("TE_KEY", "guest").strip()
CAL_WINDOW_MIN = int(os.getenv("CAL_WINDOW_MIN", "120"))    # окно для вывода событий (+/-)
QUIET_BEFORE_MIN = int(os.getenv("QUIET_BEFORE_MIN", "45"))  # тихое окно ДО high-релиза
QUIET_AFTER_MIN  = int(os.getenv("QUIET_AFTER_MIN",  "45"))  # тихое окно ПОСЛЕ high-релиза
CAL_PROVIDER = os.getenv("CAL_PROVIDER", "auto").lower()    # auto | te | fmp | ff
FMP_API_KEY  = os.getenv("FMP_API_KEY", "").strip()
CAL_TTL_SEC = int(os.getenv("CAL_TTL_SEC", "600") or "600")

COUNTRY_BY_CCY = {
    "USD": "united states",
    "JPY": "japan",
    "EUR": "euro area",
    "GBP": "united kingdom",
    "AUD": "australia",
}
PAIR_COUNTRIES = {
    "USDJPY": [COUNTRY_BY_CCY["USD"], COUNTRY_BY_CCY["JPY"]],
    "AUDUSD": [COUNTRY_BY_CCY["AUD"], COUNTRY_BY_CCY["USD"]],
    "EURUSD": [COUNTRY_BY_CCY["EUR"], COUNTRY_BY_CCY["USD"]],
    "GBPUSD": [COUNTRY_BY_CCY["GBP"], COUNTRY_BY_CCY["USD"]],
}

# -------------------- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ --------------------
_FF_CACHE = {"at": 0, "data": []}        # успешный кеш (unix time)
_FF_NEG   = {"until": 0}                 # «тихий» период после 429 (unix time)

STATE = {
    "total": 0.0,
    "weights": DEFAULT_WEIGHTS.copy(),
}

# -------------------- GOOGLE CREDS LOADER --------------------
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def _env(name: str) -> str:
    v = os.getenv(name, "")
    return v if isinstance(v, str) else ""


def _decode_b64_maybe_padded(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    padded = s + "=" * ((4 - len(s) % 4) % 4)
    return base64.b64decode(padded).decode("utf-8", "strict")


def load_google_service_info() -> Tuple[Optional[dict], str, Optional[str]]:
    b64 = _env("GOOGLE_CREDENTIALS_JSON_B64")
    if b64:
        try:
            decoded = _decode_b64_maybe_padded(b64)
            info = json.loads(decoded)
            client_email = info.get("client_email")
            return info, "env:GOOGLE_CREDENTIALS_JSON_B64", client_email
        except Exception as e:
            return None, f"b64 present but decode/json error: {e}", None

    for name in ("GOOGLE_CREDENTIALS_JSON", "GOOGLE_CREDENTIALS"):
        raw = _env(name)
        if raw:
            try:
                info = json.loads(raw)
                client_email = info.get("client_email")
                return info, f"env:{name}", client_email
            except Exception as e:
                return None, f"{name} present but invalid JSON: {e}", None

    return None, "not-found", None


def build_sheets_client(sheet_id: str):
    if not _GSHEETS_AVAILABLE:
        return None, "gsheets libs not installed"
    if not sheet_id:
        return None, "sheet_id empty"

    info, src, client_email = load_google_service_info()
    if not info:
        return None, src

    try:
        creds = service_account.Credentials.from_service_account_info(info, scopes=SHEETS_SCOPES)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_id)
        sa = client_email or getattr(getattr(gc, "auth", None), "service_account_email", "ok")
        return sh, f"{src} / sa={sa}"
    except Exception as e:
        return None, f"auth/open error: {e}"


# ---------- Sheets helpers ----------
SHEET_HEADERS = ["ts", "chat_id", "action", "total", "weights_json", "note"]


def ensure_worksheet(sh, title: str):
    """Вернёт существующий лист или создаст новый с заголовками."""
    try:
        for ws in sh.worksheets():
            if ws.title == title:
                return ws, False
        ws = sh.add_worksheet(title=title, rows=100, cols=max(10, len(SHEET_HEADERS)))
        ws.update("A1", [SHEET_HEADERS])
        return ws, True
    except Exception as e:
        raise RuntimeError(f"ensure_worksheet error: {e}")


def append_row(sh, title: str, row: list):
    ws, _ = ensure_worksheet(sh, title)
    ws.append_row(row, value_input_option="RAW")


# -------------------- УТИЛИТЫ --------------------
def human_readable_weights(w: Dict[str, int]) -> str:
    return f"JPY {w.get('JPY', 0)} / AUD {w.get('AUD', 0)} / EUR {w.get('EUR', 0)} / GBP {w.get('GBP', 0)}"


def split_total_by_weights(total: float, weights: Dict[str, int]) -> Dict[str, float]:
    s = max(sum(weights.values()), 1)
    return {
        "USDJPY": round(total * weights.get("JPY", 0) / s, 2),
        "AUDUSD": round(total * weights.get("AUD", 0) / s, 2),
        "EURUSD": round(total * weights.get("EUR", 0) / s, 2),
        "GBPUSD": round(total * weights.get("GBP", 0) / s, 2),
    }


def _fmt_tdelta_human(dt_to: datetime, now: Optional[datetime]=None) -> str:
    """Вернёт 'через 1 ч 05 мин' или '2 ч 17 мин назад' для локального текста."""
    now = now or datetime.now(timezone.utc)
    sec = int((dt_to - now).total_seconds())
    sign = "через" if sec >= 0 else "назад"
    sec = abs(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    if h and m:
        return f"{'через ' if sign=='через' else ''}{h} ч {m:02d} мин" if sign=='через' else f"{h} ч {m:02d} мин назад"
    if h:
        return f"{'через ' if sign=='через' else ''}{h} ч" if sign=='через' else f"{h} ч назад"
    return f"{'через ' if sign=='через' else ''}{m} мин" if sign=='через' else f"{m} мин назад"


def assert_master_chat(update: Update) -> bool:
    if MASTER_CHAT_ID and update.effective_chat:
        return update.effective_chat.id == MASTER_CHAT_ID
    return True


# -------------------- КОМАНДЫ --------------------
HELP_TEXT = (
    "Что я умею\n"
    "/settotal 2800 — задать общий банк (только в мастер-чате).\n"
    "/setweights jpy=40 aud=25 eur=20 gbp=15 — выставить целевые веса.\n"
    "/weights — показать целевые веса.\n"
    "/alloc — расчёт сумм и готовые команды /setbank для торговых чатов.\n"
    "/digest — утренний «инвесторский» дайджест (человеческий язык + события).\n"
    "/digest pro — краткий «трейдерский» дайджест (по цифрам, LLM).\n"
    "/init_sheet — создать/проверить лист в Google Sheets.\n"
    "/sheet_test — записать тестовую строку в лист.\n"
    "/diag — диагностика LLM и Google Sheets.\n"
    "/ping — проверить связь."
)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Привет! Я фунд-бот.\nТекущий чат id: <code>{update.effective_chat.id}</code>\n\nКоманды: /help",
        parse_mode=ParseMode.HTML,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)


async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")


async def cmd_settotal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not assert_master_chat(update):
        await update.message.reply_text("Эта команда доступна только в мастер-чате.")
        return
    try:
        parts = update.message.text.strip().split()
        if len(parts) < 2:
            raise ValueError
        total = float(parts[1])
    except Exception:
        await update.message.reply_text("Пример: /settotal 2800")
        return

    STATE["total"] = total
    await update.message.reply_text(f"OK. Общий банк = {total:.2f} USDT.\nИспользуйте /alloc для расчёта по чатам.")


async def cmd_setweights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not assert_master_chat(update):
        await update.message.reply_text("Эта команда доступна только в мастер-чате.")
        return

    text = update.message.text[len("/setweights"):].strip().lower()
    new_w = STATE["weights"].copy()
    try:
        for token in text.split():
            if "=" not in token:
                continue
            k, v = token.split("=", 1)
            k = k.strip().upper()
            v = int(v.strip())
            if k in ("JPY", "AUD", "EUR", "GBP"):
                new_w[k] = v
    except Exception:
        await update.message.reply_text("Пример: /setweights jpy=40 aud=25 eur=20 gbp=15")
        return

    STATE["weights"] = new_w
    await update.message.reply_text(
        f"Целевые веса обновлены: {human_readable_weights(new_w)}"
    )


async def cmd_weights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Целевые веса: {human_readable_weights(STATE['weights'])}")


async def cmd_alloc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total = float(STATE["total"])
    if total <= 0:
        await update.message.reply_text("Сначала задайте общий банк: /settotal 2800")
        return

    w = STATE["weights"]
    alloc = split_total_by_weights(total, w)

    lines = [
        f"Целевые веса: {human_readable_weights(w)}",
        "",
        "Распределение:",
    ]
    for sym in SYMBOLS:
        lines.append(f"{sym} → {alloc[sym]} USDT → команда в чат {sym}: /setbank {alloc[sym]}")
    msg = "\n".join(lines)
    await update.message.reply_text(msg)

    # лог в таблицу
    sh, _src = build_sheets_client(SHEET_ID)
    if sh:
        try:
            append_row(
                sh,
                SHEET_WS,
                [
                    datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    str(update.effective_chat.id),
                    "alloc",
                    f"{total:.2f}",
                    json.dumps(w, ensure_ascii=False),
                    json.dumps(alloc, ensure_ascii=False),
                ],
            )
        except Exception as e:
            log.warning("append_row alloc failed: %s", e)


async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # "pro" → краткий/профи вариант через LLM
    args = (update.message.text or "").split()
    pro = len(args) > 1 and args[1].lower() == "pro"

    if pro:
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            txt = await generate_digest(
                symbols=SYMBOLS,
                model=LLM_MINI,
                token_budget=LLM_TOKEN_BUDGET_PER_DAY,
            )
            await update.message.reply_text(txt)
        except Exception as e:
            await update.message.reply_text(f"LLM ошибка: {e}")
        return

    # Инвесторский режим (локально)
    sh, _src = build_sheets_client(SHEET_ID)
    if not sh:
        await update.message.reply_text("Sheets недоступен: не могу собрать инвесторский дайджест.")
        return

    try:
        msg = build_investor_digest(sh)
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"Ошибка сборки дайджеста: {e}")


def sheets_diag_text() -> str:
    sid_state = "set" if SHEET_ID else "empty"
    b64_len = len(_env("GOOGLE_CREDENTIALS_JSON_B64"))
    raw_json_len = len(_env("GOOGLE_CREDENTIALS_JSON")) or len(_env("GOOGLE_CREDENTIALS"))

    if not _GSHEETS_AVAILABLE:
        return f"Sheets: ❌ (libs not installed, SID={sid_state}, b64_len={b64_len}, raw_len={raw_json_len})"

    sh, src = build_sheets_client(SHEET_ID)
    if sh is None:
        return f"Sheets: ❌ (SID={sid_state}, source={src}, b64_len={b64_len}, raw_len={raw_json_len})"
    else:
        try:
            ws, created = ensure_worksheet(sh, SHEET_WS)
            mark = "created" if created else "exists"
            return f"Sheets: ✅ ok (SID={sid_state}, {src}, ws={ws.title}:{mark})"
        except Exception as e:
            return f"Sheets: ❌ (open ok, ws error: {e})"


async def cmd_diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        ok = await llm_ping()
        llm_line = "LLM: ✅ ok" if ok else "LLM: ❌ no key"
    except Exception:
        llm_line = "LLM: ❌ error"

    sheets_line = sheets_diag_text()
    await update.message.reply_text(f"{llm_line}\n{sheets_line}")


# ---- новые команды для листа ----
async def cmd_init_sheet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not SHEET_ID:
        await update.message.reply_text("SHEET_ID не задан.")
        return
    sh, src = build_sheets_client(SHEET_ID)
    if not sh:
        await update.message.reply_text(f"Sheets: ❌ {src}")
        return
    try:
        ws, created = ensure_worksheet(sh, SHEET_WS)
        await update.message.reply_text(
            f"Sheets: ✅ ws='{ws.title}' {'создан' if created else 'уже есть'} ({src})"
        )
    except Exception as e:
        await update.message.reply_text(f"Sheets: ❌ ошибка создания листа: {e}")


async def cmd_sheet_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sh, src = build_sheets_client(SHEET_ID)
    if not sh:
        await update.message.reply_text(f"Sheets: ❌ {src}")
        return
    try:
        append_row(
            sh,
            SHEET_WS,
            [
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                str(update.effective_chat.id),
                "test",
                f"{STATE['total']:.2f}",
                json.dumps(STATE['weights'], ensure_ascii=False),
                "manual /sheet_test",
            ],
        )
        await update.message.reply_text("Sheets: ✅ записано (test row).")
    except Exception as e:
        await update.message.reply_text(f"Sheets: ❌ ошибка записи: {e}")


# ---------- Digest helpers (инвесторский режим) ----------

# --- RU-дата ---
_RU_WD = ["понедельник","вторник","среда","четверг","пятница","суббота","воскресенье"]
_RU_MM = ["января","февраля","марта","апреля","мая","июня","июля","августа","сентября","октября","ноября","декабря"]

def header_ru(dt) -> str:
    wd = _RU_WD[dt.weekday()]
    mm = _RU_MM[dt.month - 1]
    return f"🧭 Утренний фон — {wd}, {dt.day} {mm} {dt.year}, {dt:%H:%M} (Europe/Belgrade)"


def _to_float(x, default=0.0) -> float:
    try:
        s = str(x).strip().replace(",", ".")
        return float(s)
    except Exception:
        return default


def _last_record(ws):
    try:
        rows = ws.get_all_records()
        return rows[-1] if rows else None
    except Exception:
        return None


def get_last_nonempty_row(sh, symbol: str, needed_fields=("Avg_Price","Next_DCA_Price","Bank_Target_USDT","Bank_Fact_USDT")) -> Optional[dict]:
    sheet_name = BMR_SHEETS.get(symbol)
    if not sheet_name:
        return None
    try:
        ws = sh.worksheet(sheet_name)
        rows = ws.get_all_records()
        if not rows:
            return None
        for r in reversed(rows):
            for f in needed_fields:
                v = r.get(f)
                if v not in (None, "", 0, "0", "0.0"):
                    return r
        return rows[-1]
    except Exception:
        return None


def latest_bank_target_fact(sh, symbol: str) -> tuple[Optional[float], Optional[float]]:
    sheet_name = BMR_SHEETS.get(symbol)
    if not sheet_name:
        return None, None
    try:
        ws = sh.worksheet(sheet_name)
        rows = ws.get_all_records()
        if not rows:
            return None, None
        tgt = fac = None
        for r in reversed(rows):
            if tgt is None:
                tv = r.get("Bank_Target_USDT")
                if tv not in (None, "", 0, "0", "0.0"):
                    tgt = _to_float(tv, None)
            if fac is None:
                fv = r.get("Bank_Fact_USDT")
                if fv not in (None, "", 0, "0", "0.0"):
                    fac = _to_float(fv, None)
            if tgt is not None and fac is not None:
                break
        return tgt, fac
    except Exception:
        return None, None


def price_fmt(symbol: str, value: Optional[float]) -> str:
    if value is None:
        return "—"
    is_jpy = symbol.endswith("JPY")
    prec = 3 if is_jpy else 5
    return f"{value:.{prec}f}"


def map_fa_level(risk: str) -> str:
    r = (risk or "").strip().lower()
    if r.startswith("red"):
        return "HIGH"
    if r.startswith("yellow") or r.startswith("amber"):
        return "CAUTION"
    return "OK"


def map_fa_bias(bias: str) -> str:
    b = (bias or "").strip().lower()
    if b.startswith("long"):
        return "LONG"
    if b.startswith("short"):
        return "SHORT"
    return "BOTH"


def policy_from_level(level: str) -> Dict[str, object]:
    L = (level or "OK").upper()
    if L == "HIGH":
        return {"reserve_off": True,  "dca_scale": 0.50, "icon": "🚧", "label": "высокий риск"}
    if L == "CAUTION":
        return {"reserve_off": False, "dca_scale": 0.75, "icon": "⚠️", "label": "умеренный риск"}
    return {"reserve_off": False, "dca_scale": 1.00, "icon": "✅", "label": "фон спокойный"}


def supertrend_dir(val: str) -> str:
    v = (val or "").strip().lower()
    return "up" if "up" in v else "down" if "down" in v else "flat"


def market_phrases(adx: float, st_dir: str, vol_z: float, atr1h: float) -> str:
    trend_txt = "умеренное" if adx < 20 else "заметное" if adx < 25 else "выраженное"
    dir_txt = "вверх" if st_dir == "up" else "вниз" if st_dir == "down" else "вбок"
    vola_txt = "ниже нормы" if atr1h < 0.8 else "около нормы" if atr1h < 1.2 else "выше нормы"
    noise_txt = "низкий" if vol_z < 0.5 else "умеренный" if vol_z < 1.5 else "повышенный"
    return f"{trend_txt} движение {dir_txt}; колебания {vola_txt}; рыночный шум {noise_txt}"


def importance_is_high(val) -> bool:
    if val is None:
        return False
    if isinstance(val, (int, float)):
        return val >= 3
    s = str(val).strip().lower()
    return "high" in s or s == "3"


def fetch_calendar_events_te(countries: List[str], d1: datetime, d2: datetime) -> List[dict]:
    if not _REQUESTS_AVAILABLE:
        return []
    try:
        path = "/calendar/country/" + ",".join(quote(c) for c in countries)
        url = TE_BASE + path
        params = {
            "d1": d1.strftime("%Y-%m-%dT%H:%M"),
            "d2": d2.strftime("%Y-%m-%dT%H:%M"),
            "importance": "3",  # High
            "c": f"{TE_CLIENT}:{TE_KEY}",
            "format": "json",
        }
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            return []
        events = []
        for it in data:
            dt_utc = None
            for k in ("DateUtc", "Date", "DateUTC"):
                val = it.get(k)
                if not val:
                    continue
                try:
                    s = str(val).replace(" ", "T")
                    dt_utc = datetime.fromisoformat(s)
                    if dt_utc.tzinfo is None:
                        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
                    else:
                        dt_utc = dt_utc.astimezone(timezone.utc)
                    break
                except Exception:
                    continue
            if not dt_utc:
                continue
            title = it.get("Event") or it.get("Title") or it.get("Category") or "Event"
            country = (it.get("Country") or "").strip()
            imp = it.get("Importance") or it.get("impact") or it.get("CategoryGroup")
            events.append({
                "utc": dt_utc,
                "country": country,
                "title": str(title),
                "importance": imp,
            })
        return events
    except Exception as e:
        log.warning("calendar fetch (TE) failed: %s", e)
        return []


def fetch_calendar_events_fmp(countries: List[str], d1: datetime, d2: datetime) -> List[dict]:
    """FMP fallback"""
    if not (_REQUESTS_AVAILABLE and FMP_API_KEY):
        return []
    try:
        url = "https://financialmodelingprep.com/api/v3/economic_calendar"
        params = { "from": d1.strftime("%Y-%m-%d"), "to": d2.strftime("%Y-%m-%d"), "apikey": FMP_API_KEY }
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list): return []
        want = [c.lower() for c in countries]
        events = []
        for it in data:
            ctry = (it.get("country") or "").lower()
            if not any(w in ctry for w in want): continue
            if "high" not in (it.get("impact") or "").lower(): continue
            val = it.get("date")
            if not val: continue
            try:
                dt_utc = datetime.fromisoformat(str(val).replace(" ", "T")).replace(tzinfo=timezone.utc)
            except Exception:
                continue
            events.append({
                "utc": dt_utc,
                "country": it.get("country") or "",
                "title": it.get("event") or "Event",
                "importance": "High",
            })
        return [e for e in events if d1 <= e["utc"] <= d2]
    except Exception as e:
        log.warning("calendar fetch (FMP) failed: %s", e)
        return []


def fetch_calendar_events_ff_all() -> list[dict]:
    """Скачиваем недельный JSON FF с кешом и защитой от 429."""
    import time
    now = int(time.time())

    # негативный кеш: если недавно был 429 — не трогаем источник
    if _FF_NEG["until"] and now < _FF_NEG["until"]:
        return _FF_CACHE["data"]  # отдаём что есть (может быть и пусто)

    # обычный кеш
    if _FF_CACHE["data"] and (now - _FF_CACHE["at"] < CAL_TTL_SEC):
        return _FF_CACHE["data"]

    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "fund-bot/1.0"})
        if r.status_code == 429:
            # тихий период 2 минуты (можно увеличить)
            _FF_NEG["until"] = now + 120
            log.warning("calendar fetch (FF) 429: backoff 120s")
            return _FF_CACHE["data"]
        r.raise_for_status()

        raw = r.json()
        data = []
        for it in raw or []:
            ts = it.get("timestamp")
            if not ts:
                continue
            dt_utc = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            data.append({
                "utc": dt_utc,
                "country": (it.get("country") or "").strip(),
                "title": it.get("title") or it.get("event") or "Event",
                "importance": it.get("impact") or "",
            })
        _FF_CACHE["data"] = data
        _FF_CACHE["at"] = now
        _FF_NEG["until"] = 0  # сбросить «тихий» режим
        return data
    except Exception as e:
        log.warning("calendar fetch (FF) failed: %s", e)
        return _FF_CACHE["data"]


def _filter_events_by(countries: list[str], d1: datetime, d2: datetime, events: list[dict]) -> list[dict]:
    want_codes = {FF_CODE_BY_COUNTRY.get(c.lower(), "").upper() for c in countries}
    want_codes.discard("")
    out = []
    for ev in events:
        if ev["country"].upper() in want_codes and importance_is_high(ev.get("importance")) and d1 <= ev["utc"] <= d2:
            out.append(ev)
    return out


def fetch_calendar_events(countries: List[str], d1: datetime, d2: datetime) -> List[dict]:
    prov = CAL_PROVIDER
    if prov == "te":
        return fetch_calendar_events_te(countries, d1, d2)
    if prov == "fmp":
        return fetch_calendar_events_fmp(countries, d1, d2)
    if prov == "ff":
        return _filter_events_by(countries, d1, d2, fetch_calendar_events_ff_all())

    # auto: TE→FMP→FF
    ev = fetch_calendar_events_te(countries, d1, d2)
    if not ev:
        ev = fetch_calendar_events_fmp(countries, d1, d2)
    if not ev:
        ev = _filter_events_by(countries, d1, d2, fetch_calendar_events_ff_all())
    return ev


def build_calendar_for_symbols(symbols: List[str], window_min: Optional[int] = None) -> Dict[str, dict]:
    """
    Для каждой пары:
      - events: High-события внутри локального окна ±window_min (по умолчанию CAL_WINDOW_MIN)
      - red_event_soon / quiet_*: как раньше
      - nearest_prev: последний High (в пределах ~вчера→завтра), если нет событий в окне
      - nearest_next: ближайший следующий High (то же), если нет событий в окне
    """
    now = datetime.now(timezone.utc)
    w = window_min if window_min is not None else CAL_WINDOW_MIN
    d1 = now - timedelta(minutes=w)
    d2 = now + timedelta(minutes=w)

    if LOCAL_TZ:
        now_loc = now.astimezone(LOCAL_TZ)
        day_start_loc = now_loc.replace(hour=0, minute=0, second=0, microsecond=0)
        d1_ext = day_start_loc - timedelta(days=1)
        d2_ext = day_start_loc + timedelta(days=2)
        d1_ext = d1_ext.astimezone(timezone.utc)
        d2_ext = d2_ext.astimezone(timezone.utc)
    else:
        d1_ext = now - timedelta(hours=36)
        d2_ext = now + timedelta(hours=36)

    out: Dict[str, dict] = {}
    
    # FF-провайдер работает с единым кешем, так что один вызов на все символы
    raw_all = fetch_calendar_events_ff_all() if CAL_PROVIDER == 'ff' else None
    
    FF_CODE_BY_COUNTRY = { "united states": "USD", "japan": "JPY", "euro area": "EUR", "united kingdom": "GBP", "australia": "AUD" }

    for sym in symbols:
        countries = PAIR_COUNTRIES.get(sym, [])
        
        # Если не FF, делаем запрос для каждой группы стран
        if raw_all is None:
            sym_raw_all = fetch_calendar_events(countries, d1_ext, d2_ext)
        else: # Фильтруем общий список
            want_codes = {FF_CODE_BY_COUNTRY.get(c.lower(), "").upper() for c in countries}
            want_codes.discard("")
            sym_raw_all = [ev for ev in raw_all if ev["country"].upper() in want_codes]

        around = []
        red_soon = False
        quiet_now = False

        for ev in sym_raw_all:
            if not importance_is_high(ev.get("importance")):
                continue
            t_utc = ev["utc"]
            if d1 <= t_utc <= d2:
                t_local = t_utc.astimezone(LOCAL_TZ) if LOCAL_TZ else t_utc
                around.append({**ev, "local": t_local})
                if abs((t_utc - now).total_seconds())/60.0 <= 60:
                    red_soon = True
                start = t_utc - timedelta(minutes=QUIET_BEFORE_MIN)
                end   = t_utc + timedelta(minutes=QUIET_AFTER_MIN)
                if start <= now <= end:
                    quiet_now = True

        around.sort(key=lambda x: x["utc"])

        nearest_prev = None
        nearest_next = None
        if not around:
            past = [ev for ev in sym_raw_all if importance_is_high(ev.get("importance")) and ev["utc"] < now]
            futr = [ev for ev in sym_raw_all if importance_is_high(ev.get("importance")) and ev["utc"] >= now]
            if past:
                p = max(past, key=lambda e: e["utc"])
                nearest_prev = {**p, "local": p["utc"].astimezone(LOCAL_TZ) if LOCAL_TZ else p["utc"]}
            if futr:
                n = min(futr, key=lambda e: e["utc"])
                nearest_next = {**n, "local": n["utc"].astimezone(LOCAL_TZ) if LOCAL_TZ else n["utc"]}

        quiet_from_to = (QUIET_BEFORE_MIN, QUIET_AFTER_MIN) if around else (0, 0)
        out[sym] = {
            "events": around,
            "red_event_soon": red_soon,
            "quiet_from_to": quiet_from_to,
            "quiet_now": quiet_now,
            "nearest_prev": nearest_prev,
            "nearest_next": nearest_next,
        }
    return out


def probability_against(side: str, fa_bias: str, adx: float, st_dir: str,
                        vol_z: float, atr1h: float, rsi: float, red_event_soon: bool) -> int:
    P = 55
    against_dir = "down" if (side or "").upper() == "LONG" else "up"

    if (fa_bias == "LONG" and against_dir == "up") or (fa_bias == "SHORT" and against_dir == "down"):
        P += 10
    elif fa_bias in ("LONG", "SHORT"):
        P -= 15

    if adx >= 25: P += 6
    elif adx >= 20: P += 3
    else: P -= 5

    if st_dir == against_dir: P += 5
    else: P -= 5

    if vol_z < 0.3: P -= 3
    elif vol_z < 1.5: P += 3
    elif vol_z > 2.5: P -= 7

    if atr1h > 1.2: P += 4
    elif atr1h < 0.8: P -= 4
    else: P += 1

    side_up = (side or "").upper() == "SHORT"
    if side_up:
        if rsi > 65: P += 3
        if rsi < 35: P -= 4
    else:
        if rsi < 35: P += 3
        if rsi > 65: P -= 4

    if red_event_soon: P -= 7

    P = max(35, min(75, int(round(P))))
    return P


def action_text(P: int, quiet_now: bool, level: str) -> str:
    if level == "HIGH":
        if P >= 64: return "готовьтесь добирать 1 шаг (после снятия ограничений и вне тихого окна)"
    if P >= 64: return "готовьтесь добирать 2 шага" + (" (вне тихого окна)" if quiet_now else "")
    if P >= 58: return "готовьтесь добирать 1 шаг" + (" (вне тихого окна)" if quiet_now else "")
    if P >= 50: return "базовый план"
    return "дополнительные доборы не приоритетны"


def delta_marker(target: float, fact: float) -> str:
    if target <= 0: return "—"
    delta_pct = (fact - target) / target
    ap = abs(delta_pct)
    if ap <= 0.02: return "✅"
    if ap <= 0.05: return f"⚠️ небольшое отклонение ({delta_pct:+.1%})"
    return f"🚧 существенное отклонение ({delta_pct:+.1%})"


def build_investor_digest(sh) -> str:
    now_utc = datetime.utcnow()
    header = header_ru(now_utc.astimezone(LOCAL_TZ)) if LOCAL_TZ else f"🧭 Утренний фон — {now_utc.strftime('%d %b %Y, %H:%M')} (UTC)"

    blocks: List[str] = [header]
    cal = build_calendar_for_symbols(SYMBOLS)

    for sym in SYMBOLS:
        row = get_last_nonempty_row(sh, sym) or {}
        side = (row.get("Side") or row.get("SIDE") or "").upper() or "LONG"

        avg_raw = row.get("Avg_Price")
        next_raw = row.get("Next_DCA_Price")

        adx = _to_float(row.get("ADX_5m"))
        rsi = _to_float(row.get("RSI_5m"), 50.0)
        volz = _to_float(row.get("Vol_z"))
        atr1h = _to_float(row.get("ATR_1h"), 1.0)
        st = supertrend_dir(row.get("Supertrend"))

        fa_level = map_fa_level(row.get("FA_Risk"))
        fa_bias = map_fa_bias(row.get("FA_Bias"))

        policy = policy_from_level(fa_level)
        reserve = "OFF" if policy["reserve_off"] else "ON"
        dca_scale = policy["dca_scale"]
        icon = policy["icon"]
        label = policy["label"]

        c = cal.get(sym, {})
        red_event_soon = bool(c.get("red_event_soon"))
        quiet_from, quiet_to = c.get("quiet_from_to", (0, 0))
        quiet_now = bool(c.get("quiet_now"))

        P = probability_against(side, fa_bias, adx, st, volz, atr1h, rsi, red_event_soon)
        act = action_text(P, quiet_now, fa_level)

        target, fact = latest_bank_target_fact(sh, sym)
        avg = None if avg_raw in (None, "", 0, "0", "0.0") else _to_float(avg_raw)
        next_dca = None if next_raw in (None, "", 0, "0", "0.0") else _to_float(next_raw)

        banks_line = "данных нет"
        if target is not None or fact is not None:
            tt = target or 0.0
            ff = fact or 0.0
            marker = delta_marker(tt, ff) if tt > 0 else "—"
            banks_line = f"Target **{tt:g}** / Fact **{ff:g}** — {marker}"

        pair_pretty = f"{sym[:3]}/{sym[3:]}"
        ev_line = ""
        events = c.get("events") or []
        if events:
            nowu = datetime.now(timezone.utc)
            nearest = min(events, key=lambda e: abs((e["utc"] - nowu).total_seconds()))
            tloc = nearest["local"]
            ev_line = f"\n• **Событие (Белград):** {tloc:%H:%M} — {nearest['country']}: {nearest['title']} (High)"
        else:
            prev_ev = c.get("nearest_prev")
            next_ev = c.get("nearest_next")
            if prev_ev:
                dtloc = prev_ev["local"]
                ev_line += f"\n• **Последний High:** {dtloc:%H:%M} — {prev_ev['country']}: {prev_ev['title']} ({_fmt_tdelta_human(prev_ev['utc'])})."
            if next_ev:
                dtloc = next_ev["local"]
                ev_line += f"\n• **Ближайший High:** {dtloc:%H:%M} — {next_ev['country']}: {next_ev['title']} ({_fmt_tdelta_human(next_ev['utc'])})."

        blocks.append(
f"""**{pair_pretty} — {icon} {label}, bias: {fa_bias}**
• **Фундаментально:** {'нейтрально' if fa_level=='OK' else ('умеренные риски' if fa_level=='CAUTION' else 'высокие риски')}.
• **Рынок сейчас:** {market_phrases(adx, st, volz, atr1h)}.
• **Наша позиция:** **{side}**, средняя {price_fmt(sym, avg)}; следующий добор {price_fmt(sym, next_dca)}.
• **Что делаем сейчас:** {"тихое окно не требуется" if not quiet_from and not quiet_to else f"тихое окно [{-quiet_from:+d};+{quiet_to:d}] мин"}; reserve **{reserve}**; dca_scale **{dca_scale:.2f}**.
• **Вероятность против позиции:** ≈ **{P}%** → {act}.
• **Цель vs факт:** {banks_line}{ev_line}"""
        )

    summary_lines: List[str] = []
    all_events = []
    for sym in SYMBOLS:
        for ev in cal.get(sym, {}).get("events", []) or []:
            all_events.append((ev["utc"], ev["local"], sym, ev["country"], ev["title"]))

    if not all_events:
        for sym in SYMBOLS:
            n = cal.get(sym, {}).get("nearest_next")
            if n: all_events.append((n["utc"], n["local"], sym, n["country"], n["title"]))

    all_events.sort(key=lambda x: x[0])

    if all_events:
        summary_lines.append("\n📅 **Ближайшие High-события (Белград):**")
        unique_events = []
        seen = set()
        for event in all_events:
            if event[0] not in seen:
                unique_events.append(event)
                seen.add(event[0])
        for _, tloc, sym, cty, title in unique_events[:8]:
            summary_lines.append(f"• {tloc:%H:%M} — {sym}: {cty}: {title}")

    return "\n\n".join(blocks + (["\n".join(summary_lines)] if summary_lines else []))


# -------------------- СТАРТ --------------------
async def _set_bot_commands(app: Application):
    cmds = [ BotCommand(c, d) for c, d in [
        ("start", "Запуск бота"), ("help", "Список команд"), ("ping", "Проверка связи"),
        ("settotal", "Задать общий банк (мастер-чат)"), ("setweights", "Задать целевые веса (мастер-чат)"),
        ("weights", "Показать целевые веса"), ("alloc", "Рассчитать распределение банка"),
        ("digest", "Утренний дайджест (investor) / pro (trader)"),
        ("init_sheet", "Создать/проверить лист в Google Sheets"),
        ("sheet_test", "Тестовая запись в лист"), ("diag", "Диагностика LLM и Sheets"),
    ]]
    try: await app.bot.set_my_commands(cmds)
    except Exception as e: log.warning("set_my_commands failed: %s", e)


async def morning_digest_scheduler(app: Application):
    import asyncio as _asyncio
    from datetime import datetime as _dt, timedelta as _td, time as _time
    while True:
        now = _dt.now(LOCAL_TZ) if LOCAL_TZ else _dt.utcnow()
        target_time = _time(MORNING_HOUR, MORNING_MINUTE, tzinfo=LOCAL_TZ)
        target = _dt.combine(now.date(), target_time)
        if now >= target: target += _td(days=1)
        wait_s = (target - now).total_seconds()
        await _asyncio.sleep(max(1.0, wait_s))
        try:
            sh, _ = build_sheets_client(SHEET_ID)
            if sh:
                msg = build_investor_digest(sh)
                await app.bot.send_message(chat_id=MASTER_CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)
            else:
                await app.bot.send_message(chat_id=MASTER_CHAT_ID, text="Sheets недоступен: утренний дайджест пропущен.")
        except Exception as e:
            try: await app.bot.send_message(chat_id=MASTER_CHAT_ID, text=f"Ошибка утреннего дайджеста: {e}")
            except Exception: pass


def build_application() -> Application:
    if not BOT_TOKEN: raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")
    builder = Application.builder().token(BOT_TOKEN)
    if _RATE_LIMITER_AVAILABLE: builder = builder.rate_limiter(AIORateLimiter())
    app = builder.build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("settotal", cmd_settotal))
    app.add_handler(CommandHandler("setweights", cmd_setweights))
    app.add_handler(CommandHandler("weights", cmd_weights))
    app.add_handler(CommandHandler("alloc", cmd_alloc))
    app.add_handler(CommandHandler("digest", cmd_digest))
    app.add_handler(CommandHandler("diag", cmd_diag))
    app.add_handler(CommandHandler("init_sheet", cmd_init_sheet))
    app.add_handler(CommandHandler("sheet_test", cmd_sheet_test))
    return app


async def main_async():
    log.info("Fund bot is running…")
    app = build_application()
    await _set_bot_commands(app)
    await app.initialize()
    await app.start()
    asyncio.create_task(morning_digest_scheduler(app))
    await app.updater.start_polling()
    await asyncio.Event().wait()


def main():
    try: asyncio.run(main_async())
    except KeyboardInterrupt: pass


if __name__ == "__main__":
    main()
