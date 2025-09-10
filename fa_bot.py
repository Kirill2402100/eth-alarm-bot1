# fa_bot.py
import os
import json
import base64
import logging
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

SHEET_ID = os.getenv("SHEET_ID", "").strip()
SHEET_WS = os.getenv("SHEET_WS", "FUND_BOT").strip() or "FUND_BOT"

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
CAL_WINDOW_MIN = int(os.getenv("CAL_WINDOW_MIN", "120") or "120")
QUIET_BEFORE_MIN = int(os.getenv("QUIET_BEFORE_MIN", "45"))
QUIET_AFTER_MIN  = int(os.getenv("QUIET_AFTER_MIN",  "45"))
CAL_PROVIDER = os.getenv("CAL_PROVIDER", "auto").lower()
FMP_API_KEY  = os.getenv("FMP_API_KEY", "").strip()
CAL_TTL_SEC = int(os.getenv("CAL_TTL_SEC", "600") or "600")

COUNTRY_BY_CCY = {
    "USD": "united states",
    "JPY": "japan",
    "EUR": "euro area",
    "GBP": "united kingdom",
    "AUD": "australia",
}
# FF country comes as currency codes; map to our canonical country names
FF_CODE2NAME = {
    "usd": "united states",
    "jpy": "japan",
    "eur": "euro area",
    "gbp": "united kingdom",
    "aud": "australia",
}
PAIR_COUNTRIES = {
    "USDJPY": [COUNTRY_BY_CCY["USD"], COUNTRY_BY_CCY["JPY"]],
    "AUDUSD": [COUNTRY_BY_CCY["AUD"], COUNTRY_BY_CCY["USD"]],
    "EURUSD": [COUNTRY_BY_CCY["EUR"], COUNTRY_BY_CCY["USD"]],
    "GBPUSD": [COUNTRY_BY_CCY["GBP"], COUNTRY_BY_CCY["USD"]],
}

# -------------------- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ --------------------
_FF_CACHE = {"at": 0, "data": []}
_FF_NEG   = {"until": 0}

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
    try:
        for ws in sh.worksheets():
            if ws.title == title:
                return ws, False
        ws = sh.add_worksheet(title=title, rows=200, cols=max(10, len(SHEET_HEADERS)))
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
    now = now or datetime.now(timezone.utc)
    sec = int((dt_to - now).total_seconds())
    sign_is_future = sec >= 0
    sec = abs(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    if h and m:
        return (f"через {h} ч {m:02d} мин" if sign_is_future else f"{h} ч {m:02d} мин назад")
    if h:
        return ("через " if sign_is_future else "") + f"{h} ч" + ("" if sign_is_future else " назад")
    return ("через " if sign_is_future else "") + f"{m} мин" + ("" if sign_is_future else " назад")


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
            if not txt or not str(txt).strip():
                txt = "⚠️ LLM не вернул текст."
            await update.message.reply_text(txt)
        except Exception as e:
            await update.message.reply_text(f"LLM ошибка: {e}")
        return

    sh, _src = build_sheets_client(SHEET_ID)
    if not sh:
        await update.message.reply_text("Sheets недоступен: не могу собрать инвесторский дайджест.")
        return

    try:
        msg = build_investor_digest(sh)
        if not msg.strip():
            msg = header_ru(datetime.now(LOCAL_TZ) if LOCAL_TZ else datetime.utcnow()) + "\n\n(событий High рядом нет)"
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
_RU_WD = ["понедельник","вторник","среда","четверг","пятница","суббота","воскресенье"]
_RU_MM = ["января","февраля","марта","апреля","мая","июня","июля","августа","сентября","октября","ноября","декабря"]

def header_ru(dt) -> str:
    wd = _RU_WD[dt.weekday()]
    mm = _RU_MM[dt.month - 1]
    return f"🧭 Утренний фон — {wd}, {dt.day} {mm} {dt.year}, {dt:%H:%M} (Europe/Belgrade)"


def _to_float(x, default=0.0) -> float:
    try:
        return float(str(x).strip().replace(",", "."))
    except Exception:
        return default


def get_last_nonempty_row(sh, symbol: str, needed_fields=("Avg_Price","Next_DCA_Price","Bank_Target_USDT","Bank_Fact_USDT")) -> Optional[dict]:
    sheet_name = BMR_SHEETS.get(symbol)
    if not sheet_name: return None
    try:
        ws = sh.worksheet(sheet_name)
        rows = ws.get_all_records()
        if not rows: return None
        for r in reversed(rows):
            if any(r.get(f) not in (None, "", 0, "0", "0.0") for f in needed_fields):
                return r
        return rows[-1]
    except Exception:
        return None


def latest_bank_target_fact(sh, symbol: str) -> tuple[Optional[float], Optional[float]]:
    sheet_name = BMR_SHEETS.get(symbol)
    if not sheet_name: return None, None
    try:
        ws = sh.worksheet(sheet_name)
        rows = ws.get_all_records()
        if not rows: return None, None
        tgt = fac = None
        for r in reversed(rows):
            if tgt is None and r.get("Bank_Target_USDT") not in (None, "", 0, "0", "0.0"):
                tgt = _to_float(r.get("Bank_Target_USDT"), None)
            if fac is None and r.get("Bank_Fact_USDT") not in (None, "", 0, "0", "0.0"):
                fac = _to_float(r.get("Bank_Fact_USDT"), None)
            if tgt is not None and fac is not None:
                break
        return tgt, fac
    except Exception:
        return None, None


def price_fmt(symbol: str, value: Optional[float]) -> str:
    if value is None: return "—"
    return f"{value:.{3 if symbol.endswith('JPY') else 5}f}"


def map_fa_level(risk: str) -> str:
    r = (risk or "").strip().lower()
    if r.startswith("red"): return "HIGH"
    if r.startswith("yellow") or r.startswith("amber"): return "CAUTION"
    return "OK"


def map_fa_bias(bias: str) -> str:
    b = (bias or "").strip().lower()
    if b.startswith("long"): return "LONG"
    if b.startswith("short"): return "SHORT"
    return "BOTH"


def policy_from_level(level: str) -> Dict[str, object]:
    L = (level or "OK").upper()
    if L == "HIGH": return {"reserve_off": True,  "dca_scale": 0.50, "icon": "🚧", "label": "высокий риск"}
    if L == "CAUTION": return {"reserve_off": False, "dca_scale": 0.75, "icon": "⚠️", "label": "умеренный риск"}
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
    if val is None: return False
    if isinstance(val, (int, float)): return val >= 3
    s = str(val).strip().lower()
    return "high" in s or s == "3"


# ---------- Источники календаря ----------
def fetch_calendar_events_te(countries: List[str], d1: datetime, d2: datetime) -> List[dict]:
    """TradingEconomics API (importance=3 — High)."""
    if not (_REQUESTS_AVAILABLE and TE_BASE):
        return []
    try:
        ctry = ",".join(quote(c) for c in countries)
        url = (
            f"{TE_BASE}/calendar/country/{ctry}"
            f"?d1={quote(d1.strftime('%Y-%m-%dT%H:%M'))}"
            f"&d2={quote(d2.strftime('%Y-%m-%dT%H:%M'))}"
            f"&importance=3&c={quote(f'{TE_CLIENT}:{TE_KEY}')}&format=json"
        )
        r = requests.get(url, timeout=15, headers={"User-Agent": "fund-bot/1.0"})
        r.raise_for_status()
        raw = r.json() or []
        out = []
        for it in raw:
            # в TE есть fields: Country, Event, Date (UTC), Importance
            dt_raw = it.get("Date")
            if not dt_raw:
                continue
            dt_utc = datetime.fromisoformat(dt_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
            out.append({
                "utc": dt_utc,
                "country": (it.get("Country") or "").strip().lower(),
                "title": it.get("Event") or "Event",
                "importance": it.get("Importance") or 3,
            })
        return out
    except Exception as e:
        log.warning("calendar fetch (TE) failed: %s", e)
        return []


def fetch_calendar_events_fmp(countries: List[str], d1: datetime, d2: datetime) -> List[dict]:
    """FinancialModelingPrep economic_calendar (High impact)."""
    if not (_REQUESTS_AVAILABLE and FMP_API_KEY):
        return []
    try:
        url = (
            "https://financialmodelingprep.com/api/v3/economic_calendar"
            f"?from={d1.date()}&to={d2.date()}&apikey={FMP_API_KEY}"
        )
        r = requests.get(url, timeout=15, headers={"User-Agent": "fund-bot/1.0"})
        r.raise_for_status()
        raw = r.json() or []
        out = []
        want = {c.lower() for c in countries}
        for it in raw:
            # FMP: fields 'country', 'event', 'date' (YYYY-MM-DD), 'impact' (e.g., "High")
            c_raw = (it.get("country") or "").strip().lower()
            if c_raw not in want:
                continue
            impact = str(it.get("impact") or "").lower()
            if "high" not in impact and it.get("importance") != 3:
                continue
            # время может быть как date или datetime; пробуем оба
            dt_s = (it.get("date") or it.get("datetime") or "").replace("Z", "+00:00")
            try:
                if "T" in dt_s:
                    dt_utc = datetime.fromisoformat(dt_s).astimezone(timezone.utc)
                else:
                    # если только дата — считаем полдень UTC, лишь бы попало в окно дня
                    dt_utc = datetime.fromisoformat(dt_s + "T12:00:00+00:00").astimezone(timezone.utc)
            except Exception:
                continue
            out.append({
                "utc": dt_utc,
                "country": c_raw,
                "title": it.get("event") or "Event",
                "importance": "High",
            })
        return out
    except Exception as e:
        log.warning("calendar fetch (FMP) failed: %s", e)
        return []


def fetch_calendar_events_ff_all() -> list[dict]:
    """Скачиваем недельный JSON FF с кешом и защитой от 429."""
    if not _REQUESTS_AVAILABLE:
        return _FF_CACHE["data"]
    now = int(datetime.now(timezone.utc).timestamp())

    if _FF_NEG["until"] and now < _FF_NEG["until"]:
        return _FF_CACHE["data"]

    if _FF_CACHE["data"] and (now - _FF_CACHE["at"] < CAL_TTL_SEC):
        return _FF_CACHE["data"]

    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "fund-bot/1.0"})
        if r.status_code == 429:
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
            country_raw = (it.get("country") or "").strip()
            country = FF_CODE2NAME.get(country_raw.lower(), country_raw).lower()
            data.append({
                "utc": dt_utc,
                "country": country,
                "title": it.get("title") or it.get("event") or "Event",
                "importance": it.get("impact") or "",
            })
        _FF_CACHE["data"] = data
        _FF_CACHE["at"] = now
        _FF_NEG["until"] = 0
        log.info("FF weekly: loaded %d events", len(data))
        return data
    except Exception as e:
        log.warning("calendar fetch (FF) failed: %s", e)
        return _FF_CACHE["data"] or []


def _filter_events_by(countries: list[str], d1: datetime, d2: datetime, events: list[dict]) -> list[dict]:
    want = {c.lower() for c in countries}
    out = []
    for ev in events:
        if ev["country"].lower() in want and importance_is_high(ev.get("importance")) and d1 <= ev["utc"] <= d2:
            out.append(ev)
    return out


def fetch_calendar_events(countries: List[str], d1: datetime, d2: datetime) -> List[dict]:
    prov = CAL_PROVIDER
    if prov == "te": return fetch_calendar_events_te(countries, d1, d2)
    if prov == "fmp": return fetch_calendar_events_fmp(countries, d1, d2)
    if prov == "ff": return _filter_events_by(countries, d1, d2, fetch_calendar_events_ff_all())

    # auto
    ev = fetch_calendar_events_te(countries, d1, d2)
    if not ev: ev = fetch_calendar_events_fmp(countries, d1, d2)
    if not ev: ev = _filter_events_by(countries, d1, d2, fetch_calendar_events_ff_all())
    return ev


def build_calendar_for_symbols(symbols: List[str], window_min: Optional[int] = None) -> Dict[str, dict]:
    now = datetime.now(timezone.utc)
    w = window_min if window_min is not None else CAL_WINDOW_MIN
    d1 = now - timedelta(minutes=w)
    d2 = now + timedelta(minutes=w)

    if LOCAL_TZ:
        now_loc = now.astimezone(LOCAL_TZ)
        day_start_loc = now_loc.replace(hour=0, minute=0, second=0, microsecond=0)
        d1_ext = day_start_loc - timedelta(days=1)
        d2_ext = day_start_loc + timedelta(days=2)
        d1_ext, d2_ext = d1_ext.astimezone(timezone.utc), d2_ext.astimezone(timezone.utc)
    else:
        d1_ext, d2_ext = now - timedelta(hours=36), now + timedelta(hours=36)

    out: Dict[str, dict] = {}
    all_raw_events = fetch_calendar_events_ff_all() if CAL_PROVIDER in ('ff', 'auto') else None

    for sym in symbols:
        countries = PAIR_COUNTRIES.get(sym, [])

        if all_raw_events is not None:
            want = {c.lower() for c in countries}
            sym_raw_all = [ev for ev in all_raw_events if ev["country"].lower() in want]
        else:
            sym_raw_all = fetch_calendar_events(countries, d1_ext, d2_ext)

        around = [
            {**ev, "local": ev["utc"].astimezone(LOCAL_TZ) if LOCAL_TZ else ev["utc"]}
            for ev in sym_raw_all
            if importance_is_high(ev.get("importance")) and d1 <= ev["utc"] <= d2
        ]
        around.sort(key=lambda x: x["utc"])

        red_soon = any(abs((ev["utc"] - now).total_seconds()) / 60.0 <= 60 for ev in around)
        quiet_now = any(
            (ev["utc"] - timedelta(minutes=QUIET_BEFORE_MIN)) <= now <= (ev["utc"] + timedelta(minutes=QUIET_AFTER_MIN))
            for ev in around
        )

        nearest_prev = nearest_next = None
        if not around:
            high_events = [ev for ev in sym_raw_all if importance_is_high(ev.get("importance"))]
            past = [ev for ev in high_events if ev["utc"] < now]
            futr = [ev for ev in high_events if ev["utc"] >= now]
            if past:
                p = max(past, key=lambda e: e["utc"])
                nearest_prev = {**p, "local": p["utc"].astimezone(LOCAL_TZ) if LOCAL_TZ else p["utc"]}
            if futr:
                n = min(futr, key=lambda e: e["utc"])
                nearest_next = {**n, "local": n["utc"].astimezone(LOCAL_TZ) if LOCAL_TZ else n["utc"]}

        out[sym] = {
            "events": around,
            "red_event_soon": red_soon,
            "quiet_from_to": (QUIET_BEFORE_MIN, QUIET_AFTER_MIN) if around else (0, 0),
            "quiet_now": quiet_now,
            "nearest_prev": nearest_prev,
            "nearest_next": nearest_next,
        }

    log.info("calendar: provider=%s, window=±%s min, utc=%s..%s",
             CAL_PROVIDER, w, d1.isoformat(timespec="minutes"), d2.isoformat(timespec="minutes"))
    for sym, pack in out.items():
        log.info("calendar[%s]: events=%d, red_soon=%s, quiet_now=%s",
                 sym, len(pack.get("events") or []), pack.get("red_event_soon"), pack.get("quiet_now"))
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

    side_up = (side or "").upper() == "SHORT"  # «против SHORT» = вверх
    if side_up:
        if rsi > 65: P += 3
        if rsi < 35: P -= 4
    else:
        if rsi < 35: P += 3
        if rsi > 65: P -= 4

    if red_event_soon: P -= 7

    return max(35, min(75, int(round(P))))


def action_text(P: int, quiet_now: bool, level: str) -> str:
    if level == "HIGH":
        if P >= 64:
            return "готовьтесь добирать 1 шаг (после снятия ограничений и вне тихого окна)"
    if P >= 64:
        return "готовьтесь добирать 2 шага" + (" (вне тихого окна)" if quiet_now else "")
    if P >= 58:
        return "готовьтесь добирать 1 шаг" + (" (вне тихого окна)" if quiet_now else "")
    if P >= 50:
        return "базовый план"
    return "дополнительные доборы не приоритетны"


def delta_marker(target: float, fact: float) -> str:
    if target <= 0:
        return "—"
    delta_pct = (fact - target) / target
    ap = abs(delta_pct)
    if ap <= 0.02:
        return "✅"
    if ap <= 0.05:
        return f"⚠️ небольшое отклонение ({delta_pct:+.1%})"
    return f"🚧 существенное отклонение ({delta_pct:+.1%})"


def build_investor_digest(sh) -> str:
    now_utc = datetime.now(timezone.utc)
    header = header_ru(now_utc.astimezone(LOCAL_TZ)) if LOCAL_TZ else f"🧭 Утренний фон — {now_utc.strftime('%d %b %Y, %H:%M')} (UTC)"

    blocks: List[str] = [header]
    cal = build_calendar_for_symbols(SYMBOLS)

    for sym in SYMBOLS:
        row = get_last_nonempty_row(sh, sym) or {}
        side = (row.get("Side") or row.get("SIDE") or "").upper() or "LONG"
        adx = _to_float(row.get("ADX_5m"))
        rsi = _to_float(row.get("RSI_5m"), 50.0)
        volz = _to_float(row.get("Vol_z"))
        atr1h = _to_float(row.get("ATR_1h"), 1.0)
        st = supertrend_dir(row.get("Supertrend"))
        fa_level = map_fa_level(row.get("FA_Risk"))
        fa_bias = map_fa_bias(row.get("FA_Bias"))
        policy = policy_from_level(fa_level)
        c = cal.get(sym, {})
        P = probability_against(side, fa_bias, adx, st, volz, atr1h, rsi, bool(c.get("red_event_soon")))
        act = action_text(P, bool(c.get("quiet_now")), fa_level)
        target, fact = latest_bank_target_fact(sh, sym)
        banks_line = "данных нет"
        if target is not None or fact is not None:
            tt, ff = target or 0.0, fact or 0.0
            banks_line = f"Target **{tt:g}** / Fact **{ff:g}** — {delta_marker(tt, ff) if tt > 0 else '—'}"

        ev_line = ""
        if events := c.get("events"):
            nearest = min(events, key=lambda e: abs((e["utc"] - now_utc).total_seconds()))
            tloc = nearest["local"]
            ev_line = f"\n• **Событие (Белград):** {tloc:%H:%M} — {nearest['country']}: {nearest['title']} (High)"
        else:
            if prev_ev := c.get("nearest_prev"):
                ev_line += f"\n• **Последний High:** {prev_ev['local']:%H:%M} — {prev_ev['country']}: {prev_ev['title']} ({_fmt_tdelta_human(prev_ev['utc'])})."
            if next_ev := c.get("nearest_next"):
                ev_line += f"\n• **Ближайший High:** {next_ev['local']:%H:%M} — {next_ev['country']}: {next_ev['title']} ({_fmt_tdelta_human(next_ev['utc'])})."

        q_from, q_to = c.get("quiet_from_to", (0, 0))
        blocks.append(
f"""**{sym[:3]}/{sym[3:]} — {policy['icon']} {policy['label']}, bias: {fa_bias}**
• **Фундаментально:** {'нейтрально' if fa_level=='OK' else ('умеренные риски' if fa_level=='CAUTION' else 'высокие риски')}.
• **Рынок сейчас:** {market_phrases(adx, st, volz, atr1h)}.
• **Наша позиция:** **{side}**, средняя {price_fmt(sym, _to_float(row.get("Avg_Price"), None))}; следующий добор {price_fmt(sym, _to_float(row.get("Next_DCA_Price"), None))}.
• **Что делаем сейчас:** {"тихое окно не требуется" if not q_from and not q_to else f"тихое окно [{-q_from:+d};+{q_to:d}] мин"}; reserve **{'OFF' if policy['reserve_off'] else 'ON'}**; dca_scale **{policy['dca_scale']:.2f}**.
• **Вероятность против позиции:** ≈ **{P}%** → {act}.
• **Цель vs факт:** {banks_line}{ev_line}"""
        )

    summary_lines, all_events = [], []
    for sym in SYMBOLS:
        all_events.extend((ev["utc"], ev["local"], sym, ev["country"], ev["title"]) for ev in cal.get(sym, {}).get("events", []))

    if not all_events:
        all_events.extend((n["utc"], n["local"], sym, n["country"], n["title"]) for sym in SYMBOLS if (n := cal.get(sym, {}).get("nearest_next")))

    if all_events:
        summary_lines.append("\n📅 **Ближайшие High-события (Белград):**")
        unique_events = {ev[0]: ev for ev in sorted(all_events)}
        for _, tloc, sym, cty, title in list(unique_events.values())[:8]:
            summary_lines.append(f"• {tloc:%H:%M} — {sym}: {cty}: {title}")

    # --- аккуратно склеиваем, чтобы не было пустого текста ---
    parts = blocks + (["\n".join(summary_lines)] if summary_lines else [])
    msg = "\n\n".join([p for p in parts if p and str(p).strip()])
    if not msg.strip():
        msg = header  # гарантируем непустой текст
    return msg


# -------------------- СТАРТ --------------------
async def _set_bot_commands(app: Application):
    cmds = [
        BotCommand("start", "Запуск бота"),
        BotCommand("help", "Список команд"),
        BotCommand("ping", "Проверка связи"),
        BotCommand("settotal", "Задать общий банк (мастер-чат)"),
        BotCommand("setweights", "Задать целевые веса (мастер-чат)"),
        BotCommand("weights", "Показать целевые веса"),
        BotCommand("alloc", "Рассчитать распределение банка"),
        BotCommand("digest", "Утренний дайджест (investor) / pro (trader)"),
        BotCommand("init_sheet", "Создать/проверить лист в Google Sheets"),
        BotCommand("sheet_test", "Тестовая запись в лист"),
        BotCommand("diag", "Диагностика LLM и Sheets"),
    ]
    try:
        await app.bot.set_my_commands(cmds)
    except Exception as e:
        log.warning("set_my_commands failed: %s", e)


async def morning_digest_scheduler(app: Application):
    from datetime import datetime as _dt, timedelta as _td, time as _time
    while True:
        now = _dt.now(LOCAL_TZ) if LOCAL_TZ else _dt.utcnow()
        target = _dt.combine(now.date(), _time(MORNING_HOUR, MORNING_MINUTE, tzinfo=LOCAL_TZ)) if LOCAL_TZ else now.replace(hour=MORNING_HOUR, minute=MORNING_MINUTE, second=0, microsecond=0)
        if now >= target:
            target = target + _td(days=1)
        await asyncio.sleep(max(1.0, (target - now).total_seconds()))
        try:
            sh, _src = build_sheets_client(SHEET_ID)
            if sh:
                msg = build_investor_digest(sh)
                await app.bot.send_message(chat_id=MASTER_CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)
            else:
                await app.bot.send_message(chat_id=MASTER_CHAT_ID, text="Sheets недоступен: утренний дайджест пропущен.")
        except Exception as e:
            try:
                await app.bot.send_message(chat_id=MASTER_CHAT_ID, text=f"Ошибка утреннего дайджеста: {e}")
            except Exception:
                pass


def build_application() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")
    builder: ApplicationBuilder = Application.builder().token(BOT_TOKEN)
    if 'AIORateLimiter' in globals() and AIORateLimiter:
        builder = builder.rate_limiter(AIORateLimiter())
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
    # PTB v20: у Application есть updater
    await app.updater.start_polling()
    await asyncio.Event().wait()


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
