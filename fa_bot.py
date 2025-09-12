# fa_bot.py
import os
import json
import base64
import logging
import re
from html import escape as _html_escape
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional, List

from telegram import Update, BotCommand
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# ============== Timezone & schedule ==============
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None

LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Europe/Belgrade")) if ZoneInfo else None
MORNING_HOUR = int(os.getenv("MORNING_HOUR", "9"))
MORNING_MINUTE = int(os.getenv("MORNING_MINUTE", "30"))

# ============== Optional rate limiter ==============
try:
    from telegram.ext import AIORateLimiter
    _RATE_LIMITER_AVAILABLE = True
except Exception:  # pragma: no cover
    AIORateLimiter = None
    _RATE_LIMITER_AVAILABLE = False

# ============== Google Sheets ==============
try:
    import gspread
    from google.oauth2 import service_account
    _GSHEETS_AVAILABLE = True
except Exception:
    gspread = None
    service_account = None
    _GSHEETS_AVAILABLE = False

# ============== HTTP for calendars ==============
try:
    import requests
    _REQUESTS_AVAILABLE = True
except Exception:
    requests = None
    _REQUESTS_AVAILABLE = False

# ============== LLM client (optional) ==============
try:
    from llm_client import generate_digest, llm_ping
except Exception:
    async def generate_digest(*args, **kwargs) -> str:
        return "⚠️ LLM сейчас недоступен (нет llm_client.py)."

    async def llm_ping() -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

# ============== Logging ==============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("fund_bot")

# ============== ENV ==============
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

# Sheet names for pairs (overridable via ENV)
BMR_SHEETS = {
    "USDJPY": os.getenv("BMR_SHEET_USDJPY", "BMR_DCA_USDJPY"),
    "AUDUSD": os.getenv("BMR_SHEET_AUDUSD", "BMR_DCA_AUDUSD"),
    "EURUSD": os.getenv("BMR_SHEET_EURUSD", "BMR_DCA_EURUSD"),
    "GBPUSD": os.getenv("BMR_SHEET_GBPUSD", "BMR_DCA_GBPUSD"),
}

# ============== Calendar config ==============
TE_BASE = os.getenv("TE_BASE", "https://api.tradingeconomics.com").rstrip("/")
TE_CLIENT = os.getenv("TE_CLIENT", "guest").strip()
TE_KEY = os.getenv("TE_KEY", "guest").strip()
CAL_WINDOW_MIN = int(os.getenv("CAL_WINDOW_MIN", "120"))
QUIET_BEFORE_MIN = int(os.getenv("QUIET_BEFORE_MIN", "45"))
QUIET_AFTER_MIN  = int(os.getenv("QUIET_AFTER_MIN",  "45"))
CAL_PROVIDER = os.getenv("CAL_PROVIDER", "auto").lower()  # te|fmp|ff|auto
FMP_API_KEY  = os.getenv("FMP_API_KEY", "").strip()
CAL_TTL_SEC = int(os.getenv("CAL_TTL_SEC", "600") or "600")
CAL_WS = os.getenv("CAL_WS", "CALENDAR").strip() or "CALENDAR"

COUNTRY_BY_CCY = {
    "USD": "united states",
    "JPY": "japan",
    "EUR": "euro area",
    "GBP": "united kingdom",
    "AUD": "australia",
}
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

# ============== Global state ==============
_FF_CACHE = {"at": 0, "data": []}
_FF_NEG   = {"until": 0}

STATE = {
    "total": 0.0,
    "weights": DEFAULT_WEIGHTS.copy(),
}

# ============== Google creds loader ==============
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


# ============== Sheets helpers ==============
SHEET_HEADERS = ["ts", "chat_id", "action", "total", "weights_json", "note"]


def ensure_worksheet(sh, title: str):
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


def read_calendar_rows_sheet(sh) -> List[dict]:
    try:
        ws = sh.worksheet(CAL_WS)
        vals = ws.get_all_records()
    except Exception:
        return []
    out = []
    for r in vals:
        try:
            utc = datetime.fromisoformat(str(r.get("utc_iso")).replace("Z","+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        out.append({
            "utc": utc,
            "country": str(r.get("country","")).strip().lower(),
            "title": str(r.get("title","")).strip(),
            "importance": str(r.get("impact","")).strip(),
        })
    return out

# ============== News (from sheet) ==============
NEWS_WS = os.getenv("NEWS_WS", "NEWS").strip() or "NEWS"
DIGEST_NEWS_LOOKBACK_MIN = int(os.getenv("DIGEST_NEWS_LOOKBACK_MIN", "720") or "720")
DIGEST_NEWS_ALLOWED_SOURCES = {
    s.strip().upper() for s in os.getenv(
        "DIGEST_NEWS_ALLOWED_SOURCES",
        "US_FED_PR,ECB_PR,BOE_PR,BOJ_PR,RBA_MR,US_TREASURY,JP_MOF_FX"
    ).split(",") if s.strip()
}
DIGEST_NEWS_KEYWORDS = re.compile(
    os.getenv(
        "DIGEST_NEWS_KEYWORDS",
        r"rate decision|monetary policy|policy|bank rate|statement|minutes|guidance|intervention|unscheduled|emergency"
    ),
    re.I
)

def read_news_rows(sh) -> List[dict]:
    try:
        ws = sh.worksheet(NEWS_WS)
        rows = ws.get_all_records()
    except Exception:
        return []
    out = []
    for r in rows:
        try:
            ts = datetime.fromisoformat(str(r.get("ts_utc")).replace("Z","+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        ctries = [c.strip().lower() for c in str(r.get("countries","")).split(",") if c.strip()]
        out.append({
            "ts": ts,
            "source": str(r.get("source","")).strip().upper(),
            "title":  str(r.get("title","")).strip(),
            "url":    str(r.get("url","")).strip(),
            "countries": set(ctries),
            "ccy":    str(r.get("ccy","")).strip().upper(),
            "tags":   str(r.get("tags","")).strip(),
            "importance": str(r.get("importance_guess","")).strip().lower(),
        })
    return out

def _ru_title_hint(title: str) -> str:
    t = title
    repl = {
        r"\bRate Decision\b": "решение по ставке",
        r"\bMonetary Policy\b": "денежно-кредитная политика",
        r"\bPolicy\b": "политика ЦБ",
        r"\bStatement\b": "заявление",
        r"\bMinutes\b": "протокол",
        r"\bGuidance\b": "прогноз/ориентир",
        r"\bIntervention\b": "интервенция",
        r"\bUnscheduled\b": "внеплановый",
        r"\bEmergency\b": "экстренный",
    }
    for pat, rep in repl.items():
        t = re.sub(pat, rep, t, flags=re.I)
    return t

def choose_top_news_for_symbol(symbol: str, news_rows: List[dict], now_utc: datetime) -> Optional[dict]:
    look_from = now_utc - timedelta(minutes=DIGEST_NEWS_LOOKBACK_MIN)
    countries = set(PAIR_COUNTRIES.get(symbol, []))
    cand = []
    for r in news_rows:
        if r["ts"] < look_from:
            continue
        if not (r["countries"] & countries):
            continue
        score = 0
        if r["source"] in DIGEST_NEWS_ALLOWED_SOURCES: score += 2
        if DIGEST_NEWS_KEYWORDS.search(r["title"] + " " + r["tags"]): score += 3
        if r["importance"] == "high": score += 1
        age_min = max(1, int((now_utc - r["ts"]).total_seconds() // 60))
        score += max(0, 3 - age_min//60)
        r["_score"] = score
        cand.append(r)
    if not cand:
        return None
    cand.sort(key=lambda x: (x["_score"], x["ts"]), reverse=True)
    best = cand[0]
    best_local = best["ts"].astimezone(LOCAL_TZ) if LOCAL_TZ else best["ts"]
    best["local_time_str"] = best_local.strftime("%H:%M")
    best["ru_title"] = _ru_title_hint(best["title"])
    return best

# ============== Utils ==============
def _h(x) -> str:
    return _html_escape(str(x), quote=True)

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


def _split_for_tg_html(msg: str, limit: int = 3500) -> List[str]:
    parts, cur = [], []
    cur_len = 0
    for para in msg.split("\n\n"):
        block = (para + "\n\n")
        if cur_len + len(block) > limit and cur:
            parts.append("".join(cur).rstrip())
            cur, cur_len = [], 0
        cur.append(block)
        cur_len += len(block)
    if cur:
        parts.append("".join(cur).rstrip())
    return parts


# ============== Commands ==============
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
        for chunk in _split_for_tg_html(msg):
            await context.bot.send_message(chat_id=update.effective_chat.id, text=chunk, parse_mode=ParseMode.HTML)
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


# ============== Digest helpers (investor mode) ==============
_RU_WD = ["понедельник","вторник","среда","четверг","пятница","суббота","воскресенье"]
_RU_MM = ["января","февраля","марта","апреля","мая","июня","июля","августа","сентября","октября","ноября","декабря"]

def header_ru(dt) -> str:
    wd = _RU_WD[dt.weekday()]
    mm = _RU_MM[dt.month - 1]
    return f"🧭 Утренний фон — {wd}, {dt.day} {mm} {dt.year}, {dt:%H:%M} (Europe/Belgrade)"


def _to_float(x, default: Optional[float] = 0.0) -> Optional[float]:
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

    return max(35, min(75, int(round(P))))


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


def fetch_calendar_events_te(countries: List[str], d1: datetime, d2: datetime) -> List[dict]:
    # Placeholder for TradingEconomics integration
    return []


def fetch_calendar_events_fmp(countries: List[str], d1: datetime, d2: datetime) -> List[dict]:
    # Placeholder for FMP integration
    return []


def fetch_calendar_events_ff_all() -> List[dict]:
    if not _REQUESTS_AVAILABLE:
        return _FF_CACHE["data"]
    import time
    now = int(time.time())

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
            if not ts: continue
            dt_utc = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            country_raw = (it.get("country") or "").strip()
            country = FF_CODE2NAME.get(country_raw.lower(), country_raw)
            data.append({
                "utc": dt_utc,
                "country": country,
                "title": it.get("title") or it.get("event") or "Event",
                "importance": it.get("impact") or "",
            })
        _FF_CACHE["data"] = data
        _FF_CACHE["at"] = now
        _FF_NEG["until"] = 0
        return data
    except Exception as e:
        log.warning("calendar fetch (FF) failed: %s", e)
        return _FF_CACHE["data"] or []


def _filter_events_by(countries: List[str], d1: datetime, d2: datetime, events: List[dict]) -> List[dict]:
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

    if (all_raw_events is not None) and not all_raw_events:
        sh, _ = build_sheets_client(SHEET_ID)
        if sh:
            all_raw_events = read_calendar_rows_sheet(sh)

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


def build_investor_digest(sh) -> str:
    now_utc = datetime.now(timezone.utc)
    header = header_ru(now_utc.astimezone(LOCAL_TZ)) if LOCAL_TZ else f"🧭 Утренний фон — {now_utc.strftime('%d %b %Y, %H:%M')} (UTC)"

    blocks: List[str] = [header]
    cal = build_calendar_for_symbols(SYMBOLS)
    news_rows = read_news_rows(sh)

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
            banks_line = f"Target <b>{tt:g}</b> / Fact <b>{ff:g}</b> — {delta_marker(tt, ff) if tt > 0 else '—'}"

        ev_line = ""
        if events := c.get("events"):
            nearest = min(events, key=lambda e: abs((e["utc"] - now_utc).total_seconds()))
            tloc = nearest["local"]
            ev_line = f"\n• <b>Событие (Белград):</b> {tloc:%H:%M} — {_h(nearest['country'])}: {_h(nearest['title'])} (High)"
        else:
            if prev_ev := c.get("nearest_prev"):
                ev_line += f"\n• <b>Последний High:</b> {prev_ev['local']:%H:%M} — {_h(prev_ev['country'])}: {_h(prev_ev['title'])} ({_fmt_tdelta_human(prev_ev['utc'])})."
            if next_ev := c.get("nearest_next"):
                ev_line += f"\n• <b>Ближайший High:</b> {next_ev['local']:%H:%M} — {_h(next_ev['country'])}: {_h(next_ev['title'])} ({_fmt_tdelta_human(next_ev['utc'])})."

        best_news = choose_top_news_for_symbol(sym, news_rows, now_utc)
        news_line = ""
        if best_news:
            news_line = f"\n• Топ-новость: {_h(best_news['ru_title'])} ({best_news['local_time_str']}) — <a href=\"{_h(best_news['url'])}\">источник</a>"

        q_from, q_to = c.get("quiet_from_to", (0, 0))
        blocks.append(
f"""<b>{sym[:3]}/{sym[3:]} — {policy['icon']} {policy['label']}, bias: {fa_bias}</b>
• <b>Фундаментально:</b> {'нейтрально' if fa_level=='OK' else ('умеренные риски' if fa_level=='CAUTION' else 'высокие риски')}.
• <b>Рынок сейчас:</b> {market_phrases(adx, st, volz, atr1h)}.
• <b>Наша позиция:</b> <b>{side}</b>, средняя {price_fmt(sym, _to_float(row.get("Avg_Price"), None))}; следующий добор {price_fmt(sym, _to_float(row.get("Next_DCA_Price"), None))}.
• <b>Что делаем сейчас:</b> {"тихое окно не требуется" if not q_from and not q_to else f"тихое окно [{-q_from:+d};+{q_to:d}] мин"}; reserve <b>{'OFF' if policy['reserve_off'] else 'ON'}</b>; dca_scale <b>{policy['dca_scale']:.2f}</b>.
• <b>Вероятность против позиции:</b> ≈ <b>{P}%</b> → {act}.
• <b>Цель vs факт:</b> {banks_line}{ev_line}{news_line}"""
        )

    summary_lines, all_events = [], []
    for sym in SYMBOLS:
        all_events.extend((ev["utc"], ev["local"], sym, ev["country"], ev["title"]) for ev in cal.get(sym, {}).get("events", []))

    if not all_events:
        all_events.extend((n["utc"], n["local"], sym, n["country"], n["title"]) for sym in SYMBOLS if (n := cal.get(sym, {}).get("nearest_next")))

    if all_events:
        summary_lines.append("\n📅 <b>Ближайшие High-события (Белград):</b>")
        unique_events = {ev[0]: ev for ev in sorted(all_events)}
        for _, tloc, sym, cty, title in list(unique_events.values())[:8]:
            summary_lines.append(f"• {tloc:%H:%M} — {sym}: {_h(cty)}: {_h(title)}")

    return "\n\n".join(blocks + ["\n".join(summary_lines)] if summary_lines else blocks)


# ============== Startup ==============
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
    import asyncio as _asyncio
    while True:
        if LOCAL_TZ:
            now = _dt.now(LOCAL_TZ)
            target_time = _time(MORNING_HOUR, MORNING_MINUTE, tzinfo=LOCAL_TZ)
        else:
            now = _dt.now(timezone.utc)
            target_time = _time(MORNING_HOUR, MORNING_MINUTE, tzinfo=timezone.utc)
        target = _dt.combine(now.date(), target_time)
        if now >= target:
            target += _td(days=1)
        wait_s = (target - now).total_seconds()
        await _asyncio.sleep(max(1.0, wait_s))
        try:
            sh, _ = build_sheets_client(SHEET_ID)
            if sh:
                msg = build_investor_digest(sh)
                for chunk in _split_for_tg_html(msg):
                    await app.bot.send_message(chat_id=MASTER_CHAT_ID, text=chunk, parse_mode=ParseMode.HTML)
            else:
                await app.bot.send_message(chat_id=MASTER_CHAT_ID, text="Sheets недоступен: утренний дайджест пропущен.")
        except Exception as e:
            try:
                await app.bot.send_message(chat_id=MASTER_CHAT_ID, text=f"Ошибка утреннего дайджеста: {e}")
            except Exception:
                pass


async def _post_init(app: Application):
    await _set_bot_commands(app)
    app.create_task(morning_digest_scheduler(app))


def build_application() -> Application:
    if not BOT_TOKEN: raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")
    builder = Application.builder().token(BOT_TOKEN)
    if _RATE_LIMITER_AVAILABLE: builder = builder.rate_limiter(AIORateLimiter())
    builder = builder.post_init(_post_init)
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


def main():
    log.info("Fund bot is running…")
    app = build_application()
    app.run_polling()


if __name__ == "__main__":
    main()
