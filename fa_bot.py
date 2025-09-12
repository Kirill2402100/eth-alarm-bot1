# fa_bot.py
import os
import json
import base64
import logging
import re
from html import escape as _html_escape
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional, List

import asyncio

from telegram import Update, BotCommand
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    Application,
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
    _REQUESTS_AVAILABLE = True
except Exception:
    requests = None
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
CAL_WINDOW_MIN = int(os.getenv("CAL_WINDOW_MIN", "120"))
QUIET_BEFORE_MIN = int(os.getenv("QUIET_BEFORE_MIN", "45"))
QUIET_AFTER_MIN  = int(os.getenv("QUIET_AFTER_MIN",  "45"))
CAL_PROVIDER = os.getenv("CAL_PROVIDER", "auto").lower()
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

# ---------- News (READ FROM SHEET) ----------
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
        r"\bRate Decision\b": "Решение по ставке",
        r"\bMonetary Policy\b": "Монетарная политика",
        r"\bPolicy\b": "Политика ЦБ",
        r"\bStatement\b": "Заявление",
        r"\bMinutes\b": "Протокол",
        r"\bGuidance\b": "Ориентиры",
        r"\bIntervention\b": "Интервенция",
        r"\bUnscheduled\b": "Внеплановое",
        r"\bEmergency\b": "Экстренное",
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

# -------------------- УТИЛИТЫ --------------------
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


def _split_for_tg_plain(msg: str, limit: int = 3500) -> List[str]:
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


# ---------- Digest helpers ----------
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
    return {"reserve_off": False, "dca_scale": 1.00, "icon": "✅", "label": "спокойно"}

def supertrend_dir(val: str) -> str:
    v = (val or "").strip().lower()
    return "up" if "up" in v else "down" if "down" in v else "flat"

def importance_is_high(val) -> bool:
    if val is None: return False
    if isinstance(val, (int, float)): return val >= 3
    s = str(val).strip().lower()
    return "high" in s or s == "3"

def fetch_calendar_events_te(countries: List[str], d1: datetime, d2: datetime) -> List[dict]:
    # плейсхолдер
    return []

def fetch_calendar_events_fmp(countries: List[str], d1: datetime, d2: datetime) -> List[dict]:
    # плейсхолдер
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

    return out

# ======== ПЛЕЙНТЕКСТ ФОРМАТ =========

def _mk_market_summary(adx: float, st_dir: str, vol_z: float, atr1h: float, next_cty: Optional[str]) -> str:
    if adx is None: adx = 20.0
    if vol_z is None: vol_z = 1.0
    if atr1h is None: atr1h = 1.0
    # Направление
    if adx < 18:
        base = "движения ровные"
    else:
        if st_dir == "up":
            base = "лёгкий подъём" if adx < 25 else "подъём заметнее обычного"
        elif st_dir == "down":
            base = "небольшой спуск" if adx < 25 else "спуск заметнее обычного"
        else:
            base = "движения ровные"
    # Волатильность
    vola = "волатильность ниже нормы" if atr1h < 0.9 else ("волатильность около нормы" if atr1h < 1.2 else "волатильность выше нормы")
    # Хвост про США
    tail = ""
    if next_cty and next_cty.lower() == "united states":
        tail = ", резких скачков не ждём до США"
    return f"{base}, {vola}{tail}."

def _speed_text(dca_scale: float) -> str:
    if dca_scale >= 0.99: return "обычная"
    if dca_scale >= 0.74: return "сниженная"
    return "минимальная"

def _pair_label(sym: str, policy_label: str, icon: str) -> str:
    return f"{sym[:3]}/{sym[3:]} — {icon} {policy_label}"

def _explain_simple(sym: str) -> str:
    if sym == "USDJPY":
        return "если ФРС жёстче ожидаемого — доллар дороже; мягче — доллар дешевле."
    if sym == "AUDUSD":
        return "если РБА не спешит снижать ставку — австралийский доллар сильнее."
    if sym == "EURUSD":
        return "ищем намёки — «больше боимся инфляции» → евро сильнее; «больше боимся слабой экономики» → евро слабее."
    if sym == "GBPUSD":
        return "больше тревоги по инфляции — фунт сильнее; меньше — слабее."
    return "жёстче — сильнее базовая валюта; мягче — слабее."

def _region_tag(sym: str) -> str:
    return {
        "USDJPY": "сегодня",
        "AUDUSD": "Азия",
        "EURUSD": "Европа",
        "GBPUSD": "Великобритания",
    }.get(sym, "сегодня")

def _banks_line_friendly(target: Optional[float], fact: Optional[float]) -> str:
    if not target and not fact:
        return "план —; факт —."
    tt = "—" if target is None else f"{target:g}"
    ff = "—" if fact is None else f"{fact:g}"
    if target and fact:
        delta = (fact - target) / (target if target else 1.0)
        ap = abs(delta)
        if ap <= 0.02:
            tail = "✅ в норме."
        elif ap <= 0.05:
            tail = f"⚠️ небольшое отклонение ({delta:+.0%})."
        else:
            tail = f"{'🚧 выше плана' if delta > 0 else '🚧 ниже плана'} ({delta:+.0%})."
        return f"план {tt} / факт {ff} — {tail}"
    return f"план {tt} / факт {ff} — —"

def _cty_short_name(country: str) -> str:
    c = (country or "").lower()
    if "united states" in c: return "ФРС"
    if "euro" in c: return "ЕЦБ"
    if "united kingdom" in c: return "Банк Англии"
    if "japan" in c: return "Банк Японии"
    if "australia" in c: return "РБА"
    return country.title() if country else "Событие"

def _fmt_hhmm(dt: datetime) -> str:
    return dt.astimezone(LOCAL_TZ).strftime("%H:%M") if LOCAL_TZ else dt.strftime("%H:%M")

def build_investor_digest(sh) -> str:
    now_utc = datetime.now(timezone.utc)
    header = header_ru(now_utc.astimezone(LOCAL_TZ)) if LOCAL_TZ else f"🧭 Утренний фон — {now_utc.strftime('%d %b %Y, %H:%M')} (UTC)"

    cal = build_calendar_for_symbols(SYMBOLS)
    news_rows = read_news_rows(sh)

    blocks: List[str] = [header]

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

        # ближайшее high-событие (для тихого окна)
        next_ev = c.get("nearest_next")
        next_cty = (next_ev or {}).get("country")
        quiet_start = quiet_end = None
        if next_ev:
            q_from, q_to = QUIET_BEFORE_MIN, QUIET_AFTER_MIN
            quiet_start = (next_ev["local"] - timedelta(minutes=q_from))
            quiet_end = (next_ev["local"] + timedelta(minutes=q_to))

        # строки
        title = _pair_label(sym, policy['label'], policy['icon'])
        market_line = _mk_market_summary(adx or 20.0, st, volz or 1.0, atr1h or 1.0, next_cty)

        side_ru = "LONG (на рост)" if side == "LONG" else "SHORT (на падение)" if side == "SHORT" else side
        avg = price_fmt(sym, _to_float(row.get("Avg_Price"), None))
        nxt = price_fmt(sym, _to_float(row.get("Next_DCA_Price"), None))

        speed = _speed_text(policy["dca_scale"])
        if c.get("quiet_now") and quiet_end:
            doing_line = f"скорость докупок — {speed}; сейчас «тихое окно» до {_fmt_hhmm(quiet_end)}."
        else:
            if quiet_start:
                doing_line = f"скорость докупок — {speed}; тихого окна нет до {_fmt_hhmm(quiet_start)}."
            else:
                doing_line = f"скорость докупок — {speed}; тихого окна нет."

        # «Что это значит для цены»
        explain = _explain_simple(sym)

        # Банки
        target, fact = latest_bank_target_fact(sh, sym)
        banks_line = _banks_line_friendly(target, fact)

        # Топ-новость
        best_news = choose_top_news_for_symbol(sym, news_rows, now_utc)
        if best_news:
            region = _region_tag(sym)
            top_news_line = f"{best_news['local_time_str']} — {best_news['ru_title']}."
            top_title = f"Топ-новость ({region})"
        elif next_ev:
            region = _region_tag(sym)
            top_news_line = f"{next_ev['local']:%H:%M} — {_ru_title_hint(next_ev['title'])}."
            top_title = f"Топ-новость ({region})"
        else:
            top_title = "Топ-новость"
            top_news_line = "—"

        # нижняя строчка про окно/важное время
        if quiet_start and quiet_end:
            tail_line = f"Тихое окно: {quiet_start:%H:%M}–{quiet_end:%H:%M}"
        elif next_ev:
            tail_line = f"Ближайшее важное время: {next_ev['local']:%H:%M} — {_cty_short_name(next_ev['country'])}."
        else:
            tail_line = "Ближайшее важное время: —"

        block = [
            f"{title}",
            f"\t•\tСводка рынка: {market_line}",
            f"\t•\tНаша позиция: {side_ru}, средняя {avg}; следующее докупление {nxt}.",
            f"\t•\tЧто делаем: {doing_line}",
        ]

        # символ-специфичная строчка «что это значит для цены»
        block.append("\t•\tЧто это значит для цены: " + (
            "до решения ФРС сильного тренда не ждём. Резкие слова ФРС — укрепляют доллар (часто рост USD/JPY); мягкие — ослабляют доллар (падение USD/JPY)."
            if sym == "USDJPY" else
            "комментарии РБА, намекающие «держим ставку дольше», обычно поддерживают AUD (AUD/USD может подрасти)."
            if sym == "AUDUSD" else
            "публикации ЕЦБ про экономику без сюрпризов — нейтрально; жёсткий тон ЕЦБ — поддержка евро (EUR/USD вверх), мягкий — давление на евро (вниз)."
            if sym == "EURUSD" else
            "если представители Банка Англии говорят «зарплаты и услуги давят на инфляцию», рынок ждёт ставку повыше дольше — фунт крепче (GBP/USD вверх). Мягче — фунт слабее."
        ))

        block.extend([
            f"\t•\tПлан vs факт по банку: {banks_line}",
            f"\t•\t{top_title}: {top_news_line}",
            f"Простыми словами: {explain}",
            f"\t•\t{tail_line}",
        ])

        blocks.append("\n".join(block))
        blocks.append("⸻")

    # Календарь «на сегодня»
    # Соберём ближайшие важные события на текущую дату (по локальному времени)
    today = (now_utc.astimezone(LOCAL_TZ) if LOCAL_TZ else now_utc).date()
    cal_lines = []
    all_events_today = []

    # попробуем взять ближайшие next по каждой паре
    cal_full = fetch_calendar_events_ff_all()
    for ev in cal_full:
        loc = ev["utc"].astimezone(LOCAL_TZ) if LOCAL_TZ else ev["utc"]
        if loc.date() == today and importance_is_high(ev.get("importance")):
            all_events_today.append((loc, ev["country"], ev["title"]))

    all_events_today.sort(key=lambda x: x[0])
    if all_events_today:
        cal_lines.append("Календарь «на сегодня»:")
        for loc, cty, title in all_events_today[:12]:
            cal_lines.append(f"\t•\t{loc:%H:%M} — {_cty_short_name(cty)}: {_ru_title_hint(title)} ({'USD' if 'united states' in cty.lower() else 'EUR' if 'euro' in cty.lower() else 'GBP' if 'united kingdom' in cty.lower() else 'JPY' if 'japan' in cty.lower() else 'AUD' if 'australia' in cty.lower() else ''})")

    if cal_lines:
        blocks.append("\n".join(cal_lines))

    # Главная мысль
    main_thought = "Главная мысль дня: до ФРС — аккуратно; после пресс-конференции вернёмся к обычному режиму, если не будет сюрпризов."
    blocks.append(main_thought)

    # убрать последний разделитель, если он последний
    if blocks[-2] == "⸻":
        blocks.pop(-2)

    return "\n\n".join(blocks)

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
        f"Привет! Я фунд-бот.\nТекущий чат id: {update.effective_chat.id}\n\nКоманды: /help"
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
    else:
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
    await update.message.reply_text(f"Целевые веса обновлены: {human_readable_weights(new_w)}")

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
            # pro-версия — оставлю как plain
            for chunk in _split_for_tg_plain(txt):
                await context.bot.send_message(chat_id=update.effective_chat.id, text=chunk)
        except Exception as e:
            await update.message.reply_text(f"LLM ошибка: {e}")
        return

    sh, _src = build_sheets_client(SHEET_ID)
    if not sh:
        await update.message.reply_text("Sheets недоступен: не могу собрать инвесторский дайджест.")
        return

    try:
        msg = build_investor_digest(sh)
        for chunk in _split_for_tg_plain(msg):
            await context.bot.send_message(chat_id=update.effective_chat.id, text=chunk)
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
                for chunk in _split_for_tg_plain(msg):
                    await app.bot.send_message(chat_id=MASTER_CHAT_ID, text=chunk)
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
