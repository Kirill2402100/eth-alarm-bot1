# fa_bot.py
# -*- coding: utf-8 -*-
import os
import json
import base64
import logging
import re
from html import escape as _html_escape
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional, List, Iterable

import asyncio

from telegram import Update, BotCommand
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# --- Таймзона ---
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ","Europe/Belgrade")) if ZoneInfo else None
MORNING_HOUR = int(os.getenv("MORNING_HOUR", "9"))
MORNING_MINUTE = int(os.getenv("MORNING_MINUTE", "30"))

# --- FA badges (трёхстатусная шкала) ---
FA_BADGE = {
    "Green": "✅ спокойно",
    "Amber": "🟡 осторожно",
    "Red":   "🛑 стоп докупок",
}

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

# --- LLM-клиент (опционально) ---
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
_FF_NEG    = {"until": 0}

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

async def get_sheets():
    sh, _ = build_sheets_client(SHEET_ID)
    return sh

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

# --- helpers: Google Sheets → FA/NEWS -------------------------------------------------
def _fa_icon(risk: str) -> str:
    return {"Green": "🟢", "Amber": "🟡", "Red": "🔴"}.get((risk or "").capitalize(), "⚪️")

def _to_local_hhmm(ts_utc: datetime) -> str:
    try:
        dt = ts_utc.astimezone(LOCAL_TZ) if LOCAL_TZ else ts_utc
        return dt.strftime("%H:%M")
    except Exception:
        return "—"

# какие источники/страны считаем профильными для каждой пары
PAIR_PREF = {
    "USDJPY": {"sources": {"US_FED_PR","JP_MOF_FX"}, "countries": {"united states","japan"}, "ccy": {"USD","JPY"}},
    "AUDUSD": {"sources": {"RBA_MR","US_FED_PR"},       "countries": {"australia","united states"}, "ccy": {"AUD","USD"}},
    "EURUSD": {"sources": {"ECB_PR","US_FED_PR"},       "countries": {"euro area","united states"}, "ccy": {"EUR","USD"}},
    "GBPUSD": {"sources": {"BOE_PR","US_FED_PR"},       "countries": {"united kingdom","united states"}, "ccy": {"GBP","USD"}},
}

def _read_fa_signals_from_sheet(sh) -> Dict[str, dict]:
    """
    Лист FA_Signals: pair,risk,bias,ttl,updated_at,scan_lock_until,reserve_off,dca_scale,reason,risk_pct
    Возвращает только НЕ протухшие по TTL записи.
    """
    try:
        ws = sh.worksheet("FA_Signals")
        rows = ws.get_all_records()
    except Exception:
        return {}
    now = datetime.now(timezone.utc)
    out: Dict[str, dict] = {}
    for r in rows:
        pair = str(r.get("pair", "")).upper()
        if not pair:
            continue
        ttl = int(r.get("ttl") or 0)
        upd_raw = str(r.get("updated_at", "")).strip()
        try:
            upd_ts = datetime.fromisoformat(upd_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            upd_ts = None
        if ttl and upd_ts and now > upd_ts + timedelta(minutes=ttl):
            # протухло — игнорируем
            continue
        out[pair] = {
            "risk":        (str(r.get("risk", "Green")).capitalize()),
            "bias":        (str(r.get("bias", "neutral")).lower()),
            "dca_scale":   float(r.get("dca_scale") or 1.0),
            "reserve_off": bool(int(r.get("reserve_off") or 0)),
            "reason":      str(r.get("reason", "base")),
            "updated_at":  upd_raw,
        }
    return out

def _pick_pair_top_news(sh, pair: str, lookback_hours: int = int(os.getenv("DIGEST_NEWS_LOOKBACK_H","48"))) -> Optional[dict]:
    """Самая свежая importance=high из NEWS для конкретной пары за окно."""
    try:
        ws = sh.worksheet("NEWS")
        rows = ws.get_all_records()
    except Exception:
        rows = []
    if not rows:
        return None
    pref = PAIR_PREF.get(pair, {"sources": set(), "countries": set(), "ccy": set()})
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    # 1) профильные источники (importance=high)
    for r in reversed(rows):
        try:
            ts = datetime.fromisoformat(str(r.get("ts_utc")).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if ts < cutoff:
            continue
        if str(r.get("importance_guess","high")).lower() != "high":
            continue
        if str(r.get("source","")).strip().upper() in pref["sources"]:
            return {"ts": ts, "title": str(r.get("title","")).strip(), "source": str(r.get("source","")).strip(), "url": str(r.get("url","")).strip()}
    # 2) совпадение стран/валют (importance=high)
    for r in reversed(rows):
        try:
            ts = datetime.fromisoformat(str(r.get("ts_utc")).replace("Z","+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if ts < cutoff:
            continue
        if str(r.get("importance_guess","high")).lower() != "high":
            continue
        ctry = {x.strip().lower() for x in str(r.get("countries","")).split(",") if x.strip()}
        ccy  = str(r.get("ccy","")).strip().upper()
        if (ctry & pref["countries"]) or (ccy and ccy in pref["ccy"]):
            return {"ts": ts, "title": str(r.get("title","")).strip(), "source": str(r.get("source","")).strip(), "url": str(r.get("url","")).strip()}
    # 3) запасной круг: берём medium, если high не нашлось
    for r in reversed(rows):
        try:
            ts = datetime.fromisoformat(str(r.get("ts_utc")).replace("Z","+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if ts < cutoff:
            continue
        if str(r.get("importance_guess","")).lower() != "medium":
            continue
        ctry = {x.strip().lower() for x in str(r.get("countries","")).split(",") if x.strip()}
        ccy  = str(r.get("ccy","")).strip().upper()
        if (ctry & pref["countries"]) or (ccy and ccy in pref["ccy"]) or (str(r.get("source","")).strip().upper() in pref["sources"]):
            return {"ts": ts, "title": str(r.get("title","")).strip(), "source": str(r.get("source","")).strip(), "url": str(r.get("url","")).strip()}
    return None

def _pick_pair_top_calendar(sh, pair: str, horizon_hours: int = 96) -> Optional[dict]:
    """Фолбэк: ближайшее high-impact событие по странам/валютам пары из CALENDAR (+/- horizon_hours)."""
    try:
        ws = sh.worksheet("CALENDAR")
        rows = ws.get_all_records()
    except Exception:
        rows = []
    if not rows:
        return None
    pref = PAIR_PREF.get(pair, {"countries": set(), "ccy": set()})
    now = datetime.now(timezone.utc)
    best = None
    for r in rows:
        try:
            ts = datetime.fromisoformat(str(r.get("utc_iso")).replace("Z","+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if abs((ts - now).total_seconds()) > horizon_hours*3600: 
            continue
        if str(r.get("impact","")).lower() != "high":
            continue
        ctry = str(r.get("country","")).strip().lower()
        ccy  = str(r.get("currency","")).strip().upper()
        if (ctry and ctry in pref["countries"]) or (ccy and ccy in pref["ccy"]):
            # выбираем ближайшее по модулю времени
            if (best is None) or (abs((ts - now).total_seconds()) < abs((best["ts"] - now).total_seconds())):
                best = {"ts": ts, "title": str(r.get("title","")).strip(), "source": str(r.get("source","")).strip() or "CAL", "url": str(r.get("url","")).strip()}
    return best

# ---------- NEWS (READ FROM SHEET) ----------
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

def _norm_iso_to_utc_dt(s: str):
    import re
    s = str(s or "").strip()
    if not s:
        return None
    s = s.replace(" ", "T").replace("Z", "+00:00")
    s = re.sub(r'([+-]\d{2})(\d{2})$', r'\1:\2', s)
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None

def read_news_rows(sh) -> List[dict]:
    try:
        ws = sh.worksheet(NEWS_WS)
        rows = ws.get_all_records()
    except Exception:
        return []
    out = []
    for r in rows:
        ts = _norm_iso_to_utc_dt(r.get("ts_utc"))
        if not ts:
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
        return f"{h} ч {m:02d} мин {sign}"
    if h:
        return f"{h} ч {sign}"
    return f"{m} мин {sign}"

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

# -------------------- КОМАНДЫ --------------------
HELP_TEXT = (
    "Что я умею\n"
    "/settotal 2800 — задать общий банк (только в мастер-чате).\n"
    "/setweights jpy=40 aud=25 eur=20 gbp=15 — выставить целевые веса.\n"
    "/weights — показать целевые веса.\n"
    "/alloc — расчёт сумм и готовые команды /setbank для торговых чатов.\n"
    "/digest — утренний дайджест (человеческий язык + события + новости).\n"
    "/digest pro — краткий «трейдерский» дайджест (по цифрам, LLM).\n"
    "/init_sheet — создать/проверить лист в Google Sheets.\n"
    "/sheet_test — записать тестовую строку в лист.\n"
    "/diag — диагностика LLM / Sheets / NEWS.\n"
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

# ---------- Вспомогательные для «инвесторского» дайджеста ----------
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

def importance_is_high(val) -> bool:
    if val is None: return False
    if isinstance(val, (int, float)): return val >= 3
    s = str(val).strip().lower()
    return "high" in s or s == "3"

def _symbol_hints(symbol: str) -> tuple[str,str]:
    # (что это значит для цены, простыми словами)
    if symbol == "USDJPY":
        return (
            "до решения ФРС сильного тренда не ждём. Резкие слова ФРС — укрепляют доллар (часто рост USD/JPY); мягкие — ослабляют доллар (падение USD/JPY).",
            "если ФРС жёстче ожидаемого — доллар дороже; мягче — доллар дешевле."
        )
    if symbol == "AUDUSD":
        return (
            "комментарии РБА, намекающие «держим ставку дольше», обычно поддерживают AUD (AUD/USD может подрасти).",
            "если РБА не спешит снижать ставку — австралийский доллар сильнее."
        )
    if symbol == "EURUSD":
        return (
            "публикации ЕЦБ про экономику без сюрпризов — нейтрально; жёсткий тон ЕЦБ — поддержка евро (EUR/USD вверх), мягкий — давление на евро (вниз).",
            "ищем намёки — «больше боимся инфляции» → евро сильнее; «больше боимся слабой экономики» → евро слабее."
        )
    # GBPUSD
    return (
        "если представители Банка Англии говорят «зарплаты и услуги давят на инфляцию», рынок ждёт ставку повыше дольше — фунт крепче (GBP/USD вверх). Мягче — фунт слабее.",
        "больше тревоги по инфляции — фунт сильнее; меньше — слабее."
    )

def _pair_to_title(symbol: str) -> str:
    return f"<b>{symbol[:3]}/{symbol[3:]}</b>"

def _plan_vs_fact_line(sh, symbol: str) -> str:
    target, fact = latest_bank_target_fact(sh, symbol)
    if target is None and fact is None:
        return "план —; факт —."
    tt = target if target is not None else 0.0
    ff = fact if fact is not None else 0.0
    if tt <= 0:
        return f"план {tt:g} / факт {ff:g} — —."
    delta = 0 if tt == 0 else (ff - tt) / tt
    ap = abs(delta)
    if ap <= 0.02: mark = "✅ в норме."
    elif ap <= 0.05: mark = f"⚠️ {'ниже' if delta<0 else 'выше'} плана ({delta:+.0%})."
    else: mark = f"🚧 {'ниже' if delta<0 else 'выше'} плана ({delta:+.0%})."
    return f"план {tt:g} / факт {ff:g} — {mark}"

def render_symbol_block(symbol: str, row: dict, fa_data: dict, sh, top_news: Optional[dict]) -> str:
    risk = fa_data.get("risk", "Green")
    bias = fa_data.get("bias", "neutral")
    dca_scale = float(fa_data.get("dca_scale", 1.0))
    reserve_on_str = "OFF" if fa_data.get("reserve_off") else "ON"
    icon = _fa_icon(risk)
    
    title = f"{_pair_to_title(symbol)} — {icon} фон {risk.lower()}, bias: {bias.upper()}"
    
    side = (row.get("Side") or row.get("SIDE") or "").upper() or "LONG"
    avg = price_fmt(symbol, _to_float(row.get("Avg_Price"), None))
    nxt = price_fmt(symbol, _to_float(row.get("Next_DCA_Price"), None))
    what_means, simple_words = _symbol_hints(symbol)
    plan_fact = _plan_vs_fact_line(sh, symbol)

    lines = [title]
    lines.append(f"•\tСводка рынка: {('подъём заметнее обычного, волатильность ниже нормы.' if bias=='short' and symbol=='AUDUSD' else 'движения ровные, резких скачков не ждём до США.')}")
    lines.append(f"•\tНаша позиция: {side} (на {'рост' if side=='LONG' else 'падение'}), средняя {avg}; следующее докупление {nxt}.")
    lines.append(f"•\tЧто делаем сейчас: {'тихое окно' if risk!='Green' else 'тихое окно не требуется'}; reserve {reserve_on_str}; dca_scale <b>{dca_scale:.2f}</b>.")
    
    if top_news:
        lines.append(f"• Топ-новость: {_to_local_hhmm(top_news['ts'])} — {_h(top_news['title'])} ({_h(top_news['source'])}).")

    lines.append(f"•\tЧто это значит для цены: {what_means}")
    lines.append(f"•\tПлан vs факт по банку: {plan_fact}.")
    lines.append(f"Простыми словами: {simple_words}")

    return "\n".join(lines)


# ---------- Digest helpers (сборка полного текста) ----------
def build_digest_text(sh, fa_sheet_data: dict) -> str:
    now_utc = datetime.now(timezone.utc)
    header = header_ru(now_utc.astimezone(LOCAL_TZ)) if LOCAL_TZ else f"🧭 Утренний фон — {now_utc.strftime('%d %b %Y, %H:%M')} (UTC)"
    
    parts: List[str] = [header]
    
    for i, sym in enumerate(SYMBOLS):
        row = get_last_nonempty_row(sh, sym) or {}
        fa_data = fa_sheet_data.get(sym, {})
        
        top_news = _pick_pair_top_news(sh, sym)
        if not top_news:
            top_news = _pick_pair_top_calendar(sh, sym)

        block = render_symbol_block(sym, row, fa_data, sh, top_news)
        parts.append(block)
        if i < len(SYMBOLS) - 1:
            parts.append("⸻")

    parts.append("Главная мысль дня: до ФРС — аккуратно; после пресс-конференции вернёмся к обычному режиму, если не будет сюрпризов.")
    return "\n".join(parts)


# -------------------- /digest --------------------
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

    sh = await get_sheets()
    if not sh:
        await update.message.reply_text("Sheets недоступен: не могу собрать инвесторский дайджест.")
        return

    try:
        fa_sheet = _read_fa_signals_from_sheet(sh)
        msg = build_digest_text(sh, fa_sheet)
        
        for chunk in _split_for_tg_html(msg, 3500):
            await context.bot.send_message(chat_id=update.effective_chat.id, text=chunk, parse_mode=ParseMode.HTML)
    except Exception as e:
        log.exception("Ошибка сборки дайджеста")
        await update.message.reply_text(f"Ошибка сборки дайджеста: {e}")


# -------------------- Диагностика --------------------
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

async def _diag_news_line() -> str:
    try:
        sh, _ = build_sheets_client(SHEET_ID)
        if not sh:
            return "NEWS: ❌ Sheets недоступен"
        rows = read_news_rows(sh)
        return f"NEWS: ✅ {len(rows)} строк в листе NEWS; окно {DIGEST_NEWS_LOOKBACK_MIN} мин."
    except Exception as e:
        return f"NEWS: ❌ ошибка: {e}"

async def cmd_diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        ok = await llm_ping()
        llm_line = "LLM: ✅ ok" if ok else "LLM: ❌ no key"
    except Exception:
        llm_line = "LLM: ❌ error"
    sheets_line = sheets_diag_text()
    news_line = await _diag_news_line()
    await update.message.reply_text(f"{llm_line}\n{sheets_line}\n{news_line}")

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
                fa_sheet_data = _read_fa_signals_from_sheet(sh)
                msg = build_digest_text(sh, fa_sheet_data)
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
