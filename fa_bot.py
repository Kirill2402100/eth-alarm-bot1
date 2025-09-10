# fa_bot.py
# -*- coding: utf-8 -*-
import os
import re
import json
import base64
import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta, timezone

import asyncio

from telegram import Update, BotCommand
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# ---------- TZ ----------
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Europe/Belgrade")) if ZoneInfo else None
MORNING_HOUR = int(os.getenv("MORNING_HOUR", "9"))
MORNING_MINUTE = int(os.getenv("MORNING_MINUTE", "30"))

# ---------- Rate limiter ----------
try:
    from telegram.ext import AIORateLimiter
    _RATE_LIMITER_AVAILABLE = True
except Exception:
    AIORateLimiter = None
    _RATE_LIMITER_AVAILABLE = False

# ---------- Google Sheets ----------
try:
    import gspread
    from google.oauth2 import service_account
    _GSHEETS_AVAILABLE = True
except Exception:
    gspread = None
    service_account = None
    _GSHEETS_AVAILABLE = False

# ---------- HTTP ----------
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from urllib.parse import urlparse
    _REQUESTS_AVAILABLE = True
except Exception:
    requests = None
    HTTPAdapter = None
    Retry = None
    urlparse = None
    _REQUESTS_AVAILABLE = False

# ---------- LLM (если есть) ----------
try:
    from llm_client import generate_digest, llm_ping
except Exception:
    async def generate_digest(*args, **kwargs) -> str:
        return "⚠️ LLM сейчас недоступен (нет llm_client.py)."

    async def llm_ping() -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

# ---------- LOG ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("fund_bot")

# ---------- ENV ----------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or os.getenv("TELEGRAM_TOKEN", "").strip()
MASTER_CHAT_ID = int(os.getenv("MASTER_CHAT_ID", "0") or "0")

SHEET_ID = os.getenv("SHEET_ID", "").strip()
SHEET_WS = os.getenv("SHEET_WS", "FUND_BOT").strip() or "FUND_BOT"

# лист для календаря (создастся автоматически)
CAL_WS = os.getenv("CAL_WS", "CALENDAR").strip() or "CALENDAR"
CAL_HEADERS = ["utc_iso", "local_time", "country", "title", "src", "impact"]

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

# ---------- Календарь/источники ----------
TE_BASE = os.getenv("TE_BASE", "https://api.tradingeconomics.com").rstrip("/")
TE_CLIENT = os.getenv("TE_CLIENT", "guest").strip()
TE_KEY = os.getenv("TE_KEY", "guest").strip()

CAL_WINDOW_MIN = int(os.getenv("CAL_WINDOW_MIN", "240") or "240")
QUIET_BEFORE_MIN = int(os.getenv("QUIET_BEFORE_MIN", "45"))
QUIET_AFTER_MIN  = int(os.getenv("QUIET_AFTER_MIN",  "45"))

# provider: auto|ff|fmp|te|free|dailyfx|investing
CAL_PROVIDER = os.getenv("CAL_PROVIDER", "free").lower()

FMP_API_KEY  = os.getenv("FMP_API_KEY", "").strip()
CAL_TTL_SEC = int(os.getenv("CAL_TTL_SEC", "600") or "600")
CAL_REFRESH_MIN = int(os.getenv("CAL_REFRESH_MIN", "180") or "180")

COUNTRY_BY_CCY = {
    "USD": "united states",
    "JPY": "japan",
    "EUR": "euro area",
    "GBP": "united kingdom",
    "AUD": "australia",
}
FF_CODE2NAME = {
    "usd": "united states", "jpy": "japan", "eur": "euro area",
    "gbp": "united kingdom", "aud": "australia",
}
PAIR_COUNTRIES = {
    "USDJPY": [COUNTRY_BY_CCY["USD"], COUNTRY_BY_CCY["JPY"]],
    "AUDUSD": [COUNTRY_BY_CCY["AUD"], COUNTRY_BY_CCY["USD"]],
    "EURUSD": [COUNTRY_BY_CCY["EUR"], COUNTRY_BY_CCY["USD"]],
    "GBPUSD": [COUNTRY_BY_CCY["GBP"], COUNTRY_BY_CCY["USD"]],
}

# ---------- STATE ----------
_FF_CACHE = {"at": 0, "data": []}
_FF_NEG   = {"until": 0}

STATE = {
    "total": 0.0,
    "weights": DEFAULT_WEIGHTS.copy(),
}

# ---------- Sheets auth ----------
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

def ensure_worksheet(sh, title: str, headers: Optional[List[str]]=None):
    try:
        for ws in sh.worksheets():
            if ws.title == title:
                if headers:
                    try:
                        ws.update(headers, "A1")
                    except Exception:
                        # ignore if not needed
                        pass
                return ws, False
        ws = sh.add_worksheet(title=title, rows=500, cols=max(10, len(headers or SHEET_HEADERS)))
        if headers:
            ws.update(headers, "A1")
        else:
            ws.update([SHEET_HEADERS], "A1")
        return ws, True
    except Exception as e:
        raise RuntimeError(f"ensure_worksheet error: {e}")

def append_row(sh, title: str, row: list):
    ws, _ = ensure_worksheet(sh, title)
    ws.append_row(row, value_input_option="RAW")

# ---------- Utils ----------
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

# ---------- Commands ----------
HELP_TEXT = (
    "Что я умею\n"
    "/settotal 2800 — задать общий банк (только в мастер-чате).\n"
    "/setweights jpy=40 aud=25 eur=20 gbp=15 — выставить целевые веса.\n"
    "/weights — показать целевые веса.\n"
    "/alloc — расчёт сумм и готовые команды /setbank для торговых чатов.\n"
    "/digest — утренний дайджест (investor) / pro (trader).\n"
    "/cal_refresh — обновить экономкалендарь (бесплатные источники).\n"
    "/init_sheet — создать/проверить листы в Google Sheets.\n"
    "/sheet_test — тестовая запись в лист.\n"
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
        if len(parts) < 2: raise ValueError
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
            if "=" not in token: continue
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
    lines = [f"Целевые веса: {human_readable_weights(w)}", "", "Распределение:"]
    for sym in SYMBOLS:
        lines.append(f"{sym} → {alloc[sym]} USDT → команда в чат {sym}: /setbank {alloc[sym]}")
    await update.message.reply_text("\n".join(lines))
    sh, _src = build_sheets_client(SHEET_ID)
    if sh:
        try:
            append_row(
                sh, SHEET_WS,
                [
                    datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    str(update.effective_chat.id), "alloc", f"{total:.2f}",
                    json.dumps(w, ensure_ascii=False), json.dumps(alloc, ensure_ascii=False),
                ],
            )
        except Exception as e:
            log.warning("append_row alloc failed: %s", e)

# ---------- Digest helpers ----------
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
    sheet_name = os.getenv(f"BMR_SHEET_{symbol}", f"BMR_DCA_{symbol}")
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
    sheet_name = os.getenv(f"BMR_SHEET_{symbol}", f"BMR_DCA_{symbol}")
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
    if (fa_bias == "LONG" and against_dir == "up") or (fa_bias == "SHORT" and against_dir == "down"): P += 10
    elif fa_bias in ("LONG", "SHORT"): P -= 15
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
        if P >= 64:
            return "готовьтесь добирать 1 шаг (после снятия ограничений и вне тихого окна)"
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

# ---------- HTTP helpers ----------
_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
]
def _ua(i=0): return _UA_POOL[i % len(_UA_POOL)]

def _build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.6, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": _ua(0),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8,ru;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "close",
        "Upgrade-Insecure-Requests": "1",
    })
    return s

def _fetch_with_fallback(url: str, referer: str | None = None, timeout=12) -> tuple[str, str]:
    """
    Возвращает (text, mode): mode in {"direct","proxy","none"}.
    При 403/429/пустой странице пробует r.jina.ai/http(s)://… (текст без JS).
    """
    if not _REQUESTS_AVAILABLE:
        return "", "none"
    s = _build_session()
    if referer:
        s.headers["Referer"] = referer
    try:
        r = s.get(url, timeout=timeout)
        if r.status_code == 200 and r.text and len(r.text) > 1000:
            return r.text, "direct"
    except Exception as e:
        log.warning("fetch error for %s: %s", url, e)
    # proxy reader
    try:
        parsed = urlparse(url)
        proxy_url = f"https://r.jina.ai/{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query: proxy_url += f"?{parsed.query}"
        r2 = s.get(proxy_url, timeout=timeout)
        if r2.status_code == 200 and r2.text and len(r2.text) > 500:
            return r2.text, "proxy"
        else:
            log.warning("proxy fetch failed %s: status=%s len=%s", proxy_url, r2.status_code, len(r2.text or ""))
    except Exception as e:
        log.warning("proxy fetch error %s: %s", url, e)
    return "", "none"

def _clean_text(x: str) -> str:
    return re.sub(r"\s+", " ", x or "").strip()

# ---------- Sources: ForexFactory ----------
def _parse_ff_text(text: str) -> list[dict]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out = []
    # Пример строки через зеркало содержит месяц: "Wed, Sep 11 14:30 USD CPI m/m High Impact Expected"
    rx = re.compile(
        r"(?P<dow>Mon|Tue|Wed|Thu|Fri|Sat|Sun)[, ]+(?P<mon>[A-Za-z]{3})\s+(?P<day>\d{1,2})\s+(?P<h>\d{1,2}):(?P<m>\d{2})\s+(?P<ccy>[A-Z]{3})\s+(?P<title>.+?)\s+(High Impact|High Impact Expected)",
        re.I)
    now = datetime.now(timezone.utc)
    year = now.year
    for l in lines:
        m = rx.search(l)
        if not m: continue
        mon = m.group("mon").title()
        day = int(m.group("day"))
        hh = int(m.group("h")); mm = int(m.group("m"))
        ccy = m.group("ccy").upper()
        title = _clean_text(m.group("title"))
        try:
            dt_loc = datetime.strptime(f"{year} {mon} {day} {hh}:{mm}", "%Y %b %d %H:%M")
            dt_utc = dt_loc.replace(tzinfo=timezone.utc)
        except Exception:
            dt_utc = now
        out.append({"utc": dt_utc, "country": COUNTRY_BY_CCY.get(ccy, ccy), "title": title, "importance": "high", "src": "ff"})
    return out

def fetch_forexfactory_week_html() -> list[dict]:
    url = "https://www.forexfactory.com/calendar?week=this"
    text, mode = _fetch_with_fallback(url, referer="https://www.forexfactory.com/")
    if not text:
        log.info("FF HTML parsed: 0 high-impact events")
        return []
    events = _parse_ff_text(text)
    log.info("FF HTML parsed: %d high-impact events (mode=%s)", len(events), mode)
    return events

# ---------- Sources: DailyFX ----------
def _extract_dailyfx_next_json(html: str) -> dict | None:
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(?P<json>{.*?})</script>', html, re.S | re.I)
    if not m: return None
    try:
        return json.loads(m.group("json"))
    except Exception:
        return None

def _parse_dailyfx_from_next(next_data: dict) -> list[dict]:
    out = []
    def walk(obj):
        if isinstance(obj, dict):
            if all(k in obj for k in ("title", "country", "date")):
                title = _clean_text(str(obj.get("title")))
                country = _clean_text(str(obj.get("country")))
                impact = _clean_text(str(obj.get("impact") or obj.get("importance") or "")).lower()
                dt_raw = str(obj.get("date") or obj.get("datetime") or "")
                try:
                    if dt_raw.isdigit():
                        ts = int(dt_raw)
                        if ts > 10_000_000_000: ts //= 1000
                        dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
                    else:
                        dt_utc = datetime.fromisoformat(dt_raw.replace("Z", "+00:00"))
                except Exception:
                    return
                if "high" in impact:
                    out.append({"utc": dt_utc, "country": country.lower(), "title": title, "importance": "high", "src": "dailyfx"})
            for v in obj.values(): walk(v)
        elif isinstance(obj, list):
            for v in obj: walk(v)
    walk(next_data)
    return out

def fetch_dailyfx_html() -> list[dict]:
    base = "https://www.dailyfx.com/economic-calendar"
    html, mode = _fetch_with_fallback(base, referer="https://www.dailyfx.com/")
    if not html:
        log.warning("DailyFX fetch failed (empty)")
        return []
    next_json = _extract_dailyfx_next_json(html)
    if next_json:
        ev = _parse_dailyfx_from_next(next_json)
        if ev:
            log.info("DailyFX parsed from NEXT_DATA: %d (mode=%s)", len(ev), mode)
            return ev
    html2, mode2 = _fetch_with_fallback(base + "?tz=0", referer=base)
    if html2:
        next_json2 = _extract_dailyfx_next_json(html2)
        if next_json2:
            ev2 = _parse_dailyfx_from_next(next_json2)
            if ev2:
                log.info("DailyFX parsed (?tz=0): %d (mode=%s)", len(ev2), mode2)
                return ev2
    # текстовый резерв
    text, mode3 = _fetch_with_fallback(base, referer="https://www.dailyfx.com/")
    out = []
    if text:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        now = datetime.now(timezone.utc)
        rx = re.compile(r"(?P<h>\d{1,2}):(?P<m>\d{2})\s+UTC\s+(?P<ccy>[A-Z]{3})\s+(?P<title>.+?)\s+(High impact|High)", re.I)
        for l in lines:
            m = rx.search(l)
            if not m: continue
            hh = int(m.group("h")); mm = int(m.group("m"))
            ccy = m.group("ccy").upper()
            title = _clean_text(m.group("title"))
            dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            out.append({"utc": dt, "country": COUNTRY_BY_CCY.get(ccy, ccy), "title": title, "importance": "high", "src": "dailyfx"})
    if out:
        log.info("DailyFX parsed (text-fallback): %d (mode=%s)", len(out), mode3)
        return out
    log.warning("DailyFX parse gave 0")
    return []

# ---------- Sources: Investing.com ----------
def fetch_investing_calendar() -> list[dict]:
    # на инвестинге жёсткий антибот; идём через зеркало сразу
    base = "https://www.investing.com/economic-calendar/"
    text, mode = _fetch_with_fallback(base + "?importance=3", referer=base)
    out = []
    if not text:
        log.warning("Investing fetch failed")
        return out
    # ищем строки с временем, валютой и High (зеркало отдаёт текст)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    now = datetime.now(timezone.utc)
    # Пример шаблона: "14:30 USD CPI (YoY) (Aug) Importance: High"
    rx = re.compile(r"(?P<h>\d{1,2}):(?P<m>\d{2})\s+(?P<ccy>[A-Z]{3})\s+(?P<title>.+?)\s+(Importance:\s*High|High)", re.I)
    for l in lines:
        m = rx.search(l)
        if not m: continue
        hh = int(m.group("h")); mm = int(m.group("m"))
        ccy = m.group("ccy").upper()
        title = _clean_text(m.group("title"))
        dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        out.append({"utc": dt, "country": COUNTRY_BY_CCY.get(ccy, ccy), "title": title, "importance": "high", "src": "investing"})
    log.info("Investing parsed: %d (mode=%s)", len(out), mode)
    return out

# ---------- Central Banks (FOMC/ECB/BoE/BoJ) ----------
def _mk_cb_event(year: int, mon: int, day: int, country: str, title: str, hour=12, minute=0) -> dict:
    try:
        dt = datetime(year, mon, day, hour, minute, tzinfo=timezone.utc)
    except Exception:
        dt = datetime(datetime.now(timezone.utc).year, mon, day, 12, 0, tzinfo=timezone.utc)
    return {"utc": dt, "country": country, "title": title, "importance": "high", "src": "cbank"}

def _parse_month(s: str) -> int | None:
    m = s.strip()[:3].title()
    mm = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}.get(m)
    return mm

def fetch_fomc() -> list[dict]:
    # https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    text, mode = _fetch_with_fallback(url, referer="https://www.federalreserve.gov/")
    if not text:
        return []
    out = []
    year = datetime.now(timezone.utc).year
    # шаблон: "September 16-17, 2025" или "September 17, 2025"
    rx = re.compile(r"(?P<mon>[A-Za-z]+)\s+(?P<d1>\d{1,2})(?:\s*[-–]\s*(?P<d2>\d{1,2}))?,\s*(?P<y>\d{4})", re.I)
    for m in rx.finditer(text):
        y = int(m.group("y"))
        if abs(y - year) > 1: continue
        mon = _parse_month(m.group("mon"))
        d1 = int(m.group("d1"))
        d2 = int(m.group("d2") or d1)
        # ставим на второй день в полдень UTC
        out.append(_mk_cb_event(y, mon, d2, COUNTRY_BY_CCY["USD"], "FOMC meeting (time TBD)"))
    log.info("FOMC parsed: %d (mode=%s)", len(out), mode)
    return out

def fetch_ecb() -> list[dict]:
    # https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html
    url = "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html"
    text, mode = _fetch_with_fallback(url, referer="https://www.ecb.europa.eu/")
    if not text:
        return []
    out = []
    year = datetime.now(timezone.utc).year
    # примеры: "Monetary policy meeting: 12 September 2025"
    rx = re.compile(r"Monetary policy meeting:\s*(?P<d>\d{1,2})\s+(?P<mon>[A-Za-z]+)\s+(?P<y>\d{4})", re.I)
    for m in rx.finditer(text):
        y = int(m.group("y"))
        if abs(y - year) > 1: continue
        mon = _parse_month(m.group("mon"))
        d = int(m.group("d"))
        out.append(_mk_cb_event(y, mon, d, COUNTRY_BY_CCY["EUR"], "ECB monetary policy meeting (time TBD)"))
    log.info("ECB parsed: %d (mode=%s)", len(out), mode)
    return out

def fetch_boe() -> list[dict]:
    # https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes  (часто есть "Next MPC meeting: 19 September 2025")
    url = "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes"
    text, mode = _fetch_with_fallback(url, referer="https://www.bankofengland.co.uk/")
    if not text:
        return []
    out = []
    year = datetime.now(timezone.utc).year
    rx = re.compile(r"(MPC meeting|Monetary Policy Committee meeting).*?(?P<d>\d{1,2})\s+(?P<mon>[A-Za-z]+)\s+(?P<y>\d{4})", re.I)
    for m in rx.finditer(text):
        y = int(m.group("y"))
        if abs(y - year) > 1: continue
        mon = _parse_month(m.group("mon")); d = int(m.group("d"))
        out.append(_mk_cb_event(y, mon, d, COUNTRY_BY_CCY["GBP"], "BoE MPC meeting (time TBD)"))
    log.info("BoE parsed: %d (mode=%s)", len(out), mode)
    return out

def fetch_boj() -> list[dict]:
    # https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm/
    url = "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm/"
    text, mode = _fetch_with_fallback(url, referer="https://www.boj.or.jp/")
    if not text:
        return []
    out = []
    year = datetime.now(timezone.utc).year
    # Пример: "Monetary Policy Meeting (MPM) September 20–21, 2025"
    rx = re.compile(r"Monetary Policy Meeting.*?(?P<mon>[A-Za-z]+)\s+(?P<d1>\d{1,2})\D+(?P<d2>\d{1,2}),\s*(?P<y>\d{4})", re.I)
    for m in rx.finditer(text):
        y = int(m.group("y"))
        if abs(y - year) > 1: continue
        mon = _parse_month(m.group("mon"))
        d2 = int(m.group("d2"))
        out.append(_mk_cb_event(y, mon, d2, COUNTRY_BY_CCY["JPY"], "BoJ Monetary Policy Meeting (time TBD)"))
    log.info("BoJ parsed: %d (mode=%s)", len(out), mode)
    return out

def fetch_cbank_all() -> list[dict]:
    events = []
    try: events += fetch_fomc()
    except Exception as e: log.warning("FOMC fetch error: %s", e)
    try: events += fetch_ecb()
    except Exception as e: log.warning("ECB fetch error: %s", e)
    try: events += fetch_boe()
    except Exception as e: log.warning("BoE fetch error: %s", e)
    try: events += fetch_boj()
    except Exception as e: log.warning("BoJ fetch error: %s", e)
    return events

# ---------- Merge & Filter ----------
def merge_sources(groups: list[list[dict]]) -> list[dict]:
    uniq = {}
    for g in groups:
        for ev in g or []:
            key = (ev["utc"], ev["country"].lower(), _clean_text(ev["title"]).lower())
            uniq[key] = ev
    return sorted(uniq.values(), key=lambda e: e["utc"])

def get_all_free_calendar_events() -> list[dict]:
    prov = CAL_PROVIDER
    ev_ff = ev_dfx = ev_inv = ev_cb = []

    if prov in ("ff", "free", "auto"):
        try:
            ev_ff = fetch_forexfactory_week_html()
        except Exception as e:
            log.warning("FF fetch failed: %s", e)

    if prov in ("dailyfx", "free", "auto"):
        try:
            ev_dfx = fetch_dailyfx_html()
        except Exception as e:
            log.warning("DailyFX fetch failed: %s", e)

    if prov in ("investing", "free", "auto"):
        try:
            ev_inv = fetch_investing_calendar()
        except Exception as e:
            log.warning("Investing fetch failed: %s", e)

    # Центральные банки — всегда полезно как резерв
    try:
        ev_cb = fetch_cbank_all()
    except Exception as e:
        log.warning("CB fetch failed: %s", e)

    if prov == "ff": out = ev_ff or ev_dfx or ev_inv or ev_cb
    elif prov == "dailyfx": out = ev_dfx or ev_ff or ev_inv or ev_cb
    elif prov == "investing": out = ev_inv or ev_ff or ev_dfx or ev_cb
    else:  # free/auto
        out = merge_sources([ev_ff, ev_dfx, ev_inv, ev_cb])

    # фильтр по валютам и окну
    want_countries = {c.lower() for ps in PAIR_COUNTRIES.values() for c in ps}
    now = datetime.now(timezone.utc)
    d1 = now - timedelta(minutes=CAL_WINDOW_MIN)
    d2 = now + timedelta(minutes=CAL_WINDOW_MIN)

    out = [ev for ev in out if ev["country"].lower() in want_countries and d1 <= ev["utc"] <= d2]
    log.info("Free calendar union: provider=%s, total=%d", prov, len(out))
    return out

# ---------- Calendar builder used by digest ----------
def build_calendar_for_symbols(symbols: List[str], window_min: Optional[int] = None) -> Dict[str, dict]:
    now = datetime.now(timezone.utc)
    w = window_min if window_min is not None else CAL_WINDOW_MIN
    d1 = now - timedelta(minutes=w)
    d2 = now + timedelta(minutes=w)

    all_events = get_all_free_calendar_events()

    out: Dict[str, dict] = {}
    for sym in symbols:
        countries = {c.lower() for c in PAIR_COUNTRIES.get(sym, [])}
        sym_all = [ev for ev in all_events if ev["country"].lower() in countries]

        around = [
            {**ev, "local": ev["utc"].astimezone(LOCAL_TZ) if LOCAL_TZ else ev["utc"]}
            for ev in sym_all if d1 <= ev["utc"] <= d2
        ]
        around.sort(key=lambda x: x["utc"])

        red_soon = any(abs((ev["utc"] - now).total_seconds()) / 60.0 <= 60 for ev in around)
        quiet_now = any(
            (ev["utc"] - timedelta(minutes=QUIET_BEFORE_MIN)) <= now <= (ev["utc"] + timedelta(minutes=QUIET_AFTER_MIN))
            for ev in around
        )

        nearest_prev = nearest_next = None
        if not around:
            past = [ev for ev in sym_all if ev["utc"] < now]
            futr = [ev for ev in sym_all if ev["utc"] >= now]
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

# ---------- Investor digest ----------
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

    summary_lines, events_list = [], []
    for sym in SYMBOLS:
        events_list.extend((ev["utc"], ev.get("local", ev["utc"]), sym, ev["country"], ev["title"]) for ev in cal.get(sym, {}).get("events", []))
    if not events_list:
        events_list.extend((n["utc"], n["local"], sym, n["country"], n["title"]) for sym in SYMBOLS if (n := cal.get(sym, {}).get("nearest_next")))
    if events_list:
        summary_lines.append("\n📅 **Ближайшие High-события (Белград):**")
        unique = {ev[0]: ev for ev in sorted(events_list)}
        for _, tloc, sym, cty, title in list(unique.values())[:8]:
            summary_lines.append(f"• {tloc:%H:%M} — {sym}: {cty}: {title}")
    return "\n\n".join(blocks + ["\n".join(summary_lines)] if summary_lines else [])

async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = (update.message.text or "").split()
    pro = len(args) > 1 and args[1].lower() == "pro"
    if pro:
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            txt = await generate_digest(symbols=SYMBOLS, model=LLM_MINI, token_budget=LLM_TOKEN_BUDGET_PER_DAY)
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
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"Ошибка сборки дайджеста: {e}")

# ---------- Sheets diag / init ----------
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
            ws, created = ensure_worksheet(sh, SHEET_WS, headers=[SHEET_HEADERS])
            _ = ensure_worksheet(sh, CAL_WS, headers=[CAL_HEADERS])
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
        ws, created = ensure_worksheet(sh, SHEET_WS, headers=[SHEET_HEADERS])
        ensure_worksheet(sh, CAL_WS, headers=[CAL_HEADERS])
        await update.message.reply_text(
            f"Sheets: ✅ ws='{ws.title}' {'создан' if created else 'уже есть'}; календарь '{CAL_WS}' готов ({src})"
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
            sh, SHEET_WS,
            [
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                str(update.effective_chat.id), "test",
                f"{STATE['total']:.2f}",
                json.dumps(STATE['weights'], ensure_ascii=False),
                "manual /sheet_test",
            ],
        )
        await update.message.reply_text("Sheets: ✅ записано (test row).")
    except Exception as e:
        await update.message.reply_text(f"Sheets: ❌ ошибка записи: {e}")

# ---------- Calendar -> Google Sheet ----------
def write_calendar_to_sheet(events: List[dict]) -> int:
    sh, src = build_sheets_client(SHEET_ID)
    if not sh:
        log.warning("Sheets not available: %s", src)
        return 0
    try:
        ws, _ = ensure_worksheet(sh, CAL_WS, headers=[CAL_HEADERS])
        # очистим всё кроме заголовка
        try:
            ws.resize(rows=1)
        except Exception:
            pass
        rows = []
        for ev in sorted(events, key=lambda e: e["utc"]):
            utc_iso = ev["utc"].isoformat(timespec="minutes").replace("+00:00","Z")
            lt = ev["utc"].astimezone(LOCAL_TZ) if LOCAL_TZ else ev["utc"]
            rows.append([utc_iso, lt.strftime("%Y-%m-%d %H:%M"), ev["country"], ev["title"], ev.get("src",""), "High"])
        if rows:
            ws.append_rows(rows, value_input_option="RAW")
        return len(rows)
    except Exception as e:
        log.warning("write_calendar_to_sheet failed: %s", e)
        return 0

async def cmd_cal_refresh(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    ev = get_all_free_calendar_events()
    n = write_calendar_to_sheet(ev)
    await update.message.reply_text(f"Календарь обновлён: записано {n} событий (High).")

# ---------- Schedulers ----------
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
        BotCommand("cal_refresh", "Обновить экономкалендарь"),
        BotCommand("init_sheet", "Создать/проверить листы в Google Sheets"),
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

async def calendar_auto_refresher(app: Application):
    # периодически обновляем календарь и лист
    while True:
        try:
            ev = get_all_free_calendar_events()
            n = write_calendar_to_sheet(ev)
            log.info("Auto calendar refresh: %d events written", n)
            if MASTER_CHAT_ID:
                try:
                    await app.bot.send_message(chat_id=MASTER_CHAT_ID, text=f"Календарь обновлён: записано {n} событий (High).")
                except Exception:
                    pass
        except Exception as e:
            log.warning("Auto calendar refresh error: %s", e)
        await asyncio.sleep(max(60, CAL_REFRESH_MIN * 60))

# ---------- App ----------
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
    app.add_handler(CommandHandler("cal_refresh", cmd_cal_refresh))
    return app

async def main_async():
    log.info("Fund bot is running…")
    app = build_application()
    await _set_bot_commands(app)
    await app.initialize()
    await app.start()
    # Автозадачи
    asyncio.create_task(morning_digest_scheduler(app))
    asyncio.create_task(calendar_auto_refresher(app))
    # polling
    await app.updater.start_polling()
    await asyncio.Event().wait()

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
