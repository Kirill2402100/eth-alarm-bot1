# fa_bot.py — investor digest (строгий формат + новости из листа NEWS)

import os, json, base64, logging, re, asyncio
from html import escape as _html_escape
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional, List

from telegram import Update, BotCommand
from telegram.constants import ParseMode, ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes

# --- TZ ---
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Europe/Belgrade")) if ZoneInfo else None
MORNING_HOUR   = int(os.getenv("MORNING_HOUR", "9"))
MORNING_MINUTE = int(os.getenv("MORNING_MINUTE", "30"))

# --- Rate limiter (optional) ---
try:
    from telegram.ext import AIORateLimiter
    _RATE_LIMITER_AVAILABLE = True
except Exception:
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

# --- HTTP (calendar fallback) ---
try:
    import requests
    _REQUESTS_AVAILABLE = True
except Exception:
    requests = None
    _REQUESTS_AVAILABLE = False

# --- LLM stub (не нужен для investor-версии, но оставим) ---
async def llm_ping() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))

# -------------------- ЛОГИ --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fund_bot")

# -------------------- ENV --------------------
BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or os.getenv("TELEGRAM_TOKEN", "").strip())
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

SYMBOLS = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]

# Листы BMR (цели/факты банка)
BMR_SHEETS = {
    "USDJPY": os.getenv("BMR_SHEET_USDJPY", "BMR_DCA_USDJPY"),
    "AUDUSD": os.getenv("BMR_SHEET_AUDUSD", "BMR_DCA_AUDUSD"),
    "EURUSD": os.getenv("BMR_SHEET_EURUSD", "BMR_DCA_EURUSD"),
    "GBPUSD": os.getenv("BMR_SHEET_GBPUSD", "BMR_DCA_GBPUSD"),
}

# --- Календарь (FF/TE/FMP + тихие окна) ---
CAL_PROVIDER = os.getenv("CAL_PROVIDER", "auto").lower()
CAL_TTL_SEC  = int(os.getenv("CAL_TTL_SEC", "600") or "600")
CAL_WINDOW_MIN   = int(os.getenv("CAL_WINDOW_MIN", "120"))
QUIET_BEFORE_MIN = int(os.getenv("QUIET_BEFORE_MIN", "45"))
QUIET_AFTER_MIN  = int(os.getenv("QUIET_AFTER_MIN",  "45"))
CAL_WS = os.getenv("CAL_WS", "CALENDAR").strip() or "CALENDAR"
TE_BASE   = os.getenv("TE_BASE", "https://api.tradingeconomics.com").rstrip("/")
TE_CLIENT = os.getenv("TE_CLIENT", "guest").strip()
TE_KEY    = os.getenv("TE_KEY", "guest").strip()
FMP_API_KEY = os.getenv("FMP_API_KEY", "").strip()

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

# --- Новости для дайджеста ---
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
        r"rate decision|monetary policy|policy|bank rate|statement|minutes|guidance|intervention|unscheduled|emergency|press conference"
    ),
    re.I
)

# -------------------- СОСТОЯНИЕ --------------------
STATE = {"total": 0.0, "weights": DEFAULT_WEIGHTS.copy()}

# -------------------- Sheets auth --------------------
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

def load_google_service_info():
    b64 = _env("GOOGLE_CREDENTIALS_JSON_B64")
    if b64:
        try:
            return json.loads(_decode_b64_maybe_padded(b64)), "env:GOOGLE_CREDENTIALS_JSON_B64"
        except Exception as e:
            return None, f"b64 decode error: {e}"
    for name in ("GOOGLE_CREDENTIALS_JSON", "GOOGLE_CREDENTIALS"):
        raw = _env(name)
        if raw:
            try:
                return json.loads(raw), f"env:{name}"
            except Exception as e:
                return None, f"{name} invalid json: {e}"
    return None, "not-found"

def build_sheets_client(sheet_id: str):
    if not _GSHEETS_AVAILABLE: return None, "gsheets libs not installed"
    if not sheet_id: return None, "sheet_id empty"
    info, src = load_google_service_info()
    if not info: return None, src
    try:
        creds = service_account.Credentials.from_service_account_info(info, scopes=SHEETS_SCOPES)
        gc = gspread.authorize(creds)
        return gc.open_by_key(sheet_id), src
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

# ---------- NEWS read ----------
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
        out.append({
            "ts": ts,
            "source": str(r.get("source","")).strip().upper(),
            "title":  str(r.get("title","")).strip(),
            "url":    str(r.get("url","")).strip(),
            "countries": {c.strip().lower() for c in str(r.get("countries","")).split(",") if c.strip()},
            "ccy":    str(r.get("ccy","")).strip().upper(),
            "tags":   str(r.get("tags","")).strip(),
            "importance": str(r.get("importance_guess","")).strip().lower(),
        })
    return out

# ---------- Small utils ----------
def _h(x) -> str: return _html_escape(str(x), quote=True)

def header_ru(dt) -> str:
    _RU_WD = ["понедельник","вторник","среда","четверг","пятница","суббота","воскресенье"]
    _RU_MM = ["января","февраля","марта","апреля","мая","июня","июля","августа","сентября","октября","ноября","декабря"]
    wd = _RU_WD[dt.weekday()]
    mm = _RU_MM[dt.month - 1]
    return f"🧭 Утренний фон — {wd}, {dt.day} {mm} {dt.year}, {dt:%H:%M} (Europe/Belgrade)"

def _to_float(x, default: Optional[float] = None) -> Optional[float]:
    try: return float(str(x).strip().replace(",", "."))
    except Exception: return default

def get_last_nonempty_row(sh, symbol: str, needed=("Avg_Price","Next_DCA_Price","Bank_Target_USDT","Bank_Fact_USDT")) -> Optional[dict]:
    sheet_name = BMR_SHEETS.get(symbol)
    if not sheet_name: return None
    try:
        ws = sh.worksheet(sheet_name)
        rows = ws.get_all_records()
        if not rows: return None
        for r in reversed(rows):
            if any(r.get(f) not in (None, "", 0, "0", "0.0") for f in needed):
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
                tgt = _to_float(r.get("Bank_Target_USDT"))
            if fac is None and r.get("Bank_Fact_USDT") not in (None, "", 0, "0", "0.0"):
                fac = _to_float(r.get("Bank_Fact_USDT"))
            if tgt is not None and fac is not None:
                break
        return tgt, fac
    except Exception:
        return None, None

def price_fmt(symbol: str, value: Optional[float]) -> str:
    if value is None: return "—"
    return f"{value:.{3 if symbol.endswith('JPY') else 5}f}"

# ---------- Calendar (using sheet CALENDAR when available) ----------
_FF_CACHE = {"at": 0, "data": []}
_FF_NEG   = {"until": 0}

def importance_is_high(val) -> bool:
    if val is None: return False
    if isinstance(val, (int,float)): return val >= 3
    s = str(val).strip().lower()
    return "high" in s or s == "3"

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

def build_calendar_for_symbols(symbols: List[str]) -> Dict[str, dict]:
    sh, _ = build_sheets_client(SHEET_ID)
    now = datetime.now(timezone.utc)
    w = CAL_WINDOW_MIN
    d1 = now - timedelta(minutes=w)
    d2 = now + timedelta(minutes=w)

    # все события из листа (предпочтительно)
    all_raw = read_calendar_rows_sheet(sh) if sh else []

    out = {}
    for sym in symbols:
        countries = {c.lower() for c in PAIR_COUNTRIES.get(sym, [])}
        sym_raw = [ev for ev in all_raw if ev["country"].lower() in countries] if all_raw else []
        around = [
            {**ev, "local": ev["utc"].astimezone(LOCAL_TZ) if LOCAL_TZ else ev["utc"]}
            for ev in sym_raw
            if importance_is_high(ev.get("importance")) and d1 <= ev["utc"] <= d2
        ]
        around.sort(key=lambda x: x["utc"])

        red_soon = any(abs((ev["utc"] - now).total_seconds())/60.0 <= 60 for ev in around)
        nearest_prev = nearest_next = None
        if not around and sym_raw:
            past = [e for e in sym_raw if importance_is_high(e.get("importance")) and e["utc"] < now]
            futr = [e for e in sym_raw if importance_is_high(e.get("importance")) and e["utc"] >= now]
            if past:
                p = max(past, key=lambda e: e["utc"])
                nearest_prev = {**p, "local": p["utc"].astimezone(LOCAL_TZ) if LOCAL_TZ else p["utc"]}
            if futr:
                n = min(futr, key=lambda e: e["utc"])
                nearest_next = {**n, "local": n["utc"].astimezone(LOCAL_TZ) if LOCAL_TZ else n["utc"]}

        out[sym] = {
            "events": around,
            "red_event_soon": red_soon,
            "nearest_prev": nearest_prev,
            "nearest_next": nearest_next,
        }
    return out

# ---------- NEWS pickers ----------
LABEL_BY_SYMBOL = {
    "USDJPY": "сегодня",
    "AUDUSD": "Азия",
    "EURUSD": "Европа",
    "GBPUSD": "Великобритания",
}

def _ru_title_hint(title: str) -> str:
    t = title
    repl = {
        r"\bRate Decision\b": "Решение по ставке",
        r"\bMonetary Policy\b": "ДКП",
        r"\bPolicy\b": "политика ЦБ",
        r"\bStatement\b": "заявление",
        r"\bMinutes\b": "протокол",
        r"\bGuidance\b": "прогноз/ориентир",
        r"\bIntervention\b": "интервенция",
        r"\bPress Conference\b": "пресс-конференция",
    }
    for pat, rep in repl.items():
        t = re.sub(pat, rep, t, flags=re.I)
    return t

def choose_top_news(symbol: str, news_rows: List[dict], now_utc: datetime) -> Optional[dict]:
    """Фильтр по окну и по источникам/ключевым словам + пересечение стран пары."""
    look_from = now_utc - timedelta(minutes=DIGEST_NEWS_LOOKBACK_MIN)
    countries = set(PAIR_COUNTRIES.get(symbol, []))
    cand = []
    for r in news_rows:
        if r["ts"] < look_from: 
            continue
        if r["source"] and r["source"].upper() not in DIGEST_NEWS_ALLOWED_SOURCES:
            continue
        if not (r["countries"] & {c.lower() for c in countries}):
            continue
        if not DIGEST_NEWS_KEYWORDS.search(r["title"] + " " + r["tags"]):
            # допускаем важные заголовки даже без ключевых слов, если impact high
            if r["importance"] != "high":
                continue
        score = 10
        # бонус за «high»
        if r["importance"] == "high": score += 5
        # свежесть
        age_min = max(1, int((now_utc - r["ts"]).total_seconds()//60))
        score += max(0, 5 - age_min//60)
        r["_score"] = score
        cand.append(r)
    if not cand:
        return None
    cand.sort(key=lambda x: (x["_score"], x["ts"]), reverse=True)
    best = cand[0]
    best_local = best["ts"].astimezone(LOCAL_TZ) if LOCAL_TZ else best["ts"]
    return {
        "time_str": best_local.strftime("%H:%M"),
        "ru_title": _ru_title_hint(best["title"]) or best["title"],
        "label": LABEL_BY_SYMBOL.get(symbol, "сегодня"),
    }

# ---------- Рендер (СТРОГИЙ ФОРМАТ) ----------
def _risk_label(fa_level: str) -> str:
    L = (fa_level or "OK").upper()
    if L == "HIGH": return "🚧 высокий риск"
    if L == "CAUTION": return "⚠️ умеренный риск"
    return "✅ спокойно"

def _market_phrase(adx, st_dir, vol_z, atr1h) -> str:
    # компактная человеческая сводка
    trend = "движения ровные" if (adx or 0) < 20 else "заметное движение" if (adx or 0) < 25 else "выраженное движение"
    dir_ = "вверх" if (st_dir or "").lower().find("up")>=0 else ("вниз" if (st_dir or "").lower().find("down")>=0 else "вбок")
    vola = "ниже нормы" if (atr1h or 1.0)<0.8 else "около нормы" if (atr1h or 1.0)<1.2 else "выше нормы"
    noise = "низкий" if (vol_z or 0)<0.5 else "умеренный" if (vol_z or 0)<1.5 else "повышенный"
    return f"{trend}, {dir_}; волатильность {vola}; рыночный шум {noise}."

def _quiet_or_next_line(sym_pack: dict) -> tuple[str, Optional[str]]:
    """Возвращает ('Тихое окно: 13:15–14:45' | 'Ближайшее важное время: …', nearest_time_for_msg)"""
    if sym_pack.get("events"):
        nearest = min(sym_pack["events"], key=lambda e: abs((e["utc"] - datetime.now(timezone.utc)).total_seconds()))
        t0 = nearest["local"] - timedelta(minutes=QUIET_BEFORE_MIN)
        t1 = nearest["local"] + timedelta(minutes=QUIET_AFTER_MIN)
        return f"Тихое окно: {t0:%H:%M}–{t1:%H:%M}", f"{nearest['local']:%H:%M} — {nearest['title']}"
    if sym_pack.get("nearest_next"):
        n = sym_pack["nearest_next"]
        return f"Ближайшее важное время: {n['local']:%H:%M} — {n['title']}", f"{n['local']:%H:%M} — {n['title']}"
    return "Тихое окно: —", None

def build_block(symbol: str, row: dict, cal_pack: dict, news_best: Optional[dict]) -> str:
    # поля из BMR-листа
    side = (row.get("Side") or row.get("SIDE") or "").upper() or "LONG"
    avg = _to_float(row.get("Avg_Price"))
    nxt = _to_float(row.get("Next_DCA_Price"))
    adx = _to_float(row.get("ADX_5m"), 18.0)
    rsi = _to_float(row.get("RSI_5m"), 50.0)
    volz = _to_float(row.get("Vol_z"), 1.0)
    atr1h = _to_float(row.get("ATR_1h"), 1.0)
    st = (row.get("Supertrend") or "").lower()
    fa_level_raw = (row.get("FA_Risk") or "").strip()
    fa_level = "OK" if fa_level_raw.lower().startswith("green") else ("CAUTION" if "yellow" in fa_level_raw.lower() or "amber" in fa_level_raw.lower() else ("HIGH" if "red" in fa_level_raw.lower() else "OK"))

    # верхняя строка
    head = f"{symbol[:3]}/{symbol[3:]} — {_risk_label(fa_level)}"

    # сводка рынка
    market = _market_phrase(adx, st, volz, atr1h)

    # позиция
    pos = f"{side} (на рост)" if side=="LONG" else "SHORT (на падение)"
    pos_line = f"Наша позиция: {pos}, средняя {price_fmt(symbol, avg)}; следующее докупление {price_fmt(symbol, nxt)}."

    # «что делаем»
    speed = "обычная" if fa_level=="OK" else ("сниженная" if fa_level=="CAUTION" else "минимальная")
    # тихое окно / ближайшее важное
    quiet_line, nearest_str = _quiet_or_next_line(cal_pack)
    act_line = f"Что делаем: скорость докупок — {speed}; " + ("тихого окна нет." if "—" in quiet_line and "Ближайшее" not in quiet_line else quiet_line.replace("Тихое окно: ", ""))

    # «что это значит для цены» — короткие подсказки
    hint_map = {
        "USDJPY": "до решения ФРС сильного тренда не ждём. Резкие слова ФРС — укрепляют доллар (часто рост USD/JPY); мягкие — ослабляют доллар (падение USD/JPY).",
        "AUDUSD": "комментарии РБА «держим ставку дольше» обычно поддерживают AUD (AUD/USD может подрасти).",
        "EURUSD": "жёсткий тон ЕЦБ — поддержка евро (EUR/USD вверх), мягкий — давление на евро (вниз).",
        "GBPUSD": "если Банку Англии «жарко» от зарплат и услуг — фунт крепче (GBP/USD вверх), мягче — фунт слабее.",
    }
    hint_line = f"Что это значит для цены: {hint_map.get(symbol,'работаем по плану; без сюрпризов движения сдержанные.')}"

    # План vs факт по банку
    tgt, fac = latest_bank_target_fact(sh, symbol)
    if tgt is None and fac is None:
        bank_line = "План vs факт по банку: —"
    else:
        mark = "—"
        if tgt and fac is not None:
            delta_pct = (fac - tgt)/tgt if tgt else 0
            if abs(delta_pct) <= 0.02: mark = "✅ в норме."
            elif delta_pct > 0.02:    mark = f"🚧 выше плана ({delta_pct:+.0%})."
            else:                     mark = f"⚠️ ниже плана ({delta_pct:+.0%})."
        bank_line = f"План vs факт по банку: план {int(tgt) if tgt else '—'} / факт {int(fac) if fac is not None else '—'} — {mark}"

    # Топ-новость
    if news_best:
        top_label = news_best["label"]
        top_line = f"Топ-новость ({top_label}): {news_best['time_str']} — {news_best['ru_title']}."
        plain_after = {
            "USDJPY": "Простыми словами: если ФРС жёстче ожидаемого — доллар дороже; мягче — доллар дешевле.",
            "AUDUSD": "Простыми словами: если РБА не спешит снижать ставку — австралийский доллар сильнее.",
            "EURUSD": "Простыми словами: «больше боимся инфляции» → евро сильнее; «больше боимся слабой экономики» → евро слабее.",
            "GBPUSD": "Простыми словами: больше тревоги по инфляции — фунт сильнее; меньше — слабее.",
        }.get(symbol, "")
    else:
        top_line = "Топ-новость: —"
        plain_after = ""

    # финальная строка (тихое окно или «ближайшее…»)
    tail_line = quiet_line

    bullets = [
        f"• Сводка рынка: {market}",
        f"• {pos_line}",
        f"• {act_line}",
        f"• {hint_line}",
        f"• {bank_line}",
        f"• {top_line}",
    ]
    if plain_after:
        bullets.append(plain_after)
    bullets.append(f"• {tail_line}")
    return "\n".join([head] + bullets)

# ---------- Основной дайджест ----------
def build_investor_digest(sh) -> str:
    now_utc = datetime.now(timezone.utc)
    header = header_ru(now_utc.astimezone(LOCAL_TZ)) if LOCAL_TZ else f"🧭 Утренний фон — {now_utc:%d %b %Y, %H:%M} (UTC)"

    # календарь и новости
    cal = build_calendar_for_symbols(SYMBOLS)
    news_rows = read_news_rows(sh)

    blocks: List[str] = [header]
    for sym in SYMBOLS:
        row = get_last_nonempty_row(sh, sym) or {}
        news_best = choose_top_news(sym, news_rows, now_utc)
        blocks.append(build_block(sym, row, cal.get(sym, {}), news_best))
        blocks.append("⸻")

    # Сводка календаря «на сегодня» (по ближайшим событиям)
    today_lines = ["Календарь «на сегодня»:"] 
    added = False
    for sym in SYMBOLS:
        pack = cal.get(sym, {})
        evs = pack.get("events") or ([pack.get("nearest_next")] if pack.get("nearest_next") else [])
        for ev in evs[:2]:
            if not ev: continue
            added = True
            today_lines.append(f"• {ev['local']:%H:%M} — {ev['title']} ({ev['country'].upper()[:3]})")
    if added:
        blocks.append("\n".join(today_lines))
        blocks.append("Главная мысль дня: до ключевых событий — аккуратно; после пресс-конференций возвращаемся к обычному режиму, если не будет сюрпризов.")

    # убрать последний разделитель
    out = "\n\n".join(blocks).rstrip()
    if out.endswith("⸻"):
        out = out[:-1].rstrip()
    return out

# ---------- Команды ----------
HELP_TEXT = (
    "Команды:\n"
    "/settotal 2800 — задать общий банк (мастер-чат)\n"
    "/setweights jpy=40 aud=25 eur=20 gbp=15 — целевые веса\n"
    "/weights — показать веса\n"
    "/alloc — расчёт сумм и запись в Sheets\n"
    "/digest — утренний инвесторский дайджест\n"
    "/news_diag — диагностика NEWS (окно и фильтры)\n"
    "/init_sheet — создать/проверить лист\n"
    "/sheet_test — тестовая запись\n"
    "/diag — диагностика LLM/Sheets/News\n"
    "/ping — pong"
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

def human_readable_weights(w: Dict[str, int]) -> str:
    return f"JPY {w.get('JPY', 0)} / AUD {w.get('AUD', 0)} / EUR {w.get('EUR', 0)} / GBP {w.get('GBP', 0)}"

async def cmd_settotal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if MASTER_CHAT_ID and update.effective_chat and update.effective_chat.id != MASTER_CHAT_ID:
        await update.message.reply_text("Эта команда доступна только в мастер-чате.")
        return
    try:
        total = float((update.message.text or "").split()[1])
    except Exception:
        await update.message.reply_text("Пример: /settotal 2800")
        return
    STATE["total"] = total
    await update.message.reply_text(f"OK. Общий банк = {total:.2f} USDT.\nИспользуйте /alloc для расчёта по чатам.")

async def cmd_setweights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if MASTER_CHAT_ID and update.effective_chat and update.effective_chat.id != MASTER_CHAT_ID:
        await update.message.reply_text("Эта команда доступна только в мастер-чате.")
        return
    text = (update.message.text or "")[len("/setweights"):].strip().lower()
    new_w = STATE["weights"].copy()
    try:
        for token in text.split():
            if "=" not in token: continue
            k, v = token.split("=", 1)
            k, v = k.strip().upper(), int(v.strip())
            if k in ("JPY","AUD","EUR","GBP"): new_w[k] = v
    except Exception:
        await update.message.reply_text("Пример: /setweights jpy=40 aud=25 eur=20 gbp=15")
        return
    STATE["weights"] = new_w
    await update.message.reply_text(f"Целевые веса обновлены: {human_readable_weights(new_w)}")

def split_total_by_weights(total: float, weights: Dict[str, int]) -> Dict[str, float]:
    s = max(sum(weights.values()), 1)
    return {
        "USDJPY": round(total * weights.get("JPY", 0) / s, 2),
        "AUDUSD": round(total * weights.get("AUD", 0) / s, 2),
        "EURUSD": round(total * weights.get("EUR", 0) / s, 2),
        "GBPUSD": round(total * weights.get("GBP", 0) / s, 2),
    }

async def cmd_alloc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total = float(STATE["total"])
    if total <= 0:
        await update.message.reply_text("Сначала задайте общий банк: /settotal 2800")
        return
    w = STATE["weights"]
    alloc = split_total_by_weights(total, w)
    lines = [f"Целевые веса: {human_readable_weights(w)}","", "Распределение:"]
    for sym,v in alloc.items():
        lines.append(f"{sym} → {v} USDT → команда в чат {sym}: /setbank {v}")
    await update.message.reply_text("\n".join(lines))
    sh, _ = build_sheets_client(SHEET_ID)
    if sh:
        try:
            append_row(sh, SHEET_WS, [
                datetime.utcnow().isoformat(timespec="seconds")+"Z",
                str(update.effective_chat.id), "alloc", f"{total:.2f}",
                json.dumps(w, ensure_ascii=False), json.dumps(alloc, ensure_ascii=False),
            ])
        except Exception as e:
            log.warning("append_row alloc failed: %s", e)

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
    sh, _ = build_sheets_client(SHEET_ID)
    if not sh: return "NEWS: ❌ sheet not available"
    rows = read_news_rows(sh)
    now_utc = datetime.now(timezone.utc)
    look_from = now_utc - timedelta(minutes=DIGEST_NEWS_LOOKBACK_MIN)
    filt = [r for r in rows if r["ts"]>=look_from and r["source"] in DIGEST_NEWS_ALLOWED_SOURCES]
    return f"NEWS: {len(filt)} in window ({DIGEST_NEWS_LOOKBACK_MIN}m, sources={len(DIGEST_NEWS_ALLOWED_SOURCES)})"

async def cmd_diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        ok = await llm_ping()
        llm_line = "LLM: ✅ ok" if ok else "LLM: ❌ no key"
    except Exception:
        llm_line = "LLM: ❌ error"
    sheets_line = sheets_diag_text()
    news_line = await _diag_news_line()
    await update.message.reply_text(f"{llm_line}\n{sh   eets_line}\n{news_line}")

async def cmd_news_diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sh, _ = build_sheets_client(SHEET_ID)
    if not sh:
        await update.message.reply_text("NEWS: sheet недоступен")
        return
    rows = read_news_rows(sh)
    now_utc = datetime.now(timezone.utc)
    picks = []
    for sym in SYMBOLS:
        top = choose_top_news(sym, rows, now_utc)
        label = LABEL_BY_SYMBOL.get(sym,"")
        picks.append(f"{sym}: " + (f"{top['time_str']} — {top['ru_title']} ({label})" if top else "—"))
    await update.message.reply_text("NEWS picks:\n"+"\n".join(picks))

async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sh, _ = build_sheets_client(SHEET_ID)
    if not sh:
        await update.message.reply_text("Sheets недоступен: не могу собрать инвесторский дайджест.")
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    try:
        msg = build_investor_digest(sh)
        # делим на части для Telegram
        parts, cur, L = [], [], 0
        for para in msg.split("\n\n"):
            block = para + "\n\n"
            if L + len(block) > 3500 and cur:
                parts.append("".join(cur).rstrip())
                cur, L = [], 0
            cur.append(block); L += len(block)
        if cur: parts.append("".join(cur).rstrip())
        for chunk in parts:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=chunk)
    except Exception as e:
        await update.message.reply_text(f"Ошибка сборки дайджеста: {e}")

# ---------- Сервис: расписание утреннего дайджеста ----------
async def _set_bot_commands(app: Application):
    cmds = [
        BotCommand("start","Запуск бота"),
        BotCommand("help","Список команд"),
        BotCommand("ping","Проверка связи"),
        BotCommand("settotal","Задать общий банк (мастер-чат)"),
        BotCommand("setweights","Задать целевые веса (мастер-чат)"),
        BotCommand("weights","Показать целевые веса"),
        BotCommand("alloc","Рассчитать распределение банка"),
        BotCommand("digest","Утренний дайджест (investor)"),
        BotCommand("news_diag","Диагностика новостей для дайджеста"),
        BotCommand("init_sheet","Создать/проверить лист в Google Sheets"),
        BotCommand("sheet_test","Тестовая запись в лист"),
        BotCommand("diag","Диагностика LLM/Sheets/News"),
    ]
    try: await app.bot.set_my_commands(cmds)
    except Exception as e: log.warning("set_my_commands failed: %s", e)

async def morning_digest_scheduler(app: Application):
    from datetime import datetime as _dt, timedelta as _td, time as _time
    while True:
        if LOCAL_TZ:
            now = _dt.now(LOCAL_TZ)
            target_time = _time(MORNING_HOUR, MORNING_MINUTE, tzinfo=LOCAL_TZ)
        else:
            now = _dt.now(timezone.utc)
            target_time = _time(MORNING_HOUR, MORNING_MINUTE, tzinfo=timezone.utc)
        target = _dt.combine(now.date(), target_time)
        if now >= target: target += _td(days=1)
        wait_s = (target - now).total_seconds()
        await asyncio.sleep(max(1.0, wait_s))
        try:
            sh, _ = build_sheets_client(SHEET_ID)
            if sh:
                msg = build_investor_digest(sh)
                for chunk in [msg[i:i+3500] for i in range(0, len(msg), 3500)]:
                    await app.bot.send_message(chat_id=MASTER_CHAT_ID, text=chunk)
            else:
                await app.bot.send_message(chat_id=MASTER_CHAT_ID, text="Sheets недоступен: утренний дайджест пропущен.")
        except Exception as e:
            try: await app.bot.send_message(chat_id=MASTER_CHAT_ID, text=f"Ошибка утреннего дайджеста: {e}")
            except Exception: pass

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
    app.add_handler(CommandHandler("weights", lambda u,c: u.message.reply_text("Целевые веса: " + human_readable_weights(STATE['weights']))))
    app.add_handler(CommandHandler("alloc", cmd_alloc))
    app.add_handler(CommandHandler("digest", cmd_digest))
    app.add_handler(CommandHandler("news_diag", cmd_news_diag))
    app.add_handler(CommandHandler("diag", cmd_diag))
    app.add_handler(CommandHandler("init_sheet", lambda u,c: u.message.reply_text("Листы создаются автоматически при первой записи.")))
    app.add_handler(CommandHandler("sheet_test", lambda u,c: u.message.reply_text("ok")))
    return app

def main():
    log.info("Fund bot is running…")
    app = build_application()
    app.run_polling()

if __name__ == "__main__":
    main()
