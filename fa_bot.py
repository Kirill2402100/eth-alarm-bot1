# fa_bot.py
# -*- coding: utf-8 -*-
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

# --- Таймзона ---
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Europe/Belgrade")) if ZoneInfo else None
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

# --- HTTP (резерв, если понадобится) ---
try:
    import requests  # noqa
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
CAL_WS_OUT = os.getenv("CAL_WS_OUT", "CALENDAR").strip() or "CALENDAR"

# страны по парам — для фильтрации NEWS/CALENDAR
PAIR_COUNTRIES = {
    "USDJPY": {"united states", "japan"},
    "AUDUSD": {"australia", "united states"},
    "EURUSD": {"euro area", "united states"},
    "GBPUSD": {"united kingdom", "united states"},
}

# Ключевые слова/источники, синхронизированы с calendar_collector.py
KW_RE = re.compile(os.getenv(
    "FA_NEWS_KEYWORDS",
    # + расширения: fomc/mpc, policy statement(s), rate statement, cash rate
    "rate decision|monetary policy|bank rate|policy decision|unscheduled|emergency|"
    "intervention|FX intervention|press conference|policy statement|policy statements|"
    "rate statement|cash rate|fomc|mpc"
), re.I)
ALLOWED_SOURCES = {
    s.strip().upper()
    for s in os.getenv("FA_NEWS_SOURCES", "US_FED_PR,ECB_PR,BOE_PR,BOJ_PR,RBA_MR,US_TREASURY,JP_MOF_FX").split(",")
    if s.strip()
}
NEWS_TTL_MIN = int(os.getenv("FA_NEWS_TTL_MIN", "120") or "120")

# -------------------- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ --------------------
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

def _top_news_for_pair(sh, pair: str, now_utc: Optional[datetime] = None) -> str:
    """
    Возвращает строку вида 'HH:MM — Title (SRC)' для КОНКРЕТНОЙ пары.
    1) Ищем свежую high-новость из NEWS за TTL (по странам пары).
    2) Если нет — ближайшее событие из CALENDAR (по странам пары) или крайнее прошедшее.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(minutes=NEWS_TTL_MIN)
    countries = PAIR_COUNTRIES.get(pair, set())

    # 1) NEWS
    try:
        rows = sh.worksheet("NEWS").get_all_records()
    except Exception:
        rows = []
    best = None
    for r in rows:
        try:
            ts = datetime.fromisoformat(str(r.get("ts_utc")).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if ts < cutoff:
            continue
        src = str(r.get("source", "")).upper().strip()
        if ALLOWED_SOURCES and src not in ALLOWED_SOURCES:
            continue
        title = str(r.get("title", "")).strip()
        tags  = str(r.get("tags", "")).strip()
        kw_ok = bool(KW_RE.search(f"{title} {tags}"))
        # Разрешаем «якорные» источники ЦБ даже без ключевых слов (иногда заголовки без явных ключей)
        if not kw_ok and src in {"US_FED_PR", "ECB_PR", "BOE_PR", "RBA_MR", "BOJ_PR"}:
            kw_ok = True
        if not kw_ok:
            continue
        row_cty = {x.strip().lower() for x in str(r.get("countries", "")).split(",") if x.strip()}
        if not (row_cty & countries):
            # Для долларовых пар разрешаем чисто американские новости (Fed/Treasury и т.п.)
            if "USD" in pair and "united states" in row_cty:
                pass
            else:
                continue
        if (best is None) or (ts > best["ts"]):
            best = {"ts": ts, "title": title, "src": src}
    if best:
        lt = best["ts"].astimezone(LOCAL_TZ) if LOCAL_TZ else best["ts"]
        return f"{lt:%H:%M} — {best['title']} ({best['src']})"

    # 2) CALENDAR (fallback: ближайшее ВПЕРЕДИ; если нет — последнее ПРОШЕДШЕЕ)
    try:
        events = sh.worksheet(CAL_WS_OUT).get_all_records()
    except Exception:
        events = []
    soon = None
    last_past = None
    for e in events:
        try:
            dt = datetime.fromisoformat(str(e.get("utc_iso")).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if str(e.get("country", "")).strip().lower() not in countries:
            continue
        if dt <= now_utc:
            if (last_past is None) or (dt > last_past["ts"]):
                last_past = {"ts": dt, "title": str(e.get("title", "")).strip(), "src": str(e.get("source", "")).strip() or "cal"}
        else:
            if (soon is None) or (dt < soon["ts"]):
                soon = {"ts": dt, "title": str(e.get("title", "")).strip(), "src": str(e.get("source", "")).strip() or "cal"}
    if soon:
        lt = soon["ts"].astimezone(LOCAL_TZ) if LOCAL_TZ else soon["ts"]
        return f"{lt:%Y-%m-%d %H:%M} — {soon['title']} ({soon['src']})"
    if last_past:
        lt = last_past["ts"].astimezone(LOCAL_TZ) if LOCAL_TZ else last_past["ts"]
        return f"{lt:%H:%M} — {last_past['title']} ({last_past['src']})"
    return ""

# -------------------- Чтение FA_Signals --------------------
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
            continue  # протухло
        out[pair] = {
            "risk":        (str(r.get("risk", "Green")).capitalize()),
            "bias":        (str(r.get("bias", "neutral")).lower()),
            "dca_scale":   float(r.get("dca_scale") or 1.0),
            "reserve_off": bool(int(r.get("reserve_off") or 0)),
            "reason":      str(r.get("reason", "base")),
            "updated_at":  upd_raw,
        }
    return out

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

def _to_float(x, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        return float(str(x).strip().replace(",", "."))
    except Exception:
        return default

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
            if any(r.get(f) not in (None, "", 0, "0", "0.0") for f in needed_fields):
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
    if value is None:
        return "—"
    return f"{value:.{3 if symbol.endswith('JPY') else 5}f}"

_RU_WD = ["понедельник","вторник","среда","четверг","пятница","суббота","воскресенье"]
_RU_MM = ["января","февраля","марта","апреля","мая","июня","июля","августа","сентября","октября","ноября","декабря"]

def header_ru(dt) -> str:
    wd = _RU_WD[dt.weekday()]
    mm = _RU_MM[dt.month - 1]
    return f"🧭 Утренний фон — {wd}, {dt.day} {mm} {dt.year}, {dt:%H:%M} (Europe/Belgrade)"

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
    if ap <= 0.02:
        mark = "✅ в норме."
    elif ap <= 0.05:
        mark = f"⚠️ {'ниже' if delta < 0 else 'выше'} плана ({delta:+.0%})."
    else:
        mark = f"🚧 {'ниже' if delta < 0 else 'выше'} плана ({delta:+.0%})."
    return f"план {tt:g} / факт {ff:g} — {mark}"

def _symbol_hints(symbol: str) -> tuple[str, str]:
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

def render_morning_pair_block(sh, pair, row: dict, fa_data: dict) -> str:
    risk = fa_data.get("risk", "Green")
    bias = fa_data.get("bias", "neutral")
    dca_scale = float(fa_data.get("dca_scale", 1.0))
    reserve_on_str = "OFF" if fa_data.get("reserve_off") else "ON"
    icon = _fa_icon(risk)

    title = f"{_pair_to_title(pair)} — {icon} фон {risk.lower()}, bias: {bias.upper()}"

    side = (row.get("Side") or row.get("SIDE") or "").upper() or "LONG"
    avg = price_fmt(pair, _to_float(row.get("Avg_Price"), None))
    nxt = price_fmt(pair, _to_float(row.get("Next_DCA_Price"), None))
    what_means, simple_words = _symbol_hints(pair)
    plan_fact = _plan_vs_fact_line(sh, pair)

    lines = [title]
    lines.append("•\tСводка рынка: движения ровные, резких скачков не ждём до США.")
    lines.append(f"•\tНаша позиция: {side} (на {'рост' if side=='LONG' else 'падение'}), средняя {avg}; следующее докупление {nxt}.")
    lines.append(f"•\tЧто делаем сейчас: {'тихое окно' if risk!='Green' else 'тихое окно не требуется'}; reserve {reserve_on_str}; dca_scale <b>{dca_scale:.2f}</b>.")

    top_news_line = _top_news_for_pair(sh, pair, now_utc=datetime.now(timezone.utc))
    if top_news_line:
        lines.append(f"•\tТоп-новость: {top_news_line}.")

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
        block = render_morning_pair_block(sh, sym, row, fa_data)
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
        try:
            rows = sh.worksheet("NEWS").get_all_records()
        except Exception:
            rows = []
        return f"NEWS: ✅ {len(rows)} строк в листе NEWS; TTL {NEWS_TTL_MIN} мин.; источники: {', '.join(sorted(ALLOWED_SOURCES))}"
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

# -------------------- Планировщик утренника --------------------
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

# -------------------- СТАРТ --------------------
def build_application() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")
    builder = Application.builder().token(BOT_TOKEN)
    if _RATE_LIMITER_AVAILABLE:
        builder = builder.rate_limiter(AIORateLimiter())
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
