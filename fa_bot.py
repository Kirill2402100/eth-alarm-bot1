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

# --- LLM-клиент (опционально) ---
try:
    from llm_client import generate_digest, explain_pair_event
    _LLM_IMPORTED = True
except Exception:
    _LLM_IMPORTED = False
    async def generate_digest(*args, **kwargs) -> str:
        return "⚠️ LLM сейчас недоступен (нет llm_client.py)."
    async def explain_pair_event(*args, **kwargs) -> str:
        return ""  # Fallback to empty string


# -------------------- ЛОГИ --------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL","INFO"),
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
LLM_TOKEN_BUDGET_PER_DAY = int(os.getenv("LLM_TOKEN_BUDGET_PER_DAY", "30000") or "30000")

SYMBOLS = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]

# --- Рекомендации по весам (ENV) ---
RECO_ENABLED              = (os.getenv("RECO_ENABLED","1").lower() in ("1","true","yes","on"))
RECO_MIN_PERSIST_MIN    = int(os.getenv("RECO_MIN_PERSIST_MIN","120") or "120")
RECO_COOLDOWN_MIN       = int(os.getenv("RECO_COOLDOWN_MIN","360") or "360")
RECO_POLL_MIN           = int(os.getenv("RECO_POLL_MIN","10") or "10")
RECO_MIN_SHIFT_PCT      = float(os.getenv("RECO_MIN_SHIFT_PCT","0.06") or "0.06")
RECO_MULT_GREEN         = float(os.getenv("RECO_MULT_GREEN","1.0") or "1.0")
RECO_MULT_AMBER         = float(os.getenv("RECO_MULT_AMBER","0.90") or "0.90")
RECO_MULT_RED           = float(os.getenv("RECO_MULT_RED","0.75") or "0.75")
RECO_TRACK_WS           = os.getenv("RECO_TRACK_WS","FA_Reco_Track").strip() or "FA_Reco_Track"

ASSET_BY_PAIR = {"USDJPY":"JPY","AUDUSD":"AUD","EURUSD":"EUR","GBPUSD":"GBP"}
PAIR_BY_ASSET = {"JPY":"USDJPY","AUD":"AUDUSD","EUR":"EURUSD","GBP":"GBPUSD"}

BMR_SHEETS = {
    "USDJPY": os.getenv("BMR_SHEET_USDJPY", "BMR_DCA_USDJPY"),
    "AUDUSD": os.getenv("BMR_SHEET_AUDUSD", "BMR_DCA_AUDUSD"),
    "EURUSD": os.getenv("BMR_SHEET_EURUSD", "BMR_DCA_EURUSD"),
    "GBPUSD": os.getenv("BMR_SHEET_GBPUSD", "BMR_DCA_GBPUSD"),
}

CAL_WS_OUT = os.getenv("CAL_WS_OUT", "CALENDAR").strip() or "CALENDAR"

_PRIMARY_COUNTRY = {
    "USDJPY": "japan",
    "AUDUSD": "australia",
    "EURUSD": "euro area",
    "GBPUSD": "united kingdom",
}

KW_RE = re.compile(os.getenv(
    "FA_NEWS_KEYWORDS",
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

def _clean_headline_for_llm(line: str) -> str:
    s = re.sub(r"^(\d{2}:\d{2}|\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*[—\-]\s*", "", line).strip()
    s = re.sub(r"\([^()]*\)\s*$", "", s).strip(" .–—-")
    return s

def _src_country_guess(src: str, title: str = "", fallback: str = "") -> str:
    s = (src or "").upper()
    if s in {"US_FED_PR", "US_TREASURY", "FOMC"}: return "united states"
    if s in {"ECB_PR", "ECB"}:                       return "euro area"
    if s in {"BOE_PR", "BOE"}:                       return "united kingdom"
    if s in {"RBA_MR", "RBA"}:                       return "australia"
    if s in {"BOJ_PR", "BOJ"}:                       return "japan"
    t = (title or "").lower()
    if "fomc" in t or "federal reserve" in t:     return "united states"
    if "ecb" in t or "european central bank" in t: return "euro area"
    if "bank of england" in t or "mpc" in t:        return "united kingdom"
    if "rba" in t or "reserve bank of australia" in t: return "australia"
    if "boj" in t or "bank of japan" in t or "mpm" in t: return "japan"
    return (fallback or "").lower().strip()

def _pick_top_event(sh, countries: set[str], now_utc: datetime, ttl_min: int = None) -> Optional[dict]:
    ttl_min = ttl_min or NEWS_TTL_MIN
    cutoff = now_utc - timedelta(minutes=ttl_min)
    try:
        rows = sh.worksheet("NEWS").get_all_records()
    except Exception:
        rows = []
    best = None
    for r in rows:
        try:
            ts = datetime.fromisoformat(str(r.get("ts_utc")).replace("Z","+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if ts < cutoff:
            continue
        src = str(r.get("source","")).upper().strip()
        if ALLOWED_SOURCES and src not in ALLOWED_SOURCES:
            continue
        title = str(r.get("title","")).strip()
        tags  = str(r.get("tags","")).strip()
        kw_ok = bool(KW_RE.search(f"{title} {tags}"))
        if not kw_ok and src in {"US_FED_PR","ECB_PR","BOE_PR","RBA_MR","BOJ_PR"}:
            kw_ok = True
        if not kw_ok:
            continue
        row_cty = {x.strip().lower() for x in str(r.get("countries","")).split(",") if x.strip()}
        if not (row_cty & countries):
            continue
        if (best is None) or (ts > best["ts"]):
            best = {
                "from": "news", "ts": ts, "title": title, "src": src, "url": str(r.get("url", "")).strip(),
                "country": (_src_country_guess(src, title) or next(iter(row_cty or []), "")),
                "consensus": (str(r.get("consensus","")).strip() or str(r.get("exp","")).strip() or ""),
            }
    if best:
        return best
    try:
        events = sh.worksheet(CAL_WS_OUT).get_all_records()
    except Exception:
        events = []
    soon, last_past = None, None
    for e in events:
        try:
            dt = datetime.fromisoformat(str(e.get("utc_iso")).replace("Z","+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        cty = str(e.get("country","")).strip().lower()
        if dt <= now_utc:
            if cty in countries and (last_past is None or dt > last_past["ts"]):
                last_past = {"from":"calendar","ts":dt,"title":str(e.get("title","")).strip(), "url": str(e.get("url","")).strip(),
                             "src":str(e.get("source","")).strip() or "CAL","country":cty, "consensus": str(e.get("consensus","")).strip()}
            continue
        if cty not in countries:
            continue
        if (soon is None) or (dt < soon["ts"]):
            soon = {"from":"calendar","ts":dt,"title":str(e.get("title","")).strip(), "url": str(e.get("url","")).strip(),
                    "src":str(e.get("source","")).strip() or "CAL","country":cty, "consensus": str(e.get("consensus","")).strip()}
    return soon or last_past

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
            return info, "env:GOOGLE_CREDENTIALS_JSON_B64", info.get("client_email")
        except Exception as e:
            return None, f"b64 present but decode/json error: {e}", None
    for name in ("GOOGLE_CREDENTIALS_JSON", "GOOGLE_CREDENTIALS"):
        raw = _env(name)
        if raw:
            try:
                info = json.loads(raw)
                return info, f"env:{name}", info.get("client_email")
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
        ws.update(range_name="A1", values=[SHEET_HEADERS])
        return ws, True
    except Exception as e:
        raise RuntimeError(f"ensure_worksheet error: {e}")

def append_row(sh, title: str, row: list):
    ws, _ = ensure_worksheet(sh, title)
    ws.append_row(row, value_input_option="RAW")

def _set_bank_target_in_bmr(sh, symbol: str, amount: float):
    sheet_name = BMR_SHEETS.get(symbol)
    if not sheet_name:
        return
    ws = sh.worksheet(sheet_name)
    hdr = ws.row_values(1)
    if "Bank_Target_USDT" not in hdr:
        return
    col = hdr.index("Bank_Target_USDT") + 1
    vals = ws.get_all_values()
    last = len(vals)
    while last > 1 and not any((c or "").strip() for c in vals[last-1]):
        last -= 1
    row_ix = max(2, last)
    ws.update_cell(row_ix, col, float(amount))

# --- helpers: Google Sheets → FA/NEWS ---
def _fa_icon(risk: str) -> str:
    return {"Green": "🟢", "Amber": "🟡", "Red": "🔴"}.get((risk or "").capitalize(), "⚪️")

def _top_news_for_pair(sh, pair: str, now_utc: datetime | None = None) -> tuple[str, Optional[str], Optional[str]]:
    now_utc = now_utc or datetime.now(timezone.utc)
    primary = _PRIMARY_COUNTRY.get(pair)
    countries = {primary} if primary else set()
    ev = _pick_top_event(sh, countries, now_utc, ttl_min=NEWS_TTL_MIN) if countries else None
    if not ev:
        return "", None, None
    ts_local = ev["ts"].astimezone(LOCAL_TZ) if LOCAL_TZ else ev["ts"]
    href = ev.get("url","").strip()
    title_text = _h(ev['title'])
    title_html = f'<a href="{_h(href)}">{title_text}</a>' if href else title_text
    if ev["from"] == "news":
        line = f"{ts_local:%H:%M} — {title_html} ({_h(ev['src'])})"
    else:
        line = f"{ts_local:%Y-%m-%d %H:%M} — {title_html} ({_h(ev['src'])})"
    return line, (ev.get("country") or None), (ev.get("consensus") or None)

def _to_bool(v) -> bool:
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def _read_fa_signals_from_sheet(sh) -> Dict[str, dict]:
    try:
        ws = sh.worksheet("FA_Signals")
        rows = ws.get_all_records()
    except Exception:
        return {}
    now = datetime.now(timezone.utc)
    out: Dict[str, dict] = {}
    for r in rows:
        pair = str(r.get("pair", "")).upper()
        if not pair: continue
        ttl = int(r.get("ttl") or 0)
        upd_raw = str(r.get("updated_at", "")).strip()
        try:
            upd_ts = datetime.fromisoformat(upd_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            upd_ts = None
        if ttl and upd_ts and now > upd_ts + timedelta(minutes=ttl):
            continue
        out[pair] = {
            "risk": (str(r.get("risk", "Green")).capitalize()),
            "bias": (str(r.get("bias", "neutral")).lower()),
            "dca_scale": float(r.get("dca_scale") or 1.0),
            "reserve_off": _to_bool(r.get("reserve_off")),
            "reason": str(r.get("reason", "base")),
            "updated_at": upd_raw,
        }
    return out

# -------------------- УТИЛИТЫ --------------------
def _h(x) -> str:
    return _html_escape(str(x), quote=True)

def _strip_html_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)

def human_readable_weights(w: Dict[str, int]) -> str:
    return f"JPY {w.get('JPY', 0)} / AUD {w.get('AUD', 0)} / EUR {w.get('EUR', 0)} / GBP {w.get('GBP', 0)}"

# ---- helpers: рекомендации по весам ----
def _normalize_int_weights(weight_floats: Dict[str, float], keep_total: int) -> Dict[str, int]:
    floors = {k: int(v) for k, v in weight_floats.items()}
    diff = keep_total - sum(floors.values())
    if diff == 0: return floors
    fracs = sorted(((k, weight_floats[k] - floors[k]) for k in floors), key=lambda x: x[1], reverse=True)
    i, step, diff = 0, 1 if diff > 0 else -1, abs(diff)
    while diff > 0 and fracs:
        k, _ = fracs[i % len(fracs)]
        floors[k] += step
        diff -= 1
        i += 1
    return floors

def _reco_multipliers_for_pair_risk(risk: str) -> float:
    r = (risk or "Green").capitalize()
    return RECO_MULT_RED if r == "Red" else RECO_MULT_AMBER if r == "Amber" else RECO_MULT_GREEN

def compute_recommended_weights(cur: Dict[str,int], fa_sheet: Dict[str,dict]) -> Dict[str,int]:
    total = max(sum(cur.values()), 1)
    floated = {}
    for asset, w in cur.items():
        pair = PAIR_BY_ASSET.get(asset)
        risk = (fa_sheet.get(pair, {}) or {}).get("risk", "Green")
        m = _reco_multipliers_for_pair_risk(risk)
        floated[asset] = w * m
    norm_sum = sum(floated.values()) or 1.0
    rescaled = {k: (v / norm_sum) * total for k, v in floated.items()}
    return _normalize_int_weights(rescaled, total)

def _weights_shift_pct(a: Dict[str,int], b: Dict[str,int]) -> float:
    total = max(sum(a.values()), 1)
    delta = sum(abs(a.get(k,0) - b.get(k,0)) for k in {"JPY","AUD","EUR","GBP"})
    return delta / total

def _fmt_age(mins: int) -> str:
    h, m = divmod(max(0, mins), 60)
    return f"{h}ч{m:02d}м"

def _load_reco_track(sh):
    headers = ["pair","risk","started_at","last_seen","announced_at"]
    ws, _ = ensure_worksheet(sh, RECO_TRACK_WS)
    try: rows = ws.get_all_records()
    except Exception: rows = []
    d = {}
    for r in rows:
        p = str(r.get("pair","")).upper()
        if p: d[p] = { "risk": str(r.get("risk","")).capitalize(), "started_at": str(r.get("started_at","")).strip(),
                       "last_seen": str(r.get("last_seen","")).strip(), "announced_at": str(r.get("announced_at","")).strip(), }
    try: ws.update(range_name="A1", values=[headers])
    except Exception: pass
    return ws, d

def _save_reco_track(ws, d: Dict[str,dict]):
    order = ["USDJPY","AUDUSD","EURUSD","GBPUSD"]
    out = [[p, d.get(p,{}).get("risk",""), d.get(p,{}).get("started_at",""),
            d.get(p,{}).get("last_seen",""), d.get(p,{}).get("announced_at","")] for p in order]
    ws.update(range_name="A2", values=out)

async def _maybe_send_reco_message(app, sh, fa_sheet: Dict[str,dict]):
    if not RECO_ENABLED or not MASTER_CHAT_ID: return
    now = datetime.now(timezone.utc)
    ws, track = _load_reco_track(sh)
    changed = False
    for pair in SYMBOLS:
        risk = (fa_sheet.get(pair, {}) or {}).get("risk","Green").capitalize()
        t = track.get(pair, {})
        now_iso = now.isoformat(timespec="seconds")+"Z"
        if risk in ("Amber","Red"):
            if not t or t.get("risk") != risk or not t.get("started_at"):
                t = {"risk": risk, "started_at": now_iso, "last_seen": now_iso, "announced_at": t.get("announced_at","")}
            else:
                t["risk"], t["last_seen"] = risk, now_iso
            track[pair] = t; changed = True
        else:
            if pair in track:
                track[pair] = {"risk":"Green","started_at":"","last_seen":now_iso,"announced_at":""}; changed = True
    if changed: _save_reco_track(ws, track)
    reasons = []
    min_ok, cd_min = RECO_MIN_PERSIST_MIN, RECO_COOLDOWN_MIN
    for pair, row in track.items():
        if row.get("risk") not in ("Amber","Red") or not row.get("started_at"): continue
        try: start = datetime.fromisoformat(row["started_at"].replace("Z","+00:00")).astimezone(timezone.utc)
        except Exception: continue
        if (now - start).total_seconds() < min_ok * 60: continue
        announced_ok = True
        if row.get("announced_at"):
            try:
                ann = datetime.fromisoformat(row["announced_at"].replace("Z","+00:00")).astimezone(timezone.utc)
                announced_ok = (now - ann).total_seconds() >= cd_min * 60
            except Exception: pass
        if announced_ok: reasons.append((pair, row.get("risk"), int((now - start).total_seconds() // 60)))
    if not reasons: return
    cur_w, reco_w = STATE["weights"].copy(), compute_recommended_weights(STATE["weights"].copy(), fa_sheet)
    if _weights_shift_pct(cur_w, reco_w) < RECO_MIN_SHIFT_PCT: return
    rr = ", ".join([f"{p} — {r} ({_fmt_age(m)})" for p, r, m in reasons])
    cmd = f"/setweights jpy={reco_w['JPY']} aud={reco_w['AUD']} eur={reco_w['EUR']} gbp={reco_w['GBP']}"
    text = (f"⚖️ Рекомендация по весам (риск&gt;{_fmt_age(min_ok)})<br>"
            f"Текущие: { _h(human_readable_weights(cur_w)) }<br>"
            f"Рекомендованные: { _h(human_readable_weights(reco_w)) }<br>"
            f"Команда, если согласен: <code>{_h(cmd)}</code><br>"
            f"Основание: { _h(rr) }")
    try:
        await app.bot.send_message(chat_id=MASTER_CHAT_ID, text=text, parse_mode=ParseMode.HTML)
        now_iso = now.isoformat(timespec="seconds")+"Z"
        for pair, _, _ in reasons: track[pair]["announced_at"] = now_iso
        _save_reco_track(ws, track)
    except Exception as e:
        log.warning("reco notify failed: %s", e)

def split_total_by_weights(total: float, weights: Dict[str, int]) -> Dict[str, float]:
    s = max(sum(weights.values()), 1)
    return { sym: round(total * weights.get(ASSET_BY_PAIR[sym], 0) / s, 2) for sym in SYMBOLS }

def assert_master_chat(update: Update) -> bool:
    return not MASTER_CHAT_ID or (update.effective_chat and update.effective_chat.id == MASTER_CHAT_ID)

def _split_for_tg_html(msg: str, limit: int = 3500) -> List[str]:
    parts, cur, cur_len = [], [], 0
    for para in msg.split("\n\n"):
        block = (para + "\n\n")
        if cur_len + len(block) > limit and cur:
            parts.append("".join(cur).rstrip()); cur, cur_len = [], 0
        cur.append(block); cur_len += len(block)
    if cur: parts.append("".join(cur).rstrip())
    return parts

# -------------------- КОМАНДЫ --------------------
HELP_TEXT = ("/settotal 2800 — задать общий банк (мастер-чат).\n"
             "/setweights jpy=40 aud=25 eur=20 gbp=15 — выставить целевые веса.\n"
             "/weights — показать целевые веса.\n"
             "/alloc — расчёт сумм и готовые команды /setbank для торговых чатов.\n"
             "/digest — утренний дайджест (человеческий язык + события + новости).\n"
             "/digest pro — краткий «трейдерский» дайджест (по цифрам, LLM).\n"
             "/diag — диагностика LLM / Sheets / NEWS.\n")

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Привет! Я фунд-бот.\nТекущий чат id: <code>{update.effective_chat.id}</code>\n\nКоманды: /help", parse_mode=ParseMode.HTML)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

async def cmd_settotal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not assert_master_chat(update): return await update.message.reply_text("Эта команда доступна только в мастер-чате.")
    try:
        parts = update.message.text.strip().split(); total = float(parts[1]) if len(parts) > 1 else None
        if total is None: raise ValueError
    except Exception: return await update.message.reply_text("Пример: /settotal 2800")
    STATE["total"] = total
    await update.message.reply_text(f"OK. Общий банк = {total:.2f} USDT.\nИспользуйте /alloc для расчёта по чатам.")

async def cmd_setweights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not assert_master_chat(update): return await update.message.reply_text("Эта команда доступна только в мастер-чате.")
    text, new_w = update.message.text[len("/setweights"):].strip().lower(), STATE["weights"].copy()
    try:
        for token in text.split():
            if "=" not in token: continue
            k, v = token.split("=", 1); k, v = k.strip().upper(), int(v.strip())
            if k in ("JPY", "AUD", "EUR", "GBP"): new_w[k] = v
    except Exception: return await update.message.reply_text("Пример: /setweights jpy=40 aud=25 eur=20 gbp=15")
    STATE["weights"] = new_w
    await update.message.reply_text(f"Целевые веса обновлены: {human_readable_weights(new_w)}")

async def cmd_weights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Целевые веса: {human_readable_weights(STATE['weights'])}")

async def cmd_alloc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total = float(STATE["total"])
    if total <= 0: return await update.message.reply_text("Сначала задайте общий банк: /settotal 2800")
    w = STATE["weights"]
    alloc = split_total_by_weights(total, w)
    lines = [f"Целевые веса: {human_readable_weights(w)}", "\nРаспределение:"]
    lines.extend(f"{sym} → {alloc[sym]} USDT → команда в чат {sym}: /setbank {alloc[sym]}" for sym in SYMBOLS)
    await update.message.reply_text("\n".join(lines))
    if sh := await get_sheets():
        try:
            append_row(sh, SHEET_WS, [datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                      str(update.effective_chat.id), "alloc", f"{total:.2f}",
                                      json.dumps(w), json.dumps(alloc)])
            for sym, amt in alloc.items(): _set_bank_target_in_bmr(sh, sym, amt)
        except Exception as e: log.warning("alloc sheets update failed: %s", e)

# ---------- Вспомогательные для дайджеста ----------
_RU_WD = ["понедельник","вторник","среда","четверг","пятница","суббота","воскресенье"]
_RU_MM = ["января","февраля","марта","апреля","мая","июня","июля","августа","сентября","октября","ноября","декабря"]

def header_ru(dt) -> str:
    wd, mm = _RU_WD[dt.weekday()], _RU_MM[dt.month - 1]
    return f"🧭 Утренний фон — {wd}, {dt.day} {mm} {dt.year}, {dt:%H:%M} (Europe/Belgrade)"

def _to_float(x, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        return float(str(x).strip().replace(",", "."))
    except Exception:
        return default

def price_fmt(symbol: str, value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.{3 if symbol.endswith('JPY') else 5}f}"

_COL_ALIASES = {
    "Side": ["Side", "Position", "Dir"],
    "Avg_Price": ["Avg_Price","Avg price","AvgPrice","Average","Avg",
                  "Средняя","Средняя цена","Средняя_Цена","Средн.","Средняя по позиции"],
    "Next_DCA_Price": ["Next_DCA_Price","Next DCA Price","NextPrice","Next_Buy_Price",
                       "Next Entry","Next_Entry_Price","Next_Buy",
                       "Следующее докупление","След. докупление","Следующее усреднение","Следующая покупка"],
    "Bank_Target_USDT": ["Bank_Target_USDT","Bank Target USDT","Target_Bank_USDT","Bank_Target",
                         "План банк","План USDT","Целевой банк"],
    "Bank_Fact_USDT": ["Bank_Fact_USDT","Bank Fact USDT","Bank_Fact","Fact_Bank_USDT",
                       "Факт банк","Факт USDT","Текущий банк","Фактический банк"],
}

_ZERO_RE = re.compile(r"^0+(?:[.,]0+)?$")
def _is_zeroish(v) -> bool:
    if v is None: return True
    if isinstance(v, (int, float)): return abs(float(v)) < 1e-12
    s = str(v).strip().replace(",", ".")
    return bool(_ZERO_RE.match(s))

def _norm_side(v: str) -> str:
    t = str(v or "").strip().lower()
    if any(w in t for w in ("short","sell","шорт","прод")):
        return "SHORT"
    return "LONG"

def _last_nonempty(rows: List[dict], cand_names: List[str], treat_zero_empty=True):
    for r in reversed(rows):
        for n in cand_names:
            if n in r:
                v = r.get(n)
                if v is None: continue
                if str(v).strip() == "": continue
                if treat_zero_empty and _is_zeroish(v): continue
                return v
    return None

def read_latest_snapshot(sh, symbol: str) -> Optional[dict]:
    """Для каждого ключевого поля берём последнее непустое значение."""
    sheet_name = BMR_SHEETS.get(symbol)
    if not sheet_name: return None
    try:
        ws = sh.worksheet(sheet_name)
        rows = ws.get_all_records()
        if not rows: return None
        return {
            "Side": _norm_side(_last_nonempty(rows, _COL_ALIASES["Side"], treat_zero_empty=False)),
            "Avg_Price": _to_float(_last_nonempty(rows, _COL_ALIASES["Avg_Price"]), None),
            "Next_DCA_Price": _to_float(_last_nonempty(rows, _COL_ALIASES["Next_DCA_Price"]), None),
            "Bank_Target_USDT": _to_float(_last_nonempty(rows, _COL_ALIASES["Bank_Target_USDT"]), None),
            "Bank_Fact_USDT": _to_float(_last_nonempty(rows, _COL_ALIASES["Bank_Fact_USDT"], treat_zero_empty=False), None),
        }
    except Exception:
        return None

def latest_bank_target_fact(sh, symbol: str) -> tuple[Optional[float], Optional[float]]:
    snap = read_latest_snapshot(sh, symbol) or {}
    return snap.get("Bank_Target_USDT"), snap.get("Bank_Fact_USDT")

def _effect_hint(pair: str, origin: Optional[str]) -> str:
    o = (origin or "").lower()
    if pair == "USDJPY":
        if o in ("united states", ""): return "жёстче ФРС → USD/JPY вверх; мягче → вниз."
        if o == "japan": return "жёстче BoJ → иена сильнее → USD/JPY вниз; мягче → вверх."
    elif pair == "AUDUSD":
        if o == "australia": return "жёстче РБА → AUD/USD вверх; мягче → вниз."
        if o == "united states": return "жёстче ФРС → доллар сильнее → AUD/USD вниз; мягче → вверх."
    elif pair == "EURUSD":
        if o == "euro area": return "жёстче ЕЦБ → EUR/USD вверх; мягче → вниз."
        if o == "united states": return "жёстче ФРС → EUR/USD вниз; мягче → вверх."
    elif pair == "GBPUSD":
        if o == "united kingdom": return "жёстче Банк Англии → GBP/USD вверх; мягче → вниз."
        if o == "united states": return "жёстче ФРС → GBP/USD вниз; мягче → вверх."
    return "ястребиный тон укрепляет валюту эмитента, голубиный — ослабляет."

def _symbol_hints(symbol: str) -> tuple[str,str]:
    # Fallback for when no news/event is found at all
    if symbol == "USDJPY": return ("Новостей нет. Ориентир — ближайшее заседание FOMC.", "")
    if symbol == "AUDUSD": return ("Новостей нет. Ориентир — ближайшее заседание РБА.", "")
    if symbol == "EURUSD": return ("Новостей нет. Ориентир — ближайшее заседание ЕЦБ.", "")
    return ("Новостей нет. Ориентир — ближайшее заседание Банка Англии.", "")

def _pair_to_title(symbol: str) -> str: return f"<b>{symbol[:3]}/{symbol[3:]}</b>"

def _plan_vs_fact_line(sh, symbol: str) -> str:
    target, fact = latest_bank_target_fact(sh, symbol)
    if target is None and fact is None: return "план —; факт —."
    tt, ff = target or 0.0, fact or 0.0
    if tt <= 0: return f"план {tt:g} / факт {ff:g} — —."
    delta = (ff - tt) / tt
    if abs(delta) <= 0.02: mark = "✅ в норме."
    elif abs(delta) <= 0.05: mark = f"⚠️ {'ниже' if delta<0 else 'выше'} плана ({delta:+.0%})."
    else: mark = f"🚧 {'ниже' if delta<0 else 'выше'} плана ({delta:+.0%})."
    return f"план {tt:g} / факт {ff:g} — {mark}"

async def render_morning_pair_block(sh, pair, row: dict, fa_data: dict) -> str:
    risk, bias = fa_data.get("risk", "Green"), fa_data.get("bias", "neutral")
    dca, reserve = float(fa_data.get("dca_scale", 1.0)), "OFF" if fa_data.get("reserve_off") else "ON"
    title = f"{_pair_to_title(pair)} — {_fa_icon(risk)} фон {risk.lower()}, bias: {bias.upper()}"
    side = row.get("Side") or "LONG"
    avg  = price_fmt(pair, (row.get("Avg_Price") if row.get("Avg_Price") is not None else None))
    nxt  = price_fmt(pair, (row.get("Next_DCA_Price") if row.get("Next_DCA_Price") is not None else None))
    lines = [title,
             f"•\tСводка рынка: {'движения ровные, резких скачков не ждём до США.'}",
             f"•\tНаша позиция: {side} (на {'рост' if side=='LONG' else 'падение'}), средняя {avg}; следующее докупление {nxt}.",
             f"•\tЧто делаем сейчас: {'тихое окно' if risk!='Green' else 'тихое окно не требуется'}; reserve {reserve}; dca_scale <b>{dca:.2f}</b>."]
    top_line, origin, consensus = _top_news_for_pair(sh, pair, now_utc=datetime.now(timezone.utc))
    if top_line:
        lines.append(f"•\tТоп-новость: {top_line}.")
        clean_hl = _clean_headline_for_llm(_strip_html_tags(top_line)).strip()
        if len(clean_hl) < 12:
            # если после чистки заголовок почти пустой — возьмём «как есть»
            clean_hl = _strip_html_tags(top_line).strip()
        
        two_lines = ""
        try:
            two_lines = await explain_pair_event(pair=pair, headline=clean_hl, origin=(origin or ""), lang="ru", consensus=consensus)
        except Exception as e:
            log.warning("explain_pair_event call failed in fa_bot: %s", e)

        if two_lines and two_lines.strip() and len(two_lines.strip()) >= 10:
            lns = [ln.strip() for ln in two_lines.splitlines() if ln.strip()]
            for ln in lns[:2]:
                lines.append("•\t" + _h(ln))
        else:
            # единый фоллбэк в любом случае
            lines.append("•\t" + _h("Что это значит: " + _effect_hint(pair, origin)))
    else:
        what_means, _ = _symbol_hints(pair)
        lines.append(f"•\tЧто это значит для цены: {what_means}")
    lines.append(f"•\tПлан vs факт по банку: {_plan_vs_fact_line(sh, pair)}.")
    return "\n".join(lines)

def render_usd_block(sh) -> str:
    now_utc = datetime.now(timezone.utc)
    ev = _pick_top_event(sh, {"united states"}, now_utc, ttl_min=NEWS_TTL_MIN)
    if not ev:
        return "USD — общий фон: свежих новостей нет; смотрим к ближайшему FOMC."
    ts_local = ev["ts"].astimezone(LOCAL_TZ) if LOCAL_TZ else ev["ts"]
    when = f"{ts_local:%H:%M}" if ev["from"] == "news" else f"{ts_local:%Y-%m-%d %H:%M}"
    href, ttl = (ev.get("url") or "").strip(), _h(ev["title"])
    ttl_html = f'<a href="{_h(href)}">{ttl}</a>' if href else ttl
    line = f"USD — общий фон: {when} — {ttl_html} ({_h(ev['src'])})."
    hint = "В общем случае: чем жёстче риторика ФРС — тем сильнее доллар; мягче — слабее."
    return f"{line}\n{hint}"

def _largest_plan_fact_delta(sh) -> Optional[tuple[str, float, float, float]]:
    """Возвращает (symbol, plan, fact, delta) с максимальным |delta|."""
    best = None
    for sym in SYMBOLS:
        plan, fact = latest_bank_target_fact(sh, sym)
        if plan and plan > 0 and fact is not None:
            d = (fact - plan) / plan
            if (best is None) or (abs(d) > abs(best[3])):
                best = (sym, plan, fact, d)
    return best

def build_main_thought(sh, fa_sheet_data: dict) -> str:
    risks = {p: (fa_sheet_data.get(p, {}) or {}).get("risk", "Green") for p in SYMBOLS}
    reds    = [p for p, r in risks.items() if r == "Red"]
    ambers = [p for p, r in risks.items() if r == "Amber"]
    parts: list[str] = []
    if reds: parts.append("стоп докупок по " + ", ".join(reds))
    if ambers: parts.append("осторожность по " + ", ".join(ambers))
    if not parts: parts.append("фон спокойный, работаем по плану")
    ev = _pick_top_event(sh, {"united states"}, datetime.now(timezone.utc), ttl_min=NEWS_TTL_MIN)
    if ev:
        ts_local = ev["ts"].astimezone(LOCAL_TZ) if LOCAL_TZ else ev["ts"]
        when = f"{ts_local:%H:%M}" if ev["from"] == "news" else f"{ts_local:%Y-%m-%d %H:%M}"
        parts.append(f"драйвер: {when} — {_h(ev['title'])} ({_h(ev['src'])})")
    worst = _largest_plan_fact_delta(sh)
    if worst and abs(worst[3]) >= 0.05:
        sym, plan, fact, d = worst
        parts.append(f"наибольшее отклонение по {sym}: план {plan:g} / факт {fact:g} ({d:+.0%})")
    return "Главная мысль дня: " + "; ".join(parts) + "."

async def build_digest_text(sh, fa_sheet_data: dict) -> str:
    now_utc = datetime.now(timezone.utc)
    header = header_ru(now_utc.astimezone(LOCAL_TZ)) if LOCAL_TZ else f"🧭 Утренний фон — {now_utc.strftime('%d %b %Y, %H:%M')} (UTC)"
    parts: List[str] = [header]
    for i, sym in enumerate(SYMBOLS):
        row = read_latest_snapshot(sh, sym) or {}
        fa_data = fa_sheet_data.get(sym, {})
        block = await render_morning_pair_block(sh, sym, row, fa_data)
        parts.append(block)
        if i < len(SYMBOLS) - 1: parts.append("⸻")
    parts.append("⸻")
    parts.append(render_usd_block(sh))
    parts.append(build_main_thought(sh, fa_sheet_data))
    return "\n".join(parts)

# -------------------- /digest --------------------
async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = (update.message.text or "").split()
    pro = len(args) > 1 and args[1].lower() == "pro"
    if pro:
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            txt = await generate_digest(symbols=SYMBOLS, model=LLM_MINI, token_budget=LLM_TOKEN_BUDGET_PER_DAY)
            await update.message.reply_text(txt)
        except Exception as e: await update.message.reply_text(f"LLM ошибка: {e}")
        return
    sh = await get_sheets()
    if not sh: return await update.message.reply_text("Sheets недоступен: не могу собрать инвесторский дайджест.")
    try:
        fa_sheet = _read_fa_signals_from_sheet(sh)
        msg = await build_digest_text(sh, fa_sheet)
        for chunk in _split_for_tg_html(msg, 3500):
            await context.bot.send_message(chat_id=update.effective_chat.id, text=chunk, parse_mode=ParseMode.HTML)
        await _maybe_send_reco_message(context.application, sh, fa_sheet)
    except Exception as e:
        log.exception("Ошибка сборки дайджеста")
        await update.message.reply_text(f"Ошибка сборки дайджеста: {e}")

# -------------------- Диагностика --------------------
async def cmd_diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async def _llm_selfcheck() -> str:
        if not _LLM_IMPORTED:
            return "LLM: ⚠️ llm_client не импортирован"
        try:
            txt = await explain_pair_event(pair="EURUSD",
                                             headline="ECB monetary policy decision",
                                             origin="euro area", lang="ru")
            if txt and txt.strip():
                return "LLM: ✅ ответ получен"
            return "LLM: ⚠️ нет ответа (используется фоллбэк) — проверь модель/ключ в llm_client.py"
        except Exception as e:
            return f"LLM: ❌ error: {e}"
    llm_line = await _llm_selfcheck()
    await update.message.reply_text(f"{llm_line}\n{sheets_diag_text()}\n{await _diag_news_line()}")

def sheets_diag_text() -> str:
    sid_state = "set" if SHEET_ID else "empty"
    if not _GSHEETS_AVAILABLE: return f"Sheets: ❌ (libs not installed)"
    sh, src = build_sheets_client(SHEET_ID)
    if sh is None: return f"Sheets: ❌ (SID={sid_state}, source={src})"
    try:
        ws, created = ensure_worksheet(sh, SHEET_WS)
        return f"Sheets: ✅ ok (SID={sid_state}, {src}, ws={ws.title}:{'created' if created else 'exists'})"
    except Exception as e:
        return f"Sheets: ❌ (open ok, ws error: {e})"

async def _diag_news_line() -> str:
    def _read_news_rows_simple(sh) -> list[dict]:
        try: return sh.worksheet("NEWS").get_all_records()
        except Exception: return []
    try:
        sh, _ = build_sheets_client(SHEET_ID)
        if not sh: return "NEWS: ❌ Sheets недоступен"
        return f"NEWS: ✅ {len(_read_news_rows_simple(sh))} строк в листе NEWS; TTL {NEWS_TTL_MIN} мин."
    except Exception as e:
        return f"NEWS: ❌ ошибка: {e}"

async def cmd_init_sheet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not SHEET_ID: return await update.message.reply_text("SHEET_ID не задан.")
    sh, src = build_sheets_client(SHEET_ID)
    if not sh: return await update.message.reply_text(f"Sheets: ❌ {src}")
    try:
        ws, created = ensure_worksheet(sh, SHEET_WS)
        await update.message.reply_text(f"Sheets: ✅ ws='{ws.title}' {'создан' if created else 'уже есть'} ({src})")
    except Exception as e: await update.message.reply_text(f"Sheets: ❌ ошибка создания листа: {e}")

async def cmd_sheet_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sh, src = build_sheets_client(SHEET_ID)
    if not sh: return await update.message.reply_text(f"Sheets: ❌ {src}")
    try:
        append_row(sh, SHEET_WS, [datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                  str(update.effective_chat.id), "test", f"{STATE['total']:.2f}",
                                  json.dumps(STATE['weights']), "manual /sheet_test"])
        await update.message.reply_text("Sheets: ✅ записано (test row).")
    except Exception as e: await update.message.reply_text(f"Sheets: ❌ ошибка записи: {e}")

async def cmd_positions_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sh = await get_sheets()
    if not sh:
        return await update.message.reply_text("Sheets недоступен")
    lines = []
    for sym in SYMBOLS:
        snap = read_latest_snapshot(sh, sym) or {}
        lines.append(
            f"{sym}: side={snap.get('Side')}; "
            f"avg={price_fmt(sym, snap.get('Avg_Price'))}; "
            f"next={price_fmt(sym, snap.get('Next_DCA_Price'))}; "
            f"plan={snap.get('Bank_Target_USDT')}; "
            f"fact={snap.get('Bank_Fact_USDT')}"
        )
    await update.message.reply_text("\n".join(lines))

# -------------------- СТАРТ --------------------
async def _set_bot_commands(app: Application):
    cmds = [BotCommand(c, d) for c, d in [
        ("start", "Запуск бота"), ("help", "Список команд"), ("ping", "Проверка связи"),
        ("settotal", "Задать общий банк (мастер-чат)"), ("setweights", "Задать целевые веса (мастер-чат)"),
        ("weights", "Показать целевые веса"), ("alloc", "Рассчитать распределение банка"),
        ("digest", "Утренний дайджест (investor) / pro (trader)"),
        ("diag", "Диагностика LLM и Sheets"),
        ("positions_debug", "Отладка позиций из Sheets"),
    ]]
    try: await app.bot.set_my_commands(cmds)
    except Exception as e: log.warning("set_my_commands failed: %s", e)

async def morning_digest_scheduler(app: Application):
    from datetime import datetime as _dt, timedelta as _td, time as _time
    while True:
        if LOCAL_TZ: now, target_time = _dt.now(LOCAL_TZ), _time(MORNING_HOUR, MORNING_MINUTE, tzinfo=LOCAL_TZ)
        else: now, target_time = _dt.now(timezone.utc), _time(MORNING_HOUR, MORNING_MINUTE, tzinfo=timezone.utc)
        target = _dt.combine(now.date(), target_time)
        if now >= target: target += _td(days=1)
        await asyncio.sleep(max(1.0, (target - now).total_seconds()))
        try:
            sh, _ = build_sheets_client(SHEET_ID)
            if sh:
                fa_sheet_data = _read_fa_signals_from_sheet(sh)
                msg = await build_digest_text(sh, fa_sheet_data)
                for chunk in _split_for_tg_html(msg, 3500):
                    await app.bot.send_message(chat_id=MASTER_CHAT_ID, text=chunk, parse_mode=ParseMode.HTML)
                await _maybe_send_reco_message(app, sh, fa_sheet_data)
            else:
                await app.bot.send_message(chat_id=MASTER_CHAT_ID, text="Sheets недоступен: утренний дайджест пропущен.")
        except Exception as e:
            try: await app.bot.send_message(chat_id=MASTER_CHAT_ID, text=f"Ошибка утреннего дайджеста: {e}")
            except Exception: pass

async def reco_watch_scheduler(app: Application):
    if not RECO_ENABLED: return
    while True:
        try:
            sh, _ = build_sheets_client(SHEET_ID)
            if sh:
                fa = _read_fa_signals_from_sheet(sh)
                await _maybe_send_reco_message(app, sh, fa)
        except Exception: log.exception("reco_watch iteration failed")
        await asyncio.sleep(max(60, RECO_POLL_MIN*60))

async def _post_init(app: Application):
    # Команды ставим сразу
    await _set_bot_commands(app)
    # Фоновые циклы запускаем после возврата из post_init,
    # чтобы не ловить PTBUserWarning и не зависеть от JobQueue.
    async def _delayed_start():
        await asyncio.sleep(0)
        asyncio.create_task(morning_digest_scheduler(app))
        asyncio.create_task(reco_watch_scheduler(app))
    asyncio.get_running_loop().create_task(_delayed_start())

def build_application() -> Application:
    if not BOT_TOKEN: raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")
    builder = Application.builder().token(BOT_TOKEN)
    if _RATE_LIMITER_AVAILABLE: builder = builder.rate_limiter(AIORateLimiter())
    app = builder.post_init(_post_init).build()
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
    app.add_handler(CommandHandler("positions_debug", cmd_positions_debug))
    return app

def main():
    log.info("Fund bot is running…")
    app = build_application()
    app.run_polling()

if __name__ == "__main__":
    main()
