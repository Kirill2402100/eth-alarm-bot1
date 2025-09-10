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

# --- TZ ---
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Europe/Belgrade")) if ZoneInfo else None
MORNING_HOUR = int(os.getenv("MORNING_HOUR", "9"))
MORNING_MINUTE = int(os.getenv("MORNING_MINUTE", "30"))

# --- Rate limiter (optional) ---
try:
    from telegram.ext import AIORateLimiter
except Exception:
    AIORateLimiter = None

# --- Google Sheets ---
try:
    import gspread
    from google.oauth2 import service_account
    _GSHEETS_AVAILABLE = True
except Exception:
    gspread = None
    service_account = None
    _GSHEETS_AVAILABLE = False

# --- LLM client (optional) ---
try:
    from llm_client import generate_digest, llm_ping
except Exception:
    async def generate_digest(*args, **kwargs) -> str:
        return "⚠️ LLM сейчас недоступен (нет llm_client.py)."
    async def llm_ping() -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

# -------------------- LOGS --------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("fund_bot")

# -------------------- ENV --------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or os.getenv("TELEGRAM_TOKEN", "").strip()
MASTER_CHAT_ID = int(os.getenv("MASTER_CHAT_ID", "0") or "0")

SHEET_ID = os.getenv("SHEET_ID", "").strip()
SHEET_WS = os.getenv("SHEET_WS", "FUND_BOT").strip() or "FUND_BOT"
CAL_WS = os.getenv("CAL_WS_OUT", "CALENDAR").strip() or "CALENDAR"

DEFAULT_WEIGHTS = {"JPY": 40, "AUD": 25, "EUR": 20, "GBP": 15}
try:
    _DEF = os.getenv("DEFAULT_WEIGHTS", "").strip()
    if _DEF:
        DEFAULT_WEIGHTS = json.loads(_DEF)
except Exception:
    pass

LLM_MINI = os.getenv("LLM_MINI", "gpt-5-mini").strip()
LLM_TOKEN_BUDGET_PER_DAY = int(os.getenv("LLM_TOKEN_BUDGET_PER_DAY", "30000") or "30000")

SYMBOLS = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]
PAIR_COUNTRIES = {
    "USDJPY": ["united states", "japan"],
    "AUDUSD": ["australia", "united states"],
    "EURUSD": ["euro area", "united states"],
    "GBPUSD": ["united kingdom", "united states"],
}

# --- Calendar window (used only for reading from sheet) ---
CAL_WINDOW_MIN = int(os.getenv("CAL_WINDOW_MIN", "120") or "120")
QUIET_BEFORE_MIN = int(os.getenv("QUIET_BEFORE_MIN", "45") or "45")
QUIET_AFTER_MIN  = int(os.getenv("QUIET_AFTER_MIN",  "45") or "45")

# -------------------- GLOBAL STATE --------------------
STATE = {"total": 0.0, "weights": DEFAULT_WEIGHTS.copy()}

# -------------------- SHEETS --------------------
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def _env(name: str) -> str:
    v = os.getenv(name, "")
    return v if isinstance(v, str) else ""

def _decode_b64(s: str) -> Optional[dict]:
    import base64
    s = (s or "").strip()
    if not s: return None
    s += "=" * ((4 - len(s) % 4) % 4)
    try:
        return json.loads(base64.b64decode(s).decode("utf-8", "strict"))
    except Exception:
        return None

def load_google_service_info() -> Tuple[Optional[dict], str, Optional[str]]:
    b64 = _env("GOOGLE_CREDENTIALS_JSON_B64")
    if b64:
        info = _decode_b64(b64)
        if info: return info, "env:GOOGLE_CREDENTIALS_JSON_B64", info.get("client_email")
        return None, "b64 present but invalid", None
    for name in ("GOOGLE_CREDENTIALS_JSON", "GOOGLE_CREDENTIALS"):
        raw = _env(name)
        if raw:
            try:
                info = json.loads(raw)
                return info, f"env:{name}", info.get("client_email")
            except Exception as e:
                return None, f"{name} invalid: {e}", None
    return None, "not-found", None

def build_sheets_client(sheet_id: str):
    if not _GSHEETS_AVAILABLE:
        return None, "gsheets libs not installed"
    if not sheet_id:
        return None, "sheet_id empty"
    info, src, _ = load_google_service_info()
    if not info:
        return None, src
    try:
        creds = service_account.Credentials.from_service_account_info(info, scopes=SHEETS_SCOPES)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_id)
        return sh, src
    except Exception as e:
        return None, f"auth/open error: {e}"

def ensure_worksheet(sh, title: str, headers: List[str]):
    for ws in sh.worksheets():
        if ws.title == title:
            try:
                cur = ws.get_values("A1:Z1") or [[]]
                if cur and cur[0] != headers:
                    ws.update(range_name="A1", values=[headers])
            except Exception:
                pass
            return ws, False
    ws = sh.add_worksheet(title=title, rows=100, cols=max(10, len(headers)))
    ws.update(range_name="A1", values=[headers])
    return ws, True

def append_row(sh, title: str, row: list):
    ws, _ = ensure_worksheet(sh, title, ["ts","chat_id","action","total","weights_json","note"])
    ws.append_row(row, value_input_option="RAW")

# -------------------- UTILS --------------------
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
    if h and m: return f"{'через ' if sign=='через' else ''}{h} ч {m:02d} мин" if sign=='через' else f"{h} ч {m:02d} мин назад"
    if h: return f"{'через ' if sign=='через' else ''}{h} ч" if sign=='через' else f"{h} ч назад"
    return f"{'через ' if sign=='через' else ''}{m} мин" if sign=='через' else f"{m} мин назад"

# -------------------- BOT COMMANDS --------------------
HELP_TEXT = (
    "Что я умею\n"
    "/settotal 2800 — задать общий банк (только в мастер-чате).\n"
    "/setweights jpy=40 aud=25 eur=20 gbp=15 — выставить целевые веса.\n"
    "/weights — показать целевые веса.\n"
    "/alloc — расчёт сумм и готовые команды /setbank для торговых чатов.\n"
    "/digest — утренний «инвесторский» дайджест (человеческий язык + события из листа).\n"
    "/digest pro — краткий «трейдерский» дайджест (по цифрам, LLM).\n"
    "/diag — диагностика LLM и Google Sheets.\n"
    "/cal_debug — показать, что вижу в листе календаря.\n"
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
    if MASTER_CHAT_ID and update.effective_chat and update.effective_chat.id != MASTER_CHAT_ID:
        await update.message.reply_text("Эта команда доступна только в мастер-чате.")
        return
    try:
        parts = (update.message.text or "").split()
        total = float(parts[1])
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
            k = k.strip().upper(); v = int(v.strip())
            if k in ("JPY","AUD","EUR","GBP"):
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

# ---------- Investor digest helpers ----------
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

def price_fmt(symbol: str, value: Optional[float]) -> str:
    if value is None: return "—"
    return f"{value:.{3 if symbol.endswith('JPY') else 5}f}"

def supertrend_dir(val: str) -> str:
    v = (val or "").strip().lower()
    return "up" if "up" in v else "down" if "down" in v else "flat"

def market_phrases(adx: float, st_dir: str, vol_z: float, atr1h: float) -> str:
    trend_txt = "умеренное" if adx < 20 else "заметное" if adx < 25 else "выраженное"
    dir_txt = "вверх" if st_dir == "up" else "вниз" if st_dir == "down" else "вбок"
    vola_txt = "ниже нормы" if atr1h < 0.8 else "около нормы" if atr1h < 1.2 else "выше нормы"
    noise_txt = "низкий" if vol_z < 0.5 else "умеренный" if vol_z < 1.5 else "повышенный"
    return f"{trend_txt} движение {dir_txt}; колебания {vola_txt}; рыночный шум {noise_txt}"

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

def probability_against(side: str, fa_bias: str, adx: float, st_dir: str,
                        vol_z: float, atr1h: float, rsi: float, red_event_soon: bool) -> int:
    P = 55
    against_dir = "down" if (side or "").upper() == "LONG" else "up"
    if (fa_bias == "LONG" and against_dir == "up") or (fa_bias == "SHORT" and against_dir == "down"): P += 10
    elif fa_bias in ("LONG","SHORT"): P -= 15
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

def delta_marker(target: float, fact: float) -> str:
    if target <= 0: return "—"
    d = (fact - target) / target
    ap = abs(d)
    if ap <= 0.02: return "✅"
    if ap <= 0.05: return f"⚠️ {d:+.1%}"
    return f"🚧 {d:+.1%}"

# ---------- Calendar: READ FROM SHEET ----------
CAL_HEADERS = ["utc_iso","local_time","country","currency","title","impact","source","url"]

def read_calendar_rows(sh) -> List[dict]:
    """Read rows from CALENDAR sheet populated by calendar_collector.py"""
    try:
        ws = sh.worksheet(CAL_WS)
        vals = ws.get_all_records()
        out = []
        for r in vals:
            try:
                utc = datetime.fromisoformat(str(r.get("utc_iso")).replace("Z","+00:00")).astimezone(timezone.utc)
            except Exception:
                continue
            out.append({
                "utc": utc,
                "country": str(r.get("country","")).strip().lower(),
                "currency": str(r.get("currency","")).strip().upper(),
                "title": str(r.get("title","")).strip(),
                "impact": str(r.get("impact","")).strip(),
                "source": str(r.get("source","")).strip(),
                "url": str(r.get("url","")).strip(),
                "local": utc.astimezone(LOCAL_TZ) if LOCAL_TZ else utc,
            })
        return out
    except Exception as e:
        log.warning("read_calendar_rows failed: %s", e)
        return []

def build_calendar_for_symbols(symbols: List[str], window_min: Optional[int] = None) -> Dict[str, dict]:
    """Now purely from sheet (no HTTP)."""
    sh, _src = build_sheets_client(SHEET_ID)
    rows = read_calendar_rows(sh) if sh else []
    now = datetime.now(timezone.utc)
    w = window_min if window_min is not None else CAL_WINDOW_MIN
    d1, d2 = now - timedelta(minutes=w), now + timedelta(minutes=w)

    out: Dict[str, dict] = {}
    for sym in symbols:
        countries = set(PAIR_COUNTRIES.get(sym, []))
        sym_rows = [r for r in rows if r["country"] in countries]
        around = [r for r in sym_rows if d1 <= r["utc"] <= d2]
        around.sort(key=lambda x: x["utc"])
        red_soon = any(abs((r["utc"] - now).total_seconds())/60.0 <= 60 for r in around)
        quiet_now = any((r["utc"] - timedelta(minutes=QUIET_BEFORE_MIN)) <= now <= (r["utc"] + timedelta(minutes=QUIET_AFTER_MIN)) for r in around)
        nearest_prev = nearest_next = None
        if not around and sym_rows:
            past = [r for r in sym_rows if r["utc"] < now]
            futr = [r for r in sym_rows if r["utc"] >= now]
            if past:
                p = max(past, key=lambda e: e["utc"])
                nearest_prev = p
            if futr:
                n = min(futr, key=lambda e: e["utc"])
                nearest_next = n
        out[sym] = {
            "events": around,
            "red_event_soon": red_soon,
            "quiet_from_to": (QUIET_BEFORE_MIN, QUIET_AFTER_MIN) if around else (0,0),
            "quiet_now": quiet_now,
            "nearest_prev": nearest_prev,
            "nearest_next": nearest_next,
        }
    return out

# ---------- Sheets data (trading logs) ----------
BMR_SHEETS = {
    "USDJPY": os.getenv("BMR_SHEET_USDJPY", "BMR_DCA_USDJPY"),
    "AUDUSD": os.getenv("BMR_SHEET_AUDUSD", "BMR_DCA_AUDUSD"),
    "EURUSD": os.getenv("BMR_SHEET_EURUSD", "BMR_DCA_EURUSD"),
    "GBPUSD": os.getenv("BMR_SHEET_GBPUSD", "BMR_DCA_GBPUSD"),
}

def _to_float(x, default=0.0) -> float:
    try:
        return float(str(x).strip().replace(",", "."))
    except Exception:
        return default

def get_last_nonempty_row(sh, symbol: str, needed=("Avg_Price","Next_DCA_Price","Bank_Target_USDT","Bank_Fact_USDT")) -> Optional[dict]:
    name = BMR_SHEETS.get(symbol)
    if not name: return None
    try:
        ws = sh.worksheet(name)
        rows = ws.get_all_records()
        if not rows: return None
        for r in reversed(rows):
            if any(r.get(f) not in (None, "", 0, "0", "0.0") for f in needed):
                return r
        return rows[-1]
    except Exception:
        return None

def latest_bank_target_fact(sh, symbol: str) -> tuple[Optional[float], Optional[float]]:
    name = BMR_SHEETS.get(symbol)
    if not name: return None, None
    try:
        ws = sh.worksheet(name)
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

# ---------- DIGEST ----------
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

    sh, _ = build_sheets_client(SHEET_ID)
    if not sh:
        await update.message.reply_text("Sheets недоступен: не могу собрать инвесторский дайджест.")
        return

    try:
        now_utc = datetime.now(timezone.utc)
        header = header_ru(now_utc.astimezone(LOCAL_TZ)) if LOCAL_TZ else f"🧭 Утренний фон — {now_utc:%d %b %Y, %H:%M} (UTC)"
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

        # summary
        summary_lines, all_events = [], []
        for sym in SYMBOLS:
            for ev in cal.get(sym, {}).get("events", []):
                all_events.append((ev["utc"], ev["local"], sym, ev["country"], ev["title"]))
        if not all_events:
            for sym in SYMBOLS:
                n = cal.get(sym, {}).get("nearest_next")
                if n:
                    all_events.append((n["utc"], n["local"], sym, n["country"], n["title"]))
        if all_events:
            summary_lines.append("\n📅 **Ближайшие High-события (Белград):**")
            unique = {ev[0]: ev for ev in sorted(all_events)}
            for _, tloc, sym, cty, title in list(unique.values())[:8]:
                summary_lines.append(f"• {tloc:%H:%M} — {sym}: {cty}: {title}")

        msg = "\n\n".join(blocks + ["\n".join(summary_lines)] if summary_lines else blocks)
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"Ошибка сборки дайджеста: {e}")

async def cmd_diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        ok = await llm_ping()
        llm_line = "LLM: ✅ ok" if ok else "LLM: ❌ no key"
    except Exception:
        llm_line = "LLM: ❌ error"
    sh_state = "SHEET_ID set" if SHEET_ID else "SHEET_ID empty"
    await update.message.reply_text(f"{llm_line}\nSheets: {sh_state}; calendar ws='{CAL_WS}' (read-only)")

async def cmd_cal_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sh, _ = build_sheets_client(SHEET_ID)
    rows = read_calendar_rows(sh) if sh else []
    now = datetime.now(timezone.utc)
    d1, d2 = now - timedelta(minutes=CAL_WINDOW_MIN), now + timedelta(minutes=CAL_WINDOW_MIN)
    tot = len(rows)
    around = [r for r in rows if d1 <= r["utc"] <= d2]
    per = {s: len([r for r in around if r["country"] in set(PAIR_COUNTRIES[s])]) for s in SYMBOLS}
    await update.message.reply_text(
        f"Calendar sheet: total={tot}, window=±{CAL_WINDOW_MIN}m → in-window={len(around)}\n" +
        "\n".join([f"{s}: {per[s]}" for s in SYMBOLS])
    )

# ---------- Bot plumbing ----------
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

async def _set_bot_commands(app: Application):
    cmds = [
        BotCommand("start", "Запуск бота"),
        BotCommand("help", "Список команд"),
        BotCommand("ping", "Проверка связи"),
        BotCommand("settotal", "Задать общий банк (мастер-чат)"),
        BotCommand("setweights", "Задать целевые веса (мастер-чат)"),
        BotCommand("weights", "Показать целевые веса"),
        BotCommand("alloc", "Рассчитать распределение банка"),
        BotCommand("digest", "Утренний дайджест (из листа) / pro"),
        BotCommand("diag", "Диагностика LLM и Sheets"),
        BotCommand("cal_debug", "Проверка данных календаря из листа"),
    ]
    try:
        await app.bot.set_my_commands(cmds)
    except Exception as e:
        log.warning("set_my_commands failed: %s", e)

async def morning_digest_scheduler(app: Application):
    while True:
        now = datetime.now(LOCAL_TZ) if LOCAL_TZ else datetime.utcnow()
        target = now.replace(hour=MORNING_HOUR, minute=MORNING_MINUTE, second=0, microsecond=0)
        if now >= target:
            target = target + timedelta(days=1)
        await asyncio.sleep(max(1.0, (target - now).total_seconds()))
        try:
            await app.bot.send_message(chat_id=MASTER_CHAT_ID, text="⏰ Утренний дайджест: /digest", parse_mode=ParseMode.MARKDOWN)
        except Exception:
            pass

def build_application() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")
    builder: ApplicationBuilder = Application.builder().token(BOT_TOKEN)
    if AIORateLimiter:
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
    app.add_handler(CommandHandler("cal_debug", cmd_cal_debug))
    return app

async def main_async():
    log.info("Fund bot is running… (calendar: from sheet '%s')", CAL_WS)
    app = build_application()
    await _set_bot_commands(app)
    await app.initialize()
    await app.start()
    asyncio.create_task(morning_digest_scheduler(app))
    await app.updater.start_polling()
    await asyncio.Event().wait()

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
