# fa_bot.py — FUND bot (RU)
# -*- coding: utf-8 -*-

import os
import json
import base64
import logging
import re
from html import escape as _html_escape
from datetime import datetime, timedelta, timezone, time as dtime
from typing import Dict, Tuple, Optional, List

from telegram import Update, BotCommand
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# ===== Настройки и окружение ==================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("fund_bot")

# Таймзона
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Europe/Belgrade")) if ZoneInfo else timezone.utc
MORNING_HOUR = int(os.getenv("MORNING_HOUR", "9"))
MORNING_MINUTE = int(os.getenv("MORNING_MINUTE", "30"))

# Telegram
BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN", "") or os.getenv("TELEGRAM_TOKEN", "")).strip()
MASTER_CHAT_ID = int(os.getenv("MASTER_CHAT_ID", "0") or "0")

# Google Sheets
try:
    import gspread
    from google.oauth2 import service_account
    _GSHEETS_AVAILABLE = True
except Exception:
    gspread = None
    service_account = None
    _GSHEETS_AVAILABLE = False

SHEET_ID = os.getenv("SHEET_ID", "").strip()
SHEET_WS = os.getenv("SHEET_WS", "FUND_BOT").strip() or "FUND_BOT"

# Символы и веса
_DEFAULT_WEIGHTS_RAW = os.getenv("DEFAULT_WEIGHTS", "").strip()
if _DEFAULT_WEIGHTS_RAW:
    try:
        DEFAULT_WEIGHTS: Dict[str, int] = json.loads(_DEFAULT_WEIGHTS_RAW)
    except Exception:
        DEFAULT_WEIGHTS = {"JPY": 40, "AUD": 25, "EUR": 20, "GBP": 15}
else:
    DEFAULT_WEIGHTS = {"JPY": 40, "AUD": 25, "EUR": 20, "GBP": 15}

SYMBOLS = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]

# Названия листов с логами по парам
BMR_SHEETS = {
    "USDJPY": os.getenv("BMR_SHEET_USDJPY", "BMR_DCA_USDJPY"),
    "AUDUSD": os.getenv("BMR_SHEET_AUDUSD", "BMR_DCA_AUDUSD"),
    "EURUSD": os.getenv("BMR_SHEET_EURUSD", "BMR_DCA_EURUSD"),
    "GBPUSD": os.getenv("BMR_SHEET_GBPUSD", "BMR_DCA_GBPUSD"),
}

# Календарь / новости (для строки «топ-новость»)
CAL_WS_OUT = os.getenv("CAL_WS_OUT", "CALENDAR").strip() or "CALENDAR"
NEWS_TTL_MIN = int(os.getenv("FA_NEWS_TTL_MIN", "120") or "120")
PAIR_COUNTRIES = {
    "USDJPY": {"united states", "japan"},
    "AUDUSD": {"australia", "united states"},
    "EURUSD": {"euro area", "united states"},
    "GBPUSD": {"united kingdom", "united states"},
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

# LLM (опционально)
try:
    from llm_client import generate_digest, llm_ping
except Exception:
    async def generate_digest(*args, **kwargs) -> str:
        return "⚠️ LLM сейчас недоступен (нет llm_client.py)."
    async def llm_ping() -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

# Состояние (в памяти)
STATE = {
    "total": 0.0,
    "weights": DEFAULT_WEIGHTS.copy(),
}

# ===== Google Sheets helpers ===================================================

SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def _env(name: str) -> str:
    v = os.getenv(name, "")
    return v if isinstance(v, str) else ""

def _decode_b64_maybe_padded(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    padded = s + "=" * ((4 - len(s) % 4) % 4)
    return base64.b64decode(padded).decode("utf-8", "strict")

def load_google_service_info() -> Tuple[Optional[dict], str]:
    b64 = _env("GOOGLE_CREDENTIALS_JSON_B64")
    if b64:
        try:
            decoded = _decode_b64_maybe_padded(b64)
            info = json.loads(decoded)
            return info, "env:GOOGLE_CREDENTIALS_JSON_B64"
        except Exception as e:
            return None, f"b64 present but decode/json error: {e}"

    for name in ("GOOGLE_CREDENTIALS_JSON", "GOOGLE_CREDENTIALS"):
        raw = _env(name)
        if raw:
            try:
                info = json.loads(raw)
                return info, f"env:{name}"
            except Exception as e:
                return None, f"{name} present but invalid JSON: {e}"

    return None, "not-found"

def build_sheets_client(sheet_id: str):
    if not _GSHEETS_AVAILABLE:
        return None, "gsheets libs not installed"
    if not sheet_id:
        return None, "sheet_id empty"

    info, src = load_google_service_info()
    if not info:
        return None, src
    try:
        creds = service_account.Credentials.from_service_account_info(info, scopes=SHEETS_SCOPES)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_id)
        return sh, "ok"
    except Exception as e:
        return None, f"auth/open error: {e}"

SHEET_HEADERS = ["ts", "chat_id", "action", "total", "weights_json", "note"]

def ensure_worksheet(sh, title: str, headers: Optional[List[str]] = None):
    for ws in sh.worksheets():
        if ws.title == title:
            if headers:
                try:
                    cur = ws.get_values("A1:Z1") or [[]]
                    if not cur or cur[0] != headers:
                        ws.update("A1", [headers])
                except Exception:
                    pass
            return ws, False
    ws = sh.add_worksheet(title=title, rows=200, cols=max(10, len(headers or [])))
    if headers:
        ws.update("A1", [headers])
    return ws, True

def append_row(sh, title: str, row: list):
    ws, _ = ensure_worksheet(sh, title, SHEET_HEADERS)
    ws.append_row(row, value_input_option="RAW")

# ===== Утилиты ================================================================

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

# ===== Данные для дайджеста (BMR, FA, NEWS/CALENDAR) ==========================

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

def _sheet_rows(sh, sheet_name: str) -> List[dict]:
    try:
        ws = sh.worksheet(sheet_name)
        return ws.get_all_records()
    except Exception:
        return []

def get_last_nonempty_row(sh, symbol: str, needed_fields=("Avg_Price","Next_DCA_Price","Bank_Target_USDT","Bank_Fact_USDT")) -> Optional[dict]:
    sheet_name = BMR_SHEETS.get(symbol)
    if not sheet_name:
        return None
    rows = _sheet_rows(sh, sheet_name)
    if not rows:
        return None
    for r in reversed(rows):
        if any(r.get(f) not in (None, "", 0, "0", "0.0") for f in needed_fields):
            return r
    return rows[-1]

def latest_bank_target_fact(sh, symbol: str) -> Tuple[Optional[float], Optional[float]]:
    sheet_name = BMR_SHEETS.get(symbol)
    if not sheet_name:
        return None, None
    rows = _sheet_rows(sh, sheet_name)
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

def price_fmt(symbol: str, value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.{3 if symbol.endswith('JPY') else 5}f}"

def _symbol_hints(symbol: str) -> Tuple[str, str]:
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
    return (
        "если представители Банка Англии говорят «зарплаты и услуги давят на инфляцию», рынок ждёт ставку повыше дольше — фунт крепче (GBP/USD вверх). Мягче — фунт слабее.",
        "больше тревоги по инфляции — фунт сильнее; меньше — слабее."
    )

def _fa_icon(risk: str) -> str:
    return {"Green": "🟢", "Amber": "🟡", "Red": "🔴"}.get((risk or "").capitalize(), "⚪️")

def _top_news_for_pair(sh, pair: str, now_utc: Optional[datetime] = None) -> str:
    """
    Возвращает строку вида 'HH:MM — Title (SRC)' для пары:
    1) свежая high-новость из NEWS за TTL (по странам пары);
    2) иначе — ближайшее событие из CALENDAR (по странам пары).
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(minutes=NEWS_TTL_MIN)
    countries = PAIR_COUNTRIES.get(pair, set())

    # NEWS
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
        tags = str(r.get("tags", "")).strip()
        kw_ok = bool(KW_RE.search(f"{title} {tags}"))
        if not kw_ok and src in {"US_FED_PR", "ECB_PR", "BOE_PR", "RBA_MR", "BOJ_PR"}:
            kw_ok = True
        if not kw_ok:
            continue
        row_cty = {x.strip().lower() for x in str(r.get("countries", "")).split(",") if x.strip()}
        if not (row_cty & countries):
            continue
        if (best is None) or (ts > best["ts"]):
            best = {"ts": ts, "title": title, "src": src}
    if best:
        lt = best["ts"].astimezone(LOCAL_TZ)
        return f"{lt:%H:%M} — {best['title']} ({best['src']})"

    # CALENDAR (fallback)
    try:
        events = sh.worksheet(CAL_WS_OUT).get_all_records()
    except Exception:
        events = []
    soon = None
    for e in events:
        try:
            dt = datetime.fromisoformat(str(e.get("utc_iso")).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if dt <= now_utc:
            continue
        if str(e.get("country", "")).strip().lower() not in countries:
            continue
        if (soon is None) or (dt < soon["ts"]):
            soon = {"ts": dt, "title": str(e.get("title", "")).strip(), "src": str(e.get("source", "")).strip() or "cal"}
    if soon:
        lt = soon["ts"].astimezone(LOCAL_TZ)
        return f"{lt:%Y-%m-%d %H:%M} — {soon['title']} ({soon['src']})"
    return ""

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

def build_digest_text(sh, fa_sheet_data: dict) -> str:
    now_utc = datetime.now(timezone.utc)
    header = header_ru(now_utc.astimezone(LOCAL_TZ))
    parts: List[str] = [header]
    for i, sym in enumerate(SYMBOLS):
        row = get_last_nonempty_row(sh, sym) or {}
        fa_data = fa_sheet_data.get(sym, {})
        parts.append(render_morning_pair_block(sh, sym, row, fa_data))
        if i < len(SYMBOLS) - 1:
            parts.append("⸻")
    parts.append("Главная мысль дня: до ФРС — аккуратно; после пресс-конференции вернёмся к обычному режиму, если не будет сюрпризов.")
    return "\n".join(parts)

def _read_fa_signals_from_sheet(sh) -> Dict[str, dict]:
    """Лист FA_Signals: pair,risk,bias,ttl,updated_at,scan_lock_until,reserve_off,dca_scale,reason,risk_pct"""
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

# ===== Автозапись целевого банка в BMR_DCA_* ==================================

def _set_bank_target_in_bmr(sh, symbol: str, amount: float):
    """Обновляет Bank_Target_USDT в листе BMR_DCA_* для пары symbol (в последней непустой строке)."""
    sheet_name = BMR_SHEETS.get(symbol)
    if not sheet_name:
        return
    try:
        ws = sh.worksheet(sheet_name)
    except Exception:
        return

    header = ws.row_values(1)
    if "Bank_Target_USDT" not in header:
        return
    col_ix = header.index("Bank_Target_USDT") + 1

    # найдём последнюю реально заполненную строку
    vals = ws.get_all_values()
    last = len(vals)
    while last > 1 and not any((c or "").strip() for c in vals[last - 1]):
        last -= 1
    row_ix = max(2, last)  # пишем в последнюю непустую строку
    try:
        ws.update_cell(row_ix, col_ix, float(amount))
    except Exception as e:
        log.warning("update BMR %s failed: %s", sheet_name, e)

# ===== Команды ================================================================

HELP_TEXT = (
    "Что я умею\n"
    "/settotal 2800 — задать общий банк (только в мастер-чате).\n"
    "/setweights jpy=40 aud=25 eur=20 gbp=15 — выставить целевые веса.\n"
    "/weights — показать целевые веса.\n"
    "/alloc — расчёт сумм и готовые команды /setbank для торговых чатов (и автозапись в BMR).\n"
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

def assert_master_chat(update: Update) -> bool:
    if MASTER_CHAT_ID and update.effective_chat:
        return update.effective_chat.id == MASTER_CHAT_ID
    return True

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
        # лог в FUND_BOT
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

        # автообновление Bank_Target_USDT в листах BMR_DCA_*
        try:
            for sym, amt in alloc.items():
                _set_bank_target_in_bmr(sh, sym, amt)
        except Exception as e:
            log.warning("auto write BMR failed: %s", e)

async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = (update.message.text or "").split()
    pro = len(args) > 1 and args[1].lower() == "pro"

    if pro:
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            txt = await generate_digest(symbols=SYMBOLS)
            await update.message.reply_text(txt)
        except Exception as e:
            await update.message.reply_text(f"LLM ошибка: {e}")
        return

    sh, _ = build_sheets_client(SHEET_ID)
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

# Диагностика / сервисные

def sheets_diag_text() -> str:
    sid_state = "set" if SHEET_ID else "empty"
    if not _GSHEETS_AVAILABLE:
        return f"Sheets: ❌ (libs not installed, SID={sid_state})"
    sh, src = build_sheets_client(SHEET_ID)
    if sh is None:
        return f"Sheets: ❌ (SID={sid_state}, source={src})"
    try:
        ws, created = ensure_worksheet(sh, SHEET_WS, SHEET_HEADERS)
        mark = "created" if created else "exists"
        return f"Sheets: ✅ ok (ws={ws.title}:{mark})"
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
        ws, created = ensure_worksheet(sh, SHEET_WS, SHEET_HEADERS)
        await update.message.reply_text(
            f"Sheets: ✅ ws='{ws.title}' {'создан' if created else 'уже есть'}"
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

# ===== Планировщик утреннего дайджеста (JobQueue) =============================

async def morning_digest_job(context: ContextTypes.DEFAULT_TYPE):
    try:
        sh, _ = build_sheets_client(SHEET_ID)
        if sh:
            fa_sheet_data = _read_fa_signals_from_sheet(sh)
            msg = build_digest_text(sh, fa_sheet_data)
            for chunk in _split_for_tg_html(msg):
                await context.bot.send_message(chat_id=MASTER_CHAT_ID, text=chunk, parse_mode=ParseMode.HTML)
        else:
            await context.bot.send_message(chat_id=MASTER_CHAT_ID, text="Sheets недоступен: утренний дайджест пропущен.")
    except Exception as e:
        try:
            await context.bot.send_message(chat_id=MASTER_CHAT_ID, text=f"Ошибка утреннего дайджеста: {e}")
        except Exception:
            pass

async def _post_init(app: Application):
    # Команды в меню
    cmds = [
        BotCommand("start", "Запуск бота"),
        BotCommand("help", "Список команд"),
        BotCommand("ping", "Проверка связи"),
        BotCommand("settotal", "Задать общий банк (мастер-чат)"),
        BotCommand("setweights", "Задать целевые веса (мастер-чат)"),
        BotCommand("weights", "Показать целевые веса"),
        BotCommand("alloc", "Рассчитать распределение банка"),
        BotCommand("digest", "Утренний дайджест / pro"),
        BotCommand("init_sheet", "Создать/проверить лист в Google Sheets"),
        BotCommand("sheet_test", "Тестовая запись в лист"),
        BotCommand("diag", "Диагностика LLM и Sheets"),
    ]
    try:
        await app.bot.set_my_commands(cmds)
    except Exception as e:
        log.warning("set_my_commands failed: %s", e)

    # Планировщик ежедневного дайджеста
    app.job_queue.run_daily(
        morning_digest_job,
        time=dtime(MORNING_HOUR, MORNING_MINUTE, tzinfo=LOCAL_TZ),
        name="morning_digest",
    )

# ===== Старт приложения ========================================================

def build_application() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")
    app = Application.builder().token(BOT_TOKEN).post_init(_post_init).build()

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
    # run_polling по умолчанию ставит delete_webhook=True и запускает JobQueue
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
