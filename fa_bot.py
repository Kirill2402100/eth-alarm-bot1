import os, json, logging, re
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple, List

import gspread
import pandas as pd
from telegram.ext import Application, CommandHandler, AIORateLimiter
from telegram import Update

log = logging.getLogger("fa_bot")
logging.basicConfig(level=logging.INFO)

SHEET_NAME = "FA_Signals"
HEADERS = [
    "pair", "risk", "bias", "ttl", "updated_at",
    "scan_lock_until", "reserve_off", "dca_scale", "notes"
]

# ---------- Google Sheets helpers ----------

def _gc_open():
    creds = os.environ.get("GOOGLE_CREDENTIALS")
    sid   = os.environ.get("SHEET_ID")
    if not creds or not sid:
        raise RuntimeError("GOOGLE_CREDENTIALS or SHEET_ID is missing")
    gc = gspread.service_account_from_dict(json.loads(creds))
    sh = gc.open_by_key(sid)
    return sh

def _ensure_ws(sh) -> gspread.Worksheet:
    try:
        ws = sh.worksheet(SHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=SHEET_NAME, rows=2000, cols=max(20, len(HEADERS)))
        ws.append_row(HEADERS)
    # гарантируем порядок и наличие заголовков
    row1 = ws.row_values(1)
    if row1 != HEADERS:
        ws.delete_rows(1)
        ws.insert_row(HEADERS, 1)
    return ws

def _find_row_index(ws, pair_upper: str) -> Optional[int]:
    colA = ws.col_values(1)  # "pair"
    for i, v in enumerate(colA, start=1):
        if i == 1:  # headers
            continue
        if str(v).upper().strip() == pair_upper:
            return i
    return None

def _row_values_for_pair(ws, pair_upper: str) -> Optional[Dict[str, Any]]:
    idx = _find_row_index(ws, pair_upper)
    if idx is None:
        return None
    values = ws.row_values(idx)
    values += [""] * (len(HEADERS) - len(values))
    return {k: values[i] for i, k in enumerate(HEADERS)}

def _values_from_dict(d: Dict[str, Any]) -> List[Any]:
    out = []
    for k in HEADERS:
        v = d.get(k, "")
        if isinstance(v, (int, float)) and pd.notna(v):
            out.append(v)
        else:
            out.append("" if v is None else str(v))
    return out

def _upsert_policy(pair: str, changes: Dict[str, Any]) -> Dict[str, Any]:
    pair_u = pair.upper().strip()
    sh = _gc_open()
    ws = _ensure_ws(sh)

    row = _row_values_for_pair(ws, pair_u)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if row is None:
        row = {k: "" for k in HEADERS}
        row["pair"] = pair_u

    # применяем изменения
    row.update(changes)
    # служебные
    if "updated_at" not in changes:
        row["updated_at"] = now_iso

    idx = _find_row_index(ws, pair_u)
    values = _values_from_dict(row)
    if idx is None:
        ws.append_row(values)
        idx = _find_row_index(ws, pair_u)
    else:
        ws.update(f"A{idx}:{chr(ord('A')+len(HEADERS)-1)}{idx}", [values])

    return row

def _clear_pair(pair: str):
    pair_u = pair.upper().strip()
    sh = _gc_open()
    ws = _ensure_ws(sh)
    idx = _find_row_index(ws, pair_u)
    if idx is not None:
        ws.delete_rows(idx)

def _read_all() -> List[Dict[str, Any]]:
    sh = _gc_open()
    ws = _ensure_ws(sh)
    recs = ws.get_all_records()
    # нормализуем пустые строки к ""
    for r in recs:
        for k in HEADERS:
            if k not in r:
                r[k] = ""
            elif r[k] is None:
                r[k] = ""
    return recs

# ---------- parsing helpers ----------

def _parse_risk(x: str) -> str:
    m = str(x or "").strip().lower()
    if m in ("g", "green"):   return "Green"
    if m in ("a", "amber", "yellow"): return "Amber"
    if m in ("r", "red"):     return "Red"
    raise ValueError("risk must be green|amber|red")

def _parse_bias(x: str) -> str:
    m = str(x or "").strip().lower()
    if m in ("n", "neutral"): return "neutral"
    if m in ("l", "long-only", "long"): return "long-only"
    if m in ("s", "short-only", "short"): return "short-only"
    raise ValueError("bias must be neutral|long-only|short-only")

def _parse_minutes(x: str) -> int:
    i = int(float(x))
    if i < 0: raise ValueError("ttl must be >= 0")
    return i

_DUR_RX = re.compile(r"^\s*(\d+)\s*([smhdw]?)\s*$", re.I)
def _parse_until(arg: str) -> str:
    """
    '30m','2h','1d','45' (минуты) или ISO '2025-09-08 14:30:00'
    → возвращаем строку ISO UTC, которую читает сканер.
    """
    s = arg.strip()
    try:
        # ISO?
        dt = pd.to_datetime(s, utc=True)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    m = _DUR_RX.match(s)
    if not m:
        raise ValueError("use 30m/2h/1d or ISO datetime")
    n = int(m.group(1))
    unit = (m.group(2) or "m").lower()
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
    dt = datetime.now(timezone.utc) + timedelta(seconds=n * mult)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def _clamp_scale(x: str) -> float:
    v = float(x)
    return max(0.0, min(1.0, v))

# ---------- Bot replies ----------

def _fmt_row(r: Dict[str, Any]) -> str:
    if not r: return "Нет политики."
    lines = [
        f"Пара: <b>{r.get('pair','')}</b>",
        f"risk: <b>{r.get('risk','')}</b>, bias: <b>{r.get('bias','')}</b>",
        f"ttl: {r.get('ttl','')} мин, updated_at: {r.get('updated_at','')}",
        f"scan_lock_until: {r.get('scan_lock_until','') or '—'}",
        f"reserve_off: {r.get('reserve_off','') or 'false'}, dca_scale: {r.get('dca_scale','') or '1.0'}",
    ]
    if r.get("notes"):
        lines.append(f"notes: {r.get('notes')}")
    return "\n".join(lines)

# ---------- Handlers ----------

async def cmd_fa(update: Update, ctx):
    if not ctx.args:
        await update.message.reply_html("Использование: <code>/fa EURUSD</code>")
        return
    pair = ctx.args[0].upper()
    sh = _gc_open(); ws = _ensure_ws(sh)
    r = _row_values_for_pair(ws, pair) or {}
    await update.message.reply_html(_fmt_row(r))

async def cmd_setrisk(update: Update, ctx):
    if len(ctx.args) < 2:
        await update.message.reply_html("Использование: <code>/setrisk EURUSD green|amber|red</code>")
        return
    pair, risk = ctx.args[0], _parse_risk(ctx.args[1])
    r = _upsert_policy(pair, {"risk": risk})
    await update.message.reply_html("OK\n" + _fmt_row(r))

async def cmd_setbias(update: Update, ctx):
    if len(ctx.args) < 2:
        await update.message.reply_html("Использование: <code>/setbias EURUSD neutral|long-only|short-only</code>")
        return
    pair, bias = ctx.args[0], _parse_bias(ctx.args[1])
    r = _upsert_policy(pair, {"bias": bias})
    await update.message.reply_html("OK\n" + _fmt_row(r))

async def cmd_setttl(update: Update, ctx):
    if len(ctx.args) < 2:
        await update.message.reply_html("Использование: <code>/setttl EURUSD 120</code> (минуты)")
        return
    pair, ttl = ctx.args[0], _parse_minutes(ctx.args[1])
    r = _upsert_policy(pair, {"ttl": ttl})
    await update.message.reply_html("OK\n" + _fmt_row(r))

async def cmd_scanlock(update: Update, ctx):
    if len(ctx.args) < 2:
        await update.message.reply_html("Использование: <code>/scanlock EURUSD 2h</code> или ISO")
        return
    pair, until_iso = ctx.args[0], _parse_until(" ".join(ctx.args[1:]))
    r = _upsert_policy(pair, {"scan_lock_until": until_iso})
    await update.message.reply_html("OK (скан заморожен)\n" + _fmt_row(r))

async def cmd_reserve(update: Update, ctx):
    if len(ctx.args) < 2:
        await update.message.reply_html("Использование: <code>/reserve EURUSD on|off</code> "
                                        "(on = разрешить резерв, off = запретить)")
        return
    pair, flag = ctx.args[0], ctx.args[1].strip().lower()
    if flag not in ("on","off"):
        await update.message.reply_html("Аргумент: on|off")
        return
    # в таблице поле называется reserve_off
    reserve_off = "true" if flag == "off" else "false"
    r = _upsert_policy(pair, {"reserve_off": reserve_off})
    await update.message.reply_html("OK\n" + _fmt_row(r))

async def cmd_dca(update: Update, ctx):
    if len(ctx.args) < 2:
        await update.message.reply_html("Использование: <code>/dca EURUSD 0.6</code>")
        return
    pair, scale = ctx.args[0], _clamp_scale(ctx.args[1])
    r = _upsert_policy(pair, {"dca_scale": scale})
    await update.message.reply_html("OK\n" + _fmt_row(r))

async def cmd_note(update: Update, ctx):
    if len(ctx.args) < 2:
        await update.message.reply_html("Использование: <code>/note EURUSD текст</code>")
        return
    pair, note = ctx.args[0], " ".join(ctx.args[1:]).strip()
    r = _upsert_policy(pair, {"notes": note})
    await update.message.reply_html("OK\n" + _fmt_row(r))

async def cmd_status(update: Update, ctx):
    arg = ctx.args[0].upper() if ctx.args else "ALL"
    rows = _read_all()
    if arg != "ALL":
        rows = [r for r in rows if str(r.get("pair","")).upper() == arg]
    if not rows:
        await update.message.reply_html("Пусто.")
        return
    lines = []
    for r in rows:
        lines.append(f"• <b>{r.get('pair')}</b>: {r.get('risk','')}/{r.get('bias','')}, "
                     f"dca={r.get('dca_scale','')}, reserve_off={r.get('reserve_off','') or 'false'}, "
                     f"lock={r.get('scan_lock_until','') or '—'}")
    await update.message.reply_html("\n".join(lines))

async def cmd_clear(update: Update, ctx):
    if not ctx.args:
        await update.message.reply_html("Использование: <code>/clear EURUSD</code>")
        return
    _clear_pair(ctx.args[0])
    await update.message.reply_html("Сброшено.")

# ---------- bootstrap ----------

def main():
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN is missing")

    app = Application.builder().token(token).rate_limiter(AIORateLimiter()).build()

    app.add_handler(CommandHandler("fa",       cmd_fa))
    app.add_handler(CommandHandler("setrisk",  cmd_setrisk))
    app.add_handler(CommandHandler("setbias",  cmd_setbias))
    app.add_handler(CommandHandler("setttl",   cmd_setttl))
    app.add_handler(CommandHandler("scanlock", cmd_scanlock))
    app.add_handler(CommandHandler("reserve",  cmd_reserve))
    app.add_handler(CommandHandler("dca",      cmd_dca))
    app.add_handler(CommandHandler("note",     cmd_note))
    app.add_handler(CommandHandler("status",   cmd_status))
    app.add_handler(CommandHandler("clear",    cmd_clear))

    log.info("FA bot started")
    app.run_polling()

if __name__ == "__main__":
    main()
