# fund_bot.py
# Мастер-бот распределения банка и фиксации рекомендаций ALLOC.
# Команды:
#  /settotal 2800
#  /setweights jpy=40 aud=25 eur=20 gbp=15
#  /alloc
#  /alloc_applied
#  /status
#
# ENV:
#  TELEGRAM_BOT_TOKEN   – токен этого бота
#  ALLOWED_CHAT_IDS     – список master-чатов, через запятую (например: -10012345,123456)
#  SHEET_ID             – Google Sheet ID
#  GOOGLE_CREDENTIALS   – JSON сервис-аккаунта (как строка)
#  SYMBOL_CHAT_MAP      – JSON-словарь {"USDJPY": -100..., "AUDUSD": -100..., "EURUSD": -100..., "GBPUSD": -100...}
#
# Примечание: бот НЕ шлёт ничего в торговые чаты автоматически – он выдаёт готовую «копипасту».

import os
import re
import json
import math
import logging
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional

import gspread
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# --- optional rate limiter (ptb v20+) ---
try:
    from telegram.ext import AIORateLimiter
    _RL = AIORateLimiter()
except Exception:
    _RL = None

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
ALLOWED_CHAT_IDS = {
    int(x) for x in os.getenv("ALLOWED_CHAT_IDS", "").split(",")
    if x.strip().lstrip("-").isdigit()
}
SHEET_ID = os.getenv("SHEET_ID")
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS")
SYMBOL_CHAT_MAP = {}
try:
    SYMBOL_CHAT_MAP = json.loads(os.getenv("SYMBOL_CHAT_MAP", "{}") or "{}")
except Exception:
    SYMBOL_CHAT_MAP = {}

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("fund_bot")

# фиксированное сопоставление валюты к инструменту
CCY2SYMBOL = {"JPY": "USDJPY", "AUD": "AUDUSD", "EUR": "EURUSD", "GBP": "GBPUSD"}
SYMBOLS = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]
CCYS = ["JPY", "AUD", "EUR", "GBP"]

# ---------------- Sheets helpers ----------------
def _gc_open() -> Optional[gspread.Spreadsheet]:
    if not (SHEET_ID and GOOGLE_CREDENTIALS):
        return None
    try:
        gc = gspread.service_account_from_dict(json.loads(GOOGLE_CREDENTIALS))
        return gc.open_by_key(SHEET_ID)
    except Exception:
        log.exception("Failed to connect Google Sheets")
        return None

def _ensure_master(sh: gspread.Spreadsheet):
    # FA_Master: key | value
    try:
        ws = sh.worksheet("FA_Master")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet("FA_Master", rows=100, cols=2)
        ws.append_row(["key", "value"])
        # дефолты
        ws.append_row(["total_bank", "0"])
        ws.append_row(["weights_json", json.dumps({"JPY": 40, "AUD": 25, "EUR": 20, "GBP": 15})])
        ws.append_row(["current_amounts_json", json.dumps({s: 0 for s in SYMBOLS})])
        ws.append_row(["last_alloc_id", ""])
    return ws

def _ensure_alloc_hist(sh: gspread.Spreadsheet):
    # FA_Alloc_History: alloc_id | ts_utc | total_bank | weights_json | amounts_json | status | note
    try:
        ws = sh.worksheet("FA_Alloc_History")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet("FA_Alloc_History", rows=2000, cols=7)
        ws.append_row(["alloc_id","ts_utc","total_bank","weights_json","amounts_json","status","note"])
    return ws

def _read_kv(ws: gspread.Worksheet) -> Dict[str, str]:
    vals = ws.get_all_values()
    out = {}
    for i, row in enumerate(vals):
        if i == 0 or len(row) < 2:
            continue
        out[row[0]] = row[1]
    return out

def _write_kv(ws: gspread.Worksheet, key: str, value: str):
    vals = ws.get_all_values()
    for i, row in enumerate(vals):
        if i == 0:
            continue
        if row and row[0] == key:
            ws.update(f"B{i+1}", value)
            return
    ws.append_row([key, value])

def _next_alloc_id(sh: gspread.Spreadsheet) -> str:
    ws = _ensure_alloc_hist(sh)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = ws.get_all_values()
    n = 0
    for i, r in enumerate(rows):
        if i == 0 or not r:
            continue
        if r[0].startswith(f"ALLOC-{today}-"):
            try:
                tail = int(r[0].split("-")[-1])
                n = max(n, tail)
            except Exception:
                pass
    return f"ALLOC-{today}-{n+1}"

# ---------------- Core math ----------------
def parse_weights(text: str) -> Dict[str, float]:
    """
    'jpy=40 aud=25 eur=20 gbp=15' -> {'JPY': 40.0, 'AUD':25.0, ...}
    допускаем разделители: пробелы/запятые/точки с запятой.
    """
    found = {}
    for chunk in re.split(r"[,\s;]+", text.strip()):
        if not chunk:
            continue
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        k = k.strip().upper()
        if k in ("USDJPY", "AUDUSD", "EURUSD", "GBPUSD"):
            # позволим задавать по символам
            # маппим обратно в валюту
            inv = {v:k for k,v in CCY2SYMBOL.items()}
            k = inv.get(k, k)
        try:
            val = float(str(v).strip().replace(",", "."))
        except Exception:
            continue
        if k in CCYS:
            found[k] = val
    return found

def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    # если не хватает — дополним нулями
    w = {c: float(w.get(c, 0.0)) for c in CCYS}
    s = sum(w.values())
    if s <= 0:
        return {c: 0.0 for c in CCYS}
    return {c: (w[c] / s) * 100.0 for c in CCYS}

def compute_amounts(total_bank: float, weights_pct: Dict[str, float]) -> Dict[str, float]:
    """По весам валют вернём суммы по символам (USDT). Сумма ровно total_bank (подправляем хвост)."""
    amounts = {}
    running = 0.0
    # три первых — обычное округление до 2 знаков, последний — остаток
    for i, c in enumerate(CCYS):
        sym = CCY2SYMBOL[c]
        if i < len(CCYS) - 1:
            amt = round(total_bank * (weights_pct.get(c, 0.0) / 100.0), 2)
            running += amt
            amounts[sym] = amt
        else:
            amounts[sym] = round(total_bank - running, 2)
    return amounts

def amounts_to_weights(amounts: Dict[str, float]) -> Dict[str, float]:
    total = sum(float(v) for v in amounts.values())
    if total <= 0:
        return {c: 0.0 for c in CCYS}
    out = {}
    for c in CCYS:
        sym = CCY2SYMBOL[c]
        out[c] = round((float(amounts.get(sym, 0.0)) / total) * 100.0, 2)
    return out

# ---------------- Messaging helpers ----------------
def fmt_weights_line(prefix: str, w: Dict[str, float]) -> str:
    return (f"{prefix}: JPY {w.get('JPY',0):.0f} / AUD {w.get('AUD',0):.0f} / "
            f"EUR {w.get('EUR',0):.0f} / GBP {w.get('GBP',0):.0f}")

def _is_allowed(update: Update) -> bool:
    if not ALLOWED_CHAT_IDS:
        return True
    cid = update.effective_chat.id if update.effective_chat else None
    return (cid in ALLOWED_CHAT_IDS)

def make_copypasta(amounts: Dict[str, float]) -> str:
    lines = []
    for sym in SYMBOLS:
        amt = amounts.get(sym, 0.0)
        chat_hint = SYMBOL_CHAT_MAP.get(sym)
        chat_txt = f" (чат: {chat_hint})" if chat_hint else ""
        lines.append(f"{sym} → {amt:.2f} USDT → команда в чат{chat_txt}: /setbank {int(amt) if amt.is_integer() else amt}")
    return "\n".join(lines)

# ---------------- Handlers ----------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update): return
    await update.message.reply_html(
        "Мастер-фунд бот готов.\n\n"
        "<b>Команды</b>\n"
        "• <code>/settotal 2800</code>\n"
        "• <code>/setweights jpy=40 aud=25 eur=20 gbp=15</code>\n"
        "• <code>/alloc</code> — показать расчёт и копипасту\n"
        "• <code>/alloc_applied</code> — пометить последнюю ALLOC как APPLIED\n"
        "• <code>/status</code>"
    )

async def _load_state():
    sh = _gc_open()
    if not sh:
        return None, None, {}
    ws = _ensure_master(sh)
    kv = _read_kv(ws)
    try:
        total = float(kv.get("total_bank", "0"))
    except Exception:
        total = 0.0
    try:
        weights = json.loads(kv.get("weights_json", "{}") or "{}")
    except Exception:
        weights = {"JPY": 40, "AUD": 25, "EUR": 20, "GBP": 15}
    try:
        current_amounts = json.loads(kv.get("current_amounts_json", "{}") or "{}")
    except Exception:
        current_amounts = {s: 0 for s in SYMBOLS}
    return sh, ws, {"total": total, "weights": weights, "current_amounts": current_amounts, "last_alloc_id": kv.get("last_alloc_id","")}

async def _save_state(ws_master, total: float=None, weights: Dict[str,float]=None, current_amounts: Dict[str,float]=None, last_alloc_id: str=None):
    if total is not None:
        _write_kv(ws_master, "total_bank", str(total))
    if weights is not None:
        _write_kv(ws_master, "weights_json", json.dumps(weights, ensure_ascii=False))
    if current_amounts is not None:
        _write_kv(ws_master, "current_amounts_json", json.dumps(current_amounts, ensure_ascii=False))
    if last_alloc_id is not None:
        _write_kv(ws_master, "last_alloc_id", last_alloc_id)

async def cmd_settotal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update): return
    if not context.args:
        await update.message.reply_text("Использование: /settotal 2800")
        return
    try:
        total = float(context.args[0].replace(",", "."))
    except Exception:
        await update.message.reply_text("Не смог распознать сумму.")
        return

    sh, ws, st = await _load_state()
    if not sh:
        await update.message.reply_text("Sheets не сконфигурирован.")
        return

    await _save_state(ws, total=total)
    # сразу показать расчёт
    w_norm = normalize_weights(st["weights"])
    amts = compute_amounts(total, w_norm)
    alloc_id = _next_alloc_id(sh)
    await _save_state(ws, last_alloc_id=alloc_id)

    fact_w = amounts_to_weights(st["current_amounts"])
    ok = all(abs(fact_w.get(c,0) - w_norm.get(c,0)) < 0.5 for c in CCYS)

    text = (
        f"Банк установлен: <b>{total:.2f} USDT</b>\n\n"
        f"{fmt_weights_line('Целевые веса', w_norm)}\n"
        f"{fmt_weights_line('Факт', fact_w)}  {'✅' if ok else '⚠️'}\n\n"
        f"ID рекомендации: <code>{alloc_id}</code>\n"
        "Готово к применению в торговых чатах:\n"
        f"{make_copypasta(amts)}\n\n"
        "После того, как применишь в чатах, нажми /alloc_applied"
    )
    await update.message.reply_html(text)

async def cmd_setweights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update): return
    if not context.args:
        await update.message.reply_text("Использование: /setweights jpy=40 aud=25 eur=20 gbp=15")
        return
    raw = " ".join(context.args)
    w = parse_weights(raw)
    if not w:
        await update.message.reply_text("Не распознал веса. Пример: /setweights jpy=40 aud=25 eur=20 gbp=15")
        return
    w_norm = normalize_weights(w)

    sh, ws, st = await _load_state()
    if not sh:
        await update.message.reply_text("Sheets не сконфигурирован.")
        return

    await _save_state(ws, weights=w_norm)

    amts = compute_amounts(st["total"], w_norm)
    alloc_id = _next_alloc_id(sh)
    await _save_state(ws, last_alloc_id=alloc_id)

    fact_w = amounts_to_weights(st["current_amounts"])
    ok = all(abs(fact_w.get(c,0) - w_norm.get(c,0)) < 0.5 for c in CCYS)

    text = (
        f"{fmt_weights_line('Целевые веса', w_norm)}\n"
        f"{fmt_weights_line('Факт', fact_w)}  {'✅' if ok else '⚠️'}\n\n"
        f"Банк: <b>{st['total']:.2f} USDT</b>\n"
        f"ID рекомендации: <code>{alloc_id}</code>\n"
        f"{make_copypasta(amts)}\n\n"
        "После применения: /alloc_applied"
    )
    await update.message.reply_html(text)

async def cmd_alloc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update): return
    sh, ws, st = await _load_state()
    if not sh:
        await update.message.reply_text("Sheets не сконфигурирован.")
        return
    w_norm = normalize_weights(st["weights"])
    amts = compute_amounts(st["total"], w_norm)
    fact_w = amounts_to_weights(st["current_amounts"])
    ok = all(abs(fact_w.get(c,0) - w_norm.get(c,0)) < 0.5 for c in CCYS)
    text = (
        f"Текущие настройки\n"
        f"Банк: <b>{st['total']:.2f} USDT</b>\n"
        f"{fmt_weights_line('Целевые веса', w_norm)}\n"
        f"{fmt_weights_line('Факт', fact_w)}  {'✅' if ok else '⚠️'}\n\n"
        f"Последний ALLOC: <code>{st.get('last_alloc_id') or '—'}</code>\n\n"
        f"{make_copypasta(amts)}"
    )
    await update.message.reply_html(text)

async def cmd_alloc_applied(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update): return
    sh, ws, st = await _load_state()
    if not sh:
        await update.message.reply_text("Sheets не сконфигурирован.")
        return
    alloc_id = st.get("last_alloc_id") or _next_alloc_id(sh)

    w_norm = normalize_weights(st["weights"])
    amts = compute_amounts(st["total"], w_norm)

    # записываем историю и помечаем APPLIED
    ws_hist = _ensure_alloc_hist(sh)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    ws_hist.append_row([
        alloc_id, ts, f"{st['total']:.2f}",
        json.dumps(w_norm, ensure_ascii=False),
        json.dumps(amts, ensure_ascii=False),
        "APPLIED",
        ""
    ])
    # фиксируем «факт» как применённые суммы (пока без запроса из торговых чатов)
    await _save_state(ws, current_amounts=amts)

    text = (
        f"Статус рекомендации <b>{alloc_id}</b>: <b>APPLIED</b>\n\n"
        f"{fmt_weights_line('Целевые веса', w_norm)}\n"
        f"{fmt_weights_line('Факт', amounts_to_weights(amts))}  ✅"
    )
    await update.message.reply_html(text)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update): return
    sh, ws, st = await _load_state()
    if not sh:
        await update.message.reply_text("Sheets не сконфигурирован.")
        return
    w_norm = normalize_weights(st["weights"])
    fact_w = amounts_to_weights(st["current_amounts"])
    ok = all(abs(fact_w.get(c,0) - w_norm.get(c,0)) < 0.5 for c in CCYS)
    await update.message.reply_html(
        f"{fmt_weights_line('Целевые веса', w_norm)}\n"
        f"{fmt_weights_line('Факт', fact_w)}  {'✅' if ok else '⚠️'}\n"
        f"Банк: <b>{st['total']:.2f} USDT</b>\n"
        f"Последний ALLOC: <code>{st.get('last_alloc_id') or '—'}</code>"
    )

# ---------------- main ----------------
def main():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    builder = Application.builder().token(BOT_TOKEN)
    if _RL is not None:
        builder = builder.rate_limiter(_RL)
    app = builder.build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("settotal", cmd_settotal))
    app.add_handler(CommandHandler("setweights", cmd_setweights))
    app.add_handler(CommandHandler("alloc", cmd_alloc))
    app.add_handler(CommandHandler("alloc_applied", cmd_alloc_applied))
    app.add_handler(CommandHandler("status", cmd_status))

    log.info("Fund master bot starting…")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
