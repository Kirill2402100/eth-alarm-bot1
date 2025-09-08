# fa_bot.py — единый мастер+фунд бот
# Команды (мастер-режим):
#   /settotal 2800
#   /setweights jpy=40 aud=25 eur=20 gbp=15
#   /alloc            — расчёт и сводка, кусочки /setbank
#   /applyalloc       — разошлёт /setbank в символ-чаты (если задана SYMBOL_CHAT_MAP)
#
# Фунд-политика (для торговых чатов читает лист FA_Signals):
#   /fa USDJPY risk=Green bias=neutral ttl=300 reserve_off=0 dca_scale=1.0 scan_lock_until=+90m
#   /fa_show USDJPY
#   /fa_clear USDJPY
#
# Переменные окружения:
#   TELEGRAM_BOT_TOKEN   — токен
#   MASTER_CHAT_ID       — id мастер-чата (int). Если задан, мастер-команды принимаются только тут
#   SHEET_ID, GOOGLE_CREDENTIALS — гугл-таблица и json сервис-аккаунта
#   SYMBOL_CHAT_MAP      — JSON {"USDJPY": -100111, "AUDUSD": -100222, ...} (опц.) для /applyalloc
#   DEFAULT_WEIGHTS      — опц. JSON {"JPY":40,"AUD":25,"EUR":20,"GBP":15}
#
# Требования: python-telegram-bot[rate-limiter]>=20.7, gspread, pandas, numpy

from __future__ import annotations
import os, json, re, math, time, logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import gspread

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, ContextTypes, AIORateLimiter
)

log = logging.getLogger("fa_bot")
logging.basicConfig(level=logging.INFO)

# ---------- ENV ----------
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN","")
MASTER_CHAT_ID = int(os.environ.get("MASTER_CHAT_ID","0") or 0)
SHEET_ID = os.environ.get("SHEET_ID","")
GOOGLE_CREDENTIALS = os.environ.get("GOOGLE_CREDENTIALS","")
SYMBOL_CHAT_MAP = {}
try:
    if os.environ.get("SYMBOL_CHAT_MAP"):
        SYMBOL_CHAT_MAP = json.loads(os.environ["SYMBOL_CHAT_MAP"])
except Exception:
    log.exception("Bad SYMBOL_CHAT_MAP JSON")

DEFAULT_WEIGHTS = {"JPY":40,"AUD":25,"EUR":20,"GBP":15}
try:
    if os.environ.get("DEFAULT_WEIGHTS"):
        DEFAULT_WEIGHTS = json.loads(os.environ["DEFAULT_WEIGHTS"])
except Exception:
    log.exception("Bad DEFAULT_WEIGHTS JSON")

# ключи веса → символы
WEIGHT_KEY_TO_SYMBOL = {"JPY":"USDJPY","AUD":"AUDUSD","EUR":"EURUSD","GBP":"GBPUSD"}

# ---------- Sheets helpers ----------
def _gc():
    if not (SHEET_ID and GOOGLE_CREDENTIALS):
        raise RuntimeError("Sheets env not configured")
    return gspread.service_account_from_dict(json.loads(GOOGLE_CREDENTIALS))

def _open_sheet():
    gc = _gc()
    return gc.open_by_key(SHEET_ID)

def _ensure_ws(sh, title: str, headers: list[str]):
    try:
        ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=2000, cols=max(20, len(headers)))
        if headers:
            ws.append_row(headers)
    return ws

def upsert_fa_row(symbol: str, fields: dict):
    """Создаёт/обновляет строку в листе FA_Signals по полю pair == symbol (UPPER)."""
    sh = _open_sheet()
    ws = _ensure_ws(sh, "FA_Signals",
                    ["pair","risk","bias","ttl","updated_at",
                     "scan_lock_until","reserve_off","dca_scale"])
    rows = ws.get_all_records()
    sym = symbol.upper()

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    fields = {**fields}
    fields.setdefault("updated_at", now_iso)

    # нормализация
    if "risk" in fields:
        fields["risk"] = str(fields["risk"]).capitalize()
    if "bias" in fields:
        fields["bias"] = str(fields["bias"]).lower()

    found = False
    for i, r in enumerate(rows, start=2):  # 2 = заголовок + 1
        if str(r.get("pair","")).upper() == sym:
            found = True
            # соберём строку по заголовкам
            headers = ws.row_values(1)
            curr = {h: r.get(h,"") for h in headers}
            curr.update({"pair": sym})
            curr.update(fields)
            row_vals = [curr.get(h,"") for h in headers]
            ws.update(f"A{i}:{gspread.utils.rowcol_to_a1(i, len(headers)).split(':')[1]}", [row_vals])
            return

    if not found:
        headers = ws.row_values(1)
        new = {h:"" for h in headers}
        new.update({"pair": sym})
        new.update(fields)
        ws.append_row([new.get(h,"") for h in headers])

def get_fa_row(symbol: str) -> dict:
    sh = _open_sheet()
    try:
        ws = sh.worksheet("FA_Signals")
    except gspread.WorksheetNotFound:
        return {}
    sym = symbol.upper()
    for r in ws.get_all_records():
        if str(r.get("pair","")).upper() == sym:
            return r
    return {}

def clear_fa_row(symbol: str):
    sh = _open_sheet()
    try:
        ws = sh.worksheet("FA_Signals")
    except gspread.WorksheetNotFound:
        return
    rows = ws.get_all_records()
    sym = symbol.upper()
    for i, r in enumerate(rows, start=2):
        if str(r.get("pair","")).upper() == sym:
            ws.delete_rows(i)
            break

# ---------- Utils ----------
def only_master(update: Update) -> bool:
    if MASTER_CHAT_ID == 0:
        return True
    return (update.effective_chat and update.effective_chat.id == MASTER_CHAT_ID)

def parse_weights(arg: str) -> dict:
    # "jpy=40 aud=25 eur=20 gbp=15"
    pairs = re.findall(r'([a-zA-Z]+)\s*=\s*([0-9]+(?:\.[0-9]+)?)', arg)
    out = {}
    for k, v in pairs:
        k2 = k.strip().upper()
        if k2 in ("JPY","AUD","EUR","GBP"):
            out[k2] = float(v)
    s = sum(out.values()) or 1.0
    # нормализуем к 100, если «рядом»
    if abs(s - 100.0) > 1e-6:
        out = {k: v * (100.0/s) for k, v in out.items()}
    return out

def human_pct_map(m: dict) -> str:
    keys = ["JPY","AUD","EUR","GBP"]
    return " / ".join(f"{k} {m.get(k,0):.0f}" for k in keys)

def parse_rel_or_iso(s: str) -> str:
    """Возвращает ISO в UTC. Поддерживает '+90m', '+2h', '+1d', '2025-09-08 12:30'."""
    s = str(s).strip()
    if s.startswith("+"):
        m = re.match(r'^\+(\d+)([mhd])$', s)
        if not m:
            return ""
        val = int(m.group(1)); unit = m.group(2)
        if unit == "m": dt = datetime.now(timezone.utc) + timedelta(minutes=val)
        elif unit == "h": dt = datetime.now(timezone.utc) + timedelta(hours=val)
        else: dt = datetime.now(timezone.utc) + timedelta(days=val)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    # иначе ISO/полу-ISO
    try:
        ts = pd.to_datetime(s, utc=True)
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""

# ---------- Telegram Handlers ----------
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "FA/ALLOC бот готов.\n"
        "Мастер:\n"
        "  /settotal 2800\n"
        "  /setweights jpy=40 aud=25 eur=20 gbp=15\n"
        "  /alloc  — расчёт и сводка\n"
        "  /applyalloc — разослать /setbank в чаты (если настроен SYMBOL_CHAT_MAP)\n\n"
        "ФА-политика для торговых ботов (лист FA_Signals):\n"
        "  /fa USDJPY risk=Green bias=neutral ttl=300 reserve_off=0 dca_scale=1.0 scan_lock_until=+90m\n"
        "  /fa_show USDJPY\n"
        "  /fa_clear USDJPY"
    )

async def cmd_settotal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not only_master(update):
        return
    if not ctx.args:
        await update.effective_message.reply_text("Формат: /settotal 2800")
        return
    try:
        total = float(ctx.args[0])
        ctx.bot_data["alloc_total"] = total
        await update.effective_message.reply_text(f"Ок. Общий банк: {total:.2f} USDT")
    except Exception:
        await update.effective_message.reply_text("Не смог разобрать число. Пример: /settotal 2800")

async def cmd_setweights(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not only_master(update):
        return
    raw = " ".join(ctx.args)
    w = parse_weights(raw)
    if not w:
        await update.effective_message.reply_text("Формат: /setweights jpy=40 aud=25 eur=20 gbp=15")
        return
    ctx.bot_data["alloc_weights"] = w
    await update.effective_message.reply_text(f"Целевые веса: {human_pct_map(w)}")

def compute_alloc(total: float, weights: dict) -> dict:
    # Вернём {SYMBOL: amount}
    res = {}
    for k, pct in weights.items():
        sym = WEIGHT_KEY_TO_SYMBOL[k]
        res[sym] = round(total * (pct/100.0), 2)
    return res

async def cmd_alloc(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not only_master(update):
        return
    total = float(ctx.bot_data.get("alloc_total", 0))
    weights = ctx.bot_data.get("alloc_weights") or DEFAULT_WEIGHTS
    if total <= 0:
        await update.effective_message.reply_text("Сначала /settotal <сумма>.")
        return
    alloc = compute_alloc(total, weights)
    lines = []
    lines.append(f"Целевые веса: {human_pct_map(weights)}")
    lines.append(f"Итого банк: {total:.2f} USDT\n")
    for sym in ["USDJPY","AUDUSD","EURUSD","GBPUSD"]:
        amt = alloc.get(sym, 0.0)
        lines.append(f"{sym} → <b>{amt:.2f} USDT</b>  → команда: <code>/setbank {amt:.2f}</code>")
    lines.append("\nЕсли чаты подключены в SYMBOL_CHAT_MAP — используйте /applyalloc для авторассылки.")
    await update.effective_message.reply_html("\n".join(lines))

async def cmd_applyalloc(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not only_master(update):
        return
    total = float(ctx.bot_data.get("alloc_total", 0))
    weights = ctx.bot_data.get("alloc_weights") or DEFAULT_WEIGHTS
    if total <= 0:
        await update.effective_message.reply_text("Сначала /settotal <сумма> и /setweights …")
        return
    alloc = compute_alloc(total, weights)
    missing = [s for s in ["USDJPY","AUDUSD","EURUSD","GBPUSD"] if s not in SYMBOL_CHAT_MAP]
    if missing:
        await update.effective_message.reply_text(
            "Не задан SYMBOL_CHAT_MAP для: " + ", ".join(missing) +
            "\n(Добавь env переменную с JSON, или разошли команды вручную.)"
        )
    sent = []
    for sym, amt in alloc.items():
        chat_id = SYMBOL_CHAT_MAP.get(sym)
        if chat_id:
            try:
                await ctx.bot.send_message(chat_id=chat_id, text=f"/setbank {amt:.2f}")
                sent.append(sym)
            except Exception as e:
                log.error(f"send to {chat_id} failed: {e}")
    await update.effective_message.reply_text(
        "Готово. Разослано в: " + (", ".join(sent) if sent else "ничего (нет настроенных чатов).")
    )
    ctx.bot_data["alloc_last"] = {"total": total, "weights": weights, "ts": int(time.time())}

# ---- FA commands ----
def parse_kv(args: list[str]) -> dict:
    txt = " ".join(args)
    pairs = re.findall(r'([a-zA-Z_]+)\s*=\s*([^\s]+)', txt)
    out = {}
    for k, v in pairs:
        out[k.strip().lower()] = v.strip()
    return out

async def cmd_fa(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not only_master(update):
        # разрешим правку и из других чатов, если хочешь — оставь только мастер
        pass
    if not ctx.args:
        await update.effective_message.reply_text(
            "Формат:\n/fa USDJPY risk=Green bias=neutral ttl=300 reserve_off=0 dca_scale=1.0 scan_lock_until=+90m"
        ); return
    symbol = ctx.args[0].upper()
    kv = parse_kv(ctx.args[1:])
    fields = {}
    if "risk" in kv: fields["risk"] = kv["risk"]
    if "bias" in kv: fields["bias"] = kv["bias"]
    if "ttl" in kv:
        try: fields["ttl"] = int(float(kv["ttl"]))
        except: pass
    if "reserve_off" in kv:
        fields["reserve_off"] = "1" if kv["reserve_off"].lower() in ("1","true","yes","on") else "0"
    if "dca_scale" in kv:
        try: fields["dca_scale"] = float(kv["dca_scale"])
        except: pass
    if "scan_lock_until" in kv:
        iso = parse_rel_or_iso(kv["scan_lock_until"])
        if iso: fields["scan_lock_until"] = iso

    if not fields:
        await update.effective_message.reply_text("Нечего писать. Укажи хотя бы risk= / bias= / ttl= …")
        return

    try:
        upsert_fa_row(symbol, fields)
        row = get_fa_row(symbol)
        pretty = "\n".join(f"{k}: {row.get(k,'')}" for k in
                           ["pair","risk","bias","ttl","updated_at","scan_lock_until","reserve_off","dca_scale"])
        await update.effective_message.reply_text("Ок, записал в FA_Signals:\n" + pretty)
    except Exception as e:
        log.exception("FA upsert failed")
        await update.effective_message.reply_text(f"Ошибка записи в лист: {e}")

async def cmd_fa_show(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.effective_message.reply_text("Формат: /fa_show USDJPY"); return
    symbol = ctx.args[0].upper()
    try:
        row = get_fa_row(symbol)
        if not row:
            await update.effective_message.reply_text("Нет записи.")
            return
        pretty = "\n".join(f"{k}: {row.get(k,'')}" for k in
                           ["pair","risk","bias","ttl","updated_at","scan_lock_until","reserve_off","dca_scale"])
        await update.effective_message.reply_text(pretty)
    except Exception as e:
        await update.effective_message.reply_text(f"Ошибка чтения: {e}")

async def cmd_fa_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.effective_message.reply_text("Формат: /fa_clear USDJPY"); return
    symbol = ctx.args[0].upper()
    try:
        clear_fa_row(symbol)
        await update.effective_message.reply_text("Запись очищена.")
    except Exception as e:
        await update.effective_message.reply_text(f"Ошибка: {e}")

async def cmd_health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("ok")

# ---------- main ----------
def main():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is empty")
    app = Application.builder().token(BOT_TOKEN).rate_limiter(AIORateLimiter()).build()

    app.add_handler(CommandHandler("start",      cmd_start))
    app.add_handler(CommandHandler("health",     cmd_health))
    app.add_handler(CommandHandler("settotal",   cmd_settotal))
    app.add_handler(CommandHandler("setweights", cmd_setweights))
    app.add_handler(CommandHandler("alloc",      cmd_alloc))
    app.add_handler(CommandHandler("applyalloc", cmd_applyalloc))

    app.add_handler(CommandHandler("fa",        cmd_fa))
    app.add_handler(CommandHandler("fa_show",   cmd_fa_show))
    app.add_handler(CommandHandler("fa_clear",  cmd_fa_clear))

    log.info("FA/ALLOC bot is running…")
    app.run_polling(allowed_updates=["message"])

if __name__ == "__main__":
    main()
