import os
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

import gspread
import pandas as pd
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update

from llm_client import summarize_pair_ru, classify_headlines_nano, deep_escalation_ru, llm_usage_today

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fund_bot")

# ========= ENV =========
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")          # токен ТГ бота (фунд-бот)
MASTER_CHAT_ID = os.getenv("MASTER_CHAT_ID", "")         # куда слать дайджест/сводки (если пусто — в чат, откуда пришла команда)
SHEET_ID = os.getenv("SHEET_ID", "")                     # Google Sheet ID
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS", "")  # JSON service account

# Пары под управлением фунд-бота
PAIRS = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]
PAIR_TO_QUOTE = {"USDJPY": "JPY", "AUDUSD": "AUD", "EURUSD": "EUR", "GBPUSD": "GBP"}

# Значения по умолчанию для весов (можно переопределить в ENV 'DEFAULT_WEIGHTS')
DEFAULT_WEIGHTS_ENV = os.getenv("DEFAULT_WEIGHTS", '{"JPY":40,"AUD":25,"EUR":20,"GBP":15}')

def _parse_weights_env(s: str) -> Dict[str, float]:
    """Поддерживаем JSON и формат 'JPY:40,AUD:25,EUR:20,GBP:15'."""
    try:
        obj = json.loads(s)
        return {k.upper(): float(v) for k, v in obj.items()}
    except Exception:
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
        out = {}
        for p in parts:
            if ":" in p:
                k, v = p.split(":", 1)
            elif "=" in p:
                k, v = p.split("=", 1)
            else:
                continue
            try:
                out[k.strip().upper()] = float(v.strip())
            except Exception:
                pass
        return out or {"JPY": 40, "AUD": 25, "EUR": 20, "GBP": 15}

DEFAULT_WEIGHTS = _parse_weights_env(DEFAULT_WEIGHTS_ENV)

# ========= Google Sheets helpers =========

def _gs_client() -> Optional[gspread.Client]:
    if not (SHEET_ID and GOOGLE_CREDENTIALS_JSON):
        return None
    try:
        creds = json.loads(GOOGLE_CREDENTIALS_JSON)
        return gspread.service_account_from_dict(creds)
    except Exception:
        log.exception("Google credentials parse error")
        return None

def _ensure_ws(sh, title: str, headers: Optional[List[str]] = None):
    try:
        ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=2000, cols=20)
        if headers:
            ws.append_row(headers)
    return ws

def _read_fa_policy_from_sheet(symbol: str) -> Dict[str, Any]:
    """
    Читает актуальные флаги из листа FA_Signals (если есть).
    Поля: pair, risk(Green/Amber/Red), bias(neutral/long-only/short-only), ttl, updated_at,
          scan_lock_until, reserve_off, dca_scale
    """
    gc = _gs_client()
    if not gc:
        return {}
    try:
        sh = gc.open_by_key(SHEET_ID)
        try:
            ws = sh.worksheet("FA_Signals")
        except gspread.WorksheetNotFound:
            return {}
        rows = ws.get_all_records()
        for r in rows:
            if str(r.get("pair", "")).upper() == symbol.upper():
                risk = (str(r.get("risk", "") or "Green").capitalize())
                bias = (str(r.get("bias", "") or "neutral").lower())
                ttl = int(r.get("ttl") or 0)
                updated_at = str(r.get("updated_at") or "").strip()
                try:
                    reserve_off = str(r.get("reserve_off") or "").strip().lower() in ("1","true","yes","on")
                except Exception:
                    reserve_off = False
                try:
                    dca_scale = float(r.get("dca_scale") or 1.0)
                except Exception:
                    dca_scale = 1.0

                # TTL: если устарело — вернём пусто (пусть торговый бот проигнорит)
                if ttl and updated_at:
                    try:
                        ts = pd.to_datetime(updated_at, utc=True)
                        if pd.Timestamp.utcnow() > ts + pd.Timedelta(minutes=ttl):
                            return {}
                    except Exception:
                        pass

                scan_lock_until = str(r.get("scan_lock_until") or "").strip()
                return {
                    "risk": risk, "bias": bias, "ttl": ttl, "updated_at": updated_at,
                    "scan_lock_until": scan_lock_until, "reserve_off": reserve_off, "dca_scale": dca_scale
                }
        return {}
    except Exception:
        log.exception("read FA_Signals failed")
        return {}

def _write_alloc_snapshot(total: float, weights: Dict[str, float], per_pair: Dict[str, float], run_id: str):
    """
    Пишем в лист 'FA_Alloc': run_id, ts_utc, total, weights_json, pair, target_bank
    """
    gc = _gs_client()
    if not gc:
        return
    try:
        sh = gc.open_by_key(SHEET_ID)
        ws = _ensure_ws(sh, "FA_Alloc", headers=["run_id","ts_utc","total_usdt","weights_json","pair","target_usdt"])
        wjson = json.dumps(weights, ensure_ascii=False)
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        rows = [[run_id, ts, float(total), wjson, pair, float(amt)] for pair, amt in per_pair.items()]
        ws.append_rows(rows)
    except Exception:
        log.exception("write FA_Alloc failed")

def _write_config(total: Optional[float], weights: Optional[Dict[str, float]]):
    """
    Сохраним последнее total/weights в лист 'FA_Config' (K/V).
    """
    if not (SHEET_ID and GOOGLE_CREDENTIALS_JSON):
        return
    try:
        gc = _gs_client()
        if not gc:
            return
        sh = gc.open_by_key(SHEET_ID)
        ws = _ensure_ws(sh, "FA_Config", headers=["key","value","updated_at_utc"])
        kv = []
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        if total is not None:
            kv.append(["total_bank", str(total), ts])
        if weights is not None:
            kv.append(["weights_json", json.dumps(weights, ensure_ascii=False), ts])
        if kv:
            ws.append_rows(kv)
    except Exception:
        log.exception("write FA_Config failed")


# ========= Внутреннее состояние (в памяти процесса) =========
_state = {
    "total_bank": None,  # float
    "weights": DEFAULT_WEIGHTS.copy(),  # {"JPY":40,...}
}

def _norm_pair(sym: str) -> str:
    return (sym or "").upper()

def _risk_emoji(risk: str) -> str:
    return {"Green": "✅", "Amber": "⚠️", "Red": "🚨"}.get((risk or "").capitalize(), "ℹ️")

def _bias_ru(bias: str) -> str:
    m = {"neutral":"оба направления", "long-only":"только LONG", "short-only":"только SHORT"}
    return m.get((bias or "").lower(), "оба направления")

def _parse_weights_arg(s: str) -> Dict[str, float]:
    """
    /setweights jpy=40 aud=25 eur=20 gbp=15
    """
    out: Dict[str, float] = {}
    s = s.strip().replace(";", " ").replace(",", " ")
    parts = [p for p in s.split() if p]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip().upper()
        try:
            out[k] = float(v.strip())
        except Exception:
            pass
    return out

def _alloc_per_pair(total: float, weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(weights.get(ccy, 0.0) for ccy in ["JPY","AUD","EUR","GBP"])
    if s <= 0:
        return {p: 0.0 for p in PAIRS}
    # нормируем на 100, если нужно
    scale = 100.0 / s
    wN = {k: v*scale for k, v in weights.items()}
    per: Dict[str, float] = {}
    for pair in PAIRS:
        ccy = PAIR_TO_QUOTE[pair]
        per[pair] = round(total * (wN.get(ccy, 0.0) / 100.0), 2)
    return per

# ========= Команды =========

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Привет! Я фунд-бот (мастер-канал).\n\n"
        "Я не меняю банки сам — только проверяю соответствие целевым весам и даю рекомендации.\n"
        "Утренний дайджест — на русском, отдельно по каждой паре.\n\n"
        "<b>Команды</b>:\n"
        "/settotal 2800 — задать общий банк\n"
        "/setweights jpy=40 aud=25 eur=20 gbp=15 — задать целевые веса\n"
        "/alloc — показать текущий расчёт распределения\n"
        "/morning — утренний дайджест по парам\n"
        "/status — статус LLM/бюджета\n"
        "/help — подсказка по командам"
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text, parse_mode="HTML")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = llm_usage_today()
    txt = (
        f"LLM usage {u['date']}: in={u['input_tokens']} out={u['output_tokens']} (budget {u['budget']})\n"
        f"Текущие веса: {json.dumps(_state['weights'], ensure_ascii=False)}\n"
        f"Общий банк: {(_state['total_bank'] if _state['total_bank'] is not None else 'не задан')}"
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=txt)

async def cmd_settotal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Использование: /settotal 2800")
        return
    try:
        total = float(context.args[0])
        if total <= 0:
            raise ValueError
    except Exception:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Нужен положительный банк, например: /settotal 2800")
        return

    _state["total_bank"] = total
    _write_config(total=total, weights=None)

    # Если есть веса — сразу покажем расчёт
    per = _alloc_per_pair(total, _state["weights"])
    lines = ["✅ Принято. Расчёт по текущим весам:"]
    for pair in PAIRS:
        amt = per[pair]
        lines.append(f"• {pair} → <b>{amt:.2f} USDT</b>  → команда в чат {pair}:  <code>/setbank {amt:.2f}</code>")
    # сводка
    w = _state["weights"]
    fact = f"Целевые веса: JPY {w.get('JPY',0):.0f} / AUD {w.get('AUD',0):.0f} / EUR {w.get('EUR',0):.0f} / GBP {w.get('GBP',0):.0f}"
    lines.append(fact)
    run_id = f"ALLOC-{datetime.utcnow().strftime('%Y-%m-%d-%H%M%S')}"
    _write_alloc_snapshot(total, w, per, run_id)

    await context.bot.send_message(chat_id=update.effective_chat.id, text="\n".join(lines), parse_mode="HTML")

async def cmd_setweights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    argline = " ".join(context.args)
    neww = _parse_weights_arg(argline)
    if not neww:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Использование: /setweights jpy=40 aud=25 eur=20 gbp=15")
        return

    # Обновим только известные ключи
    for k in ["JPY","AUD","EUR","GBP"]:
        if k in neww:
            _state["weights"][k] = float(neww[k])

    _write_config(total=None, weights=_state["weights"])

    lines = ["✅ Веса обновлены."]
    if _state["total_bank"] is not None:
        per = _alloc_per_pair(_state["total_bank"], _state["weights"])
        for pair in PAIRS:
            amt = per[pair]
            lines.append(f"• {pair} → <b>{amt:.2f} USDT</b>  → команда:  <code>/setbank {amt:.2f}</code>")
        lines.append(
            f"Целевые веса: JPY { _state['weights']['JPY']:.0f} / AUD { _state['weights']['AUD']:.0f} / "
            f"EUR { _state['weights']['EUR']:.0f} / GBP { _state['weights']['GBP']:.0f}"
        )
        run_id = f"ALLOC-{datetime.utcnow().strftime('%Y-%m-%d-%H%M%S')}"
        _write_alloc_snapshot(_state["total_bank"], _state["weights"], per, run_id)

    await context.bot.send_message(chat_id=update.effective_chat.id, text="\n".join(lines), parse_mode="HTML")

async def cmd_alloc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _state["total_bank"] is None:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Сначала задай общий банк: /settotal 2800")
        return
    per = _alloc_per_pair(_state["total_bank"], _state["weights"])
    lines = [f"Текущий расчёт распределения (total={_state['total_bank']:.2f}):"]
    for pair in PAIRS:
        lines.append(f"• {pair}: {per[pair]:.2f} USDT  (команда: /setbank {per[pair]:.2f})")
    await context.bot.send_message(chat_id=update.effective_chat.id, text="\n".join(lines))

async def cmd_morning(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = int(MASTER_CHAT_ID) if str(MASTER_CHAT_ID).strip() else update.effective_chat.id
    now_utc = datetime.utcnow().strftime("%H:%M")
    header = f"🧭 Утренний фон (UTC {now_utc})"
    blocks = [header]

    # TODO: сюда можно подставлять реальные заголовки/календарь при интеграции
    headlines_by_pair: Dict[str, List[str]] = {}

    for sym in PAIRS:
        flags = _read_fa_policy_from_sheet(sym)  # если листа нет — вернётся {}
        # если есть заголовки — можно оценить их «дёшево»
        hl = headlines_by_pair.get(sym, [])
        if hl:
            try:
                cls = await classify_headlines_nano(hl)
                # при желании можно использовать cls для корректировки пояснений в сводке
                flags = {**flags, "nano_headline_flags": cls}
            except Exception:
                pass

        try:
            text = await summarize_pair_ru(pair=sym, flags=flags, headlines=hl)
        except Exception as e:
            log.exception("summarize_pair_ru failed")
            risk = (flags.get("risk") or "Green").capitalize()
            bias = _bias_ru(flags.get("bias") or "neutral")
            roff = "выкл" if flags.get("reserve_off") else "вкл"
            scale = flags.get("dca_scale", 1.0)
            text = (
                f"{_risk_emoji(risk)} {sym} — {risk}. Режим: {bias}. "
                f"Резерв доборов: {roff}. Масштаб доборов: x{scale:.2f}."
            )

        blocks.append(f"\n<b>{sym}</b>\n{text}")

    await context.bot.send_message(chat_id=chat_id, text="\n".join(blocks), parse_mode="HTML")

async def cmd_escalate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Ручная «старшая» аналитика: /escalate USDJPY Текст/контекст
    """
    if len(context.args) < 2:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Использование: /escalate USDJPY ваш контекст")
        return
    pair = _norm_pair(context.args[0])
    ctx = " ".join(context.args[1:])
    try:
        text = await deep_escalation_ru(pair, ctx)
    except Exception:
        text = "⚠️ Не удалось выполнить глубокий разбор сейчас."
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"<b>{pair}</b>\n{text}", parse_mode="HTML")


# ========= Bootstrap =========

def build_app() -> Application:
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("settotal", cmd_settotal))
    app.add_handler(CommandHandler("setweights", cmd_setweights))
    app.add_handler(CommandHandler("alloc", cmd_alloc))
    app.add_handler(CommandHandler("morning", cmd_morning))
    app.add_handler(CommandHandler("escalate", cmd_escalate))
    return app

if __name__ == "__main__":
    if not BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN не задан")
    app = build_app()
    log.info("Fund bot is running…")
    app.run_polling(close_loop=False)
