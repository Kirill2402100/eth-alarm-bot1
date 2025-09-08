# fa_bot.py
from __future__ import annotations
import os, json, logging, asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

import gspread
from telegram import Update, BotCommand
from telegram.constants import ParseMode
from telegram.ext import Application, ContextTypes, CommandHandler

import llm_client

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fund_bot")

# ===== ENV =====
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_TOKEN", ""))
MASTER_CHAT_ID = int(os.getenv("MASTER_CHAT_ID", "0") or "0")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
SHEET_ID        = os.getenv("SHEET_ID", "")
GOOGLE_CRED     = os.getenv("GOOGLE_CREDENTIALS", "")

DEFAULT_WEIGHTS_RAW = os.getenv("DEFAULT_WEIGHTS", '{"JPY":40,"AUD":25,"EUR":20,"GBP":15}')
# валютные пары → названия для весов
PAIR_ORDER = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]
WEIGHT_KEYS = ["JPY","AUD","EUR","GBP"]

def _parse_default_weights() -> Dict[str, float]:
    try:
        if DEFAULT_WEIGHTS_RAW.strip().startswith("{"):
            m = json.loads(DEFAULT_WEIGHTS_RAW)
        else:
            # формат типа 'JPY:40,AUD:25,EUR:20,GBP:15'
            m = {}
            for part in DEFAULT_WEIGHTS_RAW.replace(" ", "").split(","):
                if not part: continue
                k, v = part.split(":")
                m[k.upper()] = float(v)
        out = {k.upper(): float(m.get(k, 0)) for k in WEIGHT_KEYS}
        s = sum(out.values()) or 1.0
        return {k: round(v * 100.0 / s, 2) for k, v in out.items()}
    except Exception:
        return {"JPY":40.0,"AUD":25.0,"EUR":20.0,"GBP":15.0}

TARGET_WEIGHTS = _parse_default_weights()

# ===== Sheets =====
def _gs_client() -> Optional[gspread.Client]:
    if not (GOOGLE_CRED and SHEET_ID):
        return None
    try:
        cred = json.loads(GOOGLE_CRED)
        return gspread.service_account_from_dict(cred)
    except Exception as e:
        log.error("Google credentials parse error: %s", e)
        return None

def ensure_fa_sheet() -> Tuple[Optional[gspread.Worksheet], str]:
    """
    Возвращает (ws, msg). Создаёт лист FA_Signals при необходимости.
    """
    if not (GOOGLE_CRED and SHEET_ID):
        return None, "Sheets: не настроено (нет SHEET_ID/GOOGLE_CREDENTIALS)."
    try:
        gc = _gs_client()
        if not gc:
            return None, "Sheets: ошибка аутентификации."
        sh = gc.open_by_key(SHEET_ID)
        headers = ["pair","risk","bias","ttl","updated_at","scan_lock_until","reserve_off","dca_scale"]
        try:
            ws = sh.worksheet("FA_Signals")
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title="FA_Signals", rows=200, cols=len(headers))
            ws.append_row(headers)
        return ws, "Sheets: OK."
    except Exception as e:
        log.exception("Sheets init failed")
        return None, f"Sheets: ошибка — {e}"

def upsert_fa_row(ws: gspread.Worksheet, pair: str, payload: Dict[str, Any]) -> None:
    """Обновить / добавить строку по pair."""
    records = ws.get_all_records()
    headers = ws.row_values(1)
    row_idx = None
    for i, r in enumerate(records, start=2):
        if str(r.get("pair","")).upper() == pair.upper():
            row_idx = i
            break
    row = []
    for h in headers:
        if h == "pair":
            row.append(pair.upper())
        elif h == "updated_at":
            row.append(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        else:
            row.append(payload.get(h, ""))
    if row_idx:
        ws.update(f"A{row_idx}:{gspread.utils.rowcol_to_a1(row_idx, len(headers))}", [row])
    else:
        ws.append_row(row)

# ===== Вспомогалки =====

def fmt_weights_line(weights: Dict[str, float]) -> str:
    return " / ".join([f"{k} {v:.0f}%" for k, v in [("JPY",weights["JPY"]),("AUD",weights["AUD"]),("EUR",weights["EUR"]),("GBP",weights["GBP"])] ])

def parse_setweights(args: List[str]) -> Dict[str, float]:
    """
    /setweights jpy=40 aud=25 eur=20 gbp=15
    """
    out = {}
    for a in args:
        if "=" not in a: continue
        k, v = a.split("=", 1)
        try:
            out[k.strip().upper()] = float(str(v).strip())
        except Exception:
            pass
    # нормализуем до 100
    s = sum(out.values()) or 1.0
    out = {k: round(v * 100.0 / s, 2) for k, v in out.items()}
    for k in WEIGHT_KEYS:
        out.setdefault(k, 0.0)
    return out

def compute_alloc(total: float, weights: Dict[str, float]) -> Dict[str, float]:
    pr = {k: weights.get(k, 0.0)/100.0 for k in WEIGHT_KEYS}
    return {
        "USDJPY": round(total * pr["JPY"], 2),
        "AUDUSD": round(total * pr["AUD"], 2),
        "EURUSD": round(total * pr["EUR"], 2),
        "GBPUSD": round(total * pr["GBP"], 2),
    }

# ===== Handlers =====

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    ws, msg = ensure_fa_sheet()
    sheet_note = msg
    hello = (
        "Привет! Я фунд-бот.\n"
        f"Текущий чат id: <code>{cid}</code>\n\n"
        "Команды: /help\n\n"
        f"{sheet_note}"
    )
    await update.effective_message.reply_html(hello)

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    txt = (
        "<b>Что я умею</b>\n"
        "• <code>/settotal 2800</code> — задать общий банк (только в мастер-чате).\n"
        "• <code>/setweights jpy=40 aud=25 eur=20 gbp=15</code> — выставить целевые веса.\n"
        "• <code>/weights</code> — показать целевые веса.\n"
        "• <code>/alloc</code> — расчёт сумм и готовые команды /setbank для торговых чатов.\n"
        "• <code>/digest</code> — короткий «человеческий» дайджест по USDJPY / AUDUSD / EURUSD / GBPUSD.\n"
        "• <code>/check_sheets</code> — быстрая проверка соединения с Google Sheets.\n"
        "• <code>/ping</code> — проверить связь."
    )
    await update.effective_message.reply_html(txt)

async def cmd_ping(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_html("pong ✅")

async def cmd_check_sheets(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ws, msg = ensure_fa_sheet()
    if not ws:
        await update.effective_message.reply_html(msg)
        return
    try:
        upsert_fa_row(ws, "USDJPY", {"risk":"OK","bias":"both","ttl":60,"scan_lock_until":"","reserve_off":"","dca_scale":1.0})
        await update.effective_message.reply_html("Sheets: запись тестовой строки в <b>FA_Signals</b> выполнена ✅")
    except Exception as e:
        await update.effective_message.reply_html(f"Sheets ошибка: {e}")

# --- состояние в памяти (простое) ---
STATE: Dict[str, Any] = {
    "total": 0.0,
    "weights": TARGET_WEIGHTS.copy(),
}

def _is_master(chat_id: int) -> bool:
    return MASTER_CHAT_ID and chat_id == MASTER_CHAT_ID

async def cmd_settotal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_master(update.effective_chat.id):
        await update.effective_message.reply_html("Эта команда доступна только в мастер-чате.")
        return
    if not ctx.args:
        await update.effective_message.reply_html("Укажи сумму: <code>/settotal 2800</code>")
        return
    try:
        total = float(ctx.args[0].replace(",", "."))
        STATE["total"] = max(0.0, total)
        await update.effective_message.reply_html(f"Общий банк установлен: <b>{STATE['total']:.2f} USDT</b>")
    except Exception:
        await update.effective_message.reply_html("Не понял сумму. Пример: <code>/settotal 2800</code>")

async def cmd_setweights(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_master(update.effective_chat.id):
        await update.effective_message.reply_html("Эта команда доступна только в мастер-чате.")
        return
    w = parse_setweights(ctx.args or [])
    if sum(w.values()) <= 0:
        await update.effective_message.reply_html("Не распознал веса. Пример: <code>/setweights jpy=40 aud=25 eur=20 gbp=15</code>")
        return
    STATE["weights"] = w
    await update.effective_message.reply_html("Целевые веса обновлены: " + fmt_weights_line(STATE["weights"]))

async def cmd_weights(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_html("Целевые веса: " + fmt_weights_line(STATE["weights"]))

async def cmd_alloc(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    total = STATE.get("total", 0.0)
    if total <= 0:
        await update.effective_message.reply_html("Сначала задай общий банк: <code>/settotal 2800</code> (в мастер-чате)")
        return
    w = STATE.get("weights", TARGET_WEIGHTS)
    alloc = compute_alloc(total, w)
    lines = []
    lines.append(f"<b>Сумма распределения</b> (банк {total:.2f} USDT)")
    for pair in PAIR_ORDER:
        val = alloc[pair]
        human = pair.replace("USD", "").replace("JPY","JPY").replace("AUD","AUD").replace("EUR","EUR").replace("GBP","GBP")
        lines.append(f"• {pair} → <b>{val:.2f} USDT</b> → команда: <code>/setbank {int(round(val))}</code>")
    lines.append("\nЦелевые веса: " + fmt_weights_line(w))
    await update.effective_message.reply_html("\n".join(lines))

async def cmd_digest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Собираем простые флаги (пока пусто/заглушка) и просим mini-модель сделать человеческий дайджест.
    """
    # Заглушка флагов, чтобы команда всегда работала:
    flags: Dict[str, Dict[str, Any]] = {}
    try:
        # пример вызова nano-анализа: заголовки ты подставишь, когда подключишь реальный сбор
        sample_titles = {
            "USDJPY": ["MinFin Japan vows to act against excessive FX volatility", "U.S. yields edge higher before ISM"],
            "AUDUSD": ["Australia CPI in line with expectations", "Commodities stable; iron ore flat"],
            "EURUSD": ["ECB minutes hint at data-dependency", "Euro area PMIs mixed"],
            "GBPUSD": ["BoE members stress inflation persistence", "Gilts volatility elevated"],
        }
        for p in PAIR_ORDER:
            fl = llm_client.nano_headlines_flags(p, sample_titles.get(p, []))
            if isinstance(fl, dict):
                flags[p] = fl
    except Exception as e:
        log.warning("nano flags failed: %s", e)

    # Вызов мини-модели для «человеческого» текста
    try:
        text = llm_client.mini_digest_ru(PAIR_ORDER, flags).strip()
        if not text:
            text = "Сводка: изменений по рынку не выявлено."
        # Дополнительная строка с флагами (компактно)
        try:
            flags_compact = {k: {"risk": v.get("risk_level"), "bias": v.get("bias")} for k, v in flags.items()}
            flags_line = "<code>" + json.dumps(flags_compact, ensure_ascii=False) + "</code>"
        except Exception:
            flags_line = "<i>(флаги не распознаны)</i>"
        await update.effective_message.reply_html(f"<b>Дайджест</b>\n{text}\n\nФлаги:\n{flags_line}")
    except Exception as e:
        await update.effective_message.reply_html(f"LLM ошибка: {e}")

# ===== Main =====

async def _set_bot_commands(app: Application):
    cmds = [
        BotCommand("start", "Запуск и проверка окружения"),
        BotCommand("help", "Справка"),
        BotCommand("weights", "Показать целевые веса"),
        BotCommand("setweights", "Задать целевые веса (только мастер-чат)"),
        BotCommand("settotal", "Задать общий банк (только мастер-чат)"),
        BotCommand("alloc", "Рассчитать распределение банка"),
        BotCommand("digest", "Короткий дайджест по 4 парам"),
        BotCommand("check_sheets", "Проверить доступ к Google Sheets"),
        BotCommand("ping", "Проверить связь"),
    ]
    await app.bot.set_my_commands(cmds)

def _env_summary() -> str:
    ok_llm = "✅" if OPENAI_API_KEY else "❌"
    ws, msg = ensure_fa_sheet()
    ok_sh = "✅" if ws else "❌"
    return f"Фунд-бот запущен {ok_llm} LLM {ok_sh} Sheets. {msg}"

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    log.info("Fund bot is running…")
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("help",        cmd_help))
    app.add_handler(CommandHandler("ping",        cmd_ping))
    app.add_handler(CommandHandler("check_sheets",cmd_check_sheets))

    app.add_handler(CommandHandler("settotal",    cmd_settotal))
    app.add_handler(CommandHandler("setweights",  cmd_setweights))
    app.add_handler(CommandHandler("weights",     cmd_weights))
    app.add_handler(CommandHandler("alloc",       cmd_alloc))
    app.add_handler(CommandHandler("digest",      cmd_digest))

    async def on_start(_):
        await _set_bot_commands(app)
        try:
            await app.bot.send_message(chat_id=MASTER_CHAT_ID or None, text=_env_summary())
        except Exception:
            pass

    app.post_init = on_start

    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
