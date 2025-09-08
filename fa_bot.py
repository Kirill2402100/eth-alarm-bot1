# fa_bot.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Dict, Tuple, Optional

from telegram import Update, BotCommand
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# Rate limiter (опционально). Если extras не поставлены — просто не используем.
try:
    from telegram.ext import AIORateLimiter  # type: ignore
except Exception:  # pragma: no cover
    AIORateLimiter = None  # noqa

# Google Sheets
import gspread

# Наш LLM клиент (лежит рядом в репо)
from llm_client import summarize_digest  # noqa: E402

log = logging.getLogger("fund_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# =========================
# ---- Конфиг/состояние ---
# =========================

# Пары, которые покрываем и отображаем (в порядке показа)
SYMBOLS = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]
KEY2PAIR = {"JPY": "USDJPY", "AUD": "AUDUSD", "EUR": "EURUSD", "GBP": "GBPUSD"}

STATE = {
    "total_bank": 0.0,                              # общий банк, задаётся в мастер-канале
    "weights": {"JPY": 40.0, "AUD": 25.0, "EUR": 20.0, "GBP": 15.0},  # по умолчанию
    "sheet_ready": False,
    "sheet_note": "not checked"
}


def _env_bot_token() -> str:
    return os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN") or ""


def _env_master_chat_id() -> Optional[int]:
    val = os.environ.get("MASTER_CHAT_ID", "").strip()
    if not val:
        return None
    try:
        return int(val)
    except Exception:
        return None


def _parse_default_weights_env() -> Dict[str, float]:
    """
    Поддерживает форматы:
    - JSON: {"JPY":40,"AUD":25,"EUR":20,"GBP":15}
    - Комма-строка: JPY:40,AUD:25,EUR:20,GBP:15
    - Пробельная строка: jpy=40 aud=25 eur=20 gbp=15
    """
    raw = (os.environ.get("DEFAULT_WEIGHTS") or "").strip()
    if not raw:
        return STATE["weights"].copy()

    # JSON?
    if raw.startswith("{"):
        try:
            d = json.loads(raw)
            norm = {k.strip().upper(): float(v) for k, v in d.items()}
            return _fix_weights(norm)
        except Exception:
            pass

    # key=val или key:val
    parts = re.split(r"[,\s]+", raw)
    out = {}
    for p in parts:
        if not p:
            continue
        if "=" in p:
            k, v = p.split("=", 1)
        elif ":" in p:
            k, v = p.split(":", 1)
        else:
            continue
        out[k.strip().upper()] = float(v)
    if out:
        return _fix_weights(out)
    return STATE["weights"].copy()


def _fix_weights(w: Dict[str, float]) -> Dict[str, float]:
    # Оставляем только JPY/AUD/EUR/GBP, отнормируем если сумма != 100
    keep = {k: float(w.get(k, 0.0)) for k in ("JPY", "AUD", "EUR", "GBP")}
    s = sum(keep.values())
    if s <= 0:
        return STATE["weights"].copy()
    if abs(s - 100.0) > 1e-6:
        keep = {k: (v / s) * 100.0 for k, v in keep.items()}
    return keep


def _fmt_money(x: float) -> str:
    return f"{x:.2f}"


def _is_master_chat(update: Update) -> bool:
    cid = update.effective_chat.id if update.effective_chat else None
    return cid is not None and _env_master_chat_id() is not None and cid == _env_master_chat_id()


# =========================
# ---- Google Sheets -------
# =========================

def try_init_sheets():
    """Инициализация и автосоздание вкладки FA_Signals.
       Возвращает (sh, note). В случае ошибки — (None, reason)."""
    creds_json = os.environ.get("GOOGLE_CREDENTIALS")
    sheet_id = os.environ.get("SHEET_ID")
    if not creds_json or not sheet_id:
        return None, "нет SHEET_ID/GOOGLE_CREDENTIALS"

    try:
        creds = json.loads(creds_json)
    except Exception as e:
        return None, f"GOOGLE_CREDENTIALS не JSON: {e.__class__.__name__}"

    try:
        gc = gspread.service_account_from_dict(creds)
        sh = gc.open_by_key(sheet_id)
        try:
            sh.worksheet("FA_Signals")
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title="FA_Signals", rows=100, cols=10)
            ws.append_row(["pair", "risk", "bias", "ttl", "updated_at", "scan_lock_until", "reserve_off", "dca_scale"])
        return sh, "OK"
    except Exception as e:
        return None, f"auth/open error: {e.__class__.__name__}"


def read_flags_from_sheet_safe() -> Dict[str, dict]:
    """Считываем политику из FA_Signals. Возвращаем dict { 'USDJPY': {...}, ... }"""
    sh, note = try_init_sheets()
    if not sh:
        log.warning("Sheets not ready: %s", note)
        return {}

    try:
        ws = sh.worksheet("FA_Signals")
    except gspread.WorksheetNotFound:
        return {}

    rows = ws.get_all_records()
    out: Dict[str, dict] = {}
    for r in rows:
        p = str(r.get("pair") or "").upper().strip()
        if not p:
            continue
        try:
            risk = str(r.get("risk") or "Green").capitalize()
            bias = str(r.get("bias") or "neutral").lower()
            ttl = int(r.get("ttl") or 0)
            updated_at = str(r.get("updated_at") or "").strip()
            scan_lock_until = str(r.get("scan_lock_until") or "").strip()
            reserve_off = str(r.get("reserve_off") or "").strip().lower() in ("1", "true", "yes", "on")
            try:
                dca_scale = float(r.get("dca_scale") or 1.0)
            except Exception:
                dca_scale = 1.0
            out[p] = {
                "risk": risk,
                "bias": bias,
                "ttl": ttl,
                "updated_at": updated_at,
                "scan_lock_until": scan_lock_until,
                "reserve_off": reserve_off,
                "dca_scale": dca_scale
            }
        except Exception:
            continue
    return out


# =========================
# ---- Telegram handlers ---
# =========================

HELP_TEXT = (
    "<b>Что я умею</b>\n"
    "• <code>/settotal 2800</code> — задать общий банк (только в мастер-чате).\n"
    "• <code>/setweights jpy=40 aud=25 eur=20 gbp=15</code> — выставить целевые веса.\n"
    "• <code>/weights</code> — показать целевые веса.\n"
    "• <code>/alloc</code> — расчёт сумм и готовые команды <b>/setbank</b> для торговых чатов.\n"
    "• <code>/digest</code> — короткий «человеческий» дайджест по USDJPY / AUDUSD / EURUSD / GBPUSD.\n"
    "• <code>/flags</code> — показать активные флаги из FA_Signals.\n"
    "• <code>/diag</code> — проверить доступ к LLM и Sheets.\n"
    "• <code>/ping</code> — проверить связь.\n\n"
    "<i>Банк по парам задаётся вручную в торговых чатах; я сверяю распределение и даю советы/дайджест.</i>"
)


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id if update.effective_chat else None
    await update.effective_chat.send_message(
        "Привет! Я фунд-бот.\nТекущий чат id: <code>{}</code>\n\nКоманды: /help".format(chat_id),
        parse_mode=ParseMode.HTML
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_chat.send_message(HELP_TEXT, parse_mode=ParseMode.HTML)


async def cmd_ping(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_chat.send_message("pong ✅")


async def cmd_diag(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # LLM
    llm_ok = bool(os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("LLM_MINI"))
    # Sheets
    sh, note = try_init_sheets()
    sheets_ok = sh is not None
    STATE["sheet_ready"] = sheets_ok
    STATE["sheet_note"] = note
    await update.effective_chat.send_message(
        f"LLM: {'✅' if llm_ok else '❌'}\nSheets: {'✅' if sheets_ok else '❌'} ({note})"
    )


async def cmd_weights(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    w = STATE["weights"]
    txt = f"Целевые веса: JPY {w['JPY']:.0f} / AUD {w['AUD']:.0f} / EUR {w['EUR']:.0f} / GBP {w['GBP']:.0f}"
    await update.effective_chat.send_message(txt)


def _parse_setweights(args: list[str]) -> Dict[str, float]:
    s = " ".join(args)
    parts = re.split(r"[,\s]+", s.strip())
    out = {}
    for p in parts:
        if not p:
            continue
        if "=" in p:
            k, v = p.split("=", 1)
        elif ":" in p:
            k, v = p.split(":", 1)
        else:
            continue
        out[k.strip().upper()] = float(v)
    return _fix_weights(out) if out else STATE["weights"].copy()


async def cmd_setweights(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_master_chat(update):
        await update.effective_chat.send_message("Команда доступна только в мастер-чате.")
        return
    try:
        nw = _parse_setweights(ctx.args or [])
        STATE["weights"] = nw
        await cmd_weights(update, ctx)
    except Exception as e:
        await update.effective_chat.send_message(f"Не удалось разобрать веса: {e}")


async def cmd_settotal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_master_chat(update):
        await update.effective_chat.send_message("Команда доступна только в мастер-чате.")
        return
    try:
        if not ctx.args:
            raise ValueError("Укажите число: /settotal 2800")
        total = float(ctx.args[0])
        if total <= 0:
            raise ValueError("Банк должен быть > 0")
        STATE["total_bank"] = total
        await update.effective_chat.send_message(f"Общий банк установлен: {total:.2f} USDT")
    except Exception as e:
        await update.effective_chat.send_message(f"Ошибка: {e}")


def _alloc_lines() -> Tuple[str, Dict[str, float]]:
    total = float(STATE["total_bank"] or 0.0)
    w = STATE["weights"]
    if total <= 0:
        return "Сначала задайте общий банк: /settotal 2800 (в мастер-чате).", {}
    res: Dict[str, float] = {}
    for k, pct in w.items():
        pair = KEY2PAIR[k]
        amt = total * (pct / 100.0)
        res[pair] = amt
    # Формируем текст с готовыми командами для копипаста
    lines = ["Распределение:"]
    for pair in SYMBOLS:
        amt = res.get(pair, 0.0)
        lines.append(f"• {pair} → <b>{_fmt_money(amt)} USDT</b>  → команда в чат <b>{pair}</b>: <code>/setbank {_fmt_money(amt)}</code>")
    return "\n".join(lines), res


async def cmd_alloc(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    txt, res = _alloc_lines()
    await update.effective_chat.send_message(txt, parse_mode=ParseMode.HTML)


async def cmd_flags(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    flags = read_flags_from_sheet_safe()
    # Покажем кратко
    rows = []
    for sym in SYMBOLS:
        f = flags.get(sym, {})
        rows.append(f"{sym}: risk={f.get('risk','?')}, bias={f.get('bias','?')}, dca_scale={f.get('dca_scale','?')}, reserve_off={f.get('reserve_off','?')}")
    out = "\n".join(rows) if rows else "{}"
    await update.effective_chat.send_message(f"Флаги:\n{out}")


async def cmd_digest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        flags = read_flags_from_sheet_safe()
        text = summarize_digest(flags, None).strip()
        if not text:
            text = "Новостей мало. Флаги без изменений; режим обычный для всех пар."
        await update.effective_chat.send_message(text)
    except Exception as e:
        await update.effective_chat.send_message(f"LLM ошибка: {e}")


# =========================
# ---------- MAIN ---------
# =========================

async def _announce_startup(app: Application):
    master_id = _env_master_chat_id()
    # Диагностика
    sh, note = try_init_sheets()
    STATE["sheet_ready"] = sh is not None
    STATE["sheet_note"] = note

    llm_ok = bool(os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("LLM_MINI"))

    banner = (
        f"Фунд-бот запущен {'✅' if True else '❌'}\n"
        f"LLM: {'✅' if llm_ok else '❌'}\n"
        f"Sheets: {'✅' if STATE['sheet_ready'] else '❌'} ({STATE['sheet_note']})."
    )
    if master_id:
        try:
            await app.bot.send_message(chat_id=master_id, text=banner)
        except Exception as e:
            log.warning("Cannot send startup banner to master chat: %s", e)


def _bot_commands() -> list[BotCommand]:
    return [
        BotCommand("start", "Поприветствовать"),
        BotCommand("help", "Что умею"),
        BotCommand("ping", "Проверка связи"),
        BotCommand("diag", "Диагностика доступов (LLM/Sheets)"),
        BotCommand("weights", "Показать целевые веса"),
        BotCommand("setweights", "Задать веса (только мастер-чат)"),
        BotCommand("settotal", "Задать общий банк (только мастер-чат)"),
        BotCommand("alloc", "Рассчитать распределение и /setbank"),
        BotCommand("flags", "Показать флаги из FA_Signals"),
        BotCommand("digest", "Короткий дайджест по 4 парам"),
    ]


def build_app() -> Application:
    token = _env_bot_token()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN/TELEGRAM_TOKEN не задан")

    builder = Application.builder().token(token)
    if AIORateLimiter is not None:
        builder = builder.rate_limiter(AIORateLimiter())
    app = builder.build()

    # Команды
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("diag", cmd_diag))
    app.add_handler(CommandHandler("weights", cmd_weights))
    app.add_handler(CommandHandler("setweights", cmd_setweights))
    app.add_handler(CommandHandler("settotal", cmd_settotal))
    app.add_handler(CommandHandler("alloc", cmd_alloc))
    app.add_handler(CommandHandler("flags", cmd_flags))
    app.add_handler(CommandHandler("digest", cmd_digest))

    async def on_startup(app: Application):
        # Подтянем дефолтные веса из ENV (если заданы)
        STATE["weights"] = _parse_default_weights_env()
        try:
            await app.bot.set_my_commands(_bot_commands())
        except Exception as e:
            log.warning("set_my_commands failed: %s", e)
        await _announce_startup(app)

    app.post_init = on_startup  # type: ignore
    return app


def main():
    # Инициализация STATE из ENV до старта
    STATE["weights"] = _parse_default_weights_env()
    try:
        STATE["total_bank"] = float(os.environ.get("TOTAL_BANK_USDT", "0") or 0)
    except Exception:
        pass

    app = build_app()
    log.info("Fund bot is running…")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
