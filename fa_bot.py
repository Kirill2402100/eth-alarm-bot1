# fa_bot.py
from __future__ import annotations

import os
import json
import logging
import asyncio
from dataclasses import dataclass
from typing import Dict, Optional, List

# Telegram
from telegram import Update, BotCommand
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes, MessageHandler, filters
)

# Rate limiter (опционально)
try:
    from telegram.ext._rate_limiter import AIORateLimiter  # type: ignore
except Exception:
    AIORateLimiter = None  # type: ignore

# Google Sheets (опционально)
try:
    import gspread
except Exception:
    gspread = None  # type: ignore

from llm_client import LLMClient, LLMError

log = logging.getLogger("fund_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ===== ENV =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("No Telegram token: set TELEGRAM_BOT_TOKEN (or TELEGRAM_TOKEN)")

MASTER_CHAT_ID: Optional[int] = None
_mc = os.getenv("MASTER_CHAT_ID")
if _mc:
    try:
        MASTER_CHAT_ID = int(_mc)
    except Exception:
        pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
SHEET_ID = os.getenv("SHEET_ID") or ""
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS") or ""

LLM_NANO = os.getenv("LLM_NANO", "gpt-5-nano")
LLM_MINI = os.getenv("LLM_MINI", "gpt-5-mini")
LLM_MAJOR = os.getenv("LLM_MAJOR", "gpt-5")
LLM_DAILY_BUDGET = int(os.getenv("LLM_TOKEN_BUDGET_PER_DAY", "30000") or "30000")

DEFAULT_WEIGHTS_ENV = os.getenv("DEFAULT_WEIGHTS", "")

PAIR_ORDER = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]
PAIR_TO_KEY = {"USDJPY": "JPY", "AUDUSD": "AUD", "EURUSD": "EUR", "GBPUSD": "GBP"}
KEY_TO_PAIR = {v: k for k, v in PAIR_TO_KEY.items()}
CONFIG_SHEET_NAME = "FA_Config"

# ===== helpers =====
def _parse_weights_any(s: str) -> Dict[str, float]:
    if not s:
        return {}
    s = s.strip()
    out: Dict[str, float] = {}
    if s.startswith("{"):
        try:
            data = json.loads(s)
            for k, v in data.items():
                out[str(k).upper()] = float(v)
            return out
        except Exception:
            pass
    parts = s.replace(",", " ").split()
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            try:
                out[k.strip().upper()] = float(v.strip().replace("%", ""))
            except Exception:
                continue
    return out

def _norm_weights(w: Dict[str, float]) -> Dict[str, float]:
    filtered = {k.upper(): float(v) for k, v in w.items() if k.upper() in ("JPY", "AUD", "EUR", "GBP")}
    s = sum(max(0.0, v) for v in filtered.values())
    if s <= 0:
        return {}
    return {k: (max(0.0, v) / s) * 100.0 for k, v in filtered.items()}

def _fmt_weights_line(w: Dict[str, float]) -> str:
    order = ["JPY", "AUD", "EUR", "GBP"]
    return " / ".join([f"{k} {int(round(w.get(k, 0.0)))}" for k in order])

def _compute_alloc(total_usdt: float, w: Dict[str, float]) -> Dict[str, float]:
    return {KEY_TO_PAIR[k]: round(total_usdt * (pct / 100.0), 2) for k, pct in w.items()}

def _parse_setweights_args(text: str) -> Dict[str, float]:
    return _norm_weights(_parse_weights_any(text))

def _read_default_weights_from_env() -> Dict[str, float]:
    w = _norm_weights(_parse_weights_any(DEFAULT_WEIGHTS_ENV))
    return w or {"JPY": 40.0, "AUD": 25.0, "EUR": 20.0, "GBP": 15.0}

# ===== storage =====
@dataclass
class MasterConfig:
    total_bank: float = 0.0
    weights_pct: Dict[str, float] = None

class ConfigStore:
    def __init__(self):
        self._cfg = MasterConfig(total_bank=0.0, weights_pct=_read_default_weights_from_env())
        self._ws = None
        self._service_email = None
        if GOOGLE_CREDENTIALS and SHEET_ID and gspread is not None:
            try:
                creds = json.loads(GOOGLE_CREDENTIALS)
                self._service_email = creds.get("client_email")
                gc = gspread.service_account_from_dict(creds)
                sh = gc.open_by_key(SHEET_ID)
                try:
                    self._ws = sh.worksheet(CONFIG_SHEET_NAME)
                except Exception:
                    self._ws = sh.add_worksheet(title=CONFIG_SHEET_NAME, rows=50, cols=10)
                    self._ws.append_row(["key", "value"])
                self._read_from_sheet()
                log.info("Sheets: connected")
            except Exception as e:
                log.warning(f"Sheets init failed: {e}")

    def sheet_status_line(self) -> str:
        if not GOOGLE_CREDENTIALS or not SHEET_ID:
            return "Sheets: не настроено (нет SHEET_ID/GOOGLE_CREDENTIALS)."
        if gspread is None:
            return "Sheets: модуль gspread не установлен."
        if self._ws is None:
            if self._service_email:
                return f"Sheets: нет доступа. Поделитесь таблицей с сервис-аккаунтом <code>{self._service_email}</code>."
            return "Sheets: нет доступа (проверьте права сервис-аккаунта)."
        return "Sheets: ок ✅ (лист FA_Config готов)."

    def _read_from_sheet(self):
        if not self._ws:
            return
        try:
            rows = self._ws.get_all_records()
            kv = {str(r.get("key")).strip(): str(r.get("value")).strip() for r in rows if r.get("key")}
            total = float(kv.get("total_bank") or 0.0)
            weights = _norm_weights(_parse_weights_any(kv.get("weights_pct") or "")) or _read_default_weights_from_env()
            self._cfg.total_bank = total
            self._cfg.weights_pct = weights
        except Exception as e:
            log.warning(f"Read FA_Config failed: {e}")

    def _write_to_sheet(self):
        if not self._ws:
            return
        try:
            self._ws.clear()
            self._ws.append_row(["key", "value"])
            self._ws.append_row(["total_bank", str(self._cfg.total_bank)])
            self._ws.append_row(["weights_pct", json.dumps(self._cfg.weights_pct, ensure_ascii=False)])
        except Exception as e:
            log.warning(f"Write FA_Config failed: {e}")

    def get(self) -> MasterConfig:
        return self._cfg

    def set_total(self, total: float):
        self._cfg.total_bank = max(0.0, float(total))
        self._write_to_sheet()

    def set_weights(self, w_pct: Dict[str, float]):
        self._cfg.weights_pct = _norm_weights(w_pct) or self._cfg.weights_pct
        self._write_to_sheet()


store = ConfigStore()
llm = LLMClient(
    api_key=OPENAI_API_KEY,
    model_nano=LLM_NANO,
    model_mini=LLM_MINI,
    model_major=LLM_MAJOR,
    daily_token_budget=LLM_DAILY_BUDGET,
)

# ===== bot init =====
async def _post_init(app: Application):
    await app.bot.set_my_commands([
        BotCommand("start", "Проверка/приветствие"),
        BotCommand("help", "Что умею"),
        BotCommand("ping", "Проверка связи"),
        BotCommand("settotal", "Задать общий банк, например: /settotal 2800"),
        BotCommand("setweights", "Задать веса: /setweights jpy=40 aud=25 eur=20 gbp=15"),
        BotCommand("weights", "Показать целевые веса"),
        BotCommand("alloc", "Рассчитать суммы и /setbank для чатов"),
        BotCommand("digest", "Дайджест по 4 парам"),
    ])
    if MASTER_CHAT_ID:
        try:
            status = store.sheet_status_line()
            await app.bot.send_message(
                MASTER_CHAT_ID,
                f"Фунд-бот запущен ✅\n{status}",
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            log.warning(f"Cannot notify master chat: {e}")

# ===== commands =====
def _is_master(update: Update) -> bool:
    if MASTER_CHAT_ID is None:
        return True
    try:
        return update.effective_chat and update.effective_chat.id == MASTER_CHAT_ID
    except Exception:
        return False

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id if update.effective_chat else None
    await update.effective_message.reply_html(
        f"Привет! Я фунд-бот.\nТекущий чат id: <code>{cid}</code>\n\nКоманды: /help"
    )

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (
        "<b>Что я умею</b>\n"
        "• <code>/settotal 2800</code> — задать общий банк (только в мастер-чате).\n"
        "• <code>/setweights jpy=40 aud=25 eur=20 gbp=15</code> — выставить целевые веса.\n"
        "• <code>/weights</code> — показать целевые веса.\n"
        "• <code>/alloc</code> — расчёт сумм и готовые /setbank для торговых чатов.\n"
        "• <code>/digest</code> — короткий дайджест на русском (USDJPY / AUDUSD / EURUSD / GBPUSD).\n"
        "• <code>/ping</code> — проверить связь.\n\n"
        "<i>Банк по парам задаётся вручную в торговых чатах; я сверяю распределение и даю советы/дайджест.</i>"
    )
    await update.effective_message.reply_html(text)

async def cmd_ping(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("pong")

async def cmd_settotal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_master(update):
        await update.effective_message.reply_text("Эта команда доступна только в мастер-чате.")
        return
    args = ctx.args or []
    if not args:
        await update.effective_message.reply_text("Использование: /settotal 2800")
        return
    try:
        total = float(args[0].replace(",", "."))
    except Exception:
        await update.effective_message.reply_text("Неверное число. Пример: /settotal 2800")
        return
    store.set_total(total)
    w = store.get().weights_pct
    await update.effective_message.reply_html(
        f"Общий банк: <b>{total:.2f} USDT</b>\nЦелевые веса: {_fmt_weights_line(w)}\n\n"
        f"Выполните <code>/alloc</code> для расчёта сумм."
    )

async def cmd_setweights(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_master(update):
        await update.effective_message.reply_text("Эта команда доступна только в мастер-чате.")
        return
    w = _parse_setweights_args(update.effective_message.text or "")
    if not w:
        await update.effective_message.reply_text("Не распознал веса. Пример: /setweights jpy=40 aud=25 eur=20 gbp=15")
        return
    store.set_weights(w)
    await update.effective_message.reply_html(
        f"Целевые веса обновлены: {_fmt_weights_line(store.get().weights_pct)}\nКоманда: <code>/alloc</code> рассчитает суммы."
    )

async def cmd_weights(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cfg = store.get()
    await update.effective_message.reply_html(
        f"Целевые веса: {_fmt_weights_line(cfg.weights_pct)}\nТекущий общий банк: <b>{cfg.total_bank:.2f} USDT</b>"
    )

async def cmd_alloc(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cfg = store.get()
    if cfg.total_bank <= 0.0:
        await update.effective_message.reply_text("Сначала задайте общий банк: /settotal 2800")
        return
    if not cfg.weights_pct:
        await update.effective_message.reply_text("Сначала задайте веса: /setweights jpy=40 aud=25 eur=20 gbp=15")
        return
    alloc = _compute_alloc(cfg.total_bank, cfg.weights_pct)
    header = f"Целевые веса: {_fmt_weights_line(cfg.weights_pct)}\n"
    lines: List[str] = []
    for pair in PAIR_ORDER:
        amt = alloc.get(pair, 0.0)
        lines.append(f"{pair} → <b>{amt:.2f} USDT</b>  → команда в чат {pair}: <code>/setbank {amt:.2f}</code>")
    text = header + "\n".join(lines) + "\n\nСтатус рекомендации: <b>APPLIED</b>."
    await update.effective_message.reply_html(text)

async def cmd_digest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    pairs = PAIR_ORDER
    try:
        digest_map = await llm.make_digest_ru(pairs)
    except LLMError as e:
        await update.effective_message.reply_text(f"LLM ошибка: {e}")
        return
    blocks = ["🧭 <b>Утренний фон</b>\n"]
    flag = {"USDJPY": "🇺🇸🇯🇵", "AUDUSD": "🇦🇺🇺🇸", "EURUSD": "🇪🇺🇺🇸", "GBPUSD": "🇬🇧🇺🇸"}
    for p in pairs:
        t = (digest_map.get(p) or "").strip()
        if not t:
            t = "✅ Нейтрально. Существенных поводов не отмечено; работаем по базовому плану."
        blocks.append(f"<b>{flag.get(p,'•')} {p}</b>\n{t}\n")
    await update.effective_message.reply_html("\n".join(blocks).strip())

async def unknown(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Не понимаю. /help — список команд.")

def main():
    builder = Application.builder().token(TELEGRAM_TOKEN)
    if AIORateLimiter:
        builder = builder.rate_limiter(AIORateLimiter())
    app = builder.post_init(_post_init).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("settotal", cmd_settotal))
    app.add_handler(CommandHandler("setweights", cmd_setweights))
    app.add_handler(CommandHandler("weights", cmd_weights))
    app.add_handler(CommandHandler("alloc", cmd_alloc))
    app.add_handler(CommandHandler("digest", cmd_digest))
    app.add_handler(MessageHandler(filters.COMMAND, unknown))

    log.info("Fund bot is running…")
    app.run_polling(allowed_updates=["message", "chat_member", "my_chat_member"])

if __name__ == "__main__":
    main()
