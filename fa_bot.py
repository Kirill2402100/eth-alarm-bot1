# fa_bot.py
from __future__ import annotations

import os
import json
import logging
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List

import pandas as pd

# --- Telegram ---
from telegram import Update, BotCommand
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes, MessageHandler, filters
)

# rate limiter: делаем безопасный импорт (без extras бот всё равно работает)
try:
    from telegram.ext._rate_limiter import AIORateLimiter  # type: ignore
except Exception:  # pragma: no cover
    AIORateLimiter = None  # fallback

# --- Google Sheets (опционально) ---
try:
    import gspread
except Exception:  # если не установлено, просто не используем
    gspread = None  # type: ignore

# --- LLM клиент ---
from llm_client import LLMClient, LLMError

log = logging.getLogger("fund_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ========= ENV & CONSTANTS =========
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
LLM_MAJOR = os.getenv("LLM_MAJOR", "gpt-5")  # опционально
LLM_DAILY_BUDGET = int(os.getenv("LLM_TOKEN_BUDGET_PER_DAY", "30000") or "30000")

# DEFAULT_WEIGHTS может быть в двух видах:
#   JSON: {"JPY":40,"AUD":25,"EUR":20,"GBP":15}
#   Строка: "jpy=40 aud=25 eur=20 gbp=15"
DEF_WEIGHTS_ENV = os.getenv("DEFAULT_WEIGHTS", "")

PAIR_ORDER = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]
PAIR_TO_KEY = {"USDJPY": "JPY", "AUDUSD": "AUD", "EURUSD": "EUR", "GBPUSD": "GBP"}
KEY_TO_PAIR = {v: k for k, v in PAIR_TO_KEY.items()}

CONFIG_SHEET_NAME = "FA_Config"   # хранение total/weights
# ========= Helpers =========

def _parse_weights_any(s: str) -> Dict[str, float]:
    """Парсит веса из JSON или из 'jpy=40 aud=25...' (без знаков %)."""
    if not s:
        return {}
    s = s.strip()
    out: Dict[str, float] = {}
    # сначала пробуем JSON
    if s.startswith("{"):
        try:
            data = json.loads(s)
            for k, v in data.items():
                kk = str(k).upper()
                out[kk] = float(v)
            return out
        except Exception:
            pass
    # затем 'jpy=40 aud=25 ...'
    parts = s.replace(",", " ").split()
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            k = k.strip().upper()
            try:
                out[k] = float(v.strip().replace("%", ""))
            except Exception:
                continue
    return out

def _norm_weights(w: Dict[str, float]) -> Dict[str, float]:
    """Нормировка и фильтр по ключам JPY/AUD/EUR/GBP."""
    filtered = {k.upper(): float(v) for k, v in w.items() if k.upper() in ("JPY", "AUD", "EUR", "GBP")}
    s = sum(max(0.0, v) for v in filtered.values())
    if s <= 0:
        return {}
    return {k: (max(0.0, v) / s) * 100.0 for k, v in filtered.items()}

def _fmt_weights_line(w: Dict[str, float]) -> str:
    order = ["JPY", "AUD", "EUR", "GBP"]
    parts = [f"{k} {int(round(w.get(k, 0.0)))}" for k in order]
    return " / ".join(parts)

def _compute_alloc(total_usdt: float, w: Dict[str, float]) -> Dict[str, float]:
    """Возвращает суммы по парам (в USD) согласно весам (в %, 0..100)."""
    res = {}
    for key, pct in w.items():
        pair = KEY_TO_PAIR.get(key, key)
        res[pair] = round(total_usdt * (pct / 100.0), 2)
    return res

def _parse_setweights_args(text: str) -> Dict[str, float]:
    # "/setweights jpy=40 aud=25 eur=20 gbp=15"
    return _norm_weights(_parse_weights_any(text))

def _read_default_weights_from_env() -> Dict[str, float]:
    w = _parse_weights_any(DEF_WEIGHTS_ENV)
    w = _norm_weights(w)
    if not w:  # дефолт
        w = {"JPY": 40.0, "AUD": 25.0, "EUR": 20.0, "GBP": 15.0}
    return w

# ========= Persistent store (in-memory + optional Sheets) =========

@dataclass
class MasterConfig:
    total_bank: float = 0.0
    weights_pct: Dict[str, float] = None  # by KEY: {"JPY":..., ...}

    def to_json(self) -> str:
        return json.dumps({"total_bank": self.total_bank, "weights_pct": self.weights_pct or {}}, ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "MasterConfig":
        try:
            data = json.loads(s)
            return MasterConfig(
                total_bank=float(data.get("total_bank") or 0.0),
                weights_pct={k.upper(): float(v) for k, v in (data.get("weights_pct") or {}).items()},
            )
        except Exception:
            return MasterConfig()

class ConfigStore:
    def __init__(self):
        self._cfg = MasterConfig(total_bank=0.0, weights_pct=_read_default_weights_from_env())
        self._gc = None
        self._sheet = None
        if GOOGLE_CREDENTIALS and SHEET_ID and gspread is not None:
            try:
                self._gc = gspread.service_account_from_dict(json.loads(GOOGLE_CREDENTIALS))
                self._sheet = self._gc.open_by_key(SHEET_ID)
                self._ensure_sheet()
                self._read_from_sheet()
                log.info("Sheets config connected")
            except Exception as e:
                log.warning(f"Sheets init failed: {e}")

    def _ensure_sheet(self):
        try:
            self._ws = self._sheet.worksheet(CONFIG_SHEET_NAME)
        except Exception:
            self._ws = self._sheet.add_worksheet(title=CONFIG_SHEET_NAME, rows=50, cols=10)
            self._ws.append_row(["key", "value"])  # simple KV
        return self._ws

    def _read_from_sheet(self):
        try:
            rows = self._ws.get_all_records()
            kv = {str(r.get("key")).strip(): str(r.get("value")).strip() for r in rows if r.get("key")}
            total = float(kv.get("total_bank") or 0.0)
            weights = _parse_weights_any(kv.get("weights_pct") or "")
            weights = _norm_weights(weights) or _read_default_weights_from_env()
            self._cfg.total_bank = total
            self._cfg.weights_pct = weights
        except Exception as e:
            log.warning(f"Read FA_Config failed: {e}")

    def _write_to_sheet(self):
        try:
            if not self._ws:
                return
            self._ws.clear()
            self._ws.append_row(["key", "value"])
            self._ws.append_row(["total_bank", str(self._cfg.total_bank)])
            # храним веса как JSON
            w_json = json.dumps(self._cfg.weights_pct, ensure_ascii=False)
            self._ws.append_row(["weights_pct", w_json])
        except Exception as e:
            log.warning(f"Write FA_Config failed: {e}")

    # public API
    def get(self) -> MasterConfig:
        return self._cfg

    def set_total(self, total: float):
        self._cfg.total_bank = max(0.0, float(total))
        self._write_to_sheet()

    def set_weights(self, w_pct: Dict[str, float]):
        self._cfg.weights_pct = _norm_weights(w_pct) or self._cfg.weights_pct
        self._write_to_sheet()

# ========= Bot core =========

store = ConfigStore()
llm = LLMClient(
    api_key=OPENAI_API_KEY,
    model_nano=LLM_NANO,
    model_mini=LLM_MINI,
    model_major=LLM_MAJOR,
    daily_token_budget=LLM_DAILY_BUDGET,
)

async def _post_init(app: Application):
    # меню команд
    await app.bot.set_my_commands([
        BotCommand("start", "Проверка/приветствие"),
        BotCommand("help", "Что умею"),
        BotCommand("ping", "Проверка связи"),
        BotCommand("settotal", "Задать общий банк, например: /settotal 2800"),
        BotCommand("setweights", "Задать веса, напр.: /setweights jpy=40 aud=25 eur=20 gbp=15"),
        BotCommand("weights", "Показать целевые веса"),
        BotCommand("alloc", "Рассчитать суммы и команды /setbank для чатов"),
        BotCommand("digest", "Сделать текстовый дайджест по 4 парам"),
    ])
    # приветствие в мастер-чат
    if MASTER_CHAT_ID:
        try:
            await app.bot.send_message(MASTER_CHAT_ID, "Фунд-бот запущен ✅", parse_mode=ParseMode.HTML)
        except Exception as e:
            log.warning(f"Cannot notify master chat: {e}")

# ----- Handlers -----

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id if update.effective_chat else None
    await update.effective_message.reply_html(
        f"Привет! Я фунд-бот.\n"
        f"Текущий чат id: <code>{cid}</code>\n\n"
        f"Команды: /help"
    )

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (
        "<b>Что я умею</b>\n"
        "• <code>/settotal 2800</code> — задать общий банк (только в мастер-чате).\n"
        "• <code>/setweights jpy=40 aud=25 eur=20 gbp=15</code> — выставить целевые веса.\n"
        "• <code>/weights</code> — показать целевые веса.\n"
        "• <code>/alloc</code> — расчёт сумм и готовые команды <code>/setbank</code> для торговых чатов.\n"
        "• <code>/digest</code> — короткий «человеческий» дайджест по USDJPY / AUDUSD / EURUSD / GBPUSD.\n"
        "• <code>/ping</code> — проверить связь.\n\n"
        "<i>Банк по парам задаётся вручную в торговых чатах; я сверяю распределение и даю советы/дайджест.</i>"
    )
    await update.effective_message.reply_html(text)

async def cmd_ping(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("pong")

def _is_master_chat(update: Update) -> bool:
    if MASTER_CHAT_ID is None:
        return True  # если не задан — не ограничиваем
    try:
        return update.effective_chat and update.effective_chat.id == MASTER_CHAT_ID
    except Exception:
        return False

async def cmd_settotal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_master_chat(update):
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
    line = _fmt_weights_line(w)
    await update.effective_message.reply_html(
        f"Общий банк: <b>{total:.2f} USDT</b>\n"
        f"Целевые веса: {line}\n\n"
        f"Теперь можете выполнить <code>/alloc</code> для расчёта сумм."
    )

async def cmd_setweights(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_master_chat(update):
        await update.effective_message.reply_text("Эта команда доступна только в мастер-чате.")
        return
    text = update.effective_message.text or ""
    w = _parse_setweights_args(text)
    if not w:
        await update.effective_message.reply_text(
            "Не распознал веса. Пример: /setweights jpy=40 aud=25 eur=20 gbp=15"
        )
        return
    store.set_weights(w)
    line = _fmt_weights_line(store.get().weights_pct)
    await update.effective_message.reply_html(f"Целевые веса обновлены: {line}\nКоманда: <code>/alloc</code> рассчитает суммы.")

async def cmd_weights(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cfg = store.get()
    line = _fmt_weights_line(cfg.weights_pct)
    await update.effective_message.reply_html(
        f"Целевые веса: {line}\nТекущий общий банк: <b>{cfg.total_bank:.2f} USDT</b>"
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
    # красивый вывод + готовые команды /setbank
    lines = []
    header = f"Целевые веса: {_fmt_weights_line(cfg.weights_pct)}\n"
    for pair in PAIR_ORDER:
        key = PAIR_TO_KEY.get(pair, pair)
        amt = alloc.get(pair, 0.0)
        lines.append(f"{pair} → <b>{amt:.2f} USDT</b>  → команда в чат {pair}: <code>/setbank {amt:.2f}</code>")
    text = header + "\n".join(lines) + "\n\nСтатус рекомендации: <b>APPLIED</b> (после выполнения команд в торговых чатах)."
    await update.effective_message.reply_html(text)

async def cmd_digest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    pairs = PAIR_ORDER
    try:
        digest_map = await llm.make_digest_ru(pairs=pairs)
    except LLMError as e:
        await update.effective_message.reply_text(f"LLM ошибка: {e}")
        return
    # компактный пост
    b = []
    b.append("🧭 <b>Утренний фон</b>\n")
    for p in pairs:
        t = digest_map.get(p, "").strip()
        if not t:
            continue
        emoji = {
            "USDJPY": "🇺🇸🇯🇵",
            "AUDUSD": "🇦🇺🇺🇸",
            "EURUSD": "🇪🇺🇺🇸",
            "GBPUSD": "🇬🇧🇺🇸",
        }.get(p, "•")
        b.append(f"<b>{emoji} {p}</b>\n{t}\n")
    await update.effective_message.reply_html("\n".join(b).strip())

# fallback: чтобы не молчал на незнакомое
async def unknown(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Не понимаю. /help — список команд.")

# ========= Main =========

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
