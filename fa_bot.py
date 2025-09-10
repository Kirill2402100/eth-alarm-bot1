# fa_bot.py
import os
import json
import base64
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional

import asyncio

from telegram import Update, BotCommand
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# --- Таймзона Белграда ---
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None

LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Europe/Belgrade")) if ZoneInfo else None
MORNING_HOUR = int(os.getenv("MORNING_HOUR", "9"))
MORNING_MINUTE = int(os.getenv("MORNING_MINUTE", "30"))

# --- Лимитер Telegram (может быть не установлен) ---
try:
    from telegram.ext import AIORateLimiter
    _RATE_LIMITER_AVAILABLE = True
except Exception:  # pragma: no cover
    AIORateLimiter = None
    _RATE_LIMITER_AVAILABLE = False

# --- Google Sheets ---
try:
    import gspread
    from google.oauth2 import service_account
    _GSHEETS_AVAILABLE = True
except Exception:
    gspread = None
    service_account = None
    _GSHEETS_AVAILABLE = False

# --- LLM-клиент ---
try:
    from llm_client import generate_digest, llm_ping
except Exception:
    async def generate_digest(*args, **kwargs) -> str:
        return "⚠️ LLM сейчас недоступен (нет llm_client.py)."

    async def llm_ping() -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

# -------------------- ЛОГИ --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("fund_bot")

# -------------------- ENV --------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or os.getenv("TELEGRAM_TOKEN", "").strip()
MASTER_CHAT_ID = int(os.getenv("MASTER_CHAT_ID", "0") or "0")

SHEET_ID = os.getenv("SHEET_ID", "").strip()                       # только ID (между /d/ и /edit)
SHEET_WS = os.getenv("SHEET_WS", "FUND_BOT").strip() or "FUND_BOT"  # имя листа/вкладки по умолчанию

_DEFAULT_WEIGHTS_RAW = os.getenv("DEFAULT_WEIGHTS", "").strip()
if _DEFAULT_WEIGHTS_RAW:
    try:
        DEFAULT_WEIGHTS: Dict[str, int] = json.loads(_DEFAULT_WEIGHTS_RAW)
    except Exception:
        DEFAULT_WEIGHTS = {"JPY": 40, "AUD": 25, "EUR": 20, "GBP": 15}
else:
    DEFAULT_WEIGHTS = {"JPY": 40, "AUD": 25, "EUR": 20, "GBP": 15}

LLM_MINI = os.getenv("LLM_MINI", "gpt-5-mini").strip()
LLM_NANO = os.getenv("LLM_NANO", "gpt-5-nano").strip()
LLM_MAJOR = os.getenv("LLM_MAJOR", "gpt-5").strip()
LLM_TOKEN_BUDGET_PER_DAY = int(os.getenv("LLM_TOKEN_BUDGET_PER_DAY", "30000") or "30000")

SYMBOLS = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]

# --- Названия листов с логами по парам (можно переопределить через ENV) ---
BMR_SHEETS = {
    "USDJPY": os.getenv("BMR_SHEET_USDJPY", "BMR_DCA_USDJPY"),
    "AUDUSD": os.getenv("BMR_SHEET_AUDUSD", "BMR_DCA_AUDUSD"),
    "EURUSD": os.getenv("BMR_SHEET_EURUSD", "BMR_DCA_EURUSD"),
    "GBPUSD": os.getenv("BMR_SHEET_GBPUSD", "BMR_DCA_GBPUSD"),
}

# -------------------- GOOGLE CREDS LOADER --------------------
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def _env(name: str) -> str:
    v = os.getenv(name, "")
    return v if isinstance(v, str) else ""


def _decode_b64_maybe_padded(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    padded = s + "=" * ((4 - len(s) % 4) % 4)
    return base64.b64decode(padded).decode("utf-8", "strict")


def load_google_service_info() -> Tuple[Optional[dict], str, Optional[str]]:
    b64 = _env("GOOGLE_CREDENTIALS_JSON_B64")
    if b64:
        try:
            decoded = _decode_b64_maybe_padded(b64)
            info = json.loads(decoded)
            client_email = info.get("client_email")
            return info, "env:GOOGLE_CREDENTIALS_JSON_B64", client_email
        except Exception as e:
            return None, f"b64 present but decode/json error: {e}", None

    for name in ("GOOGLE_CREDENTIALS_JSON", "GOOGLE_CREDENTIALS"):
        raw = _env(name)
        if raw:
            try:
                info = json.loads(raw)
                client_email = info.get("client_email")
                return info, f"env:{name}", client_email
            except Exception as e:
                return None, f"{name} present but invalid JSON: {e}", None

    return None, "not-found", None


def build_sheets_client(sheet_id: str):
    if not _GSHEETS_AVAILABLE:
        return None, "gsheets libs not installed"
    if not sheet_id:
        return None, "sheet_id empty"

    info, src, client_email = load_google_service_info()
    if not info:
        return None, src

    try:
        creds = service_account.Credentials.from_service_account_info(info, scopes=SHEETS_SCOPES)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_id)
        sa = client_email or getattr(getattr(gc, "auth", None), "service_account_email", "ok")
        return sh, f"{src} / sa={sa}"
    except Exception as e:
        return None, f"auth/open error: {e}"


# ---------- Sheets helpers ----------
SHEET_HEADERS = ["ts", "chat_id", "action", "total", "weights_json", "note"]


def ensure_worksheet(sh, title: str):
    """Вернёт существующий лист или создаст новый с заголовками."""
    try:
        for ws in sh.worksheets():
            if ws.title == title:
                return ws, False
        ws = sh.add_worksheet(title=title, rows=100, cols=max(10, len(SHEET_HEADERS)))
        ws.update("A1", [SHEET_HEADERS])
        return ws, True
    except Exception as e:
        raise RuntimeError(f"ensure_worksheet error: {e}")


def append_row(sh, title: str, row: list):
    ws, _ = ensure_worksheet(sh, title)
    ws.append_row(row, value_input_option="RAW")


# -------------------- УТИЛИТЫ --------------------
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


def assert_master_chat(update: Update) -> bool:
    if MASTER_CHAT_ID and update.effective_chat:
        return update.effective_chat.id == MASTER_CHAT_ID
    return True


# -------------------- КОМАНДЫ --------------------
HELP_TEXT = (
    "Что я умею\n"
    "/settotal 2800 — задать общий банк (только в мастер-чате).\n"
    "/setweights jpy=40 aud=25 eur=20 gbp=15 — выставить целевые веса.\n"
    "/weights — показать целевые веса.\n"
    "/alloc — расчёт сумм и готовые команды /setbank для торговых чатов.\n"
    "/digest — утренний «инвесторский» дайджест (по людям).\n"
    "/digest pro — краткий «трейдерский» дайджест (по цифрам, LLM).\n"
    "/init_sheet — создать/проверить лист в Google Sheets.\n"
    "/sheet_test — записать тестовую строку в лист.\n"
    "/diag — диагностика LLM и Google Sheets.\n"
    "/ping — проверить связь."
)

STATE = {
    "total": 0.0,
    "weights": DEFAULT_WEIGHTS.copy(),
}


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
    await update.message.reply_text(
        f"Целевые веса обновлены: {human_readable_weights(new_w)}"
    )


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

    # опционально лог в таблицу
    sh, _src = build_sheets_client(SHEET_ID)
    if sh:
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


async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Если пользователь написал "pro" → отдаём прежний краткий/профи вариант через LLM
    args = (update.message.text or "").split()
    pro = len(args) > 1 and args[1].lower() == "pro"

    if pro:
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            txt = await generate_digest(
                symbols=SYMBOLS,
                model=LLM_MINI,
                token_budget=LLM_TOKEN_BUDGET_PER_DAY,
            )
            await update.message.reply_text(txt)
        except Exception as e:
            await update.message.reply_text(f"LLM ошибка: {e}")
        return

    # Инвесторский режим (локально, без LLM)
    sh, _src = build_sheets_client(SHEET_ID)
    if not sh:
        await update.message.reply_text("Sheets недоступен: не могу собрать инвесторский дайджест.")
        return

    try:
        msg = build_investor_digest(sh)
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"Ошибка сборки дайджеста: {e}")


def sheets_diag_text() -> str:
    sid_state = "set" if SHEET_ID else "empty"
    b64_len = len(_env("GOOGLE_CREDENTIALS_JSON_B64"))
    raw_json_len = len(_env("GOOGLE_CREDENTIALS_JSON")) or len(_env("GOOGLE_CREDENTIALS"))

    if not _GSHEETS_AVAILABLE:
        return f"Sheets: ❌ (libs not installed, SID={sid_state}, b64_len={b64_len}, raw_len={raw_json_len})"

    sh, src = build_sheets_client(SHEET_ID)
    if sh is None:
        return f"Sheets: ❌ (SID={sid_state}, source={src}, b64_len={b64_len}, raw_len={raw_json_len})"
    else:
        try:
            ws, created = ensure_worksheet(sh, SHEET_WS)
            mark = "created" if created else "exists"
            return f"Sheets: ✅ ok (SID={sid_state}, {src}, ws={ws.title}:{mark})"
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


# ---- новые команды для листа ----
async def cmd_init_sheet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not SHEET_ID:
        await update.message.reply_text("SHEET_ID не задан.")
        return
    sh, src = build_sheets_client(SHEET_ID)
    if not sh:
        await update.message.reply_text(f"Sheets: ❌ {src}")
        return
    try:
        ws, created = ensure_worksheet(sh, SHEET_WS)
        await update.message.reply_text(
            f"Sheets: ✅ ws='{ws.title}' {'создан' if created else 'уже есть'} ({src})"
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


# ---------- Digest helpers (инвесторский режим) ----------

def _to_float(x, default=0.0) -> float:
    try:
        s = str(x).strip().replace(",", ".")
        return float(s)
    except Exception:
        return default


def _last_record(ws):
    """Вернуть последнюю строку как dict через get_all_records()."""
    try:
        rows = ws.get_all_records()
        return rows[-1] if rows else None
    except Exception:
        return None


def get_last_bmr_row(sh, symbol: str) -> Optional[dict]:
    sheet_name = BMR_SHEETS.get(symbol)
    if not sheet_name:
        return None
    try:
        ws = sh.worksheet(sheet_name)
        return _last_record(ws)
    except Exception:
        return None


def map_fa_level(risk: str) -> str:
    r = (risk or "").strip().lower()
    if r.startswith("red"):
        return "HIGH"
    if r.startswith("yellow") or r.startswith("amber"):
        return "CAUTION"
    return "OK"


def map_fa_bias(bias: str) -> str:
    b = (bias or "").strip().lower()
    if b.startswith("long"):
        return "LONG"
    if b.startswith("short"):
        return "SHORT"
    return "BOTH"


def policy_from_level(level: str) -> Dict[str, object]:
    L = (level or "OK").upper()
    if L == "HIGH":
        return {"quiet_min": (45, 60), "reserve_off": True,  "dca_scale": 0.50, "icon": "🚧", "label": "высокий риск"}
    if L == "CAUTION":
        return {"quiet_min": (30, 45), "reserve_off": False, "dca_scale": 0.75, "icon": "⚠️", "label": "умеренный риск"}
    return {"quiet_min": (0, 0), "reserve_off": False, "dca_scale": 1.00, "icon": "✅", "label": "фон спокойный"}


def supertrend_dir(val: str) -> str:
    v = (val or "").strip().lower()
    return "up" if "up" in v else "down" if "down" in v else "flat"


def market_phrases(adx: float, st_dir: str, vol_z: float, atr1h: float) -> str:
    # Инвесторские формулировки
    trend_txt = "умеренное" if adx < 20 else "заметное" if adx < 25 else "выраженное"
    dir_txt = "вверх" if st_dir == "up" else "вниз" if st_dir == "down" else "вбок"
    vola_txt = "ниже нормы" if atr1h < 0.8 else "около нормы" if atr1h < 1.2 else "выше нормы"
    noise_txt = "низкий" if vol_z < 0.5 else "умеренный" if vol_z < 1.5 else "повышенный"
    return f"{trend_txt} движение {dir_txt}; колебания {vola_txt}; рыночный шум {noise_txt}"


def probability_against(side: str, fa_bias: str, adx: float, st_dir: str,
                        vol_z: float, atr1h: float, rsi: float, red_event_soon: bool) -> int:
    # База
    P = 55
    against_dir = "down" if (side or "").upper() == "LONG" else "up"

    # FA bias
    if (fa_bias == "LONG" and against_dir == "up") or (fa_bias == "SHORT" and against_dir == "down"):
        P += 10
    elif fa_bias in ("LONG", "SHORT"):
        P -= 15

    # ADX
    if adx >= 25:
        P += 6
    elif adx >= 20:
        P += 3
    else:
        P -= 5

    # Supertrend
    if st_dir == against_dir:
        P += 5
    else:
        P -= 5

    # Vol_z
    if vol_z < 0.3:
        P -= 3
    elif vol_z < 1.5:
        P += 3
    elif vol_z > 2.5:
        P -= 7

    # ATR_1h (как z-скор)
    if atr1h > 1.2:
        P += 4
    elif atr1h < 0.8:
        P -= 4
    else:
        P += 1

    # RSI экстремы против позиции
    side_up = (side or "").upper() == "SHORT"  # против short = вверх
    if side_up:
        if rsi > 65: P += 3
        if rsi < 35: P -= 4
    else:
        if rsi < 35: P += 3
        if rsi > 65: P -= 4

    if red_event_soon:
        P -= 7

    P = max(35, min(75, int(round(P))))
    return P


def action_text(P: int, quiet_now: bool, level: str) -> str:
    if level == "HIGH":
        # в HIGH не обещаем 2 шага
        if P >= 64:
            return "готовьтесь добирать 1 шаг (после снятия ограничений и вне тихого окна)"
    if P >= 64:
        return "готовьтесь добирать 2 шага" + (" (вне тихого окна)" if quiet_now else "")
    if P >= 58:
        return "готовьтесь добирать 1 шаг" + (" (вне тихого окна)" if quiet_now else "")
    if P >= 50:
        return "базовый план"
    return "дополнительные доборы не приоритетны"


def delta_marker(target: float, fact: float) -> str:
    if target <= 0:
        return "—"
    delta_pct = (fact - target) / target
    ap = abs(delta_pct)
    if ap <= 0.02:
        return "✅"
    if ap <= 0.05:
        return f"⚠️ небольшое отклонение ({delta_pct:+.1%})"
    return f"🚧 существенное отклонение ({delta_pct:+.1%})"


def build_investor_digest(sh) -> str:
    # Заголовок с локальным временем
    now_utc = datetime.utcnow()
    if LOCAL_TZ:
        now_local = now_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(LOCAL_TZ)
        header = f"🧭 Утренний фон — {now_local:%A, %d %B %Y, %H:%M} (Europe/Belgrade)"
    else:
        header = f"🧭 Утренний фон — {now_utc.strftime('%A, %d %B %Y, %H:%M')} (UTC)"

    blocks = [header]

    # TODO: календарь и событие_рядом (пока без внешнего API)
    red_event_soon = False  # заглушка
    quiet_now = False       # заглушка

    for sym in SYMBOLS:
        row = get_last_bmr_row(sh, sym) or {}
        side = (row.get("Side") or row.get("SIDE") or "").upper() or "LONG"
        avg = _to_float(row.get("Avg_Price"))
        next_dca = _to_float(row.get("Next_DCA_Price"))

        adx = _to_float(row.get("ADX_5m"))
        rsi = _to_float(row.get("RSI_5m"), 50.0)
        volz = _to_float(row.get("Vol_z"))
        atr1h = _to_float(row.get("ATR_1h"), 1.0)  # трактуем как z-скор ~1.0
        st = supertrend_dir(row.get("Supertrend"))

        fa_level = map_fa_level(row.get("FA_Risk"))
        fa_bias = map_fa_bias(row.get("FA_Bias"))

        policy = policy_from_level(fa_level)
        quiet_from, quiet_to = policy["quiet_min"]
        reserve = "OFF" if policy["reserve_off"] else "ON"
        dca_scale = policy["dca_scale"]
        icon = policy["icon"]
        label = policy["label"]

        P = probability_against(side, fa_bias, adx, st, volz, atr1h, rsi, red_event_soon)
        act = action_text(P, quiet_now, fa_level)

        target = _to_float(row.get("Bank_Target_USDT"))
        fact = _to_float(row.get("Bank_Fact_USDT"))
        marker = delta_marker(target, fact)

        pair_pretty = f"{sym[:3]}/{sym[3:]}"
        blocks.append(
f"""**{pair_pretty} — {icon} {label}, bias: {fa_bias}**
• **Фундаментально:** { 'нейтрально' if fa_level=='OK' else ('умеренные риски' if fa_level=='CAUTION' else 'высокие риски') }.
• **Рынок сейчас:** {market_phrases(adx, st, volz, atr1h)}.
• **Наша позиция:** **{side}**, средняя {avg:.5f}; следующий добор {next_dca:.5f}.
• **Что делаем сейчас:** {"тихое окно не требуется" if quiet_from==0 and quiet_to==0 else f"тихое окно [{-quiet_from:+d};+{quiet_to:d}] мин"}; reserve **{reserve}**; dca_scale **{dca_scale:.2f}**.
• **Вероятность против позиции:** ≈ **{P}%** → {act}.
• **Цель vs факт:** Target **{target:g}** / Fact **{fact:g}** — {marker}"""
        )

    # Блок ближайших событий можно добавить здесь после подключения календаря
    # blocks.append("\n📅 Ближайшие события: —")

    return "\n\n".join(blocks)


# -------------------- СТАРТ --------------------
async def _set_bot_commands(app: Application):
    cmds = [
        BotCommand("start", "Запуск бота"),
        BotCommand("help", "Список команд"),
        BotCommand("ping", "Проверка связи"),
        BotCommand("settotal", "Задать общий банк (мастер-чат)"),
        BotCommand("setweights", "Задать целевые веса (мастер-чат)"),
        BotCommand("weights", "Показать целевые веса"),
        BotCommand("alloc", "Рассчитать распределение банка"),
        BotCommand("digest", "Утренний дайджест (investor) / pro (trader)"),
        BotCommand("init_sheet", "Создать/проверить лист в Google Sheets"),
        BotCommand("sheet_test", "Тестовая запись в лист"),
        BotCommand("diag", "Диагностика LLM и Sheets"),
    ]
    try:
        await app.bot.set_my_commands(cmds)
    except Exception as e:
        log.warning("set_my_commands failed: %s", e)


async def morning_digest_scheduler(app: Application):
    """Ежедневный автопост инвесторского дайджеста в MASTER_CHAT_ID в 09:30 Europe/Belgrade."""
    import asyncio as _asyncio
    from datetime import datetime as _dt, timedelta as _td, time as _time

    while True:
        now = _dt.now(LOCAL_TZ) if LOCAL_TZ else _dt.utcnow()
        target = _dt.combine(now.date(), _time(MORNING_HOUR, MORNING_MINUTE, tzinfo=LOCAL_TZ)) if LOCAL_TZ else _dt.utcnow()
        if not LOCAL_TZ:
            # без zoneinfo просто шлём через ~24 часа от старта
            target = now.replace(hour=MORNING_HOUR, minute=MORNING_MINUTE, second=0, microsecond=0)
            if now >= target:
                target = target + _td(days=1)
        else:
            if now >= target:
                target = target + _td(days=1)

        wait_s = (target - now).total_seconds()
        await _asyncio.sleep(max(1.0, wait_s))

        try:
            sh, _src = build_sheets_client(SHEET_ID)
            if sh:
                msg = build_investor_digest(sh)
                await app.bot.send_message(chat_id=MASTER_CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)
            else:
                await app.bot.send_message(chat_id=MASTER_CHAT_ID, text="Sheets недоступен: утренний дайджест пропущен.")
        except Exception as e:
            try:
                await app.bot.send_message(chat_id=MASTER_CHAT_ID, text=f"Ошибка утреннего дайджеста: {e}")
            except Exception:
                pass


def build_application() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")

    builder: ApplicationBuilder = Application.builder().token(BOT_TOKEN)
    if _RATE_LIMITER_AVAILABLE:
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
    app.add_handler(CommandHandler("init_sheet", cmd_init_sheet))
    app.add_handler(CommandHandler("sheet_test", cmd_sheet_test))

    return app


async def main_async():
    log.info("Fund bot is running…")
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
