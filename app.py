import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests
from flask import Flask, request, jsonify

# ------------ конфиг ------------

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID_RAW = os.environ.get("TELEGRAM_CHAT_ID", "")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID_RAW:
    raise RuntimeError("TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID должны быть заданы в переменных окружения")

# поддержка нескольких чатов через запятую
TELEGRAM_CHAT_IDS = [cid.strip() for cid in TELEGRAM_CHAT_ID_RAW.split(",") if cid.strip()]

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

# Имя групп для сообщений
GROUP_NAMES = {
    1: "Группа 1 — Перекупленность / перепроданность",
    2: "Группа 2 — Волатильность и зоны перегрева",
    3: "Группа 3 — Трендовые уровни и объём",
    4: "Группа 4 — Тренд и импульс",
}

# Нормальные «человеческие» названия индикаторов
# Ключи ДОЛЖНЫ совпадать с полем "indicator" в JSON от TradingView
INDICATOR_TITLES = {
    "RSI14": "RSI(14)",
    "Stoch": "Stochastic (14, 3, 3)",
    "MACD": "MACD (12, 26, 9)",
    "MFI": "MFI (Money Flow Index)",

    "BB": "Bollinger Bands",
    "KC": "Keltner Channels",
    "RSI7": "RSI(7)",

    "trendline": "Trendlines with Breaks",
    "SR": "Support/Resistance with Breaks",
    "FRVP": "Fixed Range Volume Profile (FRVP)",
    "reversal": "Reversal Signals",

    "Alligator": "Alligator",
    "AO": "Awesome Oscillator",
    "Fractals": "Fractals",
    "ATR14": "ATR(14)",
}

# ------------ утилиты ------------


def parse_time(ts_str: str) -> datetime:
    """
    Разбираем время из TradingView и ВСЕГДА возвращаем naïve UTC (без tzinfo),
    чтобы потом не было конфликтов offset-aware / offset-naive.
    """
    try:
        if not ts_str:
            return datetime.now(timezone.utc).replace(tzinfo=None)

        # если вдруг пришёл timestamp числом
        if isinstance(ts_str, (int, float)) or str(ts_str).isdigit():
            dt = datetime.fromtimestamp(float(ts_str), tz=timezone.utc)
            return dt.replace(tzinfo=None)

        # формат TV: "2025-12-02T20:15:00Z"
        s = str(ts_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        return dt.replace(tzinfo=None)
    except Exception:
        return datetime.now(timezone.utc).replace(tzinfo=None)


# простейший анти-дубликатор: не шлём одинаковый текст чаще, чем раз в 3 сек
_last_telegram_messages = []  # список (text, datetime)


def send_telegram(text: str):
    global _last_telegram_messages

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    # подчистка старых записей (старше 10с)
    _last_telegram_messages = [
        (t, ts) for (t, ts) in _last_telegram_messages if (now - ts).total_seconds() < 10
    ]

    # если такой же текст уже был за последние 3с — пропускаем
    for t, ts in _last_telegram_messages:
        if t == text and (now - ts).total_seconds() < 3:
            print("Skip duplicate telegram message")
            return

    data = {
        "text": text,
        "parse_mode": "Markdown",
    }
    for chat_id in TELEGRAM_CHAT_IDS:
        data["chat_id"] = chat_id
        try:
            r = requests.post(TELEGRAM_API_URL, json=data, timeout=5)
            if r.status_code != 200:
                print("Telegram error:", r.status_code, r.text)
        except Exception as e:
            print("Error sending telegram:", e)

    _last_telegram_messages.append((text, now))


def format_direction(direction: str) -> str:
    if direction == "BUY":
        return "BUY 🔼"
    elif direction == "SELL":
        return "SELL 🔻"
    return direction or "N/A"


def _extract_price(payload: dict) -> str:
    """Берём цену из разных возможных полей."""
    return str(payload.get("price") or payload.get("value") or "")


def format_indicator_message(payload: dict) -> str:
    group_id = int(payload.get("group_id", 0))
    indicator_code = payload.get("indicator", "unknown")
    direction = payload.get("direction")
    pair = payload.get("pair", "EURUSD")
    price = _extract_price(payload)
    time_str = str(payload.get("time", ""))
    extra = payload.get("text", "")

    indicator_name = INDICATOR_TITLES.get(indicator_code, indicator_code)
    group_title = GROUP_NAMES.get(group_id, f"Группа {group_id}")

    header = (
        f"*{group_title}*\n"
        f"Индикатор: *{indicator_name}*\n"
        f"Сигнал: *{format_direction(direction)}*\n"
    )
    body = (
        f"Пара: `{pair}`  Цена: *{price}*\n"
        f"Время бара: `{time_str}`\n\n"
        f"{extra}"
    )
    return header + body


def format_group_summary(group_id: int, direction: str, indicators: set,
                         pair: str, price: str, time_str: str) -> str:
    group_title = GROUP_NAMES.get(group_id, f"Группа {group_id}")
    arrow = "🔼" if direction == "BUY" else "🔻"
    indicators_pretty = ", ".join(INDICATOR_TITLES.get(i, i) for i in sorted(indicators))

    header = f"*Сработала {group_title}* {arrow}\n"
    meta = (
        f"Пара: `{pair}`  Цена: *{price}*\n"
        f"Время окна: `последние ~2 бара`\n\n"
    )
    body = (
        f"В этой группе *минимум два индикатора* дают сигнал в сторону {direction}:\n"
        f"- {indicators_pretty}"
    )
    return header + meta + body


def format_main_summary(direction: str, group_ids: list[int],
                        pair: str, price: str, time_str: str) -> str:
    arrow = "🚀" if direction == "BUY" else "📉"
    groups_list = ", ".join(str(g) for g in sorted(group_ids))
    header = f"*МОЩНЫЙ СИГНАЛ НА РАЗВОРОТ* {arrow}\n"
    meta = (
        f"Пара: `{pair}`  Цена: *{price}*\n"
        f"Время окна: `последние ~2 бара`\n\n"
    )
    body = (
        f"Сработали *минимум две группы* в одну сторону ({direction}).\n"
        f"Группы: *{groups_list}*.\n"
        f"Это сильная точка возможного разворота тренда."
    )
    return header + meta + body


# ------------ хранилище сигналов ------------


class SignalStore:
    """
    Храним список событий (индивидуальных индикаторов) и считаем групповые сигналы.

    Каждое событие:
    {
        "ts": datetime (UTC, naive),
        "group_id": int,
        "indicator": str,
        "direction": "BUY"/"SELL",
        "pair": str,
        "price": str,
        "time_raw": str,
    }

    Логика:
    - окно 30 минут назад от текущего события (≈ 2 бара по 15м);
    - по этому окну считаем:
        direction -> group_id -> set(indicators)
    - если в группе >=2 индикаторов в одну сторону -> групповой сигнал
    - если таких групп в одну сторону >=2 -> основной сигнал
    """

    def __init__(self):
        self.events = []
        self.max_age_minutes = 60

        # когда последний раз слали групповой / основной сигнал
        self.sent_group_last = {}  # (direction, group_id) -> datetime
        self.sent_main_last = {}   # direction -> datetime

    def _prune_old(self, now: datetime):
        cutoff = now - timedelta(minutes=self.max_age_minutes)
        self.events = [e for e in self.events if e["ts"] >= cutoff]

        # параллельно чистим историю триггеров (здесь достаточно 30 минут)
        cutoff_group = now - timedelta(minutes=30)
        self.sent_group_last = {
            k: t for k, t in self.sent_group_last.items() if t >= cutoff_group
        }
        self.sent_main_last = {
            k: t for k, t in self.sent_main_last.items() if t >= cutoff_group
        }

    def add_event(self, time_raw: str, group_id: int, indicator: str,
                  direction: str, pair: str, price: str) -> datetime:
        ts = parse_time(time_raw)
        event = {
            "ts": ts,
            "time_raw": time_raw,
            "group_id": group_id,
            "indicator": indicator,
            "direction": direction,
            "pair": pair,
            "price": price,
        }
        self.events.append(event)
        self._prune_old(ts)
        return ts

    def analyze_window(self, ts: datetime, window_minutes: int = 30):
        window_start = ts - timedelta(minutes=window_minutes)
        # direction -> group_id -> set(indicators)
        stats = defaultdict(lambda: defaultdict(set))

        for e in self.events:
            if window_start <= e["ts"] <= ts:
                stats[e["direction"]][e["group_id"]].add(e["indicator"])

        return stats

    def process_event(self, payload: dict):
        """
        Основной метод:
        - добавляет событие;
        - считает окно 30 минут;
        - возвращает:
            (new_group_triggers, main_trigger, dir_stats)
        """
        group_id = int(payload.get("group_id", 0))
        indicator = payload.get("indicator", "unknown")
        direction = payload.get("direction")
        pair = payload.get("pair", "EURUSD")
        price = _extract_price(payload)
        time_raw = str(payload.get("time", ""))

        ts = self.add_event(time_raw, group_id, indicator, direction, pair, price)
        stats = self.analyze_window(ts, window_minutes=30)
        dir_stats = stats.get(direction, {})

        # какие группы уже «сильные» в этом окне
        strong_groups = [gid for gid, inds in dir_stats.items() if len(inds) >= 2]

        new_group_triggers = []
        for gid in strong_groups:
            key = (direction, gid)
            last_ts = self.sent_group_last.get(key)
            # триггерим не чаще, чем раз в 30 минут
            if (last_ts is None) or ((ts - last_ts) >= timedelta(minutes=30)):
                self.sent_group_last[key] = ts
                new_group_triggers.append(gid)

        main_trigger = None
        if len(strong_groups) >= 2:
            last_main_ts = self.sent_main_last.get(direction)
            if (last_main_ts is None) or ((ts - last_main_ts) >= timedelta(minutes=30)):
                self.sent_main_last[direction] = ts
                main_trigger = sorted(strong_groups)

        return new_group_triggers, main_trigger, dir_stats


store = SignalStore()

# ------------ Flask app ------------

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return "TradingView webhook bot is running", 200


@app.route("/test-telegram", methods=["GET"])
def test_telegram():
    send_telegram("Test message from Railway bot (plain text)")
    return "ok", 200


@app.route("/telegram-api-debug", methods=["GET"])
def telegram_api_debug():
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
    r = requests.get(url, timeout=5)
    return jsonify({"status_code": r.status_code, "body": r.json()})


@app.route("/telegram-send-plain", methods=["GET"])
def telegram_send_plain():
    text = "Plain test message from Railway bot"
    data = {
        "chat_id": TELEGRAM_CHAT_IDS[0],
        "text": text,
    }
    r = requests.post(TELEGRAM_API_URL, json=data, timeout=5)
    return jsonify({"status_code": r.status_code, "body": r.json()})


@app.route("/tradingview-webhook", methods=["POST"])
def tradingview_webhook():
    try:
        raw = request.data.decode("utf-8")
        payload = json.loads(raw)
    except Exception as e:
        print("Bad payload:", e, "raw:", request.data)
        return jsonify({"status": "error", "detail": "invalid json"}), 400

    print("Got payload:", payload)

    p_type = payload.get("type", "indicator")
    if p_type != "indicator":
        return jsonify({"status": "ignored", "detail": "unknown type"}), 200

    group_id = int(payload.get("group_id", 0))
    indicator = payload.get("indicator")
    direction = payload.get("direction")

    # без этих полей нам нечего считать
    if not group_id or not indicator or direction not in ("BUY", "SELL"):
        return jsonify({"status": "ignored", "detail": "missing group_id/indicator/direction"}), 200

    # 1) всегда шлём индивидуальное сообщение по индикатору
    text = format_indicator_message(payload)
    send_telegram(text)

    # 2) считаем внутри окна 30 минут групповые и основной сигнал
    new_groups, main_trigger, dir_stats = store.process_event(payload)

    pair = payload.get("pair", "EURUSD")
    price = _extract_price(payload)
    time_raw = str(payload.get("time", ""))

    # 2а) новые сработавшие группы
    for gid in new_groups:
        indicators = dir_stats.get(gid, set())
        g_text = format_group_summary(gid, direction, indicators, pair, price, time_raw)
        send_telegram(g_text)

    # 2б) мощный сигнал
    if main_trigger:
        m_text = format_main_summary(direction, main_trigger, pair, price, time_raw)
        send_telegram(m_text)

    return jsonify({
        "status": "ok",
        "kind": "indicator",
        "new_groups": new_groups,
        "main_trigger": main_trigger or [],
    })


if __name__ == "__main__":
    # локальный запуск: python app.py
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
