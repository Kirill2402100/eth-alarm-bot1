import json
import os
from collections import defaultdict
from datetime import datetime, timedelta

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
INDICATOR_TITLES = {
    "rsi14": "RSI(14)",
    "stoch": "Stochastic (14, 3, 3)",
    "macd": "MACD (12, 26, 9)",
    "mfi": "MFI (Money Flow Index)",
    "bb": "Bollinger Bands",
    "kc": "Keltner Channels",
    "rsi7": "RSI(7)",
    "lux_trendline": "LuxAlgo Trendlines with Breaks",
    "lux_sr": "LuxAlgo S/R with Breaks",
    "frvp": "Fixed Range Volume Profile (FRVP)",
    "lux_reversal": "Lux Reversal Signals",
    "alligator": "Alligator",
    "ao": "Awesome Oscillator",
    "fractals": "Fractals",
    "atr14": "ATR(14)",
}


# ------------ утилиты ------------

def parse_time(ts_str: str):
    """Пытаемся разобрать ISO-дату из TradingView. Если не получилось — берём текущее UTC."""
    if not ts_str:
        return datetime.utcnow()
    try:
        # TradingView часто отдаёт что-то типа "2025-11-30T15:00:00Z"
        s = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.utcnow()


def send_telegram(text: str):
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


def format_direction(direction: str) -> str:
    if direction == "BUY":
        return "BUY 🔼"
    elif direction == "SELL":
        return "SELL 🔻"
    return direction or "N/A"


def format_indicator_message(payload: dict) -> str:
    group_id = int(payload.get("group_id", 0))
    indicator_code = payload.get("indicator", "unknown")
    direction = payload.get("direction")
    pair = payload.get("pair", "EURUSD")
    price = payload.get("price", "")
    time_str = str(payload.get("time", ""))
    extra = payload.get("text", "")

    indicator_name = INDICATOR_TITLES.get(indicator_code, indicator_code)
    group_title = GROUP_NAMES.get(group_id, f"Группа {group_id}")

    header = f"*{group_title}*\nИндикатор: *{indicator_name}*\nСигнал: *{format_direction(direction)}*\n"
    body = f"Пара: `{pair}`  Цена: *{price}*\nВремя бара: `{time_str}`\n\n{extra}"
    return header + body


def format_group_summary(group_id: int, direction: str, indicators: set, pair: str, price: str, time_str: str) -> str:
    group_title = GROUP_NAMES.get(group_id, f"Группа {group_id}")
    arrow = "🔼" if direction == "BUY" else "🔻"
    indicators_pretty = ", ".join(INDICATOR_TITLES.get(i, i) for i in sorted(indicators))

    header = f"*Сработала {group_title}* {arrow}\n"
    meta = f"Пара: `{pair}`  Цена: *{price}*\nВремя окна: `последние ~2 бара`\n\n"
    body = f"В этой группе *минимум два индикатора* дают сигнал в сторону {direction}:\n- {indicators_pretty}"
    return header + meta + body


def format_main_summary(direction: str, group_ids: list[int], pair: str, price: str, time_str: str) -> str:
    arrow = "🚀" if direction == "BUY" else "📉"
    groups_list = ", ".join(str(g) for g in sorted(group_ids))
    header = f"*МОЩНЫЙ СИГНАЛ НА РАЗВОРОТ* {arrow}\n"
    meta = f"Пара: `{pair}`  Цена: *{price}*\nВремя окна: `последние ~2 бара`\n\n"
    body = (
        f"Сработали *минимум две группы* в одну сторону ({direction}).\n"
        f"Группы: *{groups_list}*.\n"
        f"Это сильная точка возможного разворота тренда."
    )
    return header + meta + body


# ------------ хранилище сигналов ------------

class SignalStore:
    """
    Храним список событий (индивидуальных индикаторов) за последний час и считаем групповые сигналы.

    Каждое событие:
    {
        "ts": datetime,
        "group_id": int,
        "indicator": str,
        "direction": "BUY"/"SELL",
        "pair": str,
        "price": str,
        "time_raw": str,  # как пришло из TV
    }

    Логика:
    - окно 30 минут назад от текущего события (≈ 2 бара по 15м);
    - по этому окну считаем:
        direction -> group_id -> set(indicators)
    - если в группе >=2 индикаторов в одну сторону -> групповой сигнал
    - если групп с >=2 индикаторами в одну сторону >=2 -> MAIN сигнал
    """

    def __init__(self):
        self.events = []
        self.max_age_minutes = 60

        # чтобы не спамить одинаковыми сообщениями
        self.sent_group = set()  # (direction, group_id, bucket_id)
        self.sent_main = set()   # (direction, bucket_id)

    def _prune_old(self, now: datetime):
        cutoff = now - timedelta(minutes=self.max_age_minutes)
        self.events = [e for e in self.events if e["ts"] >= cutoff]

    def add_event(self, time_raw: str, group_id: int, indicator: str,
                  direction: str, pair: str, price: str):
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
        price = str(payload.get("price", ""))
        time_raw = str(payload.get("time", ""))

        ts = self.add_event(time_raw, group_id, indicator, direction, pair, price)
        stats = self.analyze_window(ts, window_minutes=30)
        dir_stats = stats.get(direction, {})

        # какие группы уже «сильные» в этом окне
        strong_groups = [gid for gid, inds in dir_stats.items() if len(inds) >= 2]

        # используем "bucket" = время текущего бара, округлённое до минут
        bucket_id = ts.replace(second=0, microsecond=0).isoformat(timespec="minutes")

        new_group_triggers = []
        for gid in strong_groups:
            key = (direction, gid, bucket_id)
            if key not in self.sent_group:
                self.sent_group.add(key)
                new_group_triggers.append(gid)

        main_trigger = None
        if len(strong_groups) >= 2:
            main_key = (direction, bucket_id)
            if main_key not in self.sent_main:
                self.sent_main.add(main_key)
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
    """
    Поддерживает два варианта:

    1) Твой старый curl / тесты:
       POST { "type": "indicator", ... }

    2) Реальный TradingView:
       POST {
         "time": "...",
         "ticker": "EURUSD",
         "exchange": "...",
         "message": "{\"type\":\"indicator\", ... }"
       }
    """
    # Пытаемся прочитать JSON "умно"
    raw_json = request.get_json(silent=True)

    # Fallback на старый способ (raw text), если вдруг get_json не сработал
    if raw_json is None:
        try:
            raw_text = request.data.decode("utf-8")
            raw_json = json.loads(raw_text)
        except Exception as e:
            print("Bad payload:", e, "raw:", request.data)
            return jsonify({"status": "error", "detail": "invalid json"}), 400

    print("RAW JSON FROM TV:", raw_json)

    # Если это TradingView-обёртка с полем message
    if isinstance(raw_json, dict) and "message" in raw_json:
        msg = raw_json["message"]

        # message может быть строкой с JSON или уже dict’ом
        if isinstance(msg, str):
            try:
                payload = json.loads(msg)
            except Exception as e:
                print("Failed to parse inner message JSON:", e, "msg:", msg)
                # если не смогли распарсить — завернём как raw
                payload = {"type": "raw", "raw_message": msg}
        elif isinstance(msg, dict):
            payload = msg
        else:
            payload = {"type": "raw", "raw_message": msg}

        # докидываем полезные поля из внешней обёртки, если их нет внутри
        if "time" in raw_json and "time" not in payload:
            payload["time"] = raw_json["time"]
        if "ticker" in raw_json and "pair" not in payload:
            payload["pair"] = raw_json["ticker"]
    else:
        # Старый вариант: тело запроса уже == нужный payload
        payload = raw_json

    print("PARSED PAYLOAD:", payload)

    p_type = payload.get("type", "indicator")
    group_id = int(payload.get("group_id", 0))
    indicator = payload.get("indicator")
    direction = payload.get("direction")

    if p_type != "indicator":
        # на будущее — можно будет добавить поддержку других типов
        return jsonify({"status": "ignored", "detail": f"unsupported type {p_type}"}), 200

    if not group_id or not indicator or direction not in ("BUY", "SELL"):
        return jsonify({"status": "ignored", "detail": "missing group_id/indicator/direction"}), 200

    # 1) всегда шлём индивидуальное сообщение по индикатору
    text = format_indicator_message(payload)
    send_telegram(text)

    # 2) считаем внутри окна 30 минут групповые и основной сигнал
    new_groups, main_trigger, dir_stats = store.process_event(payload)

    pair = payload.get("pair", "EURUSD")
    price = str(payload.get("price", ""))
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
