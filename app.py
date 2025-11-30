import json
import os
from collections import defaultdict

import requests
from flask import Flask, request, jsonify

# ------------ конфиг ------------

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID должны быть заданы в переменных окружения"
    )

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"


# ------------ хранилище сигналов (в памяти) ------------

class SignalStore:
    """
    Простое in-memory хранилище:
    - group_signals[time][group_id] = {'direction': 'BUY/SELL', 'payload': {...}}
    - micro_g3[time][direction] = set(['trendline', 'sr', 'reversal'])
    - main_sent[time] = True/False
    """

    def __init__(self):
        self.group_signals = defaultdict(dict)
        self.micro_g3 = defaultdict(lambda: defaultdict(set))
        self.main_sent = {}

    def add_group_signal(self, time_key, group_id, direction, payload):
        self.group_signals[time_key][group_id] = {
            "direction": direction,
            "payload": payload,
        }

    def add_micro_g3(self, time_key, direction, indicator):
        self.micro_g3[time_key][direction].add(indicator)

    def has_full_g3(self, time_key, direction):
        # считаем, что для группы 3 нужны ВСЕ три индикатора
        needed = {"trendline", "sr", "reversal"}
        return self.micro_g3[time_key][direction] >= needed

    def mark_main_sent(self, time_key):
        self.main_sent[time_key] = True

    def is_main_sent(self, time_key):
        return self.main_sent.get(time_key, False)


store = SignalStore()


# ------------ утилиты ------------

def send_telegram(text: str):
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }
    try:
        resp = requests.post(TELEGRAM_API_URL, json=data, timeout=5)
        print("Telegram response status:", resp.status_code, resp.text)
    except Exception as e:
        print("Error sending telegram:", e)


def format_group_message(payload: dict) -> str:
    group_id = payload.get("group_id")
    direction = payload.get("direction")
    pair = payload.get("pair", "EURUSD")
    price = payload.get("price", "")
    text_extra = payload.get("text", "")

    arrow = "🔼" if direction == "BUY" else "🔻"
    header = (
        f"*ГРУППА {group_id} — {direction}* {arrow}\n"
        f"Пара: `{pair}`  Цена: *{price}*\n\n"
    )
    return header + text_extra


def format_main_message(time_key: str, buy_groups, sell_groups, price, pair):
    if buy_groups:
        direction = "BUY"
        arrow = "🔼"
        groups_str = ", ".join(str(g) for g in buy_groups)
    else:
        direction = "SELL"
        arrow = "🔻"
        groups_str = ", ".join(str(g) for g in sell_groups)

    header = (
        f"*MAIN SIGNAL — {direction}* {arrow}\n"
        f"Пара: `{pair}`  Цена: *{price}*\n"
        f"Время бара: `{time_key}`\n\n"
    )
    body = (
        f"Совпали сигналы групп: *{groups_str}* (минимум 2 из 4).\n"
        f"Это сильная точка возможного разворота."
    )
    return header + body


def try_emit_main_signal(time_key: str, last_payload: dict):
    """
    Проверяем, есть ли на этом time >=2 групп в одну сторону.
    last_payload нужен, чтобы взять из него price/pair.
    """
    if store.is_main_sent(time_key):
        return

    groups = store.group_signals[time_key]
    buy_groups = [gid for gid, info in groups.items() if info["direction"] == "BUY"]
    sell_groups = [gid for gid, info in groups.items() if info["direction"] == "SELL"]

    if len(buy_groups) >= 2 and len(sell_groups) == 0:
        msg = format_main_message(
            time_key,
            buy_groups,
            [],
            last_payload.get("price"),
            last_payload.get("pair"),
        )
        send_telegram(msg)
        store.mark_main_sent(time_key)

    elif len(sell_groups) >= 2 and len(buy_groups) == 0:
        msg = format_main_message(
            time_key,
            [],
            sell_groups,
            last_payload.get("price"),
            last_payload.get("pair"),
        )
        send_telegram(msg)
        store.mark_main_sent(time_key)


# ------------ Flask app ------------

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return "TradingView webhook bot is running", 200


@app.route("/test-telegram", methods=["GET"])
def test_telegram():
    """
    Простой тест связи с Telegram.
    """
    send_telegram("Тест от TradingView-бота: связь с Telegram работает ✅")
    return "ok", 200


@app.route("/tradingview-webhook", methods=["POST"])
def tradingview_webhook():
    try:
        raw = request.data.decode("utf-8")
        # TradingView шлёт строку = наш JSON из Message
        payload = json.loads(raw)
    except Exception as e:
        print("Bad payload:", e, "raw:", request.data)
        return jsonify({"status": "error", "detail": "invalid json"}), 400

    print("Got payload:", payload)

    p_type = payload.get("type")
    group_id = int(payload.get("group_id", 0))
    direction = payload.get("direction")
    time_key = str(payload.get("time"))
    pair = payload.get("pair", "EURUSD")

    # ---------- type = "group" (группы 1,2,4 и позже 3) ----------
    if p_type == "group":
        store.add_group_signal(time_key, group_id, direction, payload)

        # отправляем сообщение по группе в телегу
        text = format_group_message(payload)
        send_telegram(text)

        # пробуем собрать MAIN сигнал
        try_emit_main_signal(time_key, payload)

        return jsonify({"status": "ok", "kind": "group"})

    # ---------- type = "micro" (группа 3 по LuxAlgo) ----------
    if p_type == "micro" and group_id == 3:
        indicator = payload.get("indicator")
        store.add_micro_g3(time_key, direction, indicator)

        # когда все три индикатора в одну сторону — считаем это signal group 3
        if store.has_full_g3(time_key, direction):
            # формируем виртуальный group_3 сигнал
            g3_payload = {
                "type": "group",
                "group_id": 3,
                "direction": direction,
                "pair": pair,
                "price": payload.get("price"),
                "time": time_key,
                "text": payload.get(
                    "text",
                    "ГРУППА 3 — сигнал по LuxAlgo (trendline + S/R + Reversal).",
                ),
            }

            store.add_group_signal(time_key, 3, direction, g3_payload)

            # отправляем сообщение по группе 3
            text = format_group_message(g3_payload)
            send_telegram(text)

            # пробуем собрать MAIN
            try_emit_main_signal(time_key, g3_payload)

        # по самим micro-сигналам можно в телегу пока ничего не слать
        return jsonify({"status": "ok", "kind": "micro"})

    # если тип неизвестен
    return jsonify({"status": "ignored"}), 200


@app.route("/debug-group-test", methods=["GET"])
def debug_group_test():
    """
    Простой тест: имитируем сигнал ГРУППА 1 BUY и шлём в Telegram.
    Используется только для отладки, без TradingView.
    """
    payload = {
        "type": "group",
        "group_id": 1,
        "direction": "BUY",
        "pair": "EURUSD",
        "price": "1.2345",
        "time": "TEST-DEBUG",
        "text": (
            "ГРУППА 1 — BUY (debug)\n"
            "Цена: 1.2345\n\n"
            "Направление: 🔼 Возможный разворот ВВЕРХ\n\n"
            "Условия сработки:\n"
            "• Тестовое сообщение с сервера\n"
            "• Если ты это видишь, значит webhook → Telegram работает ✅"
        ),
    }

    text = format_group_message(payload)
    send_telegram(text)

    return "debug ok", 200


if __name__ == "__main__":
    # локальный запуск: python app.py
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
