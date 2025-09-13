# llm_client.py
# -*- coding: utf-8 -*-
"""
Надёжная обёртка под gpt-5.
По умолчанию используем Chat Completions (stable), Responses можно включить флагом.
- Совместимо с твоими вызовами: quick_classify / fx_digest_ru / deep_analysis
  / generate_digest / llm_ping / explain_pair_event.
- Без temperature/max_tokens (чтобы не ловить 400).
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion

log = logging.getLogger("llm_client")
if not log.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Все «версии» сводим к gpt-5
LLM_NANO  = os.getenv("LLM_NANO",  "gpt-5")
LLM_MINI  = os.getenv("LLM_MINI",  "gpt-5")
LLM_MAJOR = os.getenv("LLM_MAJOR", "gpt-5")

# Сколько текста просим у модели
DEFAULT_MAX_OUT_TOKENS = int(os.getenv("LLM_MAX_OUT", "500") or 500)

# Управление путём вызова:
#   CHAT (по умолчанию) — используем лишь Chat Completions
#   RESPONSES          — сперва Responses, затем фоллбек в Chat
PREFERRED_API = (os.getenv("LLM_PREFERRED_API", "CHAT") or "CHAT").upper().strip()
USE_RESPONSES = PREFERRED_API == "RESPONSES"

_client: Optional[OpenAI] = None


def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ["OPENAI_API_KEY"]
        _client = OpenAI(api_key=api_key)
    return _client


# -------------------- CHAT COMPLETIONS --------------------

def _chat_ask(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_completion_tokens: int,
) -> str:
    """
    Надёжный путь для gpt-5.
    Не передаём temperature/top_p и не используем max_tokens — только max_completion_tokens.
    """
    try:
        comp: ChatCompletion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_completion_tokens=max_completion_tokens,
        )
        msg = (comp.choices[0].message.content or "").strip() if comp.choices else ""
        return msg
    except Exception as e:
        log.error("Chat Completions error: %s", e)
        # Последняя попытка — без ограничений токенов
        try:
            comp: ChatCompletion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            msg = (comp.choices[0].message.content or "").strip() if comp.choices else ""
            return msg
        except Exception as e2:
            log.error("Chat Completions retry failed: %s", e2)
            return ""


# -------------------- RESPONSES (ОПЦИОНАЛЬНО) --------------------

def _responses_ask(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_output_tokens: int,
) -> str:
    """
    Responses API у gpt-5 иногда отдаёт пустой output_text.
    Делаем две попытки, если пусто — вызывающий решит упасть в Chat.
    """
    def _create(**kwargs):
        return client.responses.create(**kwargs)

    # Попытка №1 — messages
    try:
        resp = _create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_output_tokens=max_output_tokens,
        )
        text = getattr(resp, "output_text", None)
        if text:
            return str(text).strip()
        log.warning("Responses #1: пустой текст, повторим строковым input…")
    except Exception as e:
        log.info("Responses #1 error: %s", e)

    # Попытка №2 — строка
    try:
        combined = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
        resp = _create(
            model=model,
            input=combined,
            max_output_tokens=max_output_tokens,
        )
        text = getattr(resp, "output_text", None)
        if text:
            return str(text).strip()
        log.warning("Responses #2: снова пусто — пойдём в Chat…")
    except Exception as e:
        log.info("Responses #2 error: %s", e)

    return ""


# -------------------- ЕДИНЫЙ РОУТЕР --------------------

def _respond_sync(model: str, system: str, user: str, max_out: int) -> str:
    client = _client_singleton()

    if USE_RESPONSES:
        txt = _responses_ask(client, model, system, user, max_out)
        if txt:
            return txt
        # Фоллбек
        return _chat_ask(client, model, system, user, max_out)

    # По умолчанию — только Chat (быстро и стабильно)
    return _chat_ask(client, model, system, user, max_out)


async def _respond_async(model: str, system: str, user: str, max_out: int) -> str:
    return await asyncio.to_thread(_respond_sync, model, system, user, max_out)


# -------------------- ПУБЛИЧНЫЕ ХЕЛПЕРЫ --------------------

def quick_classify(labeling_prompt: str) -> str:
    system = "Ты коротко и точно классифицируешь вход. Отвечай одной строкой."
    return _respond_sync(LLM_NANO, system, labeling_prompt, 64) or ""


def fx_digest_ru(pairs_state: Dict[str, str]) -> str:
    pairs_list = ", ".join(pairs_state.keys()) if pairs_state else "USDJPY, AUDUSD, EURUSD, GBPUSD"
    system = (
        "Ты опытный финансовый аналитик. Дай сжатый текст-дайджест по форекс-парам (RU): "
        "по каждой паре отдельная строка. Если контекста мало — 'фон спокойный; обычный режим'."
    )
    lines = ["Сформируй короткие заметки по парам: " + pairs_list, "", "Данные по парам:"]
    for p, ctx in pairs_state.items():
        lines.append(f"- {p}: {ctx or 'нет свежего контекста'}")
    user = (
        "\n".join(lines)
        + "\n\nФормат ответа строго:\nUSD/JPY — <заметка>\nAUD/USD — <заметка>\nEUR/USD — <заметка>\nGBP/USD — <заметка>"
    )
    return _respond_sync(LLM_MINI, system, user, 400) or ""


def deep_analysis(question: str, context: str = "") -> str:
    system = (
        "Ты аналитик-объяснитель: делай структурированные и практичные выводы. "
        "Сначала краткий вывод (1–2 предложения), затем маркированные пункты."
    )
    user = f"Вопрос:\n{question}\n\nКонтекст:\n{context}".strip()
    return _respond_sync(LLM_MAJOR, system, user, 800) or ""


async def llm_ping() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    # быстрая проверка chat-путём
    txt = await _respond_async(LLM_MINI, "Проверка связи. Ответь одной буквой.", "ping", 8)
    return bool(txt.strip())


async def generate_digest(
    symbols: List[str],
    model: Optional[str] = None,
    token_budget: int = 30000,
) -> str:
    mdl = (model or LLM_MINI).strip()
    max_out = max(200, min(500, (token_budget // 50) if token_budget else 400))
    pretty = {"USDJPY": "USD/JPY", "AUDUSD": "AUD/USD", "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD"}
    ordered = [s for s in (symbols or []) if isinstance(s, str)] or ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]
    want_lines = "\n".join(f"{pretty.get(s, s[:3]+'/'+s[3:])} — <заметка>" for s in ordered)
    system = (
        "Ты опытный валютный аналитик. Сделай короткий дайджест на русском, "
        "по одной строке на каждую пару. Пиши по делу: драйверы, риски, режим. Без воды."
    )
    user = "Сформируй дайджест по парам (без цен): " + ", ".join(ordered) + "\n\nФормат ответа строго:\n" + want_lines

    text = await _respond_async(mdl, system, user, max_out)
    if not text:
        return "\n".join(f"{pretty.get(s, s[:3]+'/'+s[3:])} — фон спокойный; обычный режим" for s in ordered)

    lines: List[str] = []
    by_symbol = {s: None for s in ordered}
    for raw in (text or "").splitlines():
        line = raw.strip(" \t-")
        for s in ordered:
            tag = pretty.get(s, s[:3] + "/" + s[3:])
            if line.upper().startswith(tag.upper()):
                by_symbol[s] = line
                break
    for s in ordered:
        lines.append(by_symbol[s] or f"{pretty.get(s, s[:3]+'/'+s[3:])} — фон спокойный; обычный режим")
    return "\n".join(lines)


async def explain_pair_event(
    pair: str,
    headline: str,
    origin: str = "",
    lang: str = "ru",
    model: Optional[str] = None,
) -> str:
    mdl = (model or LLM_MINI).strip()
    pair = (pair or "").upper().strip()
    origin = (origin or "").lower().strip()

    pretty = {"USDJPY": "USD/JPY", "AUDUSD": "AUD/USD", "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD"}
    pair_disp = pretty.get(pair, f"{pair[:3]}/{pair[3:]}")

    system = (
        "Ты валютный аналитик. Пиши кратко и по делу, на русском. "
        "Цель — объяснить потенциальное направление пары от новости. "
        "Никаких цен, эмодзи, вероятностей и дисклеймеров. 1–2 короткие фразы."
    )
    lang_note = "на русском" if (lang or "ru").lower().startswith("ru") else "in concise English"
    user = (
        f"Пара: {pair_disp}\n"
        f"Источник новости (страна/регулятор): {origin or 'unknown'}\n"
        f"Событие/заголовок: {headline.strip()}\n\n"
        f"Требование к ответу ({lang_note}):\n"
        f"- Объясни, как ястребиный (жёстче ожиданий) и голубиный (мягче ожиданий) исход влияет на {pair_disp}.\n"
        f"- Учитывай базовую/котируемую валюту в {pair_disp}.\n"
        f"- Формат: одна-две фразы, без списков."
    )

    text = await _respond_async(mdl, system, user, 140)
    if not text:
        return ("Жёстче ожиданий усиливает валюту источника и ведёт пару в соответствующем направлении; "
                "мягче ожиданий — наоборот.")
    text = " ".join(text.split())
    return text[:400] if len(text) <= 400 else text[:400].rsplit(". ", 1)[0] + "."


__all__ = [
    "quick_classify",
    "fx_digest_ru",
    "deep_analysis",
    "generate_digest",
    "llm_ping",
    "explain_pair_event",
]
