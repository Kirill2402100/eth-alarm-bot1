# llm_client.py
# -*- coding: utf-8 -*-
"""
Универсальная обёртка для OpenAI:
- сначала пробует Responses API;
- если текст пустой/ошибка — падает в Chat Completions;
- совместимо с gpt-5 (без mini/nano);
- не использует несовместимые параметры (temperature/max_tokens).
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

# Все модели указываем на gpt-5, чтобы не целиться в снятые mini/nano
LLM_NANO: str = os.getenv("LLM_NANO", "gpt-5")
LLM_MINI: str = os.getenv("LLM_MINI", "gpt-5")
LLM_MAJOR: str = os.getenv("LLM_MAJOR", "gpt-5")

# Токены выхода: для Responses -> max_output_tokens, для Chat -> max_completion_tokens
DEFAULT_MAX_OUT_TOKENS = 500

_client: Optional[OpenAI] = None


def _client_singleton() -> OpenAI:
    """Ленивая инициализация клиента OpenAI из переменной OPENAI_API_KEY."""
    global _client
    if _client is None:
        api_key = os.environ["OPENAI_API_KEY"]
        _client = OpenAI(api_key=api_key)
    return _client


# ---------- НИЗКОУРОВНЕВЫЕ ВЫЗОВЫ ----------

def _responses_ask(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_output_tokens: int,
) -> str:
    """
    Пытается вызвать Responses API. Делает 2 попытки:
    1) с message-структурой
    2) со строковым input
    Если ответ пустой — возвращает "" (пусть вызывающий решит делать фоллбэк).
    """

    def _create(**kwargs):
        # Иногда модели ругаются на temperature — просто не передаём его.
        # Иногда возвращают 400 на неизвестные поля — оставляем только безопасные.
        return client.responses.create(**kwargs)

    # Попытка №1 — messages формат
    try:
        resp = _create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_output_tokens=max_output_tokens,
        )
        text = getattr(resp, "output_text", None)
        if text:
            return str(text).strip()
        log.warning("Responses #1: пустой текст, повторим строковым input…")
    except Exception as e:
        # 400/… — пойдём дальше, не падаем
        log.info("Responses #1 ошибка: %s", e)

    # Попытка №2 — строковый формат (часто помогает)
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
        log.warning("Responses #2: снова пусто — пойдём в Chat Completions…")
    except Exception as e:
        log.info("Responses #2 ошибка: %s", e)

    return ""


def _chat_ask(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_completion_tokens: int,
) -> str:
    """
    Chat Completions фоллбэк.
    ВАЖНО: не передаём temperature (некоторые ревизии gpt-5 не принимают произвольные значения).
    Обязательно используем max_completion_tokens (а не max_tokens).
    """
    try:
        comp: ChatCompletion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Не указывать temperature/top_p и пр.
            max_completion_tokens=max_completion_tokens,
        )
        msg = (comp.choices[0].message.content or "").strip() if comp.choices else ""
        return msg
    except Exception as e:
        # На случай если и это не понравится модели — попробуем вообще без max_*.
        log.error("Chat Completions упал: %s", e)
        try:
            comp: ChatCompletion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            msg = (comp.choices[0].message.content or "").strip() if comp.choices else ""
            return msg
        except Exception as e2:
            log.error("Chat Completions (повтор без max_*) тоже упал: %s", e2)
            return ""


def _respond(
    model: str,
    system: str,
    user: str,
    max_output_tokens: int = DEFAULT_MAX_OUT_TOKENS,
) -> str:
    """
    Унифицированный синхронный запрос: Responses → (если пусто) Chat.
    """
    client = _client_singleton()

    # 1) Пытаемся через Responses
    text = _responses_ask(
        client=client,
        model=model,
        system=system,
        user=user,
        max_output_tokens=max_output_tokens,
    )
    if text:
        return text

    # 2) Фоллбэк в Chat Completions
    text = _chat_ask(
        client=client,
        model=model,
        system=system,
        user=user,
        max_completion_tokens=max_output_tokens,
    )
    if text:
        return text

    # Совсем fallback — пустая строка, пусть вызывающий решает
    return ""


async def _respond_async(
    model: str,
    system: str,
    user: str,
    max_output_tokens: int = DEFAULT_MAX_OUT_TOKENS,
) -> str:
    """Асинхронная обёртка над _respond (в пуле потоков)."""
    return await asyncio.to_thread(_respond, model, system, user, max_output_tokens)


# ---------- ВЫСОКОУРОВНЕВЫЕ ХЕЛПЕРЫ ----------

def quick_classify(labeling_prompt: str) -> str:
    system = "Ты коротко и точно классифицируешь вход. Отвечай одной строкой."
    out = _respond(
        model=LLM_NANO,
        system=system,
        user=labeling_prompt,
        max_output_tokens=64,
    )
    return out or ""


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
    out = _respond(
        model=LLM_MINI,
        system=system,
        user=user,
        max_output_tokens=400,
    )
    return out or ""


def deep_analysis(question: str, context: str = "") -> str:
    system = (
        "Ты аналитик-объяснитель: делай структурированные и практичные выводы. "
        "Сначала краткий вывод (1–2 предложения), затем маркированные пункты."
    )
    user = f"Вопрос:\n{question}\n\nКонтекст:\n{context}".strip()
    out = _respond(
        model=LLM_MAJOR,
        system=system,
        user=user,
        max_output_tokens=800,
    )
    return out or ""


async def llm_ping() -> bool:
    """
    Лёгкий смоук-тест, что хотя бы одна конфигурация отвечает.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return False

    client = _client_singleton()

    # Пробуем Responses коротко
    try:
        txt = await asyncio.to_thread(
            _responses_ask,
            client,
            LLM_MINI,
            "Проверка связи. Ответь одной буквой.",
            "ping",
            8,
        )
        if txt:
            return True
    except Exception:
        pass

    # Пробуем Chat коротко
    try:
        txt = await asyncio.to_thread(
            _chat_ask,
            client,
            LLM_MINI,
            "Проверка связи. Ответь одной буквой.",
            "ping",
            8,
        )
        if txt:
            return True
    except Exception:
        pass

    return False


async def generate_digest(
    symbols: List[str],
    model: Optional[str] = None,
    token_budget: int = 30000,
) -> str:
    """
    Короткий трейдерский дайджест: по одной строке на пару (RU).
    """
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

    text = await _respond_async(
        model=mdl,
        system=system,
        user=user,
        max_output_tokens=max_out,
    )

    if not text:
        return "\n".join(f"{pretty.get(s, s[:3]+'/'+s[3:])} — фон спокойный; обычный режим" for s in ordered)

    # Подчистим и приведём к ожидаемому виду
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
        if by_symbol[s]:
            lines.append(by_symbol[s])
        else:
            lines.append(f"{pretty.get(s, s[:3]+'/'+s[3:])} — фон спокойный; обычный режим")
    return "\n".join(lines)


async def explain_pair_event(
    pair: str,
    headline: str,
    origin: str = "",
    lang: str = "ru",
    model: Optional[str] = None,
) -> str:
    """
    Коротко объясняет, что означает событие для данной FX-пары.
    Возвращает 1–2 фразы (без эмодзи, цен, вероятностей).
    Пример: await explain_pair_event("USDJPY", "FOMC Meeting / Rate Decision", "united states")
    """
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
        f"- Учитывай, что усиление валюты стороны события поднимает/опускает пару в зависимости от того, "
        f"является ли она базовой или котируемой в {pair_disp}.\n"
        f"- Формат: одна-две фразы, без списков."
    )

    text = await _respond_async(
        model=mdl,
        system=system,
        user=user,
        max_output_tokens=140,
    )

    if not text:
        # Нейтральный fallback
        return (
            "Жёстче ожиданий усиливает валюту источника и сдвигает пару в соответствующем направлении; "
            "мягче ожиданий — наоборот."
        )

    text = " ".join((text or "").split())
    if len(text) > 400:
        text = text[:400].rsplit(". ", 1)[0] + "."
    return text


__all__ = [
    "quick_classify",
    "fx_digest_ru",
    "deep_analysis",
    "generate_digest",
    "llm_ping",
    "explain_pair_event",
]
