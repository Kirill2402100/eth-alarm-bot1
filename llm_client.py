# llm_client.py
# -*- coding: utf-8 -*-
"""
Надёжная обёртка над OpenAI для фонд-бота.
- По умолчанию все задачи идут в gpt-5.
- Поддержка ENV: LLM_MAJOR, LLM_MINI, LLM_NANO (иначе все = gpt-5).
- Основной путь: Responses API; фоллбэк: Chat Completions.
- На сбоях возвращает ПУСТУЮ строку (чтобы верхний уровень мог дать хинт).
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion

LOG = logging.getLogger("llm_client")
if os.getenv("DEBUG_LLM"):
    logging.basicConfig(level=logging.DEBUG)

# === Модели ===
# Если переменные не заданы — всё на gpt-5 (единая модель)
LLM_MAJOR: str = os.getenv("LLM_MAJOR", "gpt-5")
LLM_MINI: str = os.getenv("LLM_MINI", LLM_MAJOR)
LLM_NANO: str = os.getenv("LLM_NANO", LLM_MAJOR)

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_OUT_TOKENS = 500

_client: Optional[OpenAI] = None


def _client_singleton() -> OpenAI:
    """Ленивая инициализация клиента OpenAI по ключу из окружения."""
    global _client
    if _client is None:
        api_key = os.environ["OPENAI_API_KEY"]
        _client = OpenAI(api_key=api_key)
    return _client


# ---------- низкоуровневые помощники ----------

def _extract_responses_text(resp) -> str:
    """Достаём текст из Responses API-ответа."""
    if resp is None:
        return ""
    # Нормальный путь у нового SDK
    text = getattr(resp, "output_text", None)
    if text:
        return str(text).strip()

    # Фоллбэк: пройдём по структуре output/content
    out = getattr(resp, "output", None)
    parts: List[str] = []
    if isinstance(out, list):
        for item in out:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict):
                        t = c.get("text") or c.get("output_text") or ""
                        if t:
                            parts.append(str(t))
    return "\n".join(parts).strip()


def _extract_chat_text(resp: ChatCompletion) -> str:
    if not resp or not getattr(resp, "choices", None):
        return ""
    msg = resp.choices[0].message
    return (msg.content or "").strip() if msg else ""


def _create_responses(client: OpenAI, **kwargs):
    """Вызывает Responses API, аккуратно убирая temperature если не поддерживается."""
    try:
        return client.responses.create(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        # некоторые сервера ругаются на temperature у минималок
        if "temperature" in msg and "not supported" in msg:
            kwargs.pop("temperature", None)
            return client.responses.create(**kwargs)
        raise


def _respond_sync(
    model: str,
    system: str,
    user: str,
    max_output_tokens: int,
    temperature: Optional[float],
) -> str:
    """
    Основной синхронный путь:
    1) Responses API
    2) Фоллбэк: Chat Completions
    Возвращает ПУСТУЮ строку при полном провале.
    """
    client = _client_singleton()

    # --- 1) Responses API ---
    try:
        kwargs = dict(
            model=model,
            input=[{"role": "system", "content": system},
                   {"role": "user", "content": user}],
            max_output_tokens=max_output_tokens,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature

        resp = _create_responses(client, **kwargs)
        text = _extract_responses_text(resp)
        if text:
            return text
        LOG.warning("Responses API вернул пустой текст, пробуем Chat Completions…")
    except Exception as e:
        LOG.warning("Responses API ошибка: %s — пробуем Chat Completions…", e)

    # --- 2) Chat Completions фоллбэк ---
    try:
        cc = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=(temperature if temperature is not None else 0.0),
            max_tokens=max_output_tokens,
        )
        text = _extract_chat_text(cc)
        return text
    except Exception as e:
        LOG.error("Chat Completions тоже упал: %s", e)
        return ""  # ВАЖНО: пустая строка — наверху дадим хинт


async def _respond_async(
    model: str,
    system: str,
    user: str,
    max_output_tokens: int = DEFAULT_MAX_OUT_TOKENS,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
) -> str:
    """Async-обёртка над синхронным исполнением (в отдельном потоке)."""
    return await asyncio.to_thread(
        _respond_sync, model, system, user, max_output_tokens, temperature
    )


# ---------- публичные высокоуровневые функции ----------

def quick_classify(labeling_prompt: str) -> str:
    system = "Ты коротко и точно классифицируешь вход. Отвечай одной строкой."
    return _respond_sync(
        model=LLM_NANO,
        system=system,
        user=labeling_prompt,
        max_output_tokens=64,
        temperature=0.2,
    ) or ""


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
    return _respond_sync(
        model=LLM_MINI,
        system=system,
        user=user,
        max_output_tokens=400,
        temperature=0.3,
    ) or (
        "USD/JPY — фон спокойный; обычный режим\n"
        "AUD/USD — фон спокойный; обычный режим\n"
        "EUR/USD — фон спокойный; обычный режим\n"
        "GBP/USD — фон спокойный; обычный режим"
    )


def deep_analysis(question: str, context: str = "") -> str:
    system = (
        "Ты аналитик-объяснитель: делай структурированные и практичные выводы. "
        "Сначала краткий вывод (1–2 предложения), затем маркированные пункты."
    )
    user = f"Вопрос:\n{question}\n\nКонтекст:\n{context}".strip()
    return _respond_sync(
        model=LLM_MAJOR,
        system=system,
        user=user,
        max_output_tokens=800,
        temperature=0.4,
    ) or "Краткий вывод: контекст недостаточен.\n• Уточните данные и цели анализа."


async def llm_ping() -> bool:
    """Проверка, что хотя бы одна конфигурированная модель отвечает."""
    if not os.environ.get("OPENAI_API_KEY"):
        return False

    client = _client_singleton()
    models_to_try = [LLM_MINI, LLM_MAJOR, LLM_NANO]

    async def _try(mdl: str) -> bool:
        # Responses API ping
        try:
            await asyncio.to_thread(
                _create_responses, client,
                model=mdl, input="ping", max_output_tokens=8
            )
            return True
        except Exception:
            pass
        # Chat Completions ping
        try:
            await asyncio.to_thread(
                client.chat.completions.create,
                model=mdl,
                messages=[{"role": "system", "content": "Проверка связи. Ответь одной буквой."},
                          {"role": "user", "content": "ping"}],
                max_tokens=8,
                temperature=0.0,
            )
            return True
        except Exception:
            return False

    for mdl in models_to_try:
        if await _try(mdl):
            return True
    return False


async def generate_digest(
    symbols: List[str],
    model: Optional[str] = None,
    token_budget: int = 30000,
) -> str:
    """Короткий, «трейдерский» дайджест: одна строка на каждую пару (RU)."""
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
        temperature=0.3,
    )

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
    return "\n".инjoin(lines)


# ------------ краткий объяснитель «событие → пара» ------------

async def explain_pair_event(
    pair: str,
    headline: str,
    origin: str = "",
    lang: str = "ru",
    model: Optional[str] = None,
) -> str:
    """
    Коротко объясняет, что означает событие для FX-пары.
    Возвращает 1–2 фразы. Без эмодзи/цен/вероятностей.
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
        temperature=0.2,
    )

    # Нормализуем пробелы
    text = " ".join((text or "").split())

    # Если пусто — отдадим аккуратный общий хинт
    if not text:
        return (
            "Жёстче ожиданий усиливает валюту источника и сдвигает пару в соответствующем направлении; "
            "мягче ожиданий — наоборот."
        )

    # Слишком длинные обрежем до 400 символов по предложению
    if len(text) > 400:
        cut = text[:400]
        text = cut.rsplit(". ", 1)[0] + "."

    return text


__all__ = [
    "quick_classify",
    "fx_digest_ru",
    "deep_analysis",
    "generate_digest",
    "llm_ping",
    "explain_pair_event",
]
