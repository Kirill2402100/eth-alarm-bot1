# llm_client.py
# -*- coding: utf-8 -*-
"""
LLM-клиент под единственную модель GPT-5 через Responses API.
Совместим с текущим кодом бота: generate_digest, llm_ping, explain_pair_event
(+ quick_classify, fx_digest_ru, deep_analysis).

Зависимости:  pip install --upgrade openai>=1.40

ENV:
  OPENAI_API_KEY  — проектный ключ (sk-proj-…)
  LLM_MAJOR       — опц., имя основной модели (игнорируется, но может быть 'gpt-5')
  LLM_MINI        — опц., может быть задано как gpt-5-mini (в любом случае маппится на gpt-5)
  LLM_NANO        — опц., аналогично
"""

from __future__ import annotations

import os
import asyncio
from typing import Dict, List, Optional

from openai import OpenAI

# ---- Константы/настройки ----------------------------------------------------

_DEFAULT_MODEL = "gpt-5"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_OUT_TOKENS = 500

_client_singleton_obj: Optional[OpenAI] = None


def _resolve_model(_: Optional[str] = None) -> str:
    """Любые 'mini'/'nano'/кастом → всегда gpt-5."""
    return _DEFAULT_MODEL


def _client_singleton() -> OpenAI:
    """Lazy OpenAI client with API key from env."""
    global _client_singleton_obj
    if _client_singleton_obj is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _client_singleton_obj = OpenAI(api_key=api_key)
    return _client_singleton_obj


def _create_response(client: OpenAI, **kwargs):
    """
    Обёртка: если модель ругнётся на temperature — перезапросим без него.
    """
    try:
        return client.responses.create(**kwargs)
    except Exception as e:
        msg = (str(e) or "").lower()
        if "temperature" in msg and "not supported" in msg:
            kwargs.pop("temperature", None)
            return client.responses.create(**kwargs)
        raise


def _extract_text(resp) -> str:
    """
    Унифицированное извлечение текста из ответа Responses API.
    """
    # Нормальный путь
    text = getattr(resp, "output_text", None)
    if text:
        return str(text).strip()

    # Фоллбэк — собрать руками
    parts: List[str] = []
    output = getattr(resp, "output", None)
    if isinstance(output, list):
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    # у разных версий SDK тип бывает "output_text" или "text"
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                        parts.append(c.get("text", ""))
    return "\n".join(parts).strip()


def _respond(
    model: Optional[str],
    system: Optional[str],
    user: str,
    max_output_tokens: int = DEFAULT_MAX_OUT_TOKENS,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
) -> str:
    """
    Синхронный вызов Responses API → plain text.
    """
    client = _client_singleton()
    mdl = _resolve_model(model)
    kwargs = dict(
        model=mdl,
        input=(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
            if system else user
        ),
        max_output_tokens=max_output_tokens,
    )
    if temperature is not None:
        kwargs["temperature"] = temperature

    resp = _create_response(client, **kwargs)
    return _extract_text(resp)


async def _respond_async(
    model: Optional[str],
    system: Optional[str],
    user: str,
    max_output_tokens: int = DEFAULT_MAX_OUT_TOKENS,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
) -> str:
    """Async wrapper над _respond (в отдельном потоке, чтобы не блокировать event loop)."""
    return await asyncio.to_thread(
        _respond, model, system, user, max_output_tokens, temperature
    )


# ==================== ПУБЛИЧНЫЕ ФУНКЦИИ (совместимые) ========================

def quick_classify(labeling_prompt: str) -> str:
    system = "Ты коротко и точно классифицируешь вход. Отвечай одной строкой."
    return _respond(
        model="gpt-5",
        system=system,
        user=labeling_prompt,
        max_output_tokens=64,
        temperature=0.2,
    )


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
    return _respond(
        model="gpt-5",
        system=system,
        user=user,
        max_output_tokens=400,
        temperature=0.3,
    )


def deep_analysis(question: str, context: str = "") -> str:
    system = (
        "Ты аналитик-объяснитель: делай структурированные и практичные выводы. "
        "Сначала краткий вывод (1–2 предложения), затем маркированные пункты."
    )
    user = f"Вопрос:\n{question}\n\nКонтекст:\n{context}".strip()
    return _respond(
        model="gpt-5",
        system=system,
        user=user,
        max_output_tokens=800,
        temperature=0.4,
    )


async def llm_ping() -> bool:
    """
    Smoke-тест: модель отвечает и ключ валиден.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        # самый дешёвый возможный запрос
        txt = await _respond_async("gpt-5", None, "ping", max_output_tokens=8, temperature=None)
        return bool(txt)
    except Exception:
        return False


async def generate_digest(
    symbols: List[str],
    model: Optional[str] = None,
    token_budget: int = 30000,
) -> str:
    """
    Короткий трейдерский дайджест: по одной строке на каждую пару (RU).
    Параметр model игнорируется — всегда gpt-5.
    """
    mdl = "gpt-5"
    max_out = max(200, min(500, (token_budget // 50) if token_budget else 400))
    pretty = {"USDJPY": "USD/JPY", "AUDUSD": "AUD/USD", "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD"}
    ordered = [s for s in (symbols or []) if isinstance(s, str)] or ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]
    want_lines = "\n".join(f"{pretty.get(s, s[:3]+'/'+s[3:])} — <заметка>" for s in ordered)
    system = (
        "Ты опытный валютный аналитик. Сделай короткий дайджест на русском, "
        "по одной строке на каждую пару. Пиши по делу: драйверы, риски, режим. Без воды."
    )
    user = "Сформируй дайджест по парам (без цен): " + ", ".join(ordered) + "\n\nФормат ответа строго:\n" + want_lines

    try:
        text = await _respond_async(
            model=mdl,
            system=system,
            user=user,
            max_output_tokens=max_out,
            temperature=0.3,
        )
    except Exception as e:
        return f"LLM ошибка: {e}"

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
    Коротко объясняет эффект события для валютной пары.
    Возвращает 1–2 фразы, без эмодзи/цен/вероятностей.
    """
    mdl = "gpt-5"
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

    try:
        text = await _respond_async(
            model=mdl,
            system=system,
            user=user,
            max_output_tokens=140,
            temperature=0.2,
        )
    except Exception:
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
