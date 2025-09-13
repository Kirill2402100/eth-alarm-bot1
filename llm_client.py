# llm_client.py
# -*- coding: utf-8 -*-
"""
Надёжная обёртка под gpt-5 для фонд-бота.

Особенности:
- По умолчанию все модели = 'gpt-5', но ENV LLM_MAJOR/LLM_MINI/LLM_NANO поддержаны.
- 1-й путь: Responses API (input=messages, max_output_tokens).
- Если пустой ответ/ошибка — повтор и фоллбэк в Chat Completions
  (messages[], max_completion_tokens).
- На полном провале возвращается пустая строка: верхний слой решает, что показывать.
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional

from openai import OpenAI

LOG = logging.getLogger("llm_client")
if os.getenv("DEBUG_LLM"):
    logging.basicConfig(level=logging.DEBUG)

# ===== Модели =====
LLM_MAJOR: str = os.getenv("LLM_MAJOR", "gpt-5")
LLM_MINI:  str = os.getenv("LLM_MINI",  LLM_MAJOR)
LLM_NANO:  str = os.getenv("LLM_NANO",  LLM_MAJOR)

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_OUT_TOKENS = 500

_client: Optional[OpenAI] = None


def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ["OPENAI_API_KEY"]
        _client = OpenAI(api_key=api_key)
    return _client


# ---------- утилиты извлечения текста ----------

def _extract_responses_text(resp) -> str:
    if not resp:
        return ""
    txt = getattr(resp, "output_text", None)
    if txt:
        return str(txt).strip()

    # Подстрахуемся: распарсим output/content
    out = getattr(resp, "output", None)
    parts: List[str] = []
    if isinstance(out, list):
        for item in out:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict):
                        t = (c.get("text") or c.get("output_text") or "").strip()
                        if t:
                            parts.append(t)
    return "\n".join(parts).strip()


def _extract_chat_text(resp) -> str:
    if not resp or not getattr(resp, "choices", None):
        return ""
    msg = resp.choices[0].message
    return (getattr(msg, "content", "") or "").strip()


def _create_responses(client: OpenAI, **kwargs):
    """Вызов Responses API с авто-удалением temperature, если не поддерживается."""
    try:
        return client.responses.create(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        # Некоторые развёртывания ругаются на temperature
        if "temperature" in msg and "not supported" in msg:
            kwargs.pop("temperature", None)
            return client.responses.create(**kwargs)
        raise


# ---------- единый синхронный рендер ----------

def _respond_sync(
    model: str,
    system: str,
    user: str,
    max_output_tokens: int,
    temperature: Optional[float],
) -> str:
    client = _client_singleton()

    # --- 1) Responses API (попытка 1) ---
    try:
        r1 = _create_responses(
            client,
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_output_tokens=max_output_tokens,
            **({"temperature": temperature} if temperature is not None else {}),
        )
        text = _extract_responses_text(r1)
        if text:
            return text
        LOG.warning("Responses #1: пустой текст, повторяем (строковый input)…")
    except Exception as e:
        LOG.warning("Responses #1: ошибка: %s", e)

    # --- 1b) Responses API (попытка 2 с конкатенацией) ---
    try:
        stitched = f"[system]\n{system}\n\n[user]\n{user}"
        r2 = _create_responses(
            client,
            model=model,
            input=stitched,
            max_output_tokens=max_output_tokens,
            **({"temperature": temperature} if temperature is not None else {}),
        )
        text = _extract_responses_text(r2)
        if text:
            return text
        LOG.warning("Responses #2: снова пусто, падаем в Chat Completions…")
    except Exception as e:
        LOG.warning("Responses #2: ошибка: %s — падаем в Chat Completions…", e)

    # --- 2) Chat Completions фоллбэк (новые параметры!) ---
    try:
        cc = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=(temperature if temperature is not None else 0.0),
            # у gpt-5: max_completion_tokens вместо max_tokens
            max_completion_tokens=max_output_tokens,
        )
        return _extract_chat_text(cc)
    except Exception as e:
        LOG.error(
            "Chat Completions упал: %s", e
        )
        return ""  # верхний уровень сам решит, что показать


async def _respond_async(
    model: str,
    system: str,
    user: str,
    max_output_tokens: int = DEFAULT_MAX_OUT_TOKENS,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
) -> str:
    return await asyncio.to_thread(
        _respond_sync, model, system, user, max_output_tokens, temperature
    )


# ---------- публичные функции, совместимые с fa_bot.py ----------

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
    text = _respond_sync(
        model=LLM_MINI,
        system=system,
        user=user,
        max_output_tokens=400,
        temperature=0.3,
    )
    if text:
        return text
    return (
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
    """Пингуем хотя бы одну из настроенных моделей."""
    if not os.environ.get("OPENAI_API_KEY"):
        return False

    client = _client_singleton()
    models_to_try = [LLM_MINI, LLM_MAJOR, LLM_NANO]

    async def _try(mdl: str) -> bool:
        try:
            # Responses ping
            await asyncio.to_thread(
                _create_responses, client,
                model=mdl, input="ping", max_output_tokens=8
            )
            return True
        except Exception:
            pass
        try:
            # Chat ping (важно: max_completion_tokens)
            await asyncio.to_thread(
                client.chat.completions.create,
                model=mdl,
                messages=[{"role": "system", "content": "Проверка связи. Ответь одной буквой."},
                          {"role": "user", "content": "ping"}],
                max_completion_tokens=8,
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
    """Короткий «трейдерский» дайджест: одна строка на каждую пару (RU)."""
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
    return "\n".join(lines)


# ----------- краткий объяснитель «событие → пара» -----------

async def explain_pair_event(
    pair: str,
    headline: str,
    origin: str = "",
    lang: str = "ru",
    model: Optional[str] = None,
) -> str:
    """
    1–2 фразы без эмодзи/цен/вероятностей о влиянии события на FX-пару.
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

    text = " ".join((text or "").split())
    if not text:
        return (
            "Жёстче ожиданий усиливает валюту источника и сдвигает пару в соответствующем направлении; "
            "мягче ожиданий — наоборот."
        )
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
