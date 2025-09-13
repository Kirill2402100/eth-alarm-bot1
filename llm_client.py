# llm_client.py
# -*- coding: utf-8 -*-
"""
Надёжный клиент LLM под единственную модель GPT-5.
Порядок попыток:
1) Responses API (messages в input)
2) Chat Completions (fallback)
3) Повтор с упрощённым prompt
Гарантирует непустой текст и подробный лог при ошибках.

ENV:
  OPENAI_API_KEY  — ключ проекта (sk-proj-…)
  DEBUG_LLM       — '1' для подробных логов
"""

from __future__ import annotations

import os
import asyncio
import logging
from typing import Dict, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion

LOG = logging.getLogger("llm_client")
if not LOG.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))

DEFAULT_MODEL = "gpt-5"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_OUT_TOKENS = 500
DEBUG = os.getenv("DEBUG_LLM","0") in ("1","true","yes","on")

_client: Optional[OpenAI] = None


# ---------- infra ----------

def _model(_: Optional[str] = None) -> str:
    # Любые mini/nano → gpt-5
    return DEFAULT_MODEL

def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY","").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _client = OpenAI(api_key=api_key)
    return _client

def _err_has(msg: str, *needles: str) -> bool:
    m = (msg or "").lower()
    return any(n.lower() in m for n in needles)

def _extract_responses_text(resp) -> str:
    # Нормальный путь
    text = getattr(resp, "output_text", None)
    if text:
        return str(text).strip()
    # Фоллбэк
    parts: List[str] = []
    out = getattr(resp, "output", None)
    if isinstance(out, list):
        for it in out:
            content = getattr(it, "content", None)
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") in ("output_text","text"):
                        parts.append(c.get("text",""))
    return "\n".join(parts).strip()

def _extract_chat_text(resp: ChatCompletion) -> str:
    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


# ---------- core call ----------

def _call_llm_sync(
    system: Optional[str],
    user: str,
    max_output_tokens: int,
    temperature: Optional[float],
) -> str:
    """
    1) Responses API
    2) Chat Completions fallback
    """
    client = _client_singleton()
    mdl = _model()

    # ---- try Responses API
    kwargs = dict(
        model=mdl,
        input=([{"role":"system","content":system},{"role":"user","content":user}] if system else user),
        max_output_tokens=max_output_tokens,
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        resp = client.responses.create(**kwargs)
        txt = _extract_responses_text(resp)
        if txt:
            return txt
        LOG.warning("Responses API returned empty text")
    except Exception as e:
        em = str(e)
        if DEBUG:
            LOG.exception("Responses API error: %s", em)
        # если похоже на несовместимость параметров/эндпоинта — пойдём в чат
        if not _err_has(em, "not supported", "unrecognized request", "unknown parameter", "invalid_request_error"):
            # даже если другая ошибка — всё равно попробуем чат ниже
            pass

    # ---- fallback: Chat Completions
    try:
        messages = (
            [{"role":"system","content":system},{"role":"user","content":user}]
            if system else [{"role":"user","content":user}]
        )
        chat_kwargs = dict(model=mdl, messages=messages, max_tokens=max_output_tokens)
        if temperature is not None:
            chat_kwargs["temperature"] = float(temperature)
        resp = client.chat.completions.create(**chat_kwargs)  # type: ignore
        txt = _extract_chat_text(resp)
        if txt:
            return txt
        LOG.warning("Chat Completions returned empty text")
    except Exception as e:
        if DEBUG:
            LOG.exception("Chat Completions error: %s", e)

    return ""  # пусть вызывающий решает, что делать дальше


def _respond(
    model: Optional[str],
    system: Optional[str],
    user: str,
    max_output_tokens: int = DEFAULT_MAX_OUT_TOKENS,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
) -> str:
    """
    Гарантирует непустой текст: если первая попытка пустая —
    повторяет упрощённым запросом; в самом конце — стат. фоллбэк.
    """
    txt = _call_llm_sync(system, user, max_output_tokens, temperature)
    if not txt:
        # Повтор: упрощаем (без system/temperature, меньше токенов)
        LOG.warning("Empty LLM text; retrying with simplified prompt…")
        txt = _call_llm_sync(None, user, min(160, max_output_tokens), None)

    if not txt:
        LOG.error("LLM returned empty text after retries")
        return "Не удалось получить ответ от модели."
    return txt.strip()


async def _respond_async(
    model: Optional[str],
    system: Optional[str],
    user: str,
    max_output_tokens: int = DEFAULT_MAX_OUT_TOKENS,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
) -> str:
    return await asyncio.to_thread(
        _respond, model, system, user, max_output_tokens, temperature
    )


# ---------- публичные функции (совместимы с ботом) ----------

def quick_classify(labeling_prompt: str) -> str:
    system = "Ты коротко и точно классифицируешь вход. Отвечай одной строкой."
    return _respond("gpt-5", system, labeling_prompt, max_output_tokens=64, temperature=0.2)

def fx_digest_ru(pairs_state: Dict[str, str]) -> str:
    pairs_list = ", ".join(pairs_state.keys()) if pairs_state else "USDJPY, AUDUSD, EURUSD, GBPUSD"
    system = ("Ты опытный финансовый аналитик. Дай сжатый текст-дайджест по форекс-парам (RU): "
              "по каждой паре отдельная строка. Если контекста мало — 'фон спокойный; обычный режим'.")
    lines = ["Сформируй короткие заметки по парам: " + pairs_list, "", "Данные по парам:"]
    for p, ctx in pairs_state.items():
        lines.append(f"- {p}: {ctx or 'нет свежего контекста'}")
    user = ("\n".join(lines)
            + "\n\nФормат ответа строго:\nUSD/JPY — <заметка>\nAUD/USD — <заметка>\nEUR/USD — <заметка>\nGBP/USD — <заметка>")
    return _respond("gpt-5", system, user, max_output_tokens=400, temperature=0.3)

def deep_analysis(question: str, context: str = "") -> str:
    system = ("Ты аналитик-объяснитель: делай структурированные и практичные выводы. "
              "Сначала краткий вывод (1–2 предложения), затем маркированные пункты.")
    user = f"Вопрос:\n{question}\n\nКонтекст:\n{context}".strip()
    return _respond("gpt-5", system, user, max_output_tokens=800, temperature=0.4)

async def llm_ping() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        txt = await _respond_async("gpt-5", None, "ping", max_output_tokens=8, temperature=None)
        return bool(txt.strip())
    except Exception:
        return False

async def generate_digest(symbols: List[str], model: Optional[str] = None, token_budget: int = 30000) -> str:
    mdl = "gpt-5"
    max_out = max(200, min(500, (token_budget // 50) if token_budget else 400))
    pretty = {"USDJPY":"USD/JPY","AUDUSD":"AUD/USD","EURUSD":"EUR/USD","GBPUSD":"GBP/USD"}
    ordered = [s for s in (symbols or []) if isinstance(s, str)] or ["USDJPY","AUDUSD","EURUSD","GBPUSD"]
    want_lines = "\n".join(f"{pretty.get(s, s[:3]+'/'+s[3:])} — <заметка>" for s in ordered)
    system = ("Ты опытный валютный аналитик. Сделай короткий дайджест на русском, "
              "по одной строке на каждую пару. Пиши по делу: драйверы, риски, режим. Без воды.")
    user = "Сформируй дайджест по парам (без цен): " + ", ".join(ordered) + "\n\nФормат ответа строго:\n" + want_lines
    try:
        text = await _respond_async(mdl, system, user, max_output_tokens=max_out, temperature=0.3)
    except Exception as e:
        return f"LLM ошибка: {e}"

    lines: List[str] = []
    by_symbol = {s: None for s in ordered}
    for raw in (text or "").splitlines():
        line = raw.strip(" \t-")
        for s in ordered:
            tag = pretty.get(s, s[:3]+"/"+s[3:])
            if line.upper().startswith(tag.upper()):
                by_symbol[s] = line
                break
    for s in ordered:
        lines.append(by_symbol[s] or f"{pretty.get(s, s[:3]+'/'+s[3:])} — фон спокойный; обычный режим")
    return "\n".join(lines)

async def explain_pair_event(pair: str, headline: str, origin: str = "", lang: str = "ru", model: Optional[str] = None) -> str:
    mdl = "gpt-5"
    pair = (pair or "").upper().strip()
    origin = (origin or "").lower().strip()
    pretty = {"USDJPY":"USD/JPY","AUDUSD":"AUD/USD","EURUSD":"EUR/USD","GBPUSD":"GBP/USD"}
    pair_disp = pretty.get(pair, f"{pair[:3]}/{pair[3:]}")

    system = ("Ты валютный аналитик. Пиши кратко и по делу, на русском. "
              "Цель — объяснить потенциальное направление пары от новости. "
              "Никаких цен, эмодзи, вероятностей и дисклеймеров. 1–2 короткие фразы.")
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

    text = await _respond_async(mdl, system, user, max_output_tokens=140, temperature=0.2)
    text = " ".join((text or "").split())
    if not text:
        # жёсткий статический фоллбэк
        return ("Жёстче ожиданий усиливает валюту источника и сдвигает пару в соответствующем направлении; "
                "мягче ожиданий — наоборот.")
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
