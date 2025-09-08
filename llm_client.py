import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI

# Модели из ENV (заданы у тебя в Railway)
MODEL_NANO = os.getenv("LLM_NANO", "gpt-5-nano")
MODEL_MINI = os.getenv("LLM_MINI", "gpt-5-mini")
MODEL_MAJOR = os.getenv("LLM_MAJOR", "gpt-5")  # опционально

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TOKEN_BUDGET_PER_DAY = int(os.getenv("LLM_TOKEN_BUDGET_PER_DAY", "30000"))

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Примитивный лимитер по токенам на день (перезапускается при рестарте процесса)
_usage_state = {"date": None, "input_tokens": 0, "output_tokens": 0}


def _roll_usage_if_new_day():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if _usage_state.get("date") != today:
        _usage_state["date"] = today
        _usage_state["input_tokens"] = 0
        _usage_state["output_tokens"] = 0


def _budget_okay(next_tokens: int) -> bool:
    _roll_usage_if_new_day()
    # грубо проверяем, хватит ли бюджета
    return (_usage_state["input_tokens"] + _usage_state["output_tokens"] + next_tokens) <= TOKEN_BUDGET_PER_DAY


def _accumulate_usage(usage: Optional[Dict[str, Any]]):
    if not usage:
        return
    _roll_usage_if_new_day()
    _usage_state["input_tokens"] += int(usage.get("prompt_tokens", 0))
    _usage_state["output_tokens"] += int(usage.get("completion_tokens", 0))


async def _chat_json(model: str, system: str, user_obj: Dict[str, Any], max_tokens: int = 400) -> Dict[str, Any]:
    """
    Вызов чата с принудительным JSON-ответом.
    """
    approx = max_tokens + 500  # грубая оценка токенов входа+выхода
    if not _budget_okay(approx):
        return {"error": "token_budget_exceeded"}

    resp = await _client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)}
        ],
        max_tokens=max_tokens,
    )
    _accumulate_usage(getattr(resp, "usage", None))

    content = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(content)
    except Exception:
        return {"raw": content}


async def _chat_text(model: str, system: str, user_obj: Dict[str, Any], max_tokens: int = 400, temperature: float = 0.3) -> str:
    """
    Вызов чата с текстовым ответом.
    """
    approx = max_tokens + 500
    if not _budget_okay(approx):
        return "⚠️ Лимит на сегодня исчерпан. Коротко: всё стабильно; подробности позже."

    resp = await _client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)}
        ],
        max_tokens=max_tokens,
    )
    _accumulate_usage(getattr(resp, "usage", None))
    return (resp.choices[0].message.content or "").strip()


# ---------- Публичные функции ----------

async def summarize_pair_ru(pair: str, flags: Dict[str, Any], headlines: Optional[List[str]] = None) -> str:
    """
    Короткая «человеческая» сводка на РУССКОМ для одной валютной пары.
    flags: {"risk":"Green|Amber|Red","bias":"neutral|long-only|short-only","reserve_off":bool,"dca_scale":float,...}
    headlines: список заголовков (может быть None/пусто).
    """
    headlines = headlines or []
    sys = (
        "Ты опытный FX/макро-аналитик. Пиши кратко, по-русски, 3–5 предложений, без канцелярита. "
        "Структура: 1) статус (эмодзи + одно предложение), 2) что важно сегодня (календарь/заголовки), "
        "3) что делаем в торговом боте (окна тишины/резерв/смещение), 4) одно предложение про общий фон. "
        "Избегай жаргона; термины поясняй коротко в скобках."
    )
    user = {
        "pair": pair,
        "flags": flags,
        "headlines": headlines[:10],
        "task": "Собери русскую сводку по указанной структуре."
    }
    return await _chat_text(MODEL_MINI, sys, user, max_tokens=320, temperature=0.25)


async def classify_headlines_nano(headlines: List[str]) -> Dict[str, Any]:
    """
    Дёшево оценить заголовки: риск-скор 0..1, bias (long/short/neutral), горизонт (часы), уверенность 0..1.
    """
    sys = (
        "Ты классификатор новостей по FX. Верни JSON с полями: "
        "risk_score (0..1), bias ('long'|'short'|'neutral'), horizon_hours (int), confidence (0..1), reasons (array[str]). "
        "Коротко и без воды."
    )
    user = {"headlines": headlines[:24]}
    return await _chat_json(MODEL_NANO, sys, user, max_tokens=260)


async def deep_escalation_ru(pair: str, context_text: str) -> str:
    """
    Глубокий разбор редких событий (ручная кнопка). Использует старшую модель (если задана).
    """
    sys = (
        "Старший макроаналитик FX. Подробно и чётко, но без воды. Пиши по-русски. "
        "Сделай 3 блока: Резюме, Риски/сценарии (1–2 горизонта), Что делать трейд-боту."
    )
    user = {"pair": pair, "context": context_text}
    model = MODEL_MAJOR or MODEL_MINI
    return await _chat_text(model, sys, user, max_tokens=900, temperature=0.2)


def llm_usage_today() -> Dict[str, int]:
    _roll_usage_if_new_day()
    return {
        "date": _usage_state["date"],
        "input_tokens": _usage_state["input_tokens"],
        "output_tokens": _usage_state["output_tokens"],
        "budget": TOKEN_BUDGET_PER_DAY,
    }
