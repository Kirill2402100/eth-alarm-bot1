# llm_client.py
# -*- coding: utf-8 -*-
import os
import asyncio
import logging
from typing import Dict, List, Optional

from openai import OpenAI

log = logging.getLogger("llm_client")

# ---- Модели (по умолчанию везде gpt-5) ----
LLM_MAJOR: str = os.getenv("LLM_MAJOR", "gpt-5").strip() or "gpt-5"
LLM_MINI:  str = os.getenv("LLM_MINI",  "gpt-5").strip() or "gpt-5"
LLM_NANO:  str = os.getenv("LLM_NANO",  "gpt-5").strip() or "gpt-5"

DEFAULT_MAX_OUT_TOKENS = 500

_client: Optional[OpenAI] = None


def _client_singleton() -> OpenAI:
    """Ленивый клиент OpenAI с ключом из окружения."""
    global _client
    if _client is None:
        api_key = os.environ["OPENAI_API_KEY"]
        _client = OpenAI(api_key=api_key)
        log.info("llm_client: initialized OpenAI Chat client (models: major=%s mini=%s nano=%s)",
                 LLM_MAJOR, LLM_MINI, LLM_NANO)
    return _client


def _chat_completion(
    *,
    model: str,
    system: str,
    user: str,
    max_completion_tokens: int = DEFAULT_MAX_OUT_TOKENS,
) -> str:
    """
    Синхронный вызов Chat Completions.
    Важно: НЕ передаём temperature/max_tokens и т.п. — многие модели gpt-5 их игнорируют.
    """
    client = _client_singleton()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        # только поддерживаемый параметр:
        max_completion_tokens=max_completion_tokens,
    )
    try:
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        text = ""
    return text


async def _chat_completion_async(
    *,
    model: str,
    system: str,
    user: str,
    max_completion_tokens: int = DEFAULT_MAX_OUT_TOKENS,
) -> str:
    return await asyncio.to_thread(
        _chat_completion,
        model=model,
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
    )


# ----------------- High-level helpers -----------------

def quick_classify(labeling_prompt: str) -> str:
    system = "Ты коротко и точно классифицируешь вход. Отвечай одной строкой."
    try:
        return _chat_completion(model=LLM_NANO, system=system, user=labeling_prompt, max_completion_tokens=64)
    except Exception as e:
        log.error("quick_classify error: %s", e)
        return ""


def fx_digest_ru(pairs_state: Dict[str, str]) -> str:
    pairs_list = ", ".join(pairs_state.keys()) if pairs_state else "USDJPY, AUDUSD, EURUSD, GBPUSD"
    system = (
        "Ты опытный финансовый аналитик. Дай сжатый RU-дайджест по форекс-парам: "
        "на каждую пару — одна строка. Если данных мало — 'фон спокойный; обычный режим'."
    )
    lines = ["Сформируй короткие заметки по парам: " + pairs_list, "", "Данные по парам:"]
    for p, ctx in pairs_state.items():
        lines.append(f"- {p}: {ctx or 'нет свежего контекста'}")
    user = (
        "\n".join(lines)
        + "\n\nФормат ответа строго:\nUSD/JPY — <заметка>\nAUD/USD — <заметка>\nEUR/USD — <заметка>\nGBP/USD — <заметка>"
    )
    try:
        return _chat_completion(model=LLM_MINI, system=system, user=user, max_completion_tokens=400)
    except Exception as e:
        log.error("fx_digest_ru error: %s", e)
        return (
            "USD/JPY — фон спокойный; обычный режим\n"
            "AUD/USD — фон спокойный; обычный режим\n"
            "EUR/USD — фон спокойный; обычный режим\n"
            "GBP/USD — фон спокойный; обычный режим"
        )


def deep_analysis(question: str, context: str = "") -> str:
    system = (
        "Ты аналитик-объяснитель: сначала краткий вывод (1–2 предложения), затем маркированные пункты."
    )
    user = f"Вопрос:\n{question}\n\nКонтекст:\n{context}".strip()
    try:
        return _chat_completion(model=LLM_MAJOR, system=system, user=user, max_completion_tokens=800)
    except Exception as e:
        log.error("deep_analysis error: %s", e)
        return "Кратко: данных недостаточно.\n• Собери исходные факты\n• Обнови контекст\n• Повтори запрос"


async def llm_ping() -> bool:
    """Простой чат-пинг модели (true, если получили любой ответ)."""
    try:
        txt = await _chat_completion_async(model=LLM_MINI, system="Answer with a single letter.", user="ping", max_completion_tokens=8)
        return bool(txt.strip())
    except Exception as e:
        log.error("llm_ping error: %s", e)
        return False


async def generate_digest(
    symbols: List[str],
    model: Optional[str] = None,
    token_budget: int = 30000,
) -> str:
    """Короткий трейдерский дайджест: одна строка на пару (RU)."""
    mdl = (model or LLM_MINI).strip()
    max_out = max(200, min(500, (token_budget // 50) if token_budget else 400))
    pretty = {"USDJPY": "USD/JPY", "AUDUSD": "AUD/USD", "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD"}
    ordered = [s for s in (symbols or []) if isinstance(s, str)] or ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]
    want_lines = "\n".join(f"{pretty.get(s, s[:3]+'/'+s[3:])} — <заметка>" for s in ordered)
    system = (
        "Ты опытный валютный аналитик. Сделай короткий дайджест на русском — по одной строке на каждую пару. "
        "Пиши по делу: драйверы, риски, режим. Без воды и цен."
    )
    user = "Сформируй дайджест по парам: " + ", ".join(ordered) + "\n\nФормат ответа строго:\n" + want_lines

    try:
        text = await _chat_completion_async(model=mdl, system=system, user=user, max_completion_tokens=max_out)
    except Exception as e:
        log.error("generate_digest error: %s", e)
        text = ""

    lines_out: List[str] = []
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
            lines_out.append(by_symbol[s])
        else:
            lines_out.append(f"{pretty.get(s, s[:3]+'/'+s[3:])} — фон спокойный; обычный режим")
    return "\n".join(lines_out)


# ----------------- FX event explainer -----------------

def _effect_hint(pair: str, origin: str) -> str:
    p = pair.upper().strip()
    o = (origin or "").lower().strip()
    if p == "USDJPY":
        if o in ("united states", "us", "usa", "fomc", "federal reserve"):
            return "Жёстче ФРС → доллар сильнее → USD/JPY вверх; мягче ФРС → USD/JPY вниз."
        if o in ("japan", "boj", "bank of japan"):
            return "Жёстче BoJ → иена сильнее → USD/JPY вниз; мягче BoJ → USD/JPY вверх."
    if p == "AUDUSD":
        if o in ("australia", "rba", "reserve bank of australia"):
            return "Жёстче РБА → AUD/USD вверх; мягче РБА → AUD/USD вниз."
        if o in ("united states", "us", "usa", "fomc", "federal reserve"):
            return "Жёстче ФРС → доллар сильнее → AUD/USD вниз; мягче ФРС → AUD/USD вверх."
    if p == "EURUSD":
        if o in ("euro area", "eurozone", "ecb", "european central bank"):
            return "Жёстче ЕЦБ → EUR/USD вверх; мягче ЕЦБ → EUR/USD вниз."
        if o in ("united states", "us", "usa", "fomc", "federal reserve"):
            return "Жёстче ФРС → EUR/USD вниз; мягче ФРС → EUR/USD вверх."
    if p == "GBPUSD":
        if o in ("united kingdom", "uk", "boe", "bank of england"):
            return "Жёстче Банк Англии → GBP/USD вверх; мягче → GBP/USD вниз."
        if o in ("united states", "us", "usa", "fomc", "federal reserve"):
            return "Жёстче ФРС → GBP/USD вниз; мягче ФРС → GBP/USD вверх."
    return "Ястребиный исход укрепляет валюту источника; голубиный — ослабляет, пара двигается соответственно."

async def explain_pair_event(
    pair: str,
    headline: str,
    origin: str = "",
    lang: str = "ru",
    model: Optional[str] = None,
) -> str:
    """
    Коротко объясняет, что означает событие для FX-пары.
    Возвращает 1–2 фразы (без эмодзи, цен, вероятностей).
    """
    mdl = (model or LLM_MINI).strip()
    pair = (pair or "").upper().strip()
    pretty = {"USDJPY": "USD/JPY", "AUDUSD": "AUD/USD", "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD"}
    pair_disp = pretty.get(pair, f"{pair[:3]}/{pair[3:]}")

    system = (
        "Ты валютный аналитик. Пиши кратко и по делу, на русском. "
        "Цель — объяснить направление пары от новости. Никаких цен, эмодзи, вероятностей и дисклеймеров. 1–2 короткие фразы."
    )
    lang_note = "на русском" if (lang or "ru").lower().startswith("ru") else "in concise English"
    user = (
        f"Пара: {pair_disp}\n"
        f"Источник (страна/регулятор): {origin or 'unknown'}\n"
        f"Событие/заголовок: {headline.strip()}\n\n"
        f"Требование ({lang_note}):\n"
        f"- Объясни, как ястребиный (жёстче ожиданий) и голубиный (мягче ожиданий) исход влияет на {pair_disp}.\n"
        f"- Учитывай, что усиление валюты стороны события поднимает/опускает пару, "
        f"в зависимости от того, базовая или котируемая она в {pair_disp}.\n"
        f"- Формат: одна-две фразы, без списков."
    )

    try:
        text = await _chat_completion_async(
            model=mdl, system=system, user=user, max_completion_tokens=140
        )
        text = " ".join((text or "").split())
        if not text:
            raise RuntimeError("empty model output")
        return text[:400] if len(text) > 400 else text
    except Exception as e:
        # Детальный лог + умный локальный фолбэк
        log.error("explain_pair_event error: %s (pair=%s, origin=%s, headline=%s)", e, pair, origin, headline)
        return _effect_hint(pair, origin)


__all__ = [
    "quick_classify",
    "fx_digest_ru",
    "deep_analysis",
    "generate_digest",
    "llm_ping",
    "explain_pair_event",
]
