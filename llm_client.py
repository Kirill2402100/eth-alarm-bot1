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
    """Синхронный вызов Chat Completions."""
    client = _client_singleton()

    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.2,
        "seed": 1,
    }
    # gpt-5 / gpt-4.1+ требуют max_completion_tokens
    if str(model).startswith(("gpt-5", "o4", "gpt-4.1")):
        params["max_completion_tokens"] = max_completion_tokens
    else:
        params["max_tokens"] = max_completion_tokens

    resp = client.chat.completions.create(**params)

    # Робастное извлечение текста
    text = ""
    try:
        choice = resp.choices[0]
        msg = getattr(choice, "message", None)
        if getattr(msg, "tool_calls", None):  # если модель решила вызвать tool
            return ""  # заставим сработать fallback в вызывающем коде
        if msg is not None:
            c = getattr(msg, "content", None)
            if isinstance(c, str):
                text = c
            elif isinstance(c, list):  # на всякий случай, если SDK вернёт списком частей
                text = "".join([getattr(part, "text", "") or "" for part in c]).strip()
    except Exception:
        text = ""
    # Усекаем слишком длинные ответы
    return (text or "").strip()[:1200]


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

def _origin_profile(origin: str) -> dict:
    o = (origin or "").lower()
    if "japan" in o or "boj" in o:
        return {
            "ccy": "JPY",
            "consensus": "ставку оставят как есть",
            "hawk": "поднимут ставку или усилят ограничения по доходности облигаций; сократят покупки госбумаг",
            "dove": "дадут понять о паузе или снижении ставки; ослабят ограничения; увеличат покупки госбумаг",
        }
    if "australia" in o or "rba" in o:
        return {
            "ccy": "AUD",
            "consensus": "ставку оставят как есть",
            "hawk": "скажут, что бороться с инфляцией будут жёстче или поднимут ставку",
            "dove": "намекнут на паузу или снижение ставки; мягкий тон",
        }
    if "ecb" in o or "euro" in o:
        return {
            "ccy": "EUR",
            "consensus": "ставку без изменений",
            "hawk": "поднимут ставку или дадут жёсткие сигналы; быстрее сократят покупки облигаций",
            "dove": "дадут мягкие сигналы или намёк на снижение; медленнее будут сворачивать стимулы",
        }
    if "boe" in o or "england" in o or "uk" in o:
        return {
            "ccy": "GBP",
            "consensus": "ставку без изменений",
            "hawk": "поднимут ставку или результаты голосования будут за ужесточение",
            "dove": "намёк на снижение или паузу; мягкий тон",
        }
    return {
        "ccy": "USD",
        "consensus": "ставку без изменений",
        "hawk": "поднимут ставку или ужесточат риторику; быстрее сократят стимулы",
        "dove": "намекнут на паузу или снижение; смягчат риторику",
    }

def _pair_dir_phrase(pair: str, ccy: str, *, stronger: bool) -> str:
    pretty = {"USDJPY": "USD/JPY", "AUDUSD": "AUD/USD", "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD"}
    pair_disp = pretty.get(pair, f"{pair[:3]}/{pair[3:]}")
    if ccy == pair[:3]:  # базовая валюта
        direction = "вверх" if stronger else "вниз"
    else:  # котируемая валюта
        direction = "вниз" if stronger else "вверх"
    return f"{pair_disp} {direction}"

def _two_line_fallback(pair: str, origin: str, headline: str, consensus: Optional[str] = None) -> str:
    prof = _origin_profile(origin)
    cons = (consensus or prof["consensus"]).strip().rstrip(".")
    hawk_dir = _pair_dir_phrase(pair, prof["ccy"], stronger=True)
    dove_dir = _pair_dir_phrase(pair, prof["ccy"], stronger=False)
    line1 = f"Событие: {headline.strip()}. Ожидания: {cons}."
    line2 = (
        f"Что это значит: если ужесточат политику — {hawk_dir}; если смягчат — {dove_dir}. "
        f"Признаки ужесточения: {prof['hawk']}. Признаки смягчения: {prof['dove']}."
    )
    return line1 + "\n" + line2

async def explain_pair_event(
    pair: str,
    headline: str,
    origin: str = "",
    lang: str = "ru",
    model: Optional[str] = None,
    consensus: Optional[str] = None,
) -> str:
    mdl = (model or LLM_MINI).strip()
    pair = (pair or "").upper().strip()
    prof = _origin_profile(origin)
    cons = (consensus or prof["consensus"]).strip().rstrip(".")

    # Подсказки для LLM
    hawk_dir = _pair_dir_phrase(pair, prof["ccy"], stronger=True)
    dove_dir = _pair_dir_phrase(pair, prof["ccy"], stronger=False)
    pretty = {"USDJPY": "USD/JPY", "AUDUSD": "AUD/USD", "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD"}
    pair_disp = pretty.get(pair, f"{pair[:3]}/{pair[3:]}")

    system = (
        "Ты финансовый комментатор для начинающих инвесторов. Пиши очень простыми словами, без жаргона и аббревиатур. "
        "Если термин неизбежен, дай короткое пояснение в скобках: например, 'ставка (цена кредита)'. "
        "Вывод строго в ДВЕ строки:\n"
        "1) 'Событие: <что это>; Ожидания: <на что рынок рассчитывает>.'\n"
        "2) 'Что это значит: если сигналы про ужесточение — <направление пары>; если про смягчение — <направление пары>. "
        "Признаки: <2–3 простых примера для каждого случая>.'\n"
        "Никаких цен, процентов вероятности, эмодзи и дисклеймеров."
    )
    user = (
        f"Пара: {pair_disp}\n"
        f"Источник: {origin or 'unknown'}\n"
        f"Заголовок события: {headline.strip()}\n"
        f"Консенсус-ожидание: {cons}\n"
        f"Признаки ужесточения (простыми словами): {prof['hawk']}\n"
        f"Признаки смягчения (простыми словами): {prof['dove']}\n"
        f"Направления для пары: при ужесточении → {hawk_dir}; при смягчении → {dove_dir}."
    )

    try:
        text = await _chat_completion_async(
            model=mdl, system=system, user=user, max_completion_tokens=200
        )
        # Нормализация + ограничение длины каждой строки
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if len(lines) < 2:
            raise RuntimeError("empty/short model output")
        if not lines[0].lower().startswith("событие:"):
            lines[0] = "Событие: " + lines[0]
        if "Ожидания:" not in lines[0]:
            lines[0] += f" Ожидания: {cons}."
        if not (lines[1].lower().startswith("что это значит:") or lines[1].lower().startswith("импульс:")):
            lines[1] = "Что это значит: " + lines[1]
        lines = [lines[0][:220], lines[1][:240]]
        return lines[0] + "\n" + lines[1]
    except Exception as e:
        log.warning("explain_pair_event: %s; using fallback (pair=%s, origin=%s)", e, pair, origin)
        return _two_line_fallback(pair, origin, headline, consensus=cons)


__all__ = [
    "quick_classify",
    "fx_digest_ru",
    "deep_analysis",
    "generate_digest",
    "llm_ping",
    "explain_pair_event",
]
