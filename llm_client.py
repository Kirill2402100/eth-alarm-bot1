# llm_client.py
import os
import asyncio
from typing import Dict, List, Optional
from openai import OpenAI

LLM_NANO: str = os.getenv("LLM_NANO", "gpt-5-nano")
LLM_MINI: str = os.getenv("LLM_MINI", "gpt-5-mini")
LLM_MAJOR: str = os.getenv("LLM_MAJOR", "gpt-5")

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_OUT_TOKENS = 500

_client: Optional[OpenAI] = None

def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ["OPENAI_API_KEY"]
        _client = OpenAI(api_key=api_key)
    return _client

def _create_response(client: OpenAI, **kwargs):
    try:
        return client.responses.create(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        if "temperature" in msg and "not supported" in msg:
            kwargs.pop("temperature", None)
            return client.responses.create(**kwargs)
        raise

def _respond(model: str, system: str, user: str,
             max_output_tokens: int = DEFAULT_MAX_OUT_TOKENS,
             temperature: Optional[float] = DEFAULT_TEMPERATURE) -> str:
    client = _client_singleton()
    kwargs = dict(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_output_tokens=max_output_tokens,
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = _create_response(client, **kwargs)
    text = getattr(resp, "output_text", None)
    if text:
        return str(text).strip()
    parts: List[str] = []
    output = getattr(resp, "output", None)
    if isinstance(output, list):
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                        parts.append(c.get("text", ""))
    return "\n".join(parts).strip()

async def _respond_async(model: str, system: str, user: str,
                         max_output_tokens: int = DEFAULT_MAX_OUT_TOKENS,
                         temperature: Optional[float] = DEFAULT_TEMPERATURE) -> str:
    return await asyncio.to_thread(
        _respond, model, system, user, max_output_tokens, temperature
    )

def quick_classify(labeling_prompt: str) -> str:
    system = "Ты коротко и точно классифицируешь вход. Отвечай одной строкой."
    return _respond(model=LLM_NANO, system=system, user=labeling_prompt,
                    max_output_tokens=64, temperature=0.2)

def fx_digest_ru(pairs_state: Dict[str, str]) -> str:
    pairs_list = ", ".join(pairs_state.keys()) if pairs_state else "USDJPY, AUDUSD, EURUSD, GBPUSD"
    system = (
        "Ты опытный финансовый аналитик. Дай сжатый текст-дайджест по форекс-парам (RU): "
        "по каждой паре отдельная строка. Если контекста мало — 'фон спокойный; обычный режим'."
    )
    lines = ["Сформируй короткие заметки по парам: " + pairs_list, "", "Данные по парам:"]
    for p, ctx in pairs_state.items():
        lines.append(f"- {p}: {ctx or 'нет свежего контекста'}")
    user = "\n".join(lines) + "\n\nФормат ответа строго:\nUSD/JPY — <заметка>\nAUD/USD — <заметка>\nEUR/USD — <заметка>\nGBP/USD — <заметка>"
    return _respond(model=LLM_MINI, system=system, user=user, max_output_tokens=400, temperature=0.3)

def deep_analysis(question: str, context: str = "") -> str:
    system = ("Ты аналитик-объяснитель: делай структурированные и практичные выводы. "
              "Сначала краткий вывод (1–2 предложения), затем маркированные пункты.")
    user = f"Вопрос:\n{question}\n\nКонтекст:\n{context}".strip()
    return _respond(model=LLM_MAJOR, system=system, user=user, max_output_tokens=800, temperature=0.4)

async def llm_ping() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    client = _client_singleton()
    models_to_try = [LLM_MINI, LLM_MAJOR, LLM_NANO]
    async def _try(mdl: str) -> bool:
        try:
            await asyncio.to_thread(_create_response, client, model=mdl, input="ping", max_output_tokens=8)
            return True
        except Exception:
            pass
        try:
            await asyncio.to_thread(_create_response, client, model=mdl, input=[
                {"role": "system", "content": "Проверка связи. Ответь одной буквой."},
                {"role": "user", "content": "ping"},
            ], max_output_tokens=8)
            return True
        except Exception:
            return False
    for mdl in models_to_try:
        if await _try(mdl):
            return True
    return True

async def generate_digest(symbols: List[str], model: Optional[str] = None, token_budget: int = 30000) -> str:
    mdl = (model or LLM_MINI).strip()
    max_out = max(200, min(500, (token_budget // 50) if token_budget else 400))
    pretty = {"USDJPY": "USD/JPY", "AUDUSD": "AUD/USD", "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD"}
    ordered = [s for s in (symbols or []) if isinstance(s, str)] or ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]
    want_lines = "\n".join(f"{pretty.get(s, s[:3]+'/'+s[3:])} — <заметка>" for s in ordered)
    system = ("Ты опытный валютный аналитик. Сделай короткий дайджест на русском, "
              "по одной строке на каждую пару. Пиши по делу: драйверы, риски, режим. Без воды.")
    user = ("Сформируй дайджест по парам (без цен): " + ", ".join(ordered) + "\n\nФормат ответа строго:\n" + want_lines)
    try:
        text = await _respond_async(model=mdl, system=system, user=user, max_output_tokens=max_out, temperature=0.3)
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

__all__ = ["quick_classify", "fx_digest_ru", "deep_analysis", "generate_digest", "llm_ping"]
