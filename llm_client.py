from __future__ import annotations
import os, json, logging
from typing import List, Dict, Optional
from openai import OpenAI

log = logging.getLogger("llm")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=OPENAI_API_KEY or None)

def _responses_text(r) -> str:
    # 1) простой путь
    t = getattr(r, "output_text", None)
    if t:
        return t.strip()
    # 2) извлечь куски контента
    try:
        parts = []
        for item in (getattr(r, "output", None) or []):
            for c in (getattr(item, "content", None) or []):
                if getattr(c, "text", None):
                    parts.append(c.text)
        return "".join(parts).strip()
    except Exception:
        return str(r)

def _call_responses(model: str, prompt: str, temperature: float, max_new_tokens: int) -> str:
    err1 = None
    # Вариант A: max_completion_tokens
    try:
        r = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            max_completion_tokens=max_new_tokens,
        )
        return _responses_text(r)
    except Exception as e:
        err1 = e
        log.info(f"responses(max_completion_tokens) → {e}")

    # Вариант B: max_output_tokens
    try:
        r = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_new_tokens,
        )
        return _responses_text(r)
    except Exception as e2:
        log.info(f"responses(max_output_tokens) → {e2}")
        # больше НЕ пробуем chat.completions для gpt-5 — вернём осмысленную ошибку
        raise RuntimeError(f"Ошибка обращения к LLM: {e2}") from err1

def analyze_headlines_json(pair: str, headlines: List[str], model_nano: str) -> Dict:
    prompt = (
        "Проанализируй заголовки новостей по инструменту {pair} и верни ТОЛЬКО JSON с полями: "
        "level ∈ {OK, CAUTION, HIGH}, bias ∈ {BOTH, LONG, SHORT}, horizon_h (целое), confidence (0..1). "
        "Никакого текста вне JSON.\n\n"
        f"Заголовки:\n- " + "\n- ".join(headlines[:12])
    )
    raw = _call_responses(model_nano, prompt, temperature=0.1, max_new_tokens=200)
    try:
        j = json.loads(raw)
        return {
            "level": str(j.get("level", "OK")).upper(),
            "bias": str(j.get("bias", "BOTH")).upper(),
            "horizon_h": int(j.get("horizon_h", 12)),
            "confidence": float(j.get("confidence", 0.5)),
        }
    except Exception:
        return {"level": "OK", "bias": "BOTH", "horizon_h": 12, "confidence": 0.5}

async def make_digest_ru(pairs: List[str], model: str, nano_model: Optional[str] = None) -> str:
    prompt = (
        "Ты — опытный FX-аналитик. Напиши компактный дайджест на русском по инструментам: "
        f"{', '.join(pairs)}. Формат: для каждой пары 1–2 строки: текущий фон и действие трейд-бота "
        "(окна тишины/смещение/ограничения или «обычный режим»). Избегай воды и штампов. "
        "Если фактов нет — пиши аккуратно «без критичных новостей». "
        "Верни чистый текст (параграфы и маркеры), без ссылок."
    )
    text = _call_responses(model, prompt, temperature=0.4, max_new_tokens=600)
    return (text or "").strip() or "Ежедневный фон: без критичных новостей."
