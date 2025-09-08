from __future__ import annotations

import os
import json
import logging
from typing import List, Dict, Optional

from openai import OpenAI
from openai.types import CompletionUsage

log = logging.getLogger("llm")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    log.warning("OPENAI_API_KEY не задан — LLM вызовы упадут.")
client = OpenAI(api_key=OPENAI_API_KEY)

# ====== Вспомогательный универсальный вызов ======
def _try_responses_any(model: str, prompt: str, temperature: float = 0.3, max_new_tokens: int = 700) -> str:
    """
    Универсальный вызов, пережёвывающий зоопарк параметров:
    - сначала Responses API с max_completion_tokens
    - затем Responses API с max_output_tokens
    - затем chat.completions с max_tokens
    Возвращает сырой текст.
    """
    # 1) responses + max_completion_tokens
    try:
        r = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            max_completion_tokens=max_new_tokens,
        )
        out = getattr(r, "output_text", None)
        if out:
            return out.strip()
        # иногда text в первом item
        if r.output and len(r.output) and getattr(r.output[0], "content", None):
            chunks = [c.text for c in r.output[0].content if getattr(c, "text", None)]
            return "".join(chunks).strip()
    except Exception as e1:
        msg = str(e1)
        log.info(f"responses(max_completion_tokens) not used: {msg}")

    # 2) responses + max_output_tokens
    try:
        r = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_new_tokens,
        )
        out = getattr(r, "output_text", None)
        if out:
            return out.strip()
        if r.output and len(r.output) and getattr(r.output[0], "content", None):
            chunks = [c.text for c in r.output[0].content if getattr(c, "text", None)]
            return "".join(chunks).strip()
    except Exception as e2:
        msg = str(e2)
        log.info(f"responses(max_output_tokens) not used: {msg}")

    # 3) chat.completions + max_tokens
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e3:
        raise RuntimeError(f"Ошибка обращения к LLM: {e3}")

# ====== ЗАДАЧИ ======

def analyze_headlines_json(
    pair: str,
    headlines: List[str],
    model_nano: str,
) -> Dict:
    """
    Короткий «машинный» анализ: вернуть JSON с полями:
    level ∈ {OK, CAUTION, HIGH}, bias ∈ {BOTH, LONG, SHORT}, horizon_h, confidence 0..1
    """
    prompt = (
        "Проанализируй заголовки новостей по инструменту {pair} и верни ТОЛЬКО JSON с полями:\n"
        "level ∈ {OK, CAUTION, HIGH}, bias ∈ {BOTH, LONG, SHORT}, horizon_h (целое), confidence (0..1).\n"
        "Примеры причин не пиши. Никакого текста вне JSON.\n\n"
        f"Заголовки:\n- " + "\n- ".join(headlines[:12])
    )
    raw = _try_responses_any(model_nano, prompt, temperature=0.1, max_new_tokens=200)
    try:
        j = json.loads(raw)
        return {
            "level": str(j.get("level", "OK")).upper(),
            "bias": str(j.get("bias", "BOTH")).upper(),
            "horizon_h": int(j.get("horizon_h", 12)),
            "confidence": float(j.get("confidence", 0.5)),
        }
    except Exception:
        # fallback «тихий» если что-то не так
        return {"level": "OK", "bias": "BOTH", "horizon_h": 12, "confidence": 0.5}

async def make_digest_ru(
    pairs: List[str],
    model: str,
    nano_model: Optional[str] = None,
) -> str:
    """
    Короткий «человеческий» дайджест на русском для 3–4 валютных пар.
    """
    # В реальности сюда подставим факты/календарь/заголовки; сейчас — шаблон.
    bullet = []
    for p in pairs:
        bullet.append(f"{p} — без критичных новостей, режим обычный.")

    prompt = (
        "Ты — опытный FX-аналитик. Напиши компактный дайджест на русском по инструментам: "
        f"{', '.join(pairs)}. Формат: для каждой пары 1–2 строки: текущий фон и действие трейд-бота "
        "(окна тишины/смещение/ограничения или «обычный режим»). Избегай воды, будь конкретным. "
        "Если фактов нет — пиши аккуратно «без критичных новостей».\n\n"
        "Верни чистый текст для Telegram (параграфы и маркеры), без Markdown-ссылок."
    )
    text = _try_responses_any(model, prompt, temperature=0.4, max_new_tokens=600)

    # Небольшая нормализация (на случай пустого ответа)
    if not text or len(text) < 10:
        text = "Ежедневный фон:\n" + "\n".join("• " + s for s in bullet)

    # Telegram HTML безопаснее — но мы просили обычный текст. Вернём как есть.
    return text.strip()
