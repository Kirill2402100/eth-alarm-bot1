# llm_client.py
import os
from typing import Dict, List, Optional
from openai import OpenAI

# ------- Конфигурация моделей из ENV -------
LLM_NANO  = os.getenv("LLM_NANO",  "gpt-5-nano")
LLM_MINI  = os.getenv("LLM_MINI",  "gpt-5-mini")
LLM_MAJOR = os.getenv("LLM_MAJOR", "gpt-5")

DEFAULT_TEMPERATURE = 0.3
# Суточный бюджет токенов можно учитывать снаружи; здесь просто ограничение на один вызов
DEFAULT_MAX_OUT_TOKENS = 500

_client: Optional[OpenAI] = None


def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ["OPENAI_API_KEY"]
        _client = OpenAI(api_key=api_key)
    return _client


def _respond(
    model: str,
    system: str,
    user: str,
    max_output_tokens: int = DEFAULT_MAX_OUT_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """
    Вызов Responses API. ВАЖНО: используем max_output_tokens (а не max_completion_tokens).
    """
    client = _client_singleton()

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

    # Новая SDK отдаёт удобное поле output_text
    text = getattr(resp, "output_text", None)
    if text:
        return text.strip()

    # Фолбэк на явную сборку текста (на всякий случай)
    parts: List[str] = []
    for item in getattr(resp, "output", []) or []:
        # item может быть dict с ключом "content" (list с text/output)
        content = getattr(item, "content", None) or item.get("content") if isinstance(item, dict) else None
        if content and isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
    return "\n".join(parts).strip()


# ---------------- Публичные функции ----------------

def quick_classify(labeling_prompt: str) -> str:
    """
    Быстрые короткие классификации/теги — дешёвая nano-модель.
    """
    system = (
        "Ты коротко и точно классифицируешь вход. "
        "Отвечай одной строкой без лишних слов."
    )
    return _respond(
        model=LLM_NANO,
        system=system,
        user=labeling_prompt,
        max_output_tokens=64,
        temperature=0.2,
    )


def fx_digest_ru(pairs_state: Dict[str, str]) -> str:
    """
    Человеческий краткий дайджест на русском по заданным валютным парам.
    pairs_state: { 'USDJPY': '<контекст/заметки>', ... }
    """
    pairs_list = ", ".join(pairs_state.keys()) if pairs_state else "USDJPY, AUDUSD, EURUSD, GBPUSD"

    system = (
        "Ты опытный финансовый аналитик. Дай сжатый, понятный человеку текст-дайджест "
        "по форекс-парам на русском: по каждой паре отдельная строка. "
        "Без излишних украшательств, максимум пользы. Если контекста мало — напиши 'фон спокойный; обычный режим'."
    )

    lines = ["Сформируй короткие заметки по парам: " + pairs_list, "", "Данные по парам:"]
    for p, ctx in pairs_state.items():
        lines.append(f"- {p}: {ctx or 'нет свежего контекста'}")

    user = "\n".join(lines) + "\n\nФормат ответа строго:\nUSD/JPY — <заметка>\nAUD/USD — <заметка>\nEUR/USD — <заметка>\nGBP/USD — <заметка>"

    return _respond(
        model=LLM_MINI,
        system=system,
        user=user,
        max_output_tokens=400,
        temperature=0.3,
    )


def deep_analysis(question: str, context: str = "") -> str:
    """
    Глубокий разбор по кнопке — старшая модель.
    """
    system = (
        "Ты аналитик-объяснитель: делай структурированные и практичные выводы. "
        "Сначала краткий вывод (1–2 предложения), затем маркированные пункты."
    )
    user = f"Вопрос:\n{question}\n\nКонтекст:\n{context}".strip()
    return _respond(
        model=LLM_MAJOR,
        system=system,
        user=user,
        max_output_tokens=800,
        temperature=0.4,
    )
