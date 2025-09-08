# llm_client.py
# Лёгкий клиент к OpenAI Responses API без temperature/max_tokens,
# чтобы не ловить "unsupported parameter".

from __future__ import annotations
import os
import httpx
import asyncio
from typing import List

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}

class LLMError(RuntimeError):
    pass


async def _post_json(path: str, payload: dict) -> dict:
    if not OPENAI_API_KEY:
        raise LLMError("OPENAI_API_KEY is empty")

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{OPENAI_BASE_URL}{path}", headers=HEADERS, json=payload)
        # Пробуем вытащить тело при ошибке для логов
        if resp.status_code >= 400:
            try:
                data = resp.json()
            except Exception:
                data = {"text": await resp.aread()}
            raise LLMError(f"HTTP {resp.status_code}: {data}")
        return resp.json()


def _build_digest_prompt(symbols: List[str]) -> str:
    pairs_ru = {
        "USDJPY": "USD/JPY",
        "AUDUSD": "AUD/USD",
        "EURUSD": "EUR/USD",
        "GBPUSD": "GBP/USD",
    }
    lines = [
        "Ты — финансовый аналитик. Дай короткий человеческий дайджест по валютным парам.",
        "Формат: по одной строке на каждую пару, максимум 20–25 слов.",
        "Стиль — простой русский, без жаргона. Если явных факторов нет, пиши «фон спокойный; обычный режим».",
        "Не выдумывай фактов и дат. Если контекст неочевиден — давай нейтральную, осторожную формулировку.",
        "",
        "Пары для дайджеста:",
    ]
    for s in symbols:
        lines.append(f"- {pairs_ru.get(s, s)}")
    lines += [
        "",
        "Выводи ТОЛЬКО строки дайджеста без префейсов и пояснений.",
    ]
    return "\n".join(lines)


def _extract_output_text(data: dict) -> str:
    """
    Responses API (2024+): у ответа есть поле output_text.
    Делаем аккуратный парсинг на случай изменений.
    """
    if isinstance(data, dict):
        if "output_text" in data and isinstance(data["output_text"], str):
            return data["output_text"].strip()

        # Иногда бывает структура с 'output' -> list of blocks
        out = data.get("output")
        if isinstance(out, list):
            chunks = []
            for item in out:
                # text block
                t = item.get("content") if isinstance(item, dict) else None
                if isinstance(t, str):
                    chunks.append(t)
            if chunks:
                return "\n".join(chunks).strip()

    # Последняя попытка: весь json строкой
    return str(data)


async def llm_ping() -> bool:
    """Простой «пинг» — проверяем наличие ключа. Этого достаточно для /diag."""
    return bool(OPENAI_API_KEY)


async def generate_digest(
    symbols: List[str],
    model: str = "gpt-5-mini",
    token_budget: int = 1200,
) -> str:
    """
    Генерирует короткий дайджест по списку валютных пар.
    Без temperature и без max_tokens — только max_completion_tokens,
    чтобы не ловить ошибки «unsupported parameter».
    """
    prompt = _build_digest_prompt(symbols)
    payload = {
        "model": model,
        "input": prompt,
        # ограничиваем ответ: много не нужно, и это совместимо с Responses API
        "max_completion_tokens": min(500, max(64, token_budget // 10)),
    }
    data = await _post_json("/responses", payload)
    return _extract_output_text(data)
