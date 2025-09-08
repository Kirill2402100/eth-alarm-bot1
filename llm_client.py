# llm_client.py
# ----------------
# Клиент LLM для дайджеста. Использует OpenAI Responses API.
# Работает только с поддерживаемыми параметрами (model, input, max_output_tokens).

import os
from typing import List, Dict

try:
    from openai import OpenAI
except Exception:  # библиотека не установлена — дадим мягкий фолбэк
    OpenAI = None  # type: ignore

# Модели из переменных окружения
MODEL_MINI = os.environ.get("LLM_MINI", "gpt-5-mini").strip()
MODEL_NANO = os.environ.get("LLM_NANO", "gpt-5-nano").strip()

# Ограничение длины вывода; примерно ~4 слова на 1 токен
MAX_OUT = int(os.environ.get("LLM_MAX_OUT", "700"))  # хватает на 4 пары по 2–3 пункта

PAIR_NAMES: Dict[str, str] = {
    "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD",
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
}


def llm_ready() -> tuple[bool, str]:
    """Лёгкая проверка наличия ключа и клиента."""
    if not os.environ.get("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY not set"
    if OpenAI is None:
        return False, "openai SDK is not installed"
    return True, "ok"


def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "")
    return OpenAI(api_key=key)


def _pair_list_for_prompt(symbols: List[str]) -> str:
    lines = []
    for s in symbols:
        name = PAIR_NAMES.get(s.upper(), f"{s[:3]}/{s[3:]}")
        lines.append(f"- {name}")
    return "\n".join(lines)


def _postprocess_blocks(raw_text: str, ordered_symbols: List[str]) -> str:
    """
    Аккуратно собираем куски в порядке запрошенных символов.
    Если по паре ничего не нашлось — вставляем нейтральный фолбэк.
    """
    text = (raw_text or "").strip()
    if not text:
        return ""

    blocks: list[str] = []

    # Простая стратегия: ищем по подстроке "<b>PAIR</b>"
    for sym in ordered_symbols:
        name = PAIR_NAMES.get(sym, f"{sym[:3]}/{sym[3:]}")
        marker = f"<b>{name}</b>"
        ix = text.find(marker)
        if ix == -1:
            blocks.append(f"<b>{name}</b> — фон спокойный; обычный режим.")
            continue
        # граница следующего блока — следующее появление "<b>"
        jx = text.find("<b>", ix + 3)
        if jx == -1:
            seg = text[ix:].strip()
        else:
            seg = text[ix:jx].strip()
        if not seg:
            seg = f"<b>{name}</b> — фон спокойный; обычный режим."
        blocks.append(seg)

    return "\n\n".join(blocks)


async def make_digest_for_pairs(symbols: List[str]) -> str:
    """
    Возвращает короткий дайджест по 1 или нескольким парам на русском.
    Формат (на пару):
      <b>PAIR</b> — заголовок (1 строка)
      • Факт 1
      • Факт 2
      • Факт 3 (если есть)
      Что делаем: короткое правило.
    """
    ok, msg = llm_ready()
    if not ok:
        raise RuntimeError(f"LLM not ready: {msg}")

    syms = [s.upper() for s in symbols]
    prompt = f"""
Ты — макро/FX аналитик. Напиши краткий русскоязычный дайджест по парам:
{_pair_list_for_prompt(syms)}

Формат для КАЖДОЙ пары строго такой (без лишних префиксов/подзаголовков между парами):

<b>PAIR</b> — краткий заголовок (1 строка)
• Факт 1 (макро/ЦБ/доходности/сырьё/спрэд/вола)
• Факт 2
• Факт 3 (если есть)
Что делаем: короткое правило (напр. «окна тишины к CPI 12:30 UTC», «резерв OFF», «обычный режим»).

Пиши по делу, 2–3 маркера на пару. Если по паре нет важных событий — так и напиши «фон спокойный; обычный режим».
    """.strip()

    client = _client()
    # Responses API — только поддерживаемые поля
    resp = client.responses.create(
        model=MODEL_MINI,
        input=prompt,
        max_output_tokens=MAX_OUT,
    )
    raw = (resp.output_text or "").strip()

    # Приводим к стабильно читаемому формату
    final_text = _postprocess_blocks(raw, syms)
    return final_text or raw or ""
