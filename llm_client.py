# llm_client.py
import os, logging
from openai import OpenAI

log = logging.getLogger("fund_bot.llm")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL_MINI = os.getenv("LLM_MINI", "gpt-5-mini")
MODEL_NANO = os.getenv("LLM_NANO", "gpt-5-nano")
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "800"))

def _trim(s: str | None) -> str:
    return (s or "").strip()

def summarize_digest(flags: dict, headlines: dict[str, list[str]] | None = None) -> str:
    """
    Собирает «человеческий» дайджест по 4 парам.
    Всегда возвращает непустой текст (есть фолбэк).
    """
    symbols = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]

    bullets = []
    for sym in symbols:
        f = (flags or {}).get(sym, {})
        bullets.append(f"{sym}: risk={f.get('risk','Green')}, bias={f.get('bias','neutral')}")

    news_text = ""
    if headlines:
        for sym, lst in headlines.items():
            if lst:
                news_text += f"\n{sym}:\n" + "\n".join(f"- {h}" for h in lst[:6]) + "\n"

    prompt = (
        "Сделай короткий утренний дайджест по-русски для 4 FX-пар.\n"
        "Формат: по 2–3 строки на пару: контекст, риски, что делаем (окна тишины, резерв, доборы).\n\n"
        "Флаги:\n" + "\n".join(bullets) +
        ("\n\nЗаголовки:\n" + news_text if news_text else "\n\nЗаголовков нет — напиши базовую нейтральную сводку.")
    )

    resp = client.responses.create(
        model=MODEL_MINI,
        input=prompt,
        # temperature не передаём — для совместимости
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    text = _trim(resp.output_text)
    if not text:
        text = "Новостей мало. Флаги без изменений; режим обычный для всех пар."
    return text
