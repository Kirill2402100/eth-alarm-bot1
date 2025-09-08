# llm_client.py
from __future__ import annotations
import os
from typing import Optional, Dict, Any, List

from openai import OpenAI

# --------------- Конфиг ---------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_NANO  = os.getenv("LLM_NANO",  "gpt-4o-mini")   # твой env может быть gpt-5-nano — оставляю как есть
LLM_MINI  = os.getenv("LLM_MINI",  "gpt-4o-mini")
LLM_MAJOR = os.getenv("LLM_MAJOR", "gpt-4.1")       # опционально

TOKEN_BUDGET_PER_DAY = int(os.getenv("LLM_TOKEN_BUDGET_PER_DAY", "30000") or "30000")

_client: Optional[OpenAI] = None

def _client_ok() -> bool:
    return bool(OPENAI_API_KEY)

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

# --------------- Универсальный вызов ---------------

def _responses_call(model: str, prompt: str, max_output_tokens: Optional[int] = None) -> str:
    """
    Без temperature/max_tokens: некоторые 5-е модели кидают 400 на эти поля.
    Используем Responses API (SDK v1).
    """
    if not _client_ok():
        raise RuntimeError("OPENAI_API_KEY not set")

    client = _get_client()

    # Стараемся не слать пустые строки
    text = (prompt or "").strip()
    if not text:
        raise ValueError("LLM prompt is empty")

    kwargs: Dict[str, Any] = {
        "model": model,
        "input": text,
    }
    if max_output_tokens is not None:
        # у 5-й линейки корректное поле — max_output_tokens
        kwargs["max_output_tokens"] = int(max_output_tokens)

    try:
        r = client.responses.create(**kwargs)
        # Ответ может лежать в output_text, либо в первом контенте
        if hasattr(r, "output_text") and r.output_text:
            return r.output_text.strip()

        # fallback разбор
        for out in getattr(r, "output", []) or []:
            if getattr(out, "type", "") == "message":
                parts = getattr(out, "content", []) or []
                for p in parts:
                    if getattr(p, "type", "") == "output_text":
                        val = getattr(p, "text", "") or ""
                        if val.strip():
                            return val.strip()

        # ещё один мягкий fallback
        txt = str(r)
        return txt.strip() if txt.strip() else ""
    except Exception as e:
        # пробрасываем наверх — пусть вызывающая сторона красиво оформит
        raise

# --------------- Вспомогательные высокоуровневые функции ---------------

def nano_headlines_flags(pair: str, headlines: List[str]) -> Dict[str, Any]:
    """
    Анализ заголовков на 'машинном' уровне (дёшево).
    Возвращает JSON-флаг в виде словаря. Если не получилось — пустой словарь.
    """
    prompt = (
        "Ты помогаешь трейдеру по Форексу. Даны короткие заголовки новостей по инструменту {pair}.\n"
        "Верни КОМПАКТНЫЙ JSON (без лишнего текста) с ключами:\n"
        "risk_level: one of [OK, CAUTION, HIGH];\n"
        "bias: one of [both, long-only, short-only];\n"
        "horizon_min: int (оценка горизонта в минутах),\n"
        "confidence: 0..1 (float, два знака);\n"
        "reasons: короткий массив строк с 1-3 причинами.\n"
        "Заголовки:\n- " + "\n- ".join(headlines[:12])
    ).format(pair=pair)

    try:
        txt = _responses_call(LLM_NANO, prompt, max_output_tokens=450)
        import json
        # иногда модель может прислать текст до/после — попытаемся вычленить JSON
        start = txt.find("{")
        end = txt.rfind("}")
        if start != -1 and end > start:
            j = json.loads(txt[start:end + 1])
            if isinstance(j, dict):
                return j
    except Exception:
        pass
    return {}

def mini_digest_ru(pairs: List[str], flags: Dict[str, Dict[str, Any]]) -> str:
    """
    «Человеческий» дайджест на русском — коротко и по делу.
    pairs: список символов вроде ["USDJPY","AUDUSD","EURUSD","GBPUSD"]
    flags: словарь флагов на пару, как вернул nano_headlines_flags (или пусто).
    """
    lines = []
    for p in pairs:
        f = flags.get(p, {}) or {}
        risk = str(f.get("risk_level", "OK")).upper()
        bias = str(f.get("bias", "both")).lower()
        conf = f.get("confidence", "")
        rs = f.get("reasons", []) or []
        bullet = "• " + "; ".join(str(x) for x in rs[:3]) if rs else "• Без заметных факторов."
        bias_txt = {
            "both": "направление не фиксируем",
            "long-only": "смещение: long-bias",
            "short-only": "смещение: short-bias",
        }.get(bias, "направление не фиксируем")

        icon = "✅" if risk == "OK" else ("⚠️" if risk == "CAUTION" else "🚨")
        head = f"{p} — {icon} {risk}"
        tail = f"{bullet}\nЧто делаем: {bias_txt}."
        if conf:
            tail += f" (уверенность {float(conf):.2f})"
        lines.append(head)
        lines.append(tail)
        lines.append("")  # пустая строка-разделитель

    prompt = (
        "Собери короткую русскоязычную сводку из следующих пунктов. "
        "Максимум 6–8 предложений, деловой стиль, без воды, без списков рекомендаций по входам. "
        "Не добавляй префиксы вроде 'Итог:' — просто текст.\n\n"
        + "\n".join(lines)
    )

    try:
        txt = _responses_call(LLM_MINI, prompt, max_output_tokens=700)
        return txt.strip() or "Сводка: без существенных изменений."
    except Exception as e:
        return f"LLM недоступен: {e}"
