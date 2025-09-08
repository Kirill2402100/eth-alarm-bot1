# llm_client.py
import os, json, logging
from typing import Dict, List, Any

log = logging.getLogger("fund_bot.llm")

# OpenAI SDK v1.x
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("OpenAI SDK is not installed. Add `openai>=1.30.0` to requirements.") from e

_CLIENT = None

def _client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _CLIENT

def _is_responses_model(model: str) -> bool:
    m = (model or "").lower()
    # эвристика: всё новое (gpt-5*, 4.1*, omni*) чаще требует Responses API
    return m.startswith("gpt-5") or m.startswith("gpt-4.1") or "omni" in m

def _resp_text(resp: Any) -> str:
    """Достаём текст из Responses API независимо от версии SDK."""
    # у новых версий есть .output_text
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt
    # совместимость: достанем руками
    try:
        parts = []
        for item in resp.output:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", "") == "output_text" and getattr(c, "text", None):
                    parts.append(c.text)
        return "".join(parts).strip()
    except Exception:
        return ""

def _strip_json(text: str) -> str:
    """Аккуратно вынимаем JSON из ответа (допускаем обёртки, код-блоки)."""
    if not text:
        return ""
    t = text.strip()
    # ```json ... ```
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:].strip()
    # вырезаем всё до первой { и после последней }
    if "{" in t and "}" in t:
        t = t[t.index("{"): t.rindex("}")+1]
    return t

# ===================== NANO (флаги) =====================

def analyze_headlines_nano(model: str, headlines_by_ccy: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Возвращает JSON вида:
    {
      "USDJPY": {"risk":"Green/Amber/Red", "bias":"neutral/long-only/short-only", "horizon_h":12, "confidence":0.0..1.0},
      ...
    }
    """
    assert headlines_by_ccy, "Empty headlines"
    client = _client()

    ccys = ", ".join(headlines_by_ccy.keys())
    sys_prompt = (
        "Ты фунд-ассистент. На входе заголовки новостей по валютам. "
        "Верни ЧИСТЫЙ JSON без пояснений. Ключи — тикеры (USDJPY, AUDUSD, EURUSD, GBPUSD). "
        "Для каждой валюты оцени: risk (Green/Amber/Red), bias (neutral/long-only/short-only), "
        "horizon_h (целое, часы 1..48), confidence (0..1)."
    )
    user_lines = []
    for k, items in headlines_by_ccy.items():
        if not items: 
            continue
        items = list(dict.fromkeys(items))[:20]  # дедуп+лимит
        user_lines.append(f"{k}:\n- " + "\n- ".join(items))
    user_prompt = f"Валюты: {ccys}\n\nЗаголовки:\n" + "\n\n".join(user_lines) + "\n\nВерни только JSON."

    if _is_responses_model(model):
        # Responses API — без temperature; max_output_tokens вместо max_tokens
        try:
            resp = client.responses.create(
                model=model,
                input=[{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}],
                max_output_tokens=400,
            )
            txt = _resp_text(resp)
            return json.loads(_strip_json(txt) or "{}")
        except Exception as e:
            # повтор без max_output_tokens на случай “unsupported parameter”
            log.warning("Responses nano: retry without max_output_tokens due to: %s", e)
            resp = client.responses.create(
                model=model,
                input=[{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}],
            )
            txt = _resp_text(resp)
            return json.loads(_strip_json(txt) or "{}")
    else:
        # Chat Completions
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content":sys_prompt},
                    {"role":"user","content":user_prompt},
                ],
                max_tokens=400,
                temperature=0,  # детерминированный JSON
                response_format={"type":"json_object"},
            )
        except TypeError:
            # если модель не поддерживает response_format — уберём
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content":sys_prompt},
                    {"role":"user","content":user_prompt},
                ],
                max_tokens=400,
                temperature=0,
            )
        txt = (resp.choices[0].message.content or "").strip()
        return json.loads(_strip_json(txt) or "{}")

# ===================== MINI (дайджест) =====================

def daily_digest_mini(model: str, facts: Dict[str, Any]) -> str:
    """
    facts: {
      "USDJPY": {"calendar_today":[...], "notes":[...], "market":{"atr_z":..., "spread_bp":...}},
      ...
    }
    Возвращает короткий русский дайджест по 4 парам.
    """
    client = _client()

    sys_prompt = (
        "Напиши краткий человеческий дайджест на русском по парам USDJPY/AUDUSD/EURUSD/GBPUSD. "
        "Структура: на каждую пару 2–3 пункта (сильные/слабые факторы), затем строка 'Что делаем: ...' "
        "на уровне практических мер (окна тишины, смещение bias и т.п.). Без преамбулы и заключения."
    )
    user_prompt = "Факты JSON ниже. Если данных по паре мало, пиши нейтрально.\n\n" + json.dumps(facts, ensure_ascii=False)

    if _is_responses_model(model):
        try:
            resp = client.responses.create(
                model=model,
                input=[{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}],
                max_output_tokens=700,
            )
            return _resp_text(resp).strip()
        except Exception as e:
            log.warning("Responses mini: retry without max_output_tokens due to: %s", e)
            resp = client.responses.create(
                model=model,
                input=[{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}],
            )
            return _resp_text(resp).strip()
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":sys_prompt},
                {"role":"user","content":user_prompt},
            ],
            max_tokens=700,
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()
