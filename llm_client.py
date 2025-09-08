# llm_client.py — лёгкая обёртка под OpenAI
import os, httpx, json

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY","")
OPENAI_BASE = os.environ.get("OPENAI_BASE","https://api.openai.com/v1")
MODEL_MINI = os.environ.get("LLM_MINI","gpt-4o-mini")
MODEL_NANO = os.environ.get("LLM_NANO","gpt-4o-mini")  # можно такой же, если nano нет

headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"}

async def summarize_headlines(headlines_by_ccy: dict) -> dict:
    """
    headlines_by_ccy = {"JPY": ["headline1", ...], "AUD": [...], ...}
    Возвращает JSON-флаги по каждой валюте.
    """
    if not OPENAI_API_KEY:
        return {}
    sys = (
        "You are an assistant that labels FX news by currency with risk level "
        "(OK/CAUTION/HIGH), bias (short-bias/long-bias/neutral), horizon (hours) "
        "and confidence (0..1). Answer JSON."
    )
    user = "Headlines:\n" + json.dumps(headlines_by_ccy, ensure_ascii=False)
    body = {
        "model": MODEL_NANO,
        "messages": [{"role":"system","content":sys},{"role":"user","content":user}],
        "temperature": 0.2,
        "response_format": {"type":"json_object"},
    }
    async with httpx.AsyncClient(timeout=30.0) as cli:
        r = await cli.post(f"{OPENAI_BASE}/chat/completions", headers=headers, json=body)
        r.raise_for_status()
        out = r.json()
        try:
            return json.loads(out["choices"][0]["message"]["content"])
        except Exception:
            return {}

async def daily_brief(text: str) -> str:
    if not OPENAI_API_KEY:
        return ""
    sys = "Write a concise Russian morning FX brief for 4 pairs (USDJPY, AUDUSD, EURUSD, GBPUSD)."
    body = {
        "model": MODEL_MINI,
        "messages": [{"role":"system","content":sys},{"role":"user","content":text}],
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=30.0) as cli:
        r = await cli.post(f"{OPENAI_BASE}/chat/completions", headers=headers, json=body)
        r.raise_for_status()
        out = r.json()
        return out["choices"][0]["message"]["content"].strip()
