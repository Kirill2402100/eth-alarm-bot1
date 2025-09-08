# llm_client.py
import os, asyncio, json

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# ---------- OpenAI ----------
if PROVIDER == "openai":
    from openai import AsyncOpenAI
    _client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None
    )
    async def chat(system: str, user: str, model: str | None = None, json_mode: bool = False) -> str:
        model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = await _client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            **extra
        )
        return resp.choices[0].message.content

# ---------- Gemini ----------
elif PROVIDER == "gemini":
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    async def chat(system: str, user: str, model: str | None = None, json_mode: bool = False) -> str:
        mname = model or os.getenv("LLM_MODEL", "gemini-1.5-flash")
        mdl = genai.GenerativeModel(mname)
        prompt = f"{system}\n\nUser:\n{user}"
        if json_mode: prompt += "\n\nReturn ONLY valid JSON object."
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, lambda: mdl.generate_content(prompt))
        return res.text or ""

# ---------- Anthropic ----------
elif PROVIDER == "anthropic":
    import anthropic
    _client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    async def chat(system: str, user: str, model: str | None = None, json_mode: bool = False) -> str:
        model = model or os.getenv("LLM_MODEL", "claude-3-haiku-20240307")
        msg = await _client.messages.create(
            model=model, max_tokens=1024, system=system,
            messages=[{"role":"user","content":user}]
        )
        return msg.content[0].text

# ---------- Fallback ----------
else:
    async def chat(system: str, user: str, model: str | None = None, json_mode: bool = False) -> str:
        return '{"risk":"Green","bias":"neutral","ttl":60}'
