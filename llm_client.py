import os, asyncio
from typing import Dict, Any
from openai import OpenAI, BadRequestError

_CLIENT = None

def _client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _CLIENT

LLM_MINI = os.environ.get("LLM_MINI", "gpt-5-mini")
LLM_NANO = os.environ.get("LLM_NANO", "gpt-5-nano")
LLM_MAJOR = os.environ.get("LLM_MAJOR", "gpt-5")

async def _create_responses(model: str, prompt: str, max_out: int = 600) -> str:
    """
    Основной путь: Responses API.
    Без temperature и max_tokens — только max_output_tokens (как требует API).
    """
    cli = _client()
    try:
        resp = await asyncio.to_thread(
            cli.responses.create,
            model=model,
            input=prompt,
            max_output_tokens=max_out,
        )
        # у официального SDK у объекта есть удобное свойство
        out = getattr(resp, "output_text", None)
        if out is None:
            # на всякий — соберём вручную
            if resp.output and len(resp.output) and resp.output[0].content:
                out = "".join([b.text for b in (chunk.text for chunk in [resp.output[0].content]) if b])
        return out or ""
    except BadRequestError as e:
        # Перебросим наружу — верхний уровень может дать дружелюбный текст
        raise
    except Exception as e:
        raise

async def _fallback_chat(model: str, prompt: str, max_tokens: int = 600, temperature: float = 0.2) -> str:
    """
    Фоллбэк на chat.completions, если Responses ругнулся на параметры/модель.
    """
    cli = _client()
    resp = await asyncio.to_thread(
        cli.chat.completions.create,
        model=model,
        messages=[{"role":"system","content":"Кратко и по делу, на русском."},
                  {"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()

async def run_mini_digest(flags: Dict[str, Dict[str, Any]]) -> str:
    """
    flags: {"USDJPY":{"risk":"CAUTION","bias":"short-bias","notes":[...]}, ...}
    Возвращает русский дайджест по 4 парам.
    """
    def _mk_line(sym: str, f: Dict[str, Any]) -> str:
        risk = f.get("risk","OK")
        bias = f.get("bias","both")
        notes = f.get("notes") or []
        bullet = "\n".join([f"• {n}" for n in notes[:3]]) if notes else "• Без существенных новостей."
        lbl = "✅ Нейтрально" if risk=="OK" else ("⚠️ Осторожно" if risk=="CAUTION" else "🚨 Высокий риск")
        return f"{sym} — {lbl}\n{bullet}\nЧто делаем: режим {bias}."
    text_blocks = [
        "🧭 Утренний фон (кратко)",
        _mk_line("USDJPY", flags.get("USDJPY", {})),
        _mk_line("AUDUSD", flags.get("AUDUSD", {})),
        _mk_line("EURUSD", flags.get("EURUSD", {})),
        _mk_line("GBPUSD", flags.get("GBPUSD", {})),
    ]
    prompt = "\n\n".join(text_blocks) + "\n\nСформируй аккуратный, короткий дайджест для трейд-чатов."

    try:
        return await _create_responses(LLM_MINI, prompt, max_out=700)
    except BadRequestError as e:
        # Частые жалобы: unsupported parameter (если SDK/эндпоинт не совпали)
        # Пробуем фоллбэк на chat.completions
        try:
            return await _fallback_chat(LLM_MINI, prompt, max_tokens=700, temperature=0.2)
        except Exception as e2:
            raise RuntimeError(f"LLM (fallback) error: {e2}") from e
    except Exception as e:
        # общий фоллбэк на чат
        try:
            return await _fallback_chat(LLM_MINI, prompt, max_tokens=700, temperature=0.2)
        except Exception as e2:
            raise RuntimeError(f"LLM error: {e2}") from e
