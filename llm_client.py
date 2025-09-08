# llm_client.py
from __future__ import annotations

import os
import logging
from typing import Dict, List

try:
    # асинхронный клиент SDK v1.x
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

log = logging.getLogger("llm_client")


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model_nano: str = "gpt-5-nano",
        model_mini: str = "gpt-5-mini",
        model_major: str = "gpt-5",
        daily_token_budget: int = 30000,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self.model_nano = model_nano
        self.model_mini = model_mini
        self.model_major = model_major
        self.daily_budget = daily_token_budget
        self._client = None
        if AsyncOpenAI and self.api_key:
            try:
                self._client = AsyncOpenAI(api_key=self.api_key)
            except Exception as e:
                log.warning(f"OpenAI client init failed: {e}")

    def _require_client(self):
        if not self._client:
            raise LLMError("OpenAI client не инициализирован (проверьте OPENAI_API_KEY).")

    async def _call_mini(self, system: str, user: str, max_tokens: int = 600) -> str:
        """
        Универсальный вызов: сначала пробуем Responses API (max_output_tokens),
        если провал — Chat Completions (max_tokens).
        """
        self._require_client()
        # 1) Responses API
        try:
            resp = await self._client.responses.create(  # type: ignore[union-attr]
                model=self.model_mini,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_output_tokens=max_tokens,
                temperature=0.3,
            )
            # у Responses удобно вытаскивать сразу текст:
            out = getattr(resp, "output_text", None)
            if out:
                return out.strip()
            # fallback: собрать текст вручную
            try:
                chunks = []
                for item in resp.output:
                    if item.type == "message":
                        for cc in item.message.content:
                            if cc.type == "text":
                                chunks.append(cc.text)
                return "\n".join(chunks).strip()
            except Exception:
                pass
        except Exception as e1:
            msg = str(e1)
            log.info(f"Responses API failed, fallback to Chat Completions: {msg}")

        # 2) Chat Completions API
        try:
            resp = await self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model_mini,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e2:
            raise LLMError(f"Ошибка обращения к LLM: {e2}")

    async def make_digest_ru(self, pairs: List[str]) -> Dict[str, str]:
        """
        Возвращает словарь {pair: 'текст на русском'}.
        """
        sys = (
            "Ты финансовый редактор. Пиши кратко на русском, без воды, 2–4 пункта на пару. "
            "Тон — деловой. Избегай категоричных прогнозов, если уверенности нет — укажи нейтрально."
        )
        user = (
            "Сделай короткий дайджест по валютным парам: "
            f"{', '.join(pairs)}. Для каждой пары: одна строка статуса-иконки "
            "(например ✅ нейтрально / ⚠️ осторожно / 🚧 риск), затем 2–4 тезиса: "
            "ключевые риски/события сегодня, на что обратить внимание трейдеру интрадей. "
            "Если нет конкретных поводов — напиши коротко «Нейтрально, работаем по плану». "
            "Формат: для КАЖДОЙ пары отдельный блок, сначала тикер, затем текст. "
            "Аббревиатуры: ISM, CPI, NFP, BoJ, RBA, ECB, BoE можно использовать, но без фантазий."
        )

        try:
            content = await self._call_mini(sys, user, max_tokens=600)
        except LLMError:
            # полностью нейтральный дефолт
            return {p: "✅ Нейтрально. Существенных поводов не отмечено; работаем по базовому плану." for p in pairs}

        if not content:
            return {p: "✅ Нейтрально. Существенных поводов не отмечено; работаем по базовому плану." for p in pairs}

        # простая нарезка по тикерам
        result: Dict[str, str] = {p: "" for p in pairs}
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        cur_key = None
        acc: Dict[str, List[str]] = {p: [] for p in pairs}
        for ln in lines:
            up = ln.upper()
            matched = None
            for p in pairs:
                if p in up:
                    matched = p
                    break
            if matched:
                cur_key = matched
                cleaned = ln.replace(matched, "").strip("-: \t")
                if cleaned:
                    acc[cur_key].append(cleaned)
                continue
            if cur_key:
                acc[cur_key].append(ln)

        for p in pairs:
            block = "\n".join(acc.get(p) or []).strip()
            if not block:
                block = "✅ Нейтрально. Существенных поводов не отмечено; работаем по базовому плану."
            result[p] = block
        return result
