# llm_client.py
from __future__ import annotations

import os
import logging
from typing import Dict, List

try:
    # OpenAI Python SDK v1.x
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

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
        if OpenAI and self.api_key:
            try:
                self._client = OpenAI(api_key=self.api_key)
            except Exception as e:
                log.warning(f"OpenAI client init failed: {e}")

    def _require_client(self):
        if not self._client:
            raise LLMError("OpenAI client не инициализирован (проверьте OPENAI_API_KEY).")

    async def make_digest_ru(self, pairs: List[str]) -> Dict[str, str]:
        """
        Возвращает словарь {pair: 'текст на русском'}.
        Без внешних данных — просим LLM сделать короткую осмысленную выжимку в рамках 3–5 пунктов,
        избегая домыслов; если информации недостаточно — «Нейтрально».
        """
        self._require_client()
        # Компактный промпт. На mini получаем короткий «человеческий» русский текст.
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
            resp = await self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model_mini,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                temperature=0.3,
                max_tokens=600,
            )
            content = (resp.choices[0].message.content or "").strip()
        except Exception as e:  # сеть/квота/и т.д.
            raise LLMError(f"Ошибка обращения к LLM: {e}")

        # Простая нарезка по тикерам: выделяем блоки по названиям пар
        result: Dict[str, str] = {p: "" for p in pairs}
        if not content:
            # нейтральный дефолт
            for p in pairs:
                result[p] = "✅ Нейтрально. Существенных поводов не отмечено; работаем по базовому плану."
            return result

        # грубый сплит — ищем заголовки с тикерами
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
                # удалим сам тикер из строки
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
