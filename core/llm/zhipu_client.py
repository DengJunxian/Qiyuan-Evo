"""Zhipu GLM OpenAI-compatible client."""

from __future__ import annotations

import time
from typing import Any

from openai import AsyncOpenAI

from core.llm.base import LLMResponse, redact_sensitive
from core.llm.config import LLMSettings, load_llm_settings


class ZhipuClient:
    provider = "zhipu"

    def __init__(self, *, settings: LLMSettings | None = None, api_key: str | None = None, base_url: str | None = None) -> None:
        self.settings = settings or load_llm_settings()
        self.api_key = (api_key if api_key is not None else self.settings.zhipu_api_key) or ""
        self.base_url = base_url or self.settings.zhipu_base_url
        self._client: AsyncOpenAI | None = None

    def _get_client(self, timeout: float | None = None) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout or self.settings.timeout_seconds)
        return self._client

    async def complete(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout: float | None = None,
        thinking: bool | None = None,
        json_mode: bool = False,
        task_type: str | None = None,
    ) -> LLMResponse:
        model_name = model or "glm-4-flashx"
        if not self.api_key:
            return LLMResponse("", self.provider, model_name, 0.0, raw={"error": "missing_api_key"}, ok=False)

        start = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)
            if timeout is not None:
                kwargs["timeout"] = float(timeout)
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = await self._get_client(timeout).chat.completions.create(**kwargs)
            message = response.choices[0].message
            usage = response.usage.model_dump() if getattr(response, "usage", None) is not None else {}
            return LLMResponse(
                text=message.content or "",
                provider=self.provider,
                model=model_name,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                usage=usage,
                raw=response,
                ok=True,
            )
        except Exception as exc:  # noqa: BLE001
            return LLMResponse(
                text="",
                provider=self.provider,
                model=model_name,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                raw={"error": redact_sensitive(exc)},
                ok=False,
            )

