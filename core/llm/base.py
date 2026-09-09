"""Shared LLM interfaces and safe response helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol


_AUTH_HEADER_RE = re.compile(r"authorization\s*:\s*bearer\s+[^\s,;]+", re.IGNORECASE)
_BEARER_RE = re.compile(r"bearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE)
_API_KEY_RE = re.compile(r"(sk|api[_-]?key)[-_A-Za-z0-9]{8,}", re.IGNORECASE)


def redact_sensitive(value: Any) -> str:
    """Return a log-safe string with credentials and auth headers removed."""

    text = str(value or "")
    text = _AUTH_HEADER_RE.sub("[REDACTED_AUTH_HEADER]", text)
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _API_KEY_RE.sub("[REDACTED_KEY]", text)
    return text


@dataclass(slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_ms: float
    fallback_chain: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    ok: bool = True


class LLMClient(Protocol):
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
        ...
