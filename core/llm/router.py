"""Task-aware LLM router for DeepSeek and Zhipu fallback chains."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Sequence

from core.llm.base import LLMClient, LLMResponse, redact_sensitive
from core.llm.config import LLMSettings, load_llm_settings
from core.llm.deepseek_client import DeepSeekClient
from core.llm.zhipu_client import ZhipuClient
from core.reproducibility import LLMCallRecord, llm_call_record_from_response


logger = logging.getLogger("civitas.llm.router")

FAST_DIRECT_TASKS = {"belief_update", "short_extraction", "sentiment_tick", "agent_micro_decision"}


@dataclass(frozen=True, slots=True)
class RouteCandidate:
    provider: str
    model: str
    thinking: bool | None = None

    @property
    def label(self) -> str:
        if self.thinking is None:
            return f"{self.provider}:{self.model}"
        return f"{self.provider}:{self.model}:thinking={str(self.thinking).lower()}"


class LLMRouter:
    def __init__(
        self,
        *,
        settings: LLMSettings | None = None,
        deepseek_client: LLMClient | None = None,
        zhipu_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or load_llm_settings()
        self.clients: dict[str, LLMClient] = {
            "deepseek": deepseek_client or DeepSeekClient(settings=self.settings),
            "zhipu": zhipu_client or ZhipuClient(settings=self.settings),
        }
        self.call_records: list[dict[str, Any]] = []

    def get_call_records(self) -> list[dict[str, Any]]:
        return list(self.call_records)

    def _save_call_record(self, record: LLMCallRecord) -> None:
        self.call_records.append(record.to_dict())
        if len(self.call_records) > 500:
            self.call_records = self.call_records[-500:]

    def build_chain(self, mode: str, task_type: str | None = None) -> list[RouteCandidate]:
        normalized = str(mode or self.settings.default_provider or "auto").strip().lower()
        task = str(task_type or "").strip()
        if normalized in {"smart", "zhipu", "glm", "glm_only"}:
            return [RouteCandidate("zhipu", "glm-4-flashx", None)]
        if normalized == "slow":
            return [
                RouteCandidate("deepseek", "deepseek-v4-pro", True),
                RouteCandidate("deepseek", "deepseek-v4-flash", False),
                RouteCandidate("zhipu", "glm-4-flashx", None),
            ]
        if normalized == "fast":
            if task in FAST_DIRECT_TASKS:
                return [
                    RouteCandidate("deepseek", "deepseek-v4-flash", False),
                    RouteCandidate("zhipu", "glm-4-flashx", None),
                ]
            return [
                RouteCandidate("deepseek", "deepseek-v4-flash", True),
                RouteCandidate("deepseek", "deepseek-v4-flash", False),
                RouteCandidate("zhipu", "glm-4-flashx", None),
            ]
        return [
            RouteCandidate("deepseek", "deepseek-v4-flash", False),
            RouteCandidate("zhipu", "glm-4-flashx", None),
        ]

    def _offline_response(self, messages: Sequence[dict], fallback_chain: list[str], fallback_response: str | None = None) -> LLMResponse:
        if fallback_response is not None:
            text = str(fallback_response)
        else:
            seed = json.dumps(list(messages), ensure_ascii=False, sort_keys=True, default=str)
            digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
            text = json.dumps(
                {
                    "mode": "offline_fallback",
                    "ok": False,
                    "reason": "all_llm_providers_unavailable",
                    "digest": digest,
                },
                ensure_ascii=False,
            )
        return LLMResponse(
            text=text,
            provider="offline",
            model="deterministic_stub",
            latency_ms=0.0,
            fallback_chain=fallback_chain,
            usage={},
            raw={"reason": "all_llm_providers_unavailable"},
            ok=False,
        )

    async def complete(
        self,
        messages: list[dict],
        *,
        mode: str = "auto",
        task_type: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout: float | None = None,
        thinking: bool | None = None,
        json_mode: bool = False,
        fallback_response: str | None = None,
    ) -> LLMResponse:
        normalized_mode = str(mode or self.settings.default_provider or "auto").strip().lower()
        chain = self.build_chain(mode, task_type=task_type)
        if model:
            model_name = str(model)
            if normalized_mode in {"smart", "zhipu", "glm", "glm_only"} and not model_name.startswith("glm-"):
                model_name = "glm-4-flashx"
            provider = "zhipu" if model_name.startswith("glm-") else "deepseek"
            chain.insert(0, RouteCandidate(provider, model_name, thinking))

        seen: set[str] = set()
        ordered_chain: list[RouteCandidate] = []
        for candidate in chain:
            if candidate.label in seen:
                continue
            ordered_chain.append(candidate)
            seen.add(candidate.label)

        fallback_chain: list[str] = []
        for candidate in ordered_chain:
            fallback_chain.append(candidate.label)
            client = self.clients.get(candidate.provider)
            if client is None:
                continue
            try:
                response = await client.complete(
                    messages,
                    model=candidate.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout or self.settings.timeout_seconds,
                    thinking=candidate.thinking if candidate.thinking is not None else thinking,
                    json_mode=json_mode,
                    task_type=task_type,
                )
            except Exception as exc:  # noqa: BLE001
                self._save_call_record(
                    LLMCallRecord(
                        provider=candidate.provider,
                        model=candidate.model,
                        fallback_chain=list(fallback_chain),
                        latency_ms=0.0,
                        ok=False,
                        error_type=redact_sensitive(type(exc).__name__),
                    )
                )
                logger.warning(
                    "llm_provider_error provider=%s model=%s ok=false error=%s",
                    candidate.provider,
                    candidate.model,
                    redact_sensitive(exc),
                )
                continue

            if response.ok:
                response.fallback_chain = list(fallback_chain)
                self._save_call_record(llm_call_record_from_response(response))
                logger.info(
                    "llm_complete provider=%s model=%s latency_ms=%.1f fallback_chain=%s ok=true",
                    response.provider,
                    response.model,
                    float(response.latency_ms),
                    " > ".join(fallback_chain),
                )
                return response
            reason = ""
            if isinstance(response.raw, dict):
                reason = redact_sensitive(response.raw.get("error", ""))
            response.fallback_chain = list(fallback_chain)
            self._save_call_record(llm_call_record_from_response(response, error_type=reason))
            logger.warning(
                "llm_candidate_failed provider=%s model=%s latency_ms=%.1f fallback_chain=%s ok=false reason=%s",
                response.provider,
                response.model,
                float(response.latency_ms),
                " > ".join(fallback_chain),
                reason,
            )

        logger.info("llm_complete provider=offline model=deterministic_stub latency_ms=0 fallback_chain=%s ok=false", " > ".join(fallback_chain))
        offline = self._offline_response(messages, fallback_chain, fallback_response=fallback_response)
        self._save_call_record(llm_call_record_from_response(offline, error_type="all_llm_providers_unavailable"))
        return offline


async def llm_complete(messages: list[dict], mode: str = "auto", task_type: str | None = None, **kwargs: Any) -> LLMResponse:
    return await LLMRouter().complete(messages, mode=mode, task_type=task_type, **kwargs)


def _run_coro_sync(coro: Any) -> Any:
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop and running_loop.is_running():
        result_box: dict[str, Any] = {}
        error_box: dict[str, BaseException] = {}

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result_box["value"] = loop.run_until_complete(coro)
            except BaseException as exc:  # noqa: BLE001
                error_box["error"] = exc
            finally:
                loop.close()

        thread = threading.Thread(target=_runner, daemon=True, name="LLMRouterSyncBridge")
        thread.start()
        thread.join()
        if "error" in error_box:
            raise error_box["error"]
        return result_box.get("value")

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def sync_llm_complete(messages: list[dict], mode: str = "auto", task_type: str | None = None, **kwargs: Any) -> LLMResponse:
    return _run_coro_sync(llm_complete(messages, mode=mode, task_type=task_type, **kwargs))
