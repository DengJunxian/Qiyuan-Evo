"""Slow strategic-agent orchestration helpers.

This keeps slow agents batched and observable without requiring a live LLM in
default local mode. The actual per-agent decision function is injected by the
simulation loop.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Sequence


def stable_event_archetype_key(
    *,
    event_digest: Mapping[str, Any] | None,
    archetype: str,
    regime: str = "neutral",
) -> str:
    payload = {
        "event_hash": str((event_digest or {}).get("event_hash", "")),
        "active_ids": [str(item.get("event_id", "")) for item in list((event_digest or {}).get("active_events", []) or [])],
        "archetype": str(archetype or "unknown"),
        "regime": str(regime or "neutral"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class SlowBatchStats:
    llm_call_count: int = 0
    batch_count: int = 0
    cache_hit_count: int = 0
    prefix_cache_hit_count: int = 0
    avg_latency_ms: float = 0.0
    processed_agent_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm_call_count": int(self.llm_call_count),
            "batch_count": int(self.batch_count),
            "cache_hit_count": int(self.cache_hit_count),
            "prefix_cache_hit_count": int(self.prefix_cache_hit_count),
            "avg_latency_ms": float(self.avg_latency_ms),
            "processed_agent_ids": list(self.processed_agent_ids),
        }


class PrefixTemplateCache:
    """Tiny in-process prompt/prefix cache for strategic agents."""

    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}
        self.hit_count = 0
        self.miss_count = 0

    def get_or_set(self, key: str, value_factory: Callable[[], str]) -> str:
        if key in self._cache:
            self.hit_count += 1
            return self._cache[key]
        self.miss_count += 1
        value = value_factory()
        self._cache[key] = value
        return value


class SlowAgentOrchestrator:
    """Batch slow/strategic agent decisions through one async gather."""

    def __init__(self) -> None:
        self.prefix_cache = PrefixTemplateCache()

    @staticmethod
    def _agent_id(agent: Any, idx: int) -> str:
        return str(getattr(agent, "agent_id", f"agent_{idx}"))

    @staticmethod
    def _archetype(agent: Any) -> str:
        persona = getattr(agent, "persona", None)
        key = str(getattr(persona, "archetype_key", "") or "").strip()
        if key:
            return key
        archetype = getattr(persona, "archetype", None)
        if archetype is not None:
            key = str(getattr(archetype, "key", "") or "").strip()
            if key:
                return key
        return "strategic"

    async def run_batch(
        self,
        indexed_agents: Sequence[tuple[int, Any]],
        *,
        process_agent: Callable[[Any], Awaitable[Any]],
        event_digest: Mapping[str, Any] | None = None,
        regime: str = "neutral",
    ) -> tuple[Dict[int, Any], SlowBatchStats]:
        if not indexed_agents:
            return {}, SlowBatchStats()

        started = time.perf_counter()
        stats = SlowBatchStats(batch_count=1)
        for idx, agent in indexed_agents:
            archetype = self._archetype(agent)
            self.prefix_cache.get_or_set(
                f"{archetype}|{regime}",
                lambda archetype=archetype, regime=regime: f"role={archetype}; regime={regime}",
            )
            stats.processed_agent_ids.append(self._agent_id(agent, idx))
        stats.prefix_cache_hit_count = int(self.prefix_cache.hit_count)
        stats.llm_call_count = int(sum(1 for _, agent in indexed_agents if bool(getattr(agent, "use_llm", False))))

        import asyncio

        values = await asyncio.gather(*(process_agent(agent) for _, agent in indexed_agents))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        stats.avg_latency_ms = float(elapsed_ms / max(len(indexed_agents), 1))
        return {idx: value for (idx, _), value in zip(indexed_agents, values)}, stats


__all__ = ["SlowAgentOrchestrator", "SlowBatchStats", "PrefixTemplateCache", "stable_event_archetype_key"]
