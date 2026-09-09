"""Two-speed agent scheduler for Civitas market simulation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

from agents.fast_agent_kernel import FastAgentAction, FastAgentKernel, FastCohortState
from agents.slow_agent_orchestrator import SlowAgentOrchestrator


def _persona_key(agent: Any) -> str:
    persona = getattr(agent, "persona", None)
    key = str(getattr(persona, "archetype_key", "") or "").strip()
    if key:
        return key
    archetype = getattr(persona, "archetype", None)
    if archetype is not None:
        key = str(getattr(archetype, "key", "") or "").strip()
        if key:
            return key
    profile = getattr(agent, "psychology_profile", None)
    if isinstance(profile, Mapping):
        key = str(profile.get("institution_type", "") or "").strip()
        if key:
            return key
    return "retail_general"


def _hold_action(current_price: float) -> FastAgentAction:
    return FastAgentAction(action="HOLD", amount=0.0, target_price=float(current_price), metadata={"execution_path": "cadence_hold"})


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


@dataclass(slots=True)
class AgentSchedulerConfig:
    max_slow_agents: int = 4
    small_world_slow_threshold: int = 4
    slow_cadence_ticks: int = 2
    event_triggered_slow: bool = True
    daily_close_cadence: int = 10
    vector_fast_agents: bool = field(default_factory=lambda: _env_flag("CIVITAS_VECTOR_FAST_AGENTS", False))
    batched_slow_agents: bool = field(default_factory=lambda: _env_flag("CIVITAS_BATCHED_SLOW_AGENTS", True))


@dataclass(slots=True)
class AgentSchedulerResult:
    actions: List[Any]
    slow_indices: List[int] = field(default_factory=list)
    fast_indices: List[int] = field(default_factory=list)
    cohort_states: List[FastCohortState] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class AgentScheduler:
    """Bucket agents into strategic slow path and population fast path."""

    STRATEGIC_TOKENS = (
        "market_maker",
        "policy_capital",
        "state",
        "stabilization",
        "mutual_fund",
        "pension",
        "insurer",
        "institution",
        "national",
        "foreign",
    )

    def __init__(
        self,
        *,
        config: Optional[AgentSchedulerConfig] = None,
        fast_kernel: Optional[FastAgentKernel] = None,
        slow_orchestrator: Optional[SlowAgentOrchestrator] = None,
    ) -> None:
        self.config = config or AgentSchedulerConfig()
        self.fast_kernel = fast_kernel or FastAgentKernel()
        self.slow_orchestrator = slow_orchestrator or SlowAgentOrchestrator()
        self._last_slow_actions: Dict[int, Any] = {}

    def _is_strategic(self, agent: Any) -> bool:
        if bool(getattr(agent, "use_llm", False)):
            return True
        key = _persona_key(agent).lower()
        return any(token in key for token in self.STRATEGIC_TOKENS)

    def _partition(self, agents: Sequence[Any]) -> tuple[List[int], List[int]]:
        if len(agents) <= int(self.config.small_world_slow_threshold):
            return list(range(len(agents))), []
        strategic = [idx for idx, agent in enumerate(agents) if self._is_strategic(agent)]
        if not strategic:
            strategic = list(range(min(int(self.config.max_slow_agents), len(agents))))
        slow_indices = strategic[: max(1, int(self.config.max_slow_agents))]
        slow_set = set(slow_indices)
        fast_indices = [idx for idx in range(len(agents)) if idx not in slow_set]
        return slow_indices, fast_indices

    def _slow_due(self, tick: int, event_digest: Mapping[str, Any] | None) -> bool:
        if self.config.event_triggered_slow and list((event_digest or {}).get("active_events", []) or []):
            return True
        cadence = max(1, int(self.config.slow_cadence_ticks))
        if int(tick) % cadence == 0:
            return True
        daily = max(1, int(self.config.daily_close_cadence))
        return int(tick) % daily == 0

    async def collect_actions(
        self,
        agents: Sequence[Any],
        *,
        process_agent: Callable[[Any], Awaitable[Any]],
        tick: int,
        current_price: float,
        runner_symbol: str,
        event_digest: Mapping[str, Any] | None = None,
        macro_state: Mapping[str, Any] | None = None,
    ) -> AgentSchedulerResult:
        actions: List[Any] = [_hold_action(current_price) for _ in agents]
        slow_indices, fast_indices = self._partition(agents)
        due = self._slow_due(tick, event_digest)
        if slow_indices and not fast_indices:
            due = True

        slow_stats: Dict[str, Any] = {
            "llm_call_count": 0,
            "batch_count": 0,
            "cache_hit_count": 0,
            "prefix_cache_hit_count": 0,
            "avg_latency_ms": 0.0,
            "processed_agent_ids": [],
        }
        if slow_indices and due:
            slow_actions, stats = await self.slow_orchestrator.run_batch(
                [(idx, agents[idx]) for idx in slow_indices],
                process_agent=process_agent,
                event_digest=event_digest,
                regime=str((macro_state or {}).get("regime", "neutral")),
            )
            slow_stats = stats.to_dict()
            for idx, action in slow_actions.items():
                actions[idx] = action
                self._last_slow_actions[idx] = action
        else:
            for idx in slow_indices:
                if idx in self._last_slow_actions:
                    actions[idx] = self._last_slow_actions[idx]
                    slow_stats["cache_hit_count"] += 1

        fast_agents = [agents[idx] for idx in fast_indices]
        fast_actions, cohort_states = self.fast_kernel.decide(
            fast_agents,
            current_price=float(current_price),
            event_digest=event_digest,
            macro_state=macro_state,
            tick=int(tick),
        )
        for idx, action in zip(fast_indices, fast_actions):
            actions[idx] = action

        stats = {
            "architecture": "two_speed_v1",
            "slow_agent_count": int(len(slow_indices)),
            "fast_agent_count": int(len(fast_indices)),
            "slow_due": bool(due),
            "cadence": {
                "slow_cadence_ticks": int(self.config.slow_cadence_ticks),
                "daily_close_cadence": int(self.config.daily_close_cadence),
                "event_triggered": bool(self.config.event_triggered_slow),
            },
            "feature_flags": {
                "vector_fast_agents": bool(self.config.vector_fast_agents),
                "batched_slow_agents": bool(self.config.batched_slow_agents),
            },
            "llm_call_count": int(slow_stats.get("llm_call_count", 0)),
            "batch_count": int(slow_stats.get("batch_count", 0)),
            "cache_hit_count": int(slow_stats.get("cache_hit_count", 0)),
            "prefix_cache_hit_count": int(slow_stats.get("prefix_cache_hit_count", 0)),
            "avg_slow_agent_latency_ms": float(slow_stats.get("avg_latency_ms", 0.0) or 0.0),
            "fast_model_call_count": 0,
            "fast_policy_path": "heuristic_or_vector_policy_head",
            "slow_model_path": "slow_orchestrator_cadence_only",
            "cohort_count": int(len(cohort_states)),
            "cohorts": [cohort.to_dict() for cohort in cohort_states],
        }
        total_cache_checks = max(1, int(stats["cache_hit_count"]) + int(stats["batch_count"]))
        stats["cache_hit_rate"] = float(int(stats["cache_hit_count"]) / total_cache_checks)
        return AgentSchedulerResult(
            actions=actions,
            slow_indices=slow_indices,
            fast_indices=fast_indices,
            cohort_states=cohort_states,
            stats=stats,
        )


__all__ = ["AgentScheduler", "AgentSchedulerConfig", "AgentSchedulerResult"]
