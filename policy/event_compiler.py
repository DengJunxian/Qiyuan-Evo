"""Unified compiler for policy/news/rumor/macro runtime events.

This is intentionally lightweight: it translates runtime ExperimentEvent
objects into the existing structured policy package path, so the current macro,
belief, and execution layers keep working while accepting more event families.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from core.experiment_events import EventDigest, ExperimentEvent, normalize_runtime_event_type
from core.macro.government import GovernmentAgent, PolicyShock
from policy.structured import PolicyPackage


class EventCompiler:
    """Compile generic runtime events through the existing policy parser."""

    def __init__(self, government: Optional[GovernmentAgent] = None) -> None:
        self.government = government or GovernmentAgent()

    @staticmethod
    def event_text(event: ExperimentEvent | Mapping[str, Any]) -> str:
        if isinstance(event, ExperimentEvent):
            event_type = event.event_type
            title = event.title
            raw_text = event.raw_text
            strength = event.strength
        else:
            event_type = normalize_runtime_event_type(str(event.get("event_type", "")))
            title = str(event.get("title", ""))
            raw_text = str(event.get("raw_text", ""))
            strength = float(event.get("current_strength", event.get("strength", 1.0)) or 1.0)
        prefix = {
            "policy": "政策公告",
            "major_news": "重大新闻",
            "macro_shock": "宏观冲击",
            "rumor": "市场谣言",
            "refute": "官方辟谣",
            "regime_shift": "市场状态切换",
            "regulatory_action": "监管行动",
        }.get(event_type, "运行时事件")
        return f"{prefix}（强度 {float(strength):.2f}）: {title}。{raw_text}".strip()

    @staticmethod
    def type_hint(event_type: str) -> Optional[str]:
        normalized = normalize_runtime_event_type(event_type)
        if normalized == "policy":
            return None
        if normalized == "rumor":
            return "tightening"
        if normalized == "refute":
            return "rumor_refutation"
        if normalized == "macro_shock":
            return "tightening"
        if normalized == "regulatory_action":
            return "stabilization"
        if normalized == "regime_shift":
            return "stabilization"
        return None

    def compile_event(
        self,
        event: ExperimentEvent | Mapping[str, Any],
        *,
        tick: int = 0,
        market_regime: Optional[str] = None,
    ) -> PolicyPackage:
        event_type = event.event_type if isinstance(event, ExperimentEvent) else str(event.get("event_type", ""))
        strength = event.strength if isinstance(event, ExperimentEvent) else float(event.get("current_strength", event.get("strength", 1.0)) or 1.0)
        text = self.event_text(event)
        return self.government.compile_policy_package(
            text,
            tick=int(tick),
            policy_type_hint=self.type_hint(event_type),
            intensity=max(0.0, float(strength)),
            market_regime=market_regime,
            snapshot_info={
                "runtime_event_type": normalize_runtime_event_type(event_type),
                "compiler": "event_compiler_v1",
            },
        )

    def compile_digest(
        self,
        digest: EventDigest | Mapping[str, Any],
        *,
        tick: int = 0,
        market_regime: Optional[str] = None,
    ) -> Optional[PolicyPackage]:
        if isinstance(digest, EventDigest):
            text = digest.to_policy_text()
            strength = digest.aggregate_strength
            active = digest.active_events
        else:
            text = str(digest.get("combined_text", "") or "")
            strength = float(digest.get("aggregate_strength", 0.0) or 0.0)
            active = list(digest.get("active_events", []) or [])
        if not text.strip() and not active:
            return None
        if not text.strip():
            text = "；".join(self.event_text(item) for item in active)
        return self.government.compile_policy_package(
            text,
            tick=int(tick),
            intensity=max(0.05, float(strength or 1.0)),
            market_regime=market_regime,
            snapshot_info={
                "runtime_event_count": len(active),
                "compiler": "event_compiler_v1",
            },
        )

    def compile_digest_shock(
        self,
        digest: EventDigest | Mapping[str, Any],
        *,
        tick: int = 0,
        policy_id: str = "",
    ) -> Optional[PolicyShock]:
        package = self.compile_digest(digest, tick=tick)
        if package is None:
            return None
        text = digest.to_policy_text() if isinstance(digest, EventDigest) else str(digest.get("combined_text", ""))
        shock = PolicyShock(
            policy_id=policy_id or package.event.policy_id,
            policy_text=text,
            **package.to_policy_shock_fields(),
        )
        shock.metadata = {
            "policy_event": package.event.to_dict(),
            "policy_package": package.to_dict(),
            "parser_version": package.parser_version,
            "compiler": "event_compiler_v1",
        }
        return shock


__all__ = ["EventCompiler"]
