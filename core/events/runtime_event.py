"""Canonical runtime event schema for dynamic simulation injection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence

from core.experiment_events import EventImpactProfile, ExperimentEvent, normalize_runtime_event_type


def _stable_id(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def normalize_event_type(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "news": "major_news",
        "policy_text": "policy",
        "regulatory": "regulatory_action",
        "regulator": "regulatory_action",
        "liquidity": "liquidity_injection",
        "liquidity_injection": "macro_shock",
        "stamp_duty": "regulatory_action",
        "stabilization_fund": "policy",
        "external_market": "macro_shock",
        "external_market_shock": "macro_shock",
        "clarification": "refute",
        "debunk": "refute",
    }
    return normalize_runtime_event_type(aliases.get(text, text))


@dataclass(slots=True)
class RuntimeEvent:
    event_id: str
    timestamp: float
    trading_day: str
    event_type: str
    source: str
    credibility: float
    affected_sectors: Sequence[str] = field(default_factory=tuple)
    affected_agent_groups: Sequence[str] = field(default_factory=tuple)
    expected_duration: int = 1
    shock_strength: float = 1.0
    structured_payload: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    title: str = ""

    def __post_init__(self) -> None:
        self.event_id = str(self.event_id or f"runtime_{_stable_id(asdict(self))}")
        self.timestamp = float(self.timestamp)
        if not self.trading_day:
            self.trading_day = datetime.fromtimestamp(self.timestamp).date().isoformat()
        self.event_type = normalize_event_type(self.event_type)
        self.source = str(self.source or "manual")
        self.credibility = float(max(0.0, min(1.0, self.credibility)))
        self.affected_sectors = tuple(str(item) for item in self.affected_sectors)
        self.affected_agent_groups = tuple(str(item) for item in self.affected_agent_groups)
        self.expected_duration = max(1, int(self.expected_duration or 1))
        self.shock_strength = float(max(0.0, self.shock_strength))
        self.structured_payload = dict(self.structured_payload or {})
        self.raw_text = str(self.raw_text or "")
        self.title = str(self.title or self.event_type)

    @classmethod
    def create(
        cls,
        *,
        raw_text: str,
        event_type: str = "major_news",
        timestamp: Optional[float] = None,
        trading_day: str = "",
        source: str = "manual",
        credibility: float = 0.75,
        affected_sectors: Optional[Sequence[str]] = None,
        affected_agent_groups: Optional[Sequence[str]] = None,
        expected_duration: int = 1,
        shock_strength: float = 1.0,
        structured_payload: Optional[Mapping[str, Any]] = None,
        title: str = "",
        event_id: str = "",
    ) -> "RuntimeEvent":
        ts = float(timestamp if timestamp is not None else datetime.now().timestamp())
        payload = {
            "raw_text": raw_text,
            "event_type": normalize_event_type(event_type),
            "timestamp": round(ts, 6),
            "source": source,
            "structured_payload": dict(structured_payload or {}),
        }
        return cls(
            event_id=event_id or f"runtime_{_stable_id(payload)}",
            timestamp=ts,
            trading_day=trading_day or datetime.fromtimestamp(ts).date().isoformat(),
            event_type=normalize_event_type(event_type),
            source=source,
            credibility=float(credibility),
            affected_sectors=tuple(affected_sectors or ()),
            affected_agent_groups=tuple(affected_agent_groups or ()),
            expected_duration=int(expected_duration),
            shock_strength=float(shock_strength),
            structured_payload=dict(structured_payload or {}),
            raw_text=str(raw_text or ""),
            title=title or str(raw_text or "")[:24] or normalize_event_type(event_type),
        )

    @classmethod
    def from_experiment_event(cls, event: ExperimentEvent) -> "RuntimeEvent":
        created_ts = datetime.fromisoformat(str(event.created_at).replace("Z", "+00:00")).timestamp()
        impact = event.impact.to_dict() if event.impact else {}
        return cls(
            event_id=event.event_id,
            timestamp=float(created_ts),
            trading_day=str(event.metadata.get("trading_day", "")),
            event_type=event.event_type,
            source=event.source,
            credibility=float(event.confidence),
            affected_sectors=tuple(impact.get("affected_sectors", []) or []),
            affected_agent_groups=tuple(impact.get("affected_cohorts", []) or []),
            expected_duration=int(max(1, round(float(event.half_life)))),
            shock_strength=float(event.strength),
            structured_payload=dict(event.metadata or {}),
            raw_text=event.raw_text,
            title=event.title,
        )

    def to_experiment_event(
        self,
        *,
        effective_day: int = 1,
        effective_tick: Optional[int] = None,
        created_day: int = 0,
        created_tick: int = 0,
    ) -> ExperimentEvent:
        impact = EventImpactProfile(
            channels=list(self.structured_payload.get("channels", []) or []),
            affected_sectors=list(self.affected_sectors),
            affected_cohorts=list(self.affected_agent_groups),
            confidence=float(self.credibility),
            decay_half_life=float(self.expected_duration),
            sentiment_impact=float(self.structured_payload.get("sentiment_impact", 0.0) or 0.0),
            funding_stress=float(self.structured_payload.get("funding_stress", 0.0) or 0.0),
            liquidity_impact=float(self.structured_payload.get("liquidity_impact", 0.0) or 0.0),
            compliance_pressure=float(self.structured_payload.get("compliance_pressure", 0.0) or 0.0),
        )
        return ExperimentEvent.create(
            event_id=self.event_id,
            event_type=self.event_type,
            title=self.title,
            raw_text=self.raw_text,
            effective_day=max(1, int(effective_day)),
            effective_tick=effective_tick,
            strength=float(self.shock_strength),
            half_life=float(self.expected_duration),
            scope=str(self.structured_payload.get("scope", "broad_market")),
            channels=list(self.structured_payload.get("channels", []) or []),
            confidence=float(self.credibility),
            source=self.source,
            metadata={
                **dict(self.structured_payload),
                "runtime_event_schema": "runtime_event_v1",
                "trading_day": self.trading_day,
                "affected_sectors": list(self.affected_sectors),
                "affected_agent_groups": list(self.affected_agent_groups),
            },
            created_day=int(created_day),
            created_tick=int(created_tick),
            impact=impact,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = ["RuntimeEvent", "normalize_event_type"]
