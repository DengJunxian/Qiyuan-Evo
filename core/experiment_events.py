"""Runtime experiment events for policy, news, shock, and replay sessions.

The goal of this module is to keep the existing PolicySession/EventStore
pipeline while generalizing it from "policy text over time" to a reusable
runtime event stream.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from core.event_store import EventRecord, EventStore, EventType


RUNTIME_EVENT_TYPES = {
    "policy",
    "major_news",
    "macro_shock",
    "rumor",
    "refute",
    "regime_shift",
    "regulatory_action",
}

RUNTIME_EVENT_STATUSES = {"queued", "active", "expired"}


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_runtime_event_type(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "news": "major_news",
        "majornews": "major_news",
        "macro": "macro_shock",
        "shock": "macro_shock",
        "regime": "regime_shift",
        "regulation": "regulatory_action",
        "regulatory": "regulatory_action",
        "clarification": "refute",
        "refutation": "refute",
    }
    text = aliases.get(text, text)
    if text not in RUNTIME_EVENT_TYPES:
        return "major_news"
    return text


def event_store_type_for_runtime(event_type: str) -> str:
    normalized = normalize_runtime_event_type(event_type)
    if normalized == "policy":
        return EventType.POLICY.value
    if normalized == "major_news":
        return EventType.NEWS.value
    if normalized == "rumor":
        return EventType.RUMOR.value
    if normalized == "refute":
        return EventType.REFUTE.value
    if normalized == "regime_shift":
        return EventType.REGIME.value
    if normalized == "macro_shock":
        return EventType.MACRO.value
    if normalized == "regulatory_action":
        return getattr(EventType, "REGULATORY_ACTION", EventType.POLICY).value
    return EventType.NEWS.value


def _keywords(text: str, tokens: Sequence[str]) -> bool:
    lower = text.lower()
    return any(str(token).lower() in lower or str(token) in text for token in tokens)


@dataclass(slots=True)
class EventImpactProfile:
    """Compiled impact hints consumed by macro/social/market layers."""

    channels: List[str] = field(default_factory=list)
    macro_delta: Dict[str, float] = field(default_factory=dict)
    social_delta: Dict[str, float] = field(default_factory=dict)
    market_delta: Dict[str, float] = field(default_factory=dict)
    affected_symbols: List[str] = field(default_factory=list)
    affected_sectors: List[str] = field(default_factory=list)
    affected_cohorts: List[str] = field(default_factory=list)
    direction: str = "neutral"
    confidence: float = 0.75
    lag_days: int = 0
    decay_half_life: float = 7.0
    liquidity_impact: float = 0.0
    funding_stress: float = 0.0
    sentiment_impact: float = 0.0
    compliance_pressure: float = 0.0

    def scaled(self, multiplier: float) -> "EventImpactProfile":
        factor = float(multiplier)

        def _scale_map(payload: Mapping[str, float]) -> Dict[str, float]:
            return {str(k): float(v) * factor for k, v in payload.items()}

        return EventImpactProfile(
            channels=list(self.channels),
            macro_delta=_scale_map(self.macro_delta),
            social_delta=_scale_map(self.social_delta),
            market_delta=_scale_map(self.market_delta),
            affected_symbols=list(self.affected_symbols),
            affected_sectors=list(self.affected_sectors),
            affected_cohorts=list(self.affected_cohorts),
            direction=self.direction,
            confidence=float(self.confidence),
            lag_days=int(self.lag_days),
            decay_half_life=float(self.decay_half_life),
            liquidity_impact=float(self.liquidity_impact) * factor,
            funding_stress=float(self.funding_stress) * factor,
            sentiment_impact=float(self.sentiment_impact) * factor,
            compliance_pressure=float(self.compliance_pressure) * factor,
        )

    def merge(self, other: "EventImpactProfile") -> "EventImpactProfile":
        def _add(left: Mapping[str, float], right: Mapping[str, float]) -> Dict[str, float]:
            out = {str(k): float(v) for k, v in left.items()}
            for key, value in right.items():
                out[str(key)] = float(out.get(str(key), 0.0) + float(value))
            return out

        return EventImpactProfile(
            channels=list(dict.fromkeys([*self.channels, *other.channels])),
            macro_delta=_add(self.macro_delta, other.macro_delta),
            social_delta=_add(self.social_delta, other.social_delta),
            market_delta=_add(self.market_delta, other.market_delta),
            affected_symbols=list(dict.fromkeys([*self.affected_symbols, *other.affected_symbols])),
            affected_sectors=list(dict.fromkeys([*self.affected_sectors, *other.affected_sectors])),
            affected_cohorts=list(dict.fromkeys([*self.affected_cohorts, *other.affected_cohorts])),
            direction="mixed" if self.direction != other.direction else self.direction,
            confidence=float((self.confidence + other.confidence) / 2.0),
            lag_days=min(int(self.lag_days), int(other.lag_days)),
            decay_half_life=float(max(self.decay_half_life, other.decay_half_life)),
            liquidity_impact=float(self.liquidity_impact + other.liquidity_impact),
            funding_stress=float(self.funding_stress + other.funding_stress),
            sentiment_impact=float(self.sentiment_impact + other.sentiment_impact),
            compliance_pressure=float(self.compliance_pressure + other.compliance_pressure),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExperimentEvent:
    """One runtime event injected into an experiment session."""

    event_id: str
    event_type: str
    title: str
    raw_text: str
    effective_day: int = 1
    effective_tick: Optional[int] = None
    visibility_time: Any = ""
    strength: float = 1.0
    half_life: float = 7.0
    scope: str = "broad_market"
    channels: List[str] = field(default_factory=list)
    confidence: float = 0.75
    source: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_day: int = 0
    created_tick: int = 0
    created_at: str = field(default_factory=_utc_now_iso)
    impact: EventImpactProfile = field(default_factory=EventImpactProfile)

    def __post_init__(self) -> None:
        self.event_type = normalize_runtime_event_type(self.event_type)
        self.event_id = str(self.event_id or f"event_{uuid.uuid4().hex[:10]}")
        self.title = str(self.title or self.event_type)
        self.raw_text = str(self.raw_text or "")
        self.effective_day = max(1, int(self.effective_day or 1))
        self.effective_tick = None if self.effective_tick is None else max(0, int(self.effective_tick))
        self.visibility_time = self.visibility_time or self.created_at
        self.strength = float(max(0.0, self.strength))
        self.half_life = float(max(float(self.half_life or 1.0), 1e-6))
        self.scope = str(self.scope or "broad_market")
        self.channels = list(dict.fromkeys(str(ch) for ch in (self.channels or []) if str(ch).strip()))
        self.confidence = float(_clip(self.confidence, 0.0, 1.0))
        self.source = str(self.source or "manual")
        if not self.impact.channels and self.channels:
            self.impact.channels = list(self.channels)
        if self.impact.decay_half_life <= 0:
            self.impact.decay_half_life = float(self.half_life)
        if self.impact.confidence <= 0:
            self.impact.confidence = float(self.confidence)

    @classmethod
    def create(
        cls,
        *,
        event_type: str,
        title: str,
        raw_text: str,
        effective_day: int = 1,
        effective_tick: Optional[int] = None,
        visibility_time: Any = "",
        strength: float = 1.0,
        half_life: float = 7.0,
        scope: str = "broad_market",
        channels: Optional[Sequence[str]] = None,
        confidence: float = 0.75,
        source: str = "manual",
        metadata: Optional[Mapping[str, Any]] = None,
        created_day: int = 0,
        created_tick: int = 0,
        event_id: Optional[str] = None,
        impact: Optional[EventImpactProfile] = None,
    ) -> "ExperimentEvent":
        normalized = normalize_runtime_event_type(event_type)
        payload = {
            "event_type": normalized,
            "title": title,
            "raw_text": raw_text,
            "effective_day": int(effective_day),
            "effective_tick": effective_tick,
            "source": source,
            "metadata": dict(metadata or {}),
            "nonce": uuid.uuid4().hex[:8],
        }
        return cls(
            event_id=event_id or f"{normalized}_{_stable_hash(payload)[:10]}",
            event_type=normalized,
            title=str(title or normalized),
            raw_text=str(raw_text or ""),
            effective_day=int(effective_day),
            effective_tick=effective_tick,
            visibility_time=visibility_time,
            strength=float(strength),
            half_life=float(half_life),
            scope=str(scope or "broad_market"),
            channels=list(channels or []),
            confidence=float(confidence),
            source=str(source or "manual"),
            metadata=dict(metadata or {}),
            created_day=int(created_day),
            created_tick=int(created_tick),
            impact=impact or compile_event_impact(
                event_type=normalized,
                raw_text=raw_text,
                strength=strength,
                scope=scope,
                channels=channels or [],
                confidence=confidence,
            ),
        )

    def age_for_time(self, day_index: int, tick: Optional[int] = None, *, ticks_per_day: int = 1) -> float:
        if self.effective_tick is not None and tick is not None:
            return max(0.0, (int(tick) - int(self.effective_tick)) / max(1.0, float(ticks_per_day)))
        return max(0.0, float(int(day_index) - int(self.effective_day)))

    def intensity_for_time(self, day_index: int, tick: Optional[int] = None, *, ticks_per_day: int = 1) -> float:
        if int(day_index) < int(self.effective_day):
            return 0.0
        if self.effective_tick is not None and tick is not None and int(tick) < int(self.effective_tick):
            return 0.0
        age = self.age_for_time(day_index, tick, ticks_per_day=ticks_per_day)
        decay = math.pow(0.5, age / max(float(self.half_life), 1e-6))
        return float(max(0.0, self.strength) * decay)

    def state_for_time(self, day_index: int, tick: Optional[int] = None, *, expire_threshold: float = 0.02) -> str:
        if int(day_index) < int(self.effective_day):
            return "queued"
        if self.effective_tick is not None and tick is not None and int(tick) < int(self.effective_tick):
            return "queued"
        if self.intensity_for_time(day_index, tick) <= float(expire_threshold):
            return "expired"
        return "active"

    def to_timeline_row(self, day_index: Optional[int] = None, tick: Optional[int] = None) -> Dict[str, Any]:
        payload = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "title": self.title,
            "raw_text": self.raw_text,
            "effective_day": int(self.effective_day),
            "effective_tick": self.effective_tick,
            "visibility_time": self.visibility_time,
            "strength": float(self.strength),
            "half_life": float(self.half_life),
            "scope": self.scope,
            "channels": list(self.channels),
            "confidence": float(self.confidence),
            "source": self.source,
            "metadata": dict(self.metadata),
            "created_day": int(self.created_day),
            "created_tick": int(self.created_tick),
            "impact": self.impact.to_dict(),
        }
        if day_index is not None:
            payload["status"] = self.state_for_time(int(day_index), tick)
            payload["current_strength"] = round(self.intensity_for_time(int(day_index), tick), 6)
        return payload

    def to_event_record(self) -> EventRecord:
        payload = self.to_timeline_row(None)
        payload["runtime_event_schema"] = "experiment_event_v1"
        return EventRecord(
            event_id=self.event_id,
            timestamp=self.created_at,
            visibility_time=str(self.visibility_time or self.created_at),
            source=self.source,
            confidence=float(self.confidence),
            event_type=event_store_type_for_runtime(self.event_type),
            payload=payload,
            metadata={"module": "experiment_events", **dict(self.metadata or {})},
        )


@dataclass(slots=True)
class EventDigest:
    """Aggregated day/tick view consumed by macro, social, agents, and UI."""

    day_index: int
    tick: Optional[int] = None
    active_events: List[Dict[str, Any]] = field(default_factory=list)
    queued_events: List[Dict[str, Any]] = field(default_factory=list)
    expired_events: List[Dict[str, Any]] = field(default_factory=list)
    by_type: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    policy_digest: str = ""
    news_digest: str = ""
    rumor_digest: str = ""
    refute_digest: str = ""
    macro_shock_digest: str = ""
    aggregate_strength: float = 0.0
    aggregate_impact: EventImpactProfile = field(default_factory=EventImpactProfile)
    event_hash: str = ""

    def to_policy_text(self) -> str:
        parts = []
        for label, text in (
            ("政策", self.policy_digest),
            ("重大新闻", self.news_digest),
            ("谣言", self.rumor_digest),
            ("辟谣", self.refute_digest),
            ("宏观冲击", self.macro_shock_digest),
        ):
            if text:
                parts.append(f"{label}: {text}")
        return "；".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day_index": int(self.day_index),
            "tick": self.tick,
            "active_events": list(self.active_events),
            "queued_events": list(self.queued_events),
            "expired_events": list(self.expired_events),
            "by_type": {str(k): list(v) for k, v in self.by_type.items()},
            "policy_digest": self.policy_digest,
            "news_digest": self.news_digest,
            "rumor_digest": self.rumor_digest,
            "refute_digest": self.refute_digest,
            "macro_shock_digest": self.macro_shock_digest,
            "aggregate_strength": float(self.aggregate_strength),
            "aggregate_impact": self.aggregate_impact.to_dict(),
            "event_hash": self.event_hash,
            "combined_text": self.to_policy_text(),
            "active_count": len(self.active_events),
            "queued_count": len(self.queued_events),
            "expired_count": len(self.expired_events),
        }


def compile_event_impact(
    *,
    event_type: str,
    raw_text: str,
    strength: float = 1.0,
    scope: str = "broad_market",
    channels: Sequence[str] = (),
    confidence: float = 0.75,
) -> EventImpactProfile:
    """Small deterministic event compiler with no heavy dependencies."""

    normalized = normalize_runtime_event_type(event_type)
    text = str(raw_text or "")
    intensity = float(_clip(strength, 0.0, 2.5))
    base_channels = list(dict.fromkeys([str(ch) for ch in channels if str(ch).strip()]))
    macro_delta: Dict[str, float] = {}
    social_delta: Dict[str, float] = {}
    market_delta: Dict[str, float] = {}
    affected_sectors: List[str] = []
    affected_symbols: List[str] = []
    affected_cohorts: List[str] = []
    direction = "neutral"
    liquidity = 0.0
    funding_stress = 0.0
    sentiment = 0.0
    compliance = 0.0

    positive = _keywords(text, ("支持", "稳", "降准", "降息", "流动性", "回购", "补贴", "利好", "support", "easing", "stimulus"))
    negative = _keywords(text, ("风险", "暴跌", "违约", "爆雷", "收紧", "加息", "抛售", "panic", "selloff", "default"))
    rumorish = normalized == "rumor" or _keywords(text, ("传闻", "谣言", "未经证实", "rumor"))
    refuting = normalized == "refute" or _keywords(text, ("辟谣", "澄清", "refute", "clarify"))

    if positive and not negative:
        direction = "supportive"
        sentiment += 0.08 * intensity
        liquidity += 0.05 * intensity
    if negative and not positive:
        direction = "restrictive"
        sentiment -= 0.10 * intensity
        liquidity -= 0.04 * intensity
        funding_stress += 0.08 * intensity
    if positive and negative:
        direction = "mixed"
        sentiment -= 0.02 * intensity
        funding_stress += 0.04 * intensity

    if normalized == "policy":
        base_channels.extend(["policy_signal", "risk_appetite"])
        macro_delta["sentiment_index"] = sentiment or 0.04 * intensity
        macro_delta["liquidity_index"] = liquidity
        market_delta["order_flow_bias"] = 0.06 * intensity if direction != "restrictive" else -0.06 * intensity
    elif normalized == "major_news":
        base_channels.extend(["information_network", "expectations"])
        social_delta["news_attention"] = 0.18 * intensity
        macro_delta["sentiment_index"] = sentiment
        market_delta["volatility_bias"] = (0.08 if negative else 0.03) * intensity
    elif normalized == "macro_shock":
        base_channels.extend(["macro_state", "funding_conditions"])
        macro_delta["liquidity_index"] = liquidity
        macro_delta["credit_spread"] = funding_stress * 0.010
        macro_delta["sentiment_index"] = sentiment
        market_delta["volatility_bias"] = 0.12 * intensity
    elif normalized == "rumor":
        base_channels.extend(["rumor_network", "social_contagion"])
        direction = "restrictive" if direction == "neutral" else direction
        sentiment -= 0.12 * intensity
        funding_stress += 0.06 * intensity
        social_delta["rumor_pressure"] = 0.25 * intensity
        macro_delta["sentiment_index"] = -0.10 * intensity
        macro_delta["liquidity_index"] = -0.03 * intensity
        market_delta["volatility_bias"] = 0.15 * intensity
        affected_cohorts.extend(["retail", "rumor_trader", "leveraged_capital"])
    elif normalized == "refute":
        base_channels.extend(["authority_signal", "rumor_suppression", "compliance_intensity"])
        direction = "stabilizing"
        sentiment += 0.10 * intensity
        compliance += 0.12 * intensity
        social_delta["rumor_pressure"] = -0.22 * intensity
        macro_delta["sentiment_index"] = 0.09 * intensity
        macro_delta["liquidity_index"] = 0.02 * intensity
        market_delta["volatility_bias"] = -0.08 * intensity
        affected_cohorts.extend(["retail", "rumor_trader", "market_maker"])
    elif normalized == "regime_shift":
        base_channels.extend(["regime_transition", "risk_budget"])
        market_delta["regime_transition"] = 1.0 * intensity
        market_delta["volatility_bias"] = 0.10 * intensity
        macro_delta["sentiment_index"] = sentiment
    elif normalized == "regulatory_action":
        base_channels.extend(["regulatory_signal", "compliance_intensity"])
        direction = "stabilizing" if direction == "neutral" else direction
        compliance += 0.18 * intensity
        macro_delta["sentiment_index"] = 0.04 * intensity
        market_delta["compliance_pressure"] = compliance
        affected_cohorts.extend(["market_maker", "leveraged_capital", "rumor_trader"])

    if _keywords(text, ("银行", "券商", "金融", "bank", "financial")):
        affected_sectors.append("financials")
    if _keywords(text, ("科技", "芯片", "ai", "growth", "tech")):
        affected_sectors.append("growth")
    if _keywords(text, ("地产", "房地产", "property", "real estate")):
        affected_sectors.append("property")
    if _keywords(text, ("消费", "consumer")):
        affected_sectors.append("consumer")
    if str(scope).startswith("symbol:"):
        affected_symbols.append(str(scope).split(":", 1)[-1])

    if not affected_sectors and scope in {"financials", "growth", "consumer", "property", "defensive"}:
        affected_sectors.append(str(scope))

    return EventImpactProfile(
        channels=list(dict.fromkeys(base_channels or ["expectations"])),
        macro_delta={k: float(_clip(v, -1.0, 1.0)) for k, v in macro_delta.items()},
        social_delta={k: float(_clip(v, -1.0, 1.0)) for k, v in social_delta.items()},
        market_delta={k: float(_clip(v, -1.0, 1.0)) for k, v in market_delta.items()},
        affected_symbols=list(dict.fromkeys(affected_symbols)),
        affected_sectors=list(dict.fromkeys(affected_sectors)),
        affected_cohorts=list(dict.fromkeys(affected_cohorts)),
        direction=direction,
        confidence=float(_clip(confidence, 0.0, 1.0)),
        lag_days=0,
        decay_half_life=7.0,
        liquidity_impact=float(_clip(liquidity, -1.0, 1.0)),
        funding_stress=float(_clip(funding_stress, -1.0, 1.0)),
        sentiment_impact=float(_clip(sentiment, -1.0, 1.0)),
        compliance_pressure=float(_clip(compliance, -1.0, 1.0)),
    )


class ScenarioEventQueue:
    """Runtime event queue with EventStore/EventBus integration."""

    def __init__(
        self,
        events: Optional[Iterable[ExperimentEvent]] = None,
        *,
        event_store: Optional[EventStore] = None,
        dataset_version: str = "policy_lab_runtime",
        persist_events: bool = True,
        event_bus: Any = None,
    ) -> None:
        self.events: List[ExperimentEvent] = list(events or [])
        self.event_store = event_store or EventStore()
        self.dataset_version = str(dataset_version or "policy_lab_runtime")
        self.persist_events = bool(persist_events)
        self.event_bus = event_bus
        self._persisted_ids: set[str] = set()

    def append_event(self, event: ExperimentEvent) -> str:
        self.events.append(event)
        if self.persist_events and event.event_id not in self._persisted_ids:
            try:
                self.event_store.append_events(self.dataset_version, [event.to_event_record()])
                self._persisted_ids.add(event.event_id)
            except Exception:
                pass
        if self.event_bus is not None:
            try:
                self.event_bus.publish(
                    event_type="runtime_event_injected",
                    stage="event_queue",
                    tick=int(event.created_tick),
                    payload=event.to_timeline_row(event.created_day, event.created_tick),
                )
            except Exception:
                pass
        return event.event_id

    def append(
        self,
        *,
        event_type: str,
        title: str,
        raw_text: str,
        effective_day: int,
        effective_tick: Optional[int] = None,
        strength: float = 1.0,
        half_life: float = 7.0,
        scope: str = "broad_market",
        channels: Optional[Sequence[str]] = None,
        confidence: float = 0.75,
        source: str = "manual",
        metadata: Optional[Mapping[str, Any]] = None,
        created_day: int = 0,
        created_tick: int = 0,
    ) -> str:
        return self.append_event(
            ExperimentEvent.create(
                event_type=event_type,
                title=title,
                raw_text=raw_text,
                effective_day=effective_day,
                effective_tick=effective_tick,
                strength=strength,
                half_life=half_life,
                scope=scope,
                channels=channels,
                confidence=confidence,
                source=source,
                metadata=metadata,
                created_day=created_day,
                created_tick=created_tick,
            )
        )

    def digest_for_time(self, day_index: int, tick: Optional[int] = None) -> EventDigest:
        active: List[Dict[str, Any]] = []
        queued: List[Dict[str, Any]] = []
        expired: List[Dict[str, Any]] = []
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        aggregate = EventImpactProfile()
        aggregate_strength = 0.0

        for event in sorted(self.events, key=lambda item: (item.effective_day, item.effective_tick or -1, item.created_at)):
            state = event.state_for_time(day_index, tick)
            row = event.to_timeline_row(day_index, tick)
            if state == "active":
                intensity = event.intensity_for_time(day_index, tick)
                row["current_strength"] = round(float(intensity), 6)
                active.append(row)
                by_type.setdefault(event.event_type, []).append(row)
                aggregate_strength += float(intensity)
                aggregate = aggregate.merge(event.impact.scaled(float(intensity)))
            elif state == "queued":
                queued.append(row)
            else:
                expired.append(row)

        def _join(event_type: str, limit: int = 4) -> str:
            rows = by_type.get(event_type, [])
            parts = [str(item.get("raw_text", "") or item.get("title", "")).strip() for item in rows]
            return "；".join([part for part in parts if part][:limit])

        digest_payload = {
            "day_index": int(day_index),
            "tick": tick,
            "active_ids": [item["event_id"] for item in active],
            "strength": round(float(aggregate_strength), 8),
        }
        return EventDigest(
            day_index=int(day_index),
            tick=tick,
            active_events=active,
            queued_events=queued,
            expired_events=expired,
            by_type=by_type,
            policy_digest=_join("policy"),
            news_digest=_join("major_news"),
            rumor_digest=_join("rumor"),
            refute_digest=_join("refute"),
            macro_shock_digest=_join("macro_shock"),
            aggregate_strength=float(aggregate_strength),
            aggregate_impact=aggregate,
            event_hash=_stable_hash(digest_payload),
        )

    def timeline(self, day_index: int, tick: Optional[int] = None) -> List[Dict[str, Any]]:
        return [
            event.to_timeline_row(day_index, tick)
            for event in sorted(self.events, key=lambda item: (item.effective_day, item.effective_tick or -1, item.created_at))
        ]

    def active_events(self, day_index: int, tick: Optional[int] = None) -> List[ExperimentEvent]:
        return [event for event in self.events if event.state_for_time(day_index, tick) == "active"]


__all__ = [
    "EventDigest",
    "EventImpactProfile",
    "ExperimentEvent",
    "ScenarioEventQueue",
    "RUNTIME_EVENT_TYPES",
    "RUNTIME_EVENT_STATUSES",
    "compile_event_impact",
    "event_store_type_for_runtime",
    "normalize_runtime_event_type",
]
