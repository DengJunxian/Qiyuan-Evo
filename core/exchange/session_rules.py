"""China A-share session timeline helpers.

This module is intentionally independent from the existing market rule schema:
the schema remains the backward-compatible source for legacy order-book tests,
while this helper provides the richer A-share trading-day phases used by the
session-aware market kernel and replay tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence


ACTIVE_MATCHING_PHASES = {"continuous"}
CALL_AUCTION_PHASES = {"open_call", "close_call"}
ORDER_ACCEPTING_PHASES = {"open_call", "continuous", "midday_break", "close_call"}


def _parse_hhmm(value: str, fallback: dt_time) -> dt_time:
    try:
        return datetime.strptime(str(value), "%H:%M").time()
    except Exception:
        return fallback


@dataclass(frozen=True)
class AShareSessionEvent:
    """One segment of a China A-share trading day."""

    name: str
    phase: str
    start: str
    end: str
    accepts_orders: bool = True
    matches_immediately: bool = False
    auction_uncross: bool = False
    notes: str = ""

    @property
    def start_time(self) -> dt_time:
        return _parse_hhmm(self.start, dt_time(0, 0))

    @property
    def end_time(self) -> dt_time:
        return _parse_hhmm(self.end, dt_time(23, 59))

    def contains(self, timestamp: float | datetime) -> bool:
        current = timestamp if isinstance(timestamp, datetime) else datetime.fromtimestamp(float(timestamp))
        t = current.time()
        return self.start_time <= t < self.end_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "phase": self.phase,
            "start": self.start,
            "end": self.end,
            "accepts_orders": bool(self.accepts_orders),
            "matches_immediately": bool(self.matches_immediately),
            "auction_uncross": bool(self.auction_uncross),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class AShareSessionRules:
    """Default A-share day with open/close calls and midday break."""

    trading_day: Optional[str] = None
    events: Sequence[AShareSessionEvent] = field(default_factory=tuple)

    @classmethod
    def default(cls, *, trading_day: Optional[str] = None) -> "AShareSessionRules":
        return cls(
            trading_day=trading_day,
            events=(
                AShareSessionEvent(
                    name="opening_call_auction",
                    phase="open_call",
                    start="09:15",
                    end="09:25",
                    accepts_orders=True,
                    matches_immediately=False,
                    auction_uncross=True,
                ),
                AShareSessionEvent(
                    name="pre_open_pause",
                    phase="pre_open_pause",
                    start="09:25",
                    end="09:30",
                    accepts_orders=False,
                    matches_immediately=False,
                ),
                AShareSessionEvent(
                    name="continuous_am",
                    phase="continuous",
                    start="09:30",
                    end="11:30",
                    accepts_orders=True,
                    matches_immediately=True,
                ),
                AShareSessionEvent(
                    name="midday_break",
                    phase="midday_break",
                    start="11:30",
                    end="13:00",
                    accepts_orders=True,
                    matches_immediately=False,
                    notes="Orders are accepted into the queue but not matched.",
                ),
                AShareSessionEvent(
                    name="continuous_pm",
                    phase="continuous",
                    start="13:00",
                    end="14:57",
                    accepts_orders=True,
                    matches_immediately=True,
                ),
                AShareSessionEvent(
                    name="closing_call_auction",
                    phase="close_call",
                    start="14:57",
                    end="15:00",
                    accepts_orders=True,
                    matches_immediately=False,
                    auction_uncross=True,
                ),
                AShareSessionEvent(
                    name="market_close",
                    phase="closed",
                    start="15:00",
                    end="23:59",
                    accepts_orders=False,
                    matches_immediately=False,
                ),
            ),
        )

    @classmethod
    def from_events(
        cls,
        events: Iterable[AShareSessionEvent | Dict[str, Any]],
        *,
        trading_day: Optional[str] = None,
    ) -> "AShareSessionRules":
        normalized: List[AShareSessionEvent] = []
        for item in events:
            normalized.append(item if isinstance(item, AShareSessionEvent) else AShareSessionEvent(**dict(item)))
        return cls(trading_day=trading_day, events=tuple(normalized))

    def event_at(self, timestamp: float | datetime) -> Optional[AShareSessionEvent]:
        for event in self.events:
            if event.contains(timestamp):
                return event
        return None

    def phase_at(self, timestamp: float | datetime) -> str:
        event = self.event_at(timestamp)
        return event.phase if event is not None else "closed"

    def accepts_orders(self, timestamp: float | datetime) -> bool:
        event = self.event_at(timestamp)
        return bool(event and event.accepts_orders)

    def matches_immediately(self, timestamp: float | datetime) -> bool:
        event = self.event_at(timestamp)
        return bool(event and event.matches_immediately)

    def is_call_auction(self, timestamp: float | datetime) -> bool:
        return self.phase_at(timestamp) in CALL_AUCTION_PHASES

    def next_order_accepting_timestamp(self, timestamp: float | datetime) -> float:
        current = timestamp if isinstance(timestamp, datetime) else datetime.fromtimestamp(float(timestamp))
        for event in self.events:
            if event.accepts_orders and current.time() < event.start_time:
                return float(datetime.combine(current.date(), event.start_time).timestamp())
            if event.contains(current):
                if event.accepts_orders:
                    return float(current.timestamp())
                for later in self.events:
                    if later.accepts_orders and later.start_time >= event.end_time:
                        return float(datetime.combine(current.date(), later.start_time).timestamp())
        next_day = current.date() + timedelta(days=1)
        for event in self.events:
            if event.accepts_orders:
                return float(datetime.combine(next_day, event.start_time).timestamp())
        return float(current.timestamp())

    def next_matching_timestamp(self, timestamp: float | datetime) -> float:
        current = timestamp if isinstance(timestamp, datetime) else datetime.fromtimestamp(float(timestamp))
        for event in self.events:
            if event.matches_immediately and current.time() < event.start_time:
                return float(datetime.combine(current.date(), event.start_time).timestamp())
            if event.contains(current):
                if event.matches_immediately:
                    return float(current.timestamp())
                for later in self.events:
                    if later.matches_immediately and later.start_time >= event.end_time:
                        return float(datetime.combine(current.date(), later.start_time).timestamp())
        next_day = current.date() + timedelta(days=1)
        for event in self.events:
            if event.matches_immediately:
                return float(datetime.combine(next_day, event.start_time).timestamp())
        return float(current.timestamp())

    def trading_day_for(self, timestamp: float | datetime) -> str:
        if self.trading_day:
            return str(self.trading_day)
        current = timestamp if isinstance(timestamp, datetime) else datetime.fromtimestamp(float(timestamp))
        return current.date().isoformat()

    def timeline_for_day(self, day: date | str) -> List[Dict[str, Any]]:
        day_obj = datetime.strptime(day, "%Y-%m-%d").date() if isinstance(day, str) else day
        rows: List[Dict[str, Any]] = []
        for event in self.events:
            payload = event.to_dict()
            payload["trading_day"] = day_obj.isoformat()
            payload["start_timestamp"] = float(datetime.combine(day_obj, event.start_time).timestamp())
            payload["end_timestamp"] = float(datetime.combine(day_obj, event.end_time).timestamp())
            rows.append(payload)
        return rows

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trading_day": self.trading_day,
            "schema": "a_share_session_rules/v1",
            "events": [event.to_dict() for event in self.events],
        }


def normalize_phase(phase: str) -> str:
    text = str(phase or "").strip().lower()
    aliases = {
        "call_auction": "open_call",
        "opening_call_auction": "open_call",
        "closing_call_auction": "close_call",
        "auction_flush": "open_call",
    }
    return aliases.get(text, text or "closed")


__all__ = [
    "ACTIVE_MATCHING_PHASES",
    "CALL_AUCTION_PHASES",
    "ORDER_ACCEPTING_PHASES",
    "AShareSessionEvent",
    "AShareSessionRules",
    "normalize_phase",
]
