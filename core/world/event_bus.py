"""Event bus primitives for staged simulation orchestration."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional


@dataclass(slots=True)
class SimulationEvent:
    """A typed event emitted by one stage and consumed by another stage."""

    event_type: str
    stage: str
    tick: int
    payload: Dict[str, Any]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event as a plain dict."""
        return asdict(self)


class EventBus:
    """In-memory event bus used by policy/macro/social/micro layers."""

    def __init__(self, max_history: int = 1024):
        self._queue: Deque[SimulationEvent] = deque()
        self._history: Deque[SimulationEvent] = deque(maxlen=max_history)

    def publish(self, *, event_type: str, stage: str, tick: int, payload: Dict[str, Any]) -> SimulationEvent:
        """Publish a new event to the bus."""
        event = SimulationEvent(event_type=event_type, stage=stage, tick=tick, payload=payload)
        self._queue.append(event)
        self._history.append(event)
        return event

    def consume_next(self) -> Optional[SimulationEvent]:
        """Consume next event in FIFO order."""
        if not self._queue:
            return None
        return self._queue.popleft()

    def consume_stage(self, stage: str) -> List[SimulationEvent]:
        """Consume and return all queued events that belong to one stage."""
        stage = str(stage)
        remain: Deque[SimulationEvent] = deque()
        selected: List[SimulationEvent] = []
        while self._queue:
            event = self._queue.popleft()
            if event.stage == stage:
                selected.append(event)
            else:
                remain.append(event)
        self._queue = remain
        return selected

    def drain(self) -> List[SimulationEvent]:
        """Consume all queued events."""
        items = list(self._queue)
        self._queue.clear()
        return items

    def recent(self, limit: int = 50, stage: Optional[str] = None) -> List[SimulationEvent]:
        """Return recent event history, optionally filtered by stage."""
        events = list(self._history)
        if stage is not None:
            stage = str(stage)
            events = [item for item in events if item.stage == stage]
        return events[-max(0, int(limit)) :]

    def __len__(self) -> int:
        return len(self._queue)


if __name__ == "__main__":
    bus = EventBus()
    bus.publish(event_type="policy_shock", stage="policy", tick=1, payload={"text": "印花税下调"})
    bus.publish(event_type="macro_state", stage="macro_update", tick=1, payload={"liquidity_index": 1.05})
    for event in bus.drain():
        print(event.to_dict())
