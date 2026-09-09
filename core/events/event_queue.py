"""Runtime event queue facade with time-based triggering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from core.events.event_compiler import RuntimeEventCompiler
from core.events.runtime_event import RuntimeEvent
from core.experiment_events import ScenarioEventQueue


@dataclass
class RuntimeEventQueue:
    compiler: RuntimeEventCompiler = field(default_factory=RuntimeEventCompiler)
    events: List[RuntimeEvent] = field(default_factory=list)
    triggered_ids: set[str] = field(default_factory=set)

    def append(self, event: RuntimeEvent | Mapping[str, Any] | str, **kwargs: Any) -> RuntimeEvent:
        compiled = self.compiler.compile(event, **kwargs)
        self.events.append(compiled)
        self.events.sort(key=lambda item: (float(item.timestamp), str(item.event_id)))
        return compiled

    def extend(self, events: Iterable[RuntimeEvent | Mapping[str, Any] | str]) -> List[RuntimeEvent]:
        return [self.append(event) for event in events]

    def due_events(self, timestamp: float, *, include_triggered: bool = False) -> List[RuntimeEvent]:
        rows = [event for event in self.events if float(event.timestamp) <= float(timestamp)]
        if include_triggered:
            return rows
        return [event for event in rows if event.event_id not in self.triggered_ids]

    def trigger_due(self, timestamp: float) -> List[RuntimeEvent]:
        due = self.due_events(timestamp)
        for event in due:
            self.triggered_ids.add(event.event_id)
        return due

    def to_scenario_event_queue(
        self,
        *,
        start_day: int = 1,
        event_store: Any = None,
        dataset_version: str = "runtime_events",
        persist_events: bool = False,
    ) -> ScenarioEventQueue:
        queue = ScenarioEventQueue(
            event_store=event_store,
            dataset_version=dataset_version,
            persist_events=bool(persist_events),
        )
        for offset, event in enumerate(self.events):
            queue.append_event(
                event.to_experiment_event(
                    effective_day=max(1, int(start_day) + offset),
                    created_day=max(0, int(start_day) - 1),
                )
            )
        return queue

    def effect_summary(self, events: Optional[Sequence[RuntimeEvent]] = None) -> Dict[str, Any]:
        rows = list(events if events is not None else self.events)
        sentiment = sum(float(row.structured_payload.get("sentiment_impact", 0.0) or 0.0) * row.shock_strength for row in rows)
        liquidity = sum(float(row.structured_payload.get("liquidity_impact", 0.0) or 0.0) * row.shock_strength for row in rows)
        funding = sum(float(row.structured_payload.get("funding_stress", 0.0) or 0.0) * row.shock_strength for row in rows)
        compliance = sum(float(row.structured_payload.get("compliance_pressure", 0.0) or 0.0) * row.shock_strength for row in rows)
        affected_groups = sorted({group for row in rows for group in row.affected_agent_groups})
        affected_sectors = sorted({sector for row in rows for sector in row.affected_sectors})
        return {
            "active_count": int(len(rows)),
            "macro_state_delta": {
                "sentiment_index": float(sentiment),
                "liquidity_index": float(liquidity),
                "funding_stress": float(funding),
            },
            "agent_belief_delta": {
                "credibility_weighted_shock": float(sum(row.credibility * row.shock_strength for row in rows)),
                "belief_dispersion": float(min(1.0, abs(sentiment) + funding * 0.5)),
            },
            "order_flow_delta": {
                "buy_pressure": float(max(0.0, sentiment + liquidity)),
                "sell_pressure": float(max(0.0, -sentiment + funding)),
                "compliance_pressure": float(compliance),
            },
            "report_narrative": "; ".join(row.raw_text for row in rows if row.raw_text),
            "affected_agent_groups": affected_groups,
            "affected_sectors": affected_sectors,
        }


__all__ = ["RuntimeEventQueue"]
