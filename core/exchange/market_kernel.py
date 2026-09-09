"""Event-driven market kernel layered on top of the existing matching engine."""

from __future__ import annotations

import heapq
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from config import GLOBAL_CONFIG
from core.exchange.bar_builder import TradeTapeBarBuilder, TradeTapeEntry
from core.exchange.session_rules import AShareSessionRules
from core.exchange.trade_tape import TradeTape, TradeTapeRecord
from core.exchange.market_rules import resolve_market_rule_schema
from core.types import ExecutionPlan, Order, OrderSide, OrderStatus, OrderType, Trade

if TYPE_CHECKING:
    from core.time_manager import SimulationClock


def _stable_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(order=True)
class MarketEvent:
    """A scheduled market event in the kernel queue."""

    sort_key: Tuple[float, int, int] = field(init=False, repr=False)
    scheduled_ts: float
    priority: int
    sequence: int
    event_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        self.sort_key = (float(self.scheduled_ts), int(self.priority), int(self.sequence))


@dataclass
class MarketKernelConfig:
    """Kernel behavior switches and market microstructure parameters."""

    seed: int = 42
    board_lot: int = 100
    allow_odd_lots: bool = True
    enforce_board_lot: bool = False
    order_latency_ticks: int = 0
    cancel_latency_ticks: int = 0
    modify_latency_ticks: int = 0
    commission_rate: float = GLOBAL_CONFIG.TAX_RATE_COMMISSION
    stamp_duty_rate: float = GLOBAL_CONFIG.TAX_RATE_STAMP
    call_auction_start: str = "09:15"
    call_auction_end: str = "09:25"
    midday_break_start: str = "11:30"
    midday_break_end: str = "13:00"
    continuous_start: str = "09:30"
    continuous_end: str = "15:00"
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    market_rules: Optional[Dict[str, Any]] = None
    data_snapshot_hash: str = ""


def _parse_clock_time(value: str, fallback: dt_time) -> dt_time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except Exception:
        return fallback


class MarketKernel:
    """Event queue, session state machine, and trade-tape aggregator."""

    PRIORITY = {
        "halt": 0,
        "resume": 1,
        "cancel": 10,
        "modify": 20,
        "arrival": 30,
        "auction_flush": 40,
    }

    def __init__(
        self,
        *,
        symbol: str,
        prev_close: float,
        clock: Optional["SimulationClock"] = None,
        matching_engine: Optional[Any] = None,
        regulatory_module: Optional[Any] = None,
        config: Optional[MarketKernelConfig] = None,
    ) -> None:
        self.symbol = symbol
        self.prev_close = float(prev_close)
        self.clock = clock
        self.regulatory_module = regulatory_module
        self.config = config or MarketKernelConfig()
        self.rng = random.Random(int(self.config.seed))
        self.feature_flags = dict(self.config.feature_flags or {})
        self.a_share_sessions = AShareSessionRules.default()
        self.market_rules = resolve_market_rule_schema(
            symbol=symbol,
            prev_close=prev_close,
            overrides=self._build_market_rule_overrides(),
            feature_flags=self.feature_flags,
        )

        if matching_engine is not None:
            self.engine = matching_engine
        else:
            from core.market_engine import MatchingEngine

            self.engine = MatchingEngine(symbol=symbol, prev_close=prev_close, clock=clock)

        self._event_queue: List[MarketEvent] = []
        self._deferred_events: List[MarketEvent] = []
        self._auction_orders: List[Any] = []
        self._auction_phase: str = "open_call"
        self._live_orders: Dict[str, Any] = {}
        self._sequence = 0
        self._halted = False
        self._last_phase = "closed"
        self._current_ts = self._now_ts()
        self._step_trades: List[Trade] = []
        self.trade_tape: List[TradeTapeEntry] = []
        self.canonical_trade_tape: List[TradeTapeRecord] = []
        self._step_trade_tape: List[TradeTapeEntry] = []
        self._step_canonical_trade_tape: List[TradeTapeRecord] = []
        self.config_hash = _stable_hash(
            {
                "symbol": self.symbol,
                "prev_close": self.prev_close,
                "seed": self.config.seed,
                "feature_flags": self.feature_flags,
                "board_lot": self.config.board_lot,
                "allow_odd_lots": self.config.allow_odd_lots,
                "enforce_board_lot": self.config.enforce_board_lot,
                "market_rules": self.market_rules.to_dict(),
            }
        )
        self.bar_builder = TradeTapeBarBuilder(
            seed=self.config.seed,
            config_hash=self.config_hash,
            feature_flags=self.feature_flags,
            snapshot_info={"symbol": self.symbol, "prev_close": self.prev_close},
        )
        self.canonical_tape = TradeTape(
            symbol=self.symbol,
            seed=self.config.seed,
            config_hash=self.config_hash,
            data_snapshot_hash=self.config.data_snapshot_hash,
        )
        self.replay_metrics: Dict[str, Any] = {}

    def _now_ts(self) -> float:
        if self.clock is not None:
            return float(self.clock.timestamp)
        return float(time.time())

    def _now_tick(self) -> int:
        if self.clock is not None:
            return int(self.clock.ticks)
        return int(self._sequence)

    def _build_market_rule_overrides(self) -> Dict[str, Any]:
        overrides = dict(self.config.market_rules or {})
        overrides.setdefault("board_lot", self.config.board_lot)
        overrides.setdefault("allow_odd_lots", self.config.allow_odd_lots)
        overrides.setdefault("enforce_board_lot", self.config.enforce_board_lot)
        overrides.setdefault("commission_rate", self.config.commission_rate)
        overrides.setdefault("stamp_duty_rate", self.config.stamp_duty_rate)
        overrides.setdefault("call_auction_start", self.config.call_auction_start)
        overrides.setdefault("call_auction_end", self.config.call_auction_end)
        overrides.setdefault("midday_break_start", self.config.midday_break_start)
        overrides.setdefault("midday_break_end", self.config.midday_break_end)
        overrides.setdefault("continuous_start", self.config.continuous_start)
        overrides.setdefault("continuous_end", self.config.continuous_end)
        return overrides

    def _session_phase_from_schema(self, timestamp: float) -> str:
        session = self.market_rules.find_session(timestamp)
        if session is None:
            return "closed"
        return str(session.phase)

    def _next_active_ts_from_schema(self, timestamp: float) -> float:
        dt = datetime.fromtimestamp(float(timestamp))
        current = dt.time()
        sessions = list(self.market_rules.sessions)
        for session in sessions:
            if session.accepts_orders and current < session.start_time:
                return float(datetime.combine(dt.date(), session.start_time).timestamp())
            if session.contains(timestamp):
                for later in sessions:
                    if later.accepts_orders and later.start_time >= session.end_time:
                        return float(datetime.combine(dt.date(), later.start_time).timestamp())
                break
        next_day = dt.date() + timedelta(days=1)
        for session in sessions:
            if session.accepts_orders:
                return float(datetime.combine(next_day, session.start_time).timestamp())
        return float(timestamp)

    def _session_phase(self, timestamp: float) -> str:
        if self.feature_flags.get("a_share_session_v1", False):
            return self.a_share_sessions.phase_at(timestamp)
        if self.feature_flags.get("session_rules_v1", False):
            return self._session_phase_from_schema(timestamp)
        dt = datetime.fromtimestamp(float(timestamp))
        session_time = dt.time()
        call_start = _parse_clock_time(self.config.call_auction_start, dt_time(9, 15))
        call_end = _parse_clock_time(self.config.call_auction_end, dt_time(9, 25))
        cont_start = _parse_clock_time(self.config.continuous_start, dt_time(9, 30))
        break_start = _parse_clock_time(self.config.midday_break_start, dt_time(11, 30))
        break_end = _parse_clock_time(self.config.midday_break_end, dt_time(13, 0))
        cont_end = _parse_clock_time(self.config.continuous_end, dt_time(15, 0))

        if call_start <= session_time < call_end:
            return "call_auction"
        if break_start <= session_time < break_end:
            return "midday_break"
        if cont_start <= session_time < break_start or break_end <= session_time <= cont_end:
            return "continuous"
        return "closed"

    def _next_active_ts(self, timestamp: float) -> float:
        if self.feature_flags.get("a_share_session_v1", False):
            return self.a_share_sessions.next_matching_timestamp(timestamp)
        if self.feature_flags.get("session_rules_v1", False):
            return self._next_active_ts_from_schema(timestamp)
        dt = datetime.fromtimestamp(float(timestamp))
        session_time = dt.time()
        break_start = _parse_clock_time(self.config.midday_break_start, dt_time(11, 30))
        break_end = _parse_clock_time(self.config.midday_break_end, dt_time(13, 0))
        call_start = _parse_clock_time(self.config.call_auction_start, dt_time(9, 15))
        call_end = _parse_clock_time(self.config.call_auction_end, dt_time(9, 25))
        cont_start = _parse_clock_time(self.config.continuous_start, dt_time(9, 30))
        cont_end = _parse_clock_time(self.config.continuous_end, dt_time(15, 0))

        if break_start <= session_time < break_end:
            return float(datetime.combine(dt.date(), break_end).timestamp())
        if call_start <= session_time < call_end:
            return float(datetime.combine(dt.date(), cont_start).timestamp())
        if call_end <= session_time < cont_start:
            return float(datetime.combine(dt.date(), cont_start).timestamp())
        if session_time > cont_end:
            next_day = dt.date() + timedelta(days=1)
            return float(datetime.combine(next_day, call_start).timestamp())
        if session_time < call_start:
            return float(datetime.combine(dt.date(), call_start).timestamp())
        return float(timestamp)

    def _normalize_order(self, order: Any) -> Any:
        if isinstance(order, ExecutionPlan):
            qty = order.resolved_qty(self.engine.last_price)
            if self.market_rules.enforce_board_lot and not self.market_rules.allow_odd_lots and qty < self.market_rules.board_lot:
                return None
            if self.market_rules.is_feature_enabled("market_rules_v1"):
                qty = self.market_rules.normalize_quantity(qty)
            elif self.config.enforce_board_lot and qty >= self.config.board_lot:
                qty = max(self.config.board_lot, (qty // self.config.board_lot) * self.config.board_lot)
            if qty > 0:
                order.target_qty = qty
            return order

        if not isinstance(order, Order):
            return None

        if self.market_rules.enforce_board_lot and not self.market_rules.allow_odd_lots and order.quantity < self.market_rules.board_lot:
            order.status = OrderStatus.REJECTED
            order.reason = "board lot constraint"
            return None
        if self.market_rules.is_feature_enabled("market_rules_v1"):
            normalized_qty = self.market_rules.normalize_quantity(order.quantity)
            if normalized_qty <= 0:
                order.status = OrderStatus.REJECTED
                order.reason = "market rule quantity constraint"
                return None
            order.quantity = normalized_qty
            order.timestamp = self.market_rules.normalize_timestamp(order.timestamp)
        elif self.config.enforce_board_lot and order.quantity >= self.config.board_lot:
            order.quantity = max(self.config.board_lot, (int(order.quantity) // self.config.board_lot) * self.config.board_lot)
        return order

    def _estimate_queue_position(self, order: Any) -> int:
        try:
            return int(self.engine.lob.estimate_queue_position(order))
        except Exception:
            depth = self.engine.get_order_book_depth(10)
            queue_ahead = 0
            if isinstance(order, ExecutionPlan):
                ref_price = float(order.resolved_reference_price(self.engine.last_price))
                side = order.side
                qty = int(order.resolved_qty(ref_price))
            else:
                ref_price = float(order.price)
                side = order.side
                qty = int(order.quantity)
            if side == OrderSide.BUY:
                for level in depth.get("bids", []):
                    price = float(level.get("price", 0.0))
                    level_qty = int(level.get("qty", 0))
                    if price > ref_price or price == ref_price:
                        queue_ahead += level_qty
            else:
                for level in depth.get("asks", []):
                    price = float(level.get("price", 0.0))
                    level_qty = int(level.get("qty", 0))
                    if price < ref_price or price == ref_price:
                        queue_ahead += level_qty
            return int(max(0, queue_ahead))

    def _enqueue(self, event: MarketEvent) -> None:
        heapq.heappush(self._event_queue, event)

    def _enqueue_control_events(self, result: Optional[Dict[str, Any]]) -> None:
        if not result:
            return
        for payload in result.get("market_events", []) or []:
            event = MarketEvent(
                scheduled_ts=self._current_ts,
                priority=self.PRIORITY.get(str(payload.get("event_type", "")).lower(), 5),
                sequence=self._sequence,
                event_type=str(payload.get("event_type", "")),
                payload=dict(payload),
                reason=str(payload.get("reason", "")),
            )
            self._sequence += 1
            self._enqueue(event)

    def _sync_regulatory_state(self) -> None:
        if self.regulatory_module is None:
            return
        try:
            current_tick = self._now_tick()
            last_price = float(getattr(self.engine, "last_price", self.prev_close))
            panic_level = float(getattr(self.regulatory_module, "panic_level", 0.0))
            result = self.regulatory_module.process_tick(
                current_tick=current_tick,
                price=last_price,
                prev_close=self.prev_close,
                panic_level=panic_level,
            )
            self._enqueue_control_events(result)
        except Exception:
            return

    def _record_trade_tape(
        self,
        trades: Sequence[Trade],
        *,
        event: MarketEvent,
        phase: str,
    ) -> None:
        queue_position = int(event.payload.get("queue_position", 0))
        latency_ticks = int(event.payload.get("latency_ticks", 0))
        depth = {}
        spread = 0.0
        depth_imbalance = 0.0
        try:
            depth = self.engine.get_order_book_depth(5)
            bids = depth.get("bids", []) if isinstance(depth, dict) else []
            asks = depth.get("asks", []) if isinstance(depth, dict) else []
            bid_qty = sum(int(row.get("qty", 0)) for row in bids if isinstance(row, dict))
            ask_qty = sum(int(row.get("qty", 0)) for row in asks if isinstance(row, dict))
            denom = max(1, bid_qty + ask_qty)
            depth_imbalance = float((bid_qty - ask_qty) / denom)
            best_bid = self.engine.lob.get_best_bid()
            best_ask = self.engine.lob.get_best_ask()
            if best_bid is not None and best_ask is not None:
                spread = float(best_ask - best_bid)
        except Exception:
            depth = {}
        trading_day = self.a_share_sessions.trading_day_for(event.scheduled_ts)
        for trade in trades:
            metadata = {
                "symbol": self.symbol,
                "trading_day": trading_day,
                "seed": int(self.config.seed),
                "config_hash": self.config_hash,
                "data_snapshot_hash": self.config.data_snapshot_hash,
                "spread": spread,
                "depth_imbalance": depth_imbalance,
                **dict(event.payload),
            }
            entry = TradeTapeEntry(
                trade=trade,
                tick=self._now_tick(),
                phase=phase,
                event_type=str(event.event_type),
                queue_position=queue_position,
                latency_ticks=latency_ticks,
                market_timestamp=float(event.scheduled_ts),
                metadata=metadata,
            )
            canonical = self.canonical_tape.append_trade(
                trade,
                tick=self._now_tick(),
                trading_day=trading_day,
                phase=phase,
                queue_position=queue_position,
                latency_ticks=latency_ticks,
                market_timestamp=float(event.scheduled_ts),
                metadata=metadata,
            )
            self.trade_tape.append(entry)
            self.canonical_trade_tape.append(canonical)
            self._step_trade_tape.append(entry)
            self._step_canonical_trade_tape.append(canonical)
            self._step_trades.append(trade)

    def _flush_auction_if_needed(self, timestamp: float) -> List[Trade]:
        phase = self._session_phase(timestamp)
        auction_phases = {"call_auction", "open_call", "close_call"}
        if phase in auction_phases or not self._auction_orders:
            self._last_phase = phase
            return []

        trades: List[Trade] = []
        if self._auction_orders:
            opening_price, auction_trades = self.engine.run_call_auction(self._auction_orders, market_time=timestamp)
            self.engine.last_price = opening_price
            trades.extend(auction_trades)
            self._record_trade_tape(
                auction_trades,
                event=MarketEvent(
                    scheduled_ts=timestamp,
                    priority=self.PRIORITY["auction_flush"],
                    sequence=self._sequence,
                    event_type="auction_flush",
                    payload={"auction_orders": len(self._auction_orders), "phase": self._auction_phase},
                    reason="call auction flush",
                ),
                phase=self._auction_phase,
            )
            self._auction_orders = []
            self._auction_phase = "open_call"
        self._last_phase = phase
        return trades

    def _process_arrival(self, event: MarketEvent) -> List[Trade]:
        order = self._normalize_order(event.payload.get("order"))
        if order is None:
            return []
        phase = self._session_phase(event.scheduled_ts)
        queue_position = self._estimate_queue_position(order)
        event.payload["queue_position"] = queue_position
        event.payload["latency_ticks"] = int(event.payload.get("latency_ticks", 0))
        event.payload["phase"] = phase
        if hasattr(order, "metadata"):
            order.metadata.update(
                {
                    "queue_position": queue_position,
                    "latency_ticks": int(event.payload["latency_ticks"]),
                    "market_phase": phase,
                    "kernel_config_hash": self.config_hash,
                }
            )
        self._live_orders[getattr(order, "order_id", str(id(order)))] = order

        if self._halted or phase in {"midday_break", "closed"}:
            event.payload["deferred_to"] = self._next_active_ts(event.scheduled_ts)
            self._deferred_events.append(event)
            return []

        if phase in {"call_auction", "open_call", "close_call"}:
            self._auction_orders.append(order)
            self._auction_phase = phase
            return []

        trades = self.engine.submit_order(order, liquidity_injection_prob=float(event.payload.get("liquidity_injection_prob", 0.0)))
        self._record_trade_tape(trades, event=event, phase=phase)
        if getattr(order, "is_filled", False):
            self._live_orders.pop(getattr(order, "order_id", ""), None)
        return list(trades)

    def _process_cancel(self, event: MarketEvent) -> List[Trade]:
        order_id = str(event.payload.get("order_id", ""))
        if not order_id:
            return []
        cancel_fn = getattr(self.engine.lob, "cancel_order", None)
        if callable(cancel_fn):
            cancel_fn(order_id)
        self._live_orders.pop(order_id, None)
        return []

    def _process_modify(self, event: MarketEvent) -> List[Trade]:
        order_id = str(event.payload.get("order_id", ""))
        original = self._live_orders.get(order_id)
        if original is None:
            return []
        replacement = getattr(self.engine.lob, "modify_order", None)
        if callable(replacement):
            maybe_replacement = replacement(
                order_id,
                price=event.payload.get("price"),
                quantity=event.payload.get("quantity"),
                timestamp=event.scheduled_ts,
                original_order=original,
            )
            if maybe_replacement is not None:
                self._live_orders.pop(order_id, None)
                event.payload["order"] = maybe_replacement
                event.event_type = "arrival"
                return self._process_arrival(event)

        cancel_fn = getattr(self.engine.lob, "cancel_order", None)
        if callable(cancel_fn):
            cancel_fn(order_id)
        self._live_orders.pop(order_id, None)

        order_type = getattr(original, "order_type", OrderType.LIMIT)
        new_order = Order(
            symbol=getattr(original, "symbol", self.symbol),
            price=float(event.payload.get("price", getattr(original, "price", self.engine.last_price))),
            quantity=int(event.payload.get("quantity", getattr(original, "remaining_qty", getattr(original, "quantity", 0)))),
            side=getattr(original, "side", OrderSide.BUY),
            order_type=order_type,
            agent_id=getattr(original, "agent_id", "unknown"),
            timestamp=float(event.scheduled_ts),
            metadata=dict(getattr(original, "metadata", {})),
        )
        new_order.metadata.update(
            {
                "replaces_order_id": order_id,
                "replace_reason": "kernel_modify",
            }
        )
        event.payload["order"] = new_order
        event.event_type = "arrival"
        return self._process_arrival(event)

    def _process_control(self, event: MarketEvent) -> List[Trade]:
        if event.event_type == "halt":
            self._halted = True
        elif event.event_type == "resume":
            self._halted = False
        elif event.event_type == "intervention":
            self._deferred_events.extend([])
        return []

    def _process_event(self, event: MarketEvent) -> List[Trade]:
        if event.event_type == "arrival":
            return self._process_arrival(event)
        if event.event_type == "cancel":
            return self._process_cancel(event)
        if event.event_type == "modify":
            return self._process_modify(event)
        if event.event_type in {"halt", "resume", "intervention"}:
            return self._process_control(event)
        return []

    def _release_deferred(self, timestamp: float) -> None:
        if not self._deferred_events:
            return
        pending = self._deferred_events
        self._deferred_events = []
        for event in pending:
            event.scheduled_ts = max(float(timestamp), float(event.scheduled_ts))
            event.sort_key = (event.scheduled_ts, event.priority, event.sequence)
            self._enqueue(event)

    def advance_to(self, timestamp: Optional[float] = None) -> List[Trade]:
        """Process all events scheduled up to ``timestamp``."""
        target_ts = float(self._now_ts() if timestamp is None else timestamp)
        self._current_ts = target_ts
        self._sync_regulatory_state()
        phase = self._session_phase(target_ts)
        if not self._halted and phase in {"call_auction", "open_call", "close_call", "continuous"}:
            self._release_deferred(target_ts)

        generated: List[Trade] = []
        while self._event_queue and self._event_queue[0].sort_key <= (target_ts, 999, 999999):
            event = heapq.heappop(self._event_queue)
            if event.event_type in {"halt", "resume", "intervention"}:
                generated.extend(self._process_event(event))
                if not self._halted and self._session_phase(event.scheduled_ts) in {"call_auction", "open_call", "close_call", "continuous"}:
                    self._release_deferred(event.scheduled_ts)
                continue

            if self._halted:
                event.payload["deferred_to"] = self._next_active_ts(event.scheduled_ts)
                self._deferred_events.append(event)
                continue

            phase = self._session_phase(event.scheduled_ts)
            if phase in {"midday_break", "closed"}:
                event.payload["deferred_to"] = self._next_active_ts(event.scheduled_ts)
                self._deferred_events.append(event)
                continue

            generated.extend(self._process_event(event))

        generated.extend(self._flush_auction_if_needed(target_ts))
        return generated

    def submit_order(
        self,
        order: Order | ExecutionPlan,
        *,
        current_timestamp: Optional[float] = None,
        latency_ticks: Optional[int] = None,
        liquidity_injection_prob: float = 0.0,
    ) -> List[Trade]:
        ts = float(self._now_ts() if current_timestamp is None else current_timestamp)
        order_latency = int(self.config.order_latency_ticks if latency_ticks is None else latency_ticks)
        if isinstance(order, (Order, ExecutionPlan)):
            queue_position = self._estimate_queue_position(order)
            payload = {
                "order": order,
                "queue_position": queue_position,
                "latency_ticks": order_latency,
                "liquidity_injection_prob": float(liquidity_injection_prob),
            }
            event = MarketEvent(
                scheduled_ts=ts + (order_latency * max(1.0, getattr(self.clock, "time_step_seconds", 1.0))),
                priority=self.PRIORITY["arrival"],
                sequence=self._sequence,
                event_type="arrival",
                payload=payload,
                reason="order arrival",
            )
            self._sequence += 1
            self._enqueue(event)
            return self.advance_to(ts)
        return []

    def cancel_order(self, order_id: str, *, current_timestamp: Optional[float] = None) -> List[Trade]:
        ts = float(self._now_ts() if current_timestamp is None else current_timestamp)
        event = MarketEvent(
            scheduled_ts=ts + (self.config.cancel_latency_ticks * max(1.0, getattr(self.clock, "time_step_seconds", 1.0))),
            priority=self.PRIORITY["cancel"],
            sequence=self._sequence,
            event_type="cancel",
            payload={"order_id": str(order_id)},
            reason="cancel request",
        )
        self._sequence += 1
        self._enqueue(event)
        return self.advance_to(ts)

    def modify_order(
        self,
        order_id: str,
        *,
        price: Optional[float] = None,
        quantity: Optional[int] = None,
        current_timestamp: Optional[float] = None,
    ) -> List[Trade]:
        ts = float(self._now_ts() if current_timestamp is None else current_timestamp)
        event = MarketEvent(
            scheduled_ts=ts + (self.config.modify_latency_ticks * max(1.0, getattr(self.clock, "time_step_seconds", 1.0))),
            priority=self.PRIORITY["modify"],
            sequence=self._sequence,
            event_type="modify",
            payload={"order_id": str(order_id), "price": price, "quantity": quantity},
            reason="modify request",
        )
        self._sequence += 1
        self._enqueue(event)
        return self.advance_to(ts)

    def inject_halt(self, *, reason: str = "manual halt", timestamp: Optional[float] = None, level: int = 1) -> None:
        ts = float(self._now_ts() if timestamp is None else timestamp)
        self._sequence += 1
        self._enqueue(
            MarketEvent(
                scheduled_ts=ts,
                priority=self.PRIORITY["halt"],
                sequence=self._sequence,
                event_type="halt",
                payload={"level": int(level)},
                reason=reason,
            )
        )

    def inject_resume(self, *, reason: str = "manual resume", timestamp: Optional[float] = None) -> None:
        ts = float(self._now_ts() if timestamp is None else timestamp)
        self._sequence += 1
        self._enqueue(
            MarketEvent(
                scheduled_ts=ts,
                priority=self.PRIORITY["resume"],
                sequence=self._sequence,
                event_type="resume",
                payload={},
                reason=reason,
            )
        )

    def flush_step_trades(self) -> List[Trade]:
        trades = self._step_trades[:]
        self._step_trades = []
        return trades

    def flush_step_trade_tape(self) -> List[TradeTapeEntry]:
        tape = self._step_trade_tape[:]
        self._step_trade_tape = []
        return tape

    def get_trade_tape(self) -> List[TradeTapeEntry]:
        return list(self.trade_tape)

    def flush_step_canonical_trade_tape(self) -> List[TradeTapeRecord]:
        tape = self._step_canonical_trade_tape[:]
        self._step_canonical_trade_tape = []
        return tape

    def get_canonical_trade_tape(self) -> List[TradeTapeRecord]:
        return list(self.canonical_trade_tape)

    def build_replay_metrics(self) -> Dict[str, Any]:
        bars = self.bar_builder.build_bars_from_trade_tape(
            self.trade_tape,
            symbol=self.symbol,
            prev_close=self.prev_close,
            is_simulated=True,
        )
        self.replay_metrics = self.bar_builder.build_replay_metrics(self.trade_tape, bars)
        self.replay_metrics["bars"] = [bar.__dict__ for bar in bars]
        self.replay_metrics["trade_tape_size"] = len(self.trade_tape)
        return dict(self.replay_metrics)

    def clear(self) -> None:
        self._event_queue.clear()
        self._deferred_events.clear()
        self._auction_orders.clear()
        self._live_orders.clear()
        self._step_trades.clear()
        self._step_trade_tape.clear()
        self._step_canonical_trade_tape.clear()
        self.trade_tape.clear()
        self.canonical_trade_tape.clear()
        self.canonical_tape = TradeTape(
            symbol=self.symbol,
            seed=self.config.seed,
            config_hash=self.config_hash,
            data_snapshot_hash=self.config.data_snapshot_hash,
        )
        self.replay_metrics = {}
        self._sequence = 0
        self._halted = False
        self._last_phase = "closed"
        self._auction_phase = "open_call"


__all__ = ["MarketEvent", "MarketKernel", "MarketKernelConfig"]
