"""A-share session-aware helpers layered on the existing matching engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.exchange.session_rules import AShareSessionRules
from core.exchange.trade_tape import TradeTape, TradeTapeRecord
from core.types import Order, OrderSide, OrderStatus, Trade


def find_price_maximizing_volume_and_minimizing_imbalance(
    orders: Sequence[Order],
    *,
    prev_close: float,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Select an A-share call-auction uncross price.

    Tie-break order: maximum executable volume, minimum imbalance, closest to
    previous close, then lower price. The final lower-price tie is deterministic
    and keeps replay stable.
    """

    buys = [order for order in orders if order.side == OrderSide.BUY and order.remaining_qty > 0]
    sells = [order for order in orders if order.side == OrderSide.SELL and order.remaining_qty > 0]
    if not buys or not sells:
        return float(prev_close), {"match_volume": 0, "imbalance": 0, "candidate_count": 0}

    prices = sorted({float(order.price) for order in [*buys, *sells]})
    if lower_limit is not None:
        prices = [price for price in prices if float(price) >= float(lower_limit)]
    if upper_limit is not None:
        prices = [price for price in prices if float(price) <= float(upper_limit)]
    if not prices:
        return float(prev_close), {"match_volume": 0, "imbalance": 0, "candidate_count": 0}

    candidates: List[Tuple[int, int, float, float]] = []
    for price in prices:
        buy_volume = sum(int(order.remaining_qty) for order in buys if float(order.price) >= price)
        sell_volume = sum(int(order.remaining_qty) for order in sells if float(order.price) <= price)
        match_volume = min(buy_volume, sell_volume)
        imbalance = abs(buy_volume - sell_volume)
        candidates.append((int(match_volume), int(imbalance), abs(float(price) - float(prev_close)), float(price)))

    best = sorted(candidates, key=lambda item: (-item[0], item[1], item[2], item[3]))[0]
    return float(best[3]), {
        "match_volume": int(best[0]),
        "imbalance": int(best[1]),
        "distance_to_prev_close": float(best[2]),
        "candidate_count": int(len(candidates)),
    }


def call_auction_match(
    orders: Sequence[Order],
    *,
    price: float,
    timestamp: float,
    commission_rate: float,
    stamp_duty_rate: float,
    seller_only_stamp_duty: bool = True,
) -> List[Trade]:
    executable_buys = sorted(
        [order for order in orders if order.side == OrderSide.BUY and order.price >= price and order.remaining_qty > 0],
        key=lambda order: (-float(order.price), float(order.timestamp), str(order.order_id)),
    )
    executable_sells = sorted(
        [order for order in orders if order.side == OrderSide.SELL and order.price <= price and order.remaining_qty > 0],
        key=lambda order: (float(order.price), float(order.timestamp), str(order.order_id)),
    )
    trades: List[Trade] = []
    buy_idx = 0
    sell_idx = 0
    while buy_idx < len(executable_buys) and sell_idx < len(executable_sells):
        buy_order = executable_buys[buy_idx]
        sell_order = executable_sells[sell_idx]
        qty = min(int(buy_order.remaining_qty), int(sell_order.remaining_qty))
        if qty <= 0:
            if buy_order.remaining_qty <= 0:
                buy_idx += 1
            if sell_order.remaining_qty <= 0:
                sell_idx += 1
            continue

        buy_order.filled_qty += qty
        sell_order.filled_qty += qty
        for order in (buy_order, sell_order):
            order.status = OrderStatus.FILLED if order.is_filled else OrderStatus.PARTIAL
        notional = float(price) * qty
        maker = buy_order if (buy_order.timestamp, buy_order.order_id) <= (sell_order.timestamp, sell_order.order_id) else sell_order
        taker = sell_order if maker is buy_order else buy_order
        trades.append(
            Trade(
                trade_id=f"auction_{len(trades):06d}",
                price=float(price),
                quantity=int(qty),
                maker_id=str(maker.order_id),
                taker_id=str(taker.order_id),
                maker_agent_id=str(maker.agent_id),
                taker_agent_id=str(taker.agent_id),
                buyer_agent_id=str(buy_order.agent_id),
                seller_agent_id=str(sell_order.agent_id),
                timestamp=float(timestamp),
                buyer_fee=notional * float(commission_rate),
                seller_fee=notional * float(commission_rate),
                seller_tax=notional * float(stamp_duty_rate) if seller_only_stamp_duty else 0.0,
            )
        )
        if buy_order.remaining_qty <= 0:
            buy_idx += 1
        if sell_order.remaining_qty <= 0:
            sell_idx += 1
    return trades


@dataclass
class AShareSessionEngine:
    """Thin facade for tests and callers that need session/tape primitives."""

    symbol: str
    prev_close: float
    seed: int = 42
    config_hash: str = ""
    data_snapshot_hash: str = ""
    session_rules: AShareSessionRules = field(default_factory=AShareSessionRules.default)

    def __post_init__(self) -> None:
        self.trade_tape = TradeTape(
            symbol=self.symbol,
            seed=self.seed,
            config_hash=self.config_hash,
            data_snapshot_hash=self.data_snapshot_hash,
        )
        self.halted = False

    def halt(self) -> None:
        self.halted = True

    def resume(self) -> None:
        self.halted = False

    def phase_at(self, timestamp: float) -> str:
        if self.halted:
            return "halted"
        return self.session_rules.phase_at(timestamp)

    def record_trades(
        self,
        trades: Sequence[Trade],
        *,
        tick: int,
        timestamp: float,
        phase: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TradeTapeRecord]:
        trading_day = self.session_rules.trading_day_for(timestamp)
        return [
            self.trade_tape.append_trade(
                trade,
                tick=tick,
                trading_day=trading_day,
                phase=phase,
                market_timestamp=timestamp,
                metadata=metadata or {},
            )
            for trade in trades
        ]


__all__ = [
    "AShareSessionEngine",
    "call_auction_match",
    "find_price_maximizing_volume_and_minimizing_imbalance",
]
