"""交易所状态封装与步进执行器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from core.exchange.order_book import Order, OrderBook, Trade


@dataclass(slots=True)
class MarketState:
    """单步撮合后的盘口快照。"""

    step_id: int
    last_price: float
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    depth: Dict[str, List[Dict[str, float]]]
    activity_stats: Dict[str, int]


@dataclass(slots=True)
class ExchangeStepResult:
    """交易所单步执行结果。"""

    step_id: int
    trades: List[Trade]
    market_state: MarketState


class Exchange:
    """围绕订单簿的轻量交易所模拟器。"""

    def __init__(self, symbol: str = "A_SHARE_IDX", prev_close: float = 3000.0) -> None:
        self.order_book = OrderBook(symbol=symbol, prev_close=prev_close)
        self.step_id = 0
        self._vwap_numerator: float = 0.0
        self._vwap_denominator: int = 0

    def _build_market_state(self) -> MarketState:
        return MarketState(
            step_id=self.step_id,
            last_price=float(self.order_book.last_price),
            best_bid=self.order_book.get_best_bid(),
            best_ask=self.order_book.get_best_ask(),
            spread=self.order_book.get_spread(),
            depth=self.order_book.get_depth(5),
            activity_stats=self.order_book.get_activity_stats(),
        )

    def step(self, actions: Sequence[Order]) -> ExchangeStepResult:
        """执行一个撮合步。"""
        self.step_id += 1
        all_trades: List[Trade] = []

        for action in actions:
            trades = self.order_book.add_order(action)
            all_trades.extend(trades)

        for trade in all_trades:
            self._vwap_numerator += float(trade.price) * int(trade.quantity)
            self._vwap_denominator += int(trade.quantity)

        market_state = self._build_market_state()
        return ExchangeStepResult(step_id=self.step_id, trades=all_trades, market_state=market_state)

    def get_vwap(self) -> float:
        """返回当前会话 VWAP。无成交时回退到昨收。"""
        if self._vwap_denominator <= 0:
            return float(self.order_book.prev_close)
        return float(self._vwap_numerator / self._vwap_denominator)

    def end_of_day(self, close_price: float) -> None:
        """日终结算并重置会话 VWAP。"""
        self.order_book.update_prev_close(float(close_price))
        self._vwap_numerator = 0.0
        self._vwap_denominator = 0


__all__ = ["Exchange", "MarketState", "ExchangeStepResult"]
