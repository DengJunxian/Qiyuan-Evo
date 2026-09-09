from __future__ import annotations

from typing import Dict, List, Optional

from core.exchange.order_book import OrderBook
from core.types import Order, OrderSide, OrderStatus, OrderType, Trade

try:
    import _civitas_lob
except ImportError:
    _civitas_lob = None


class OrderBookCPP(OrderBook):
    """C++ optimized order book wrapper with schema-compatible rules."""

    def __init__(
        self,
        symbol: str = "A_SHARE_IDX",
        prev_close: float = 3000.0,
        market_rules: Optional[dict] = None,
        feature_flags: Optional[dict] = None,
    ):
        super().__init__(symbol, prev_close, market_rules=market_rules, feature_flags=feature_flags)
        if _civitas_lob is None:
            raise ImportError("C++ extension _civitas_lob not found. Please build with setup.py.")
        self._cpp_lob = _civitas_lob.LimitOrderBook(symbol, self._build_cpp_rule_config())
        self.submitted_orders = 0
        self.cancelled_orders = 0
        self.trade_count = 0

    def _build_cpp_rule_config(self):
        config = _civitas_lob.RuleConfig()
        config.commission_rate = float(self.market_rules.commission_rate)
        config.stamp_duty_rate = float(self.market_rules.stamp_duty_rate)
        config.min_price_tick = float(self.market_rules.min_price_tick)
        config.min_trade_unit = int(self.market_rules.min_trade_unit)
        config.board_lot = int(self.market_rules.board_lot)
        config.enforce_min_trade_unit = bool(self.market_rules.enforce_min_trade_unit)
        config.enforce_board_lot = bool(self.market_rules.enforce_board_lot)
        config.allow_odd_lots = bool(self.market_rules.allow_odd_lots)
        config.strict_queue_timestamps = bool(
            self.market_rules.is_feature_enabled("strict_queue_timestamps")
            or self.market_rules.is_feature_enabled("market_rules_v1")
        )
        config.timestamp_precision = str(self.market_rules.timestamp_precision)
        return config

    def estimate_available_qty(self, side: OrderSide, price: float) -> int:
        depth = self.get_depth(levels=50)
        total = 0
        if side == OrderSide.BUY:
            for level in depth.get("asks", []):
                if float(level.get("price", 0.0)) <= float(price):
                    total += int(level.get("qty", 0))
        else:
            for level in depth.get("bids", []):
                if float(level.get("price", 0.0)) >= float(price):
                    total += int(level.get("qty", 0))
        return int(total)

    def add_order(self, order: Order) -> List[Trade]:
        if not self._preprocess_order(order):
            return []

        self.submitted_orders += 1
        side_str = order.side.value if hasattr(order.side, "value") else str(order.side)
        type_str = order.order_type.value if hasattr(order.order_type, "value") else str(order.order_type)
        filled_qty, status_str, trades_tuples = self._cpp_lob.add_order_exploded(
            str(order.order_id),
            str(order.agent_id),
            float(order.timestamp),
            str(order.symbol),
            side_str,
            type_str,
            float(order.price),
            float(order.quantity),
        )

        order.filled_qty = int(filled_qty)
        status_map = {
            "filled": OrderStatus.FILLED,
            "partial": OrderStatus.PARTIAL,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "pending": OrderStatus.PENDING,
        }
        order.status = status_map.get(str(status_str).lower(), OrderStatus.PENDING)
        if order.status == OrderStatus.CANCELLED and order.remaining_qty > 0 and order.order_type in {OrderType.IOC, OrderType.MARKET}:
            order.reason = order.reason or "unfilled remainder canceled"

        trades: List[Trade] = []
        for trade_tuple in trades_tuples:
            tid, price, qty, maker_id, taker_id, maker_agent_id, taker_agent_id, ts, buyer_fee, seller_fee, seller_tax = trade_tuple
            trade = Trade(
                trade_id=str(tid),
                price=float(price),
                quantity=int(qty),
                maker_id=str(maker_id),
                taker_id=str(taker_id),
                maker_agent_id=str(maker_agent_id),
                taker_agent_id=str(taker_agent_id),
                buyer_agent_id=order.agent_id if order.side == OrderSide.BUY else str(maker_agent_id),
                seller_agent_id=str(maker_agent_id) if order.side == OrderSide.BUY else order.agent_id,
                timestamp=float(ts),
                buyer_fee=float(buyer_fee),
                seller_fee=float(seller_fee),
                seller_tax=float(seller_tax),
            )
            trades.append(trade)
            self.last_price = trade.price
            self.total_volume += int(trade.quantity)
            self.trades_history.append(trade)
            self._step_trades.append(trade)
            self.trade_count += 1
        return trades

    def cancel_order(self, order_id: str) -> bool:
        success = bool(self._cpp_lob.cancel_order(str(order_id)))
        if success:
            self.cancelled_orders += 1
        return success

    def estimate_queue_position(self, order: Order) -> int:
        depth = self.get_depth(levels=10)
        queue_ahead = 0
        if order.side == OrderSide.BUY:
            for level in depth.get("bids", []):
                price = float(level.get("price", 0.0))
                qty = int(level.get("qty", 0))
                if price > order.price or price == order.price:
                    queue_ahead += qty
        else:
            for level in depth.get("asks", []):
                price = float(level.get("price", 0.0))
                qty = int(level.get("qty", 0))
                if price < order.price or price == order.price:
                    queue_ahead += qty
        return int(queue_ahead)

    def get_best_bid(self) -> Optional[float]:
        p = self._cpp_lob.get_best_bid()
        return float(p) if p > 0 else None

    def get_best_ask(self) -> Optional[float]:
        p = self._cpp_lob.get_best_ask()
        return float(p) if p > 0 else None

    def get_depth(self, levels: int = 5) -> Dict[str, List[Dict[str, float]]]:
        depth = self._cpp_lob.get_depth(int(levels))
        return {
            "bids": [{"price": float(price), "qty": int(qty)} for price, qty in depth.get("bids", [])],
            "asks": [{"price": float(price), "qty": int(qty)} for price, qty in depth.get("asks", [])],
        }

    def clear(self):
        super().clear()
        self._cpp_lob.clear()
        self.submitted_orders = 0
        self.cancelled_orders = 0
        self.trade_count = 0

    def get_activity_stats(self) -> Dict[str, int]:
        return {
            "submitted_orders": int(self.submitted_orders),
            "cancelled_orders": int(self.cancelled_orders),
            "trade_count": int(self.trade_count),
        }
