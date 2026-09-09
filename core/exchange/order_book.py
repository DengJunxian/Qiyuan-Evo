"""
High-performance central limit order book with optional schema-driven microstructure rules.

The default path preserves the historical behavior. When `market_rules_v1` or
explicit schema overrides are enabled, price tick, board lot, timestamp
precision, T+1 checks, and session-aware metadata normalization are all sourced
from the same serializable `market_rule_schema`.
"""

from __future__ import annotations

import heapq
import time
import uuid
from typing import Dict, List, Optional, Tuple

from core.exchange.market_rules import MarketRuleSchema, resolve_market_rule_schema
from core.types import Order, OrderSide, OrderStatus, OrderType, Trade


class OrderBook:
    """ABIDES-style order book using heaps for O(log N) matching."""

    def __init__(
        self,
        symbol: str = "A_SHARE_IDX",
        prev_close: float = 3000.0,
        market_rules: Optional[dict | MarketRuleSchema] = None,
        feature_flags: Optional[dict] = None,
    ):
        self.symbol = symbol
        self.prev_close = float(prev_close)
        self.market_rules = resolve_market_rule_schema(
            symbol=symbol,
            prev_close=prev_close,
            overrides=market_rules,
            feature_flags=feature_flags,
        )

        self.bids: List[Tuple[float, float, str]] = []
        self.asks: List[Tuple[float, float, str]] = []
        self.orders: Dict[str, Order] = {}

        self.last_price: float = float(prev_close)
        self.total_volume: int = 0
        self.trades_history: List[Trade] = []
        self._step_trades: List[Trade] = []
        self.submitted_orders: int = 0
        self.cancelled_orders: int = 0
        self.trade_count: int = 0

    @property
    def feature_flags(self) -> Dict[str, bool]:
        return dict(self.market_rules.feature_flags)

    def get_market_rule_schema(self) -> Dict[str, object]:
        return self.market_rules.to_dict()

    def _get_dynamic_limit(self) -> float:
        return float(self.market_rules.price_limit_pct)

    def get_limit_prices(self) -> Tuple[float, float]:
        return self.market_rules.get_limit_prices()

    def _check_price_limit(self, price: float) -> bool:
        return self.market_rules.check_price_limit(price)

    def _normalize_order_timestamp(self, order: Order) -> None:
        if self.market_rules.is_feature_enabled("strict_queue_timestamps") or self.market_rules.is_feature_enabled("market_rules_v1"):
            order.timestamp = self.market_rules.normalize_timestamp(order.timestamp)

    def _normalize_order_price(self, order: Order) -> None:
        if order.order_type in {OrderType.LIMIT, OrderType.IOC, OrderType.FOK, OrderType.POST_ONLY}:
            if self.market_rules.is_feature_enabled("market_rules_v1"):
                order.price = self.market_rules.quantize_price(order.price)

    def _normalize_order_quantity(self, order: Order) -> bool:
        if not self.market_rules.is_feature_enabled("market_rules_v1"):
            return int(order.quantity) > 0
        original_qty = int(order.quantity)
        if self.market_rules.enforce_board_lot and not self.market_rules.allow_odd_lots:
            lot = max(1, int(self.market_rules.board_lot or 1))
            if original_qty < lot or original_qty % lot != 0:
                order.status = OrderStatus.REJECTED
                order.reason = "board lot constraint"
                return False
        normalized = self.market_rules.normalize_quantity(order.quantity)
        if normalized <= 0:
            order.status = OrderStatus.REJECTED
            order.reason = "quantity violates market rule schema"
            return False
        order.quantity = int(normalized)
        return True

    def _check_t_plus_one(self, order: Order) -> bool:
        if not (self.market_rules.is_feature_enabled("market_rules_v1") and self.market_rules.t_plus_one and order.side == OrderSide.SELL):
            return True
        acquired = order.metadata.get("position_acquired_ts")
        if acquired is None:
            acquired = order.metadata.get("inventory_acquired_ts")
        if acquired is None:
            return True
        acquired_day = time.strftime("%Y-%m-%d", time.localtime(float(acquired)))
        order_day = time.strftime("%Y-%m-%d", time.localtime(float(order.timestamp)))
        if acquired_day == order_day:
            order.status = OrderStatus.REJECTED
            order.reason = "t+1 sell restriction"
            return False
        return True

    def _would_cross_book(self, order: Order) -> bool:
        if order.side == OrderSide.BUY:
            best_ask = self.get_best_ask()
            return best_ask is not None and order.price >= best_ask
        best_bid = self.get_best_bid()
        return best_bid is not None and order.price <= best_bid

    def estimate_available_qty(self, side: OrderSide, price: float) -> int:
        if side == OrderSide.BUY:
            candidates = [
                o for o in self.orders.values()
                if o.side == OrderSide.SELL
                and o.status not in [OrderStatus.CANCELLED, OrderStatus.FILLED, OrderStatus.REJECTED]
                and o.price <= price
            ]
            candidates.sort(key=lambda o: (o.price, o.timestamp, o.order_id))
        else:
            candidates = [
                o for o in self.orders.values()
                if o.side == OrderSide.BUY
                and o.status not in [OrderStatus.CANCELLED, OrderStatus.FILLED, OrderStatus.REJECTED]
                and o.price >= price
            ]
            candidates.sort(key=lambda o: (-o.price, o.timestamp, o.order_id))
        return int(sum(max(0, o.remaining_qty) for o in candidates))

    def estimate_queue_position(self, order: Order) -> int:
        if order.side == OrderSide.BUY:
            ahead = [
                o for o in self.orders.values()
                if o.side == OrderSide.BUY
                and o.status not in [OrderStatus.CANCELLED, OrderStatus.FILLED, OrderStatus.REJECTED]
                and (o.price > order.price or (o.price == order.price and (o.timestamp, o.order_id) < (order.timestamp, order.order_id)))
            ]
        else:
            ahead = [
                o for o in self.orders.values()
                if o.side == OrderSide.SELL
                and o.status not in [OrderStatus.CANCELLED, OrderStatus.FILLED, OrderStatus.REJECTED]
                and (o.price < order.price or (o.price == order.price and (o.timestamp, o.order_id) < (order.timestamp, order.order_id)))
            ]
        return int(sum(max(0, o.remaining_qty) for o in ahead))

    def modify_order(
        self,
        order_id: str,
        *,
        price: Optional[float] = None,
        quantity: Optional[int] = None,
        timestamp: Optional[float] = None,
        original_order: Optional[Order] = None,
    ) -> Optional[Order]:
        source = original_order or self.orders.get(order_id)
        if source is None or source.status in [OrderStatus.CANCELLED, OrderStatus.FILLED, OrderStatus.REJECTED]:
            return None

        self.cancel_order(order_id)
        replacement = Order(
            symbol=source.symbol,
            price=float(price if price is not None else source.price),
            quantity=int(quantity if quantity is not None else source.remaining_qty or source.quantity),
            side=source.side,
            order_type=source.order_type,
            agent_id=source.agent_id,
            timestamp=float(timestamp if timestamp is not None else time.time()),
            metadata=dict(source.metadata),
        )
        replacement.metadata.update({"replaces_order_id": order_id, "replace_reason": "modify_order"})
        self._normalize_order_timestamp(replacement)
        self._normalize_order_price(replacement)
        self._normalize_order_quantity(replacement)
        return replacement

    def _preprocess_order(self, order: Order) -> bool:
        if order.symbol != self.symbol:
            order.status = OrderStatus.REJECTED
            return False

        self._normalize_order_timestamp(order)

        if order.order_type == OrderType.MARKET:
            lower, upper = self.get_limit_prices()
            order.price = upper if order.side == OrderSide.BUY else lower
        else:
            self._normalize_order_price(order)

        if not self._normalize_order_quantity(order):
            return False

        if order.order_type == OrderType.LIMIT and not self._check_price_limit(order.price):
            order.status = OrderStatus.REJECTED
            return False
        if order.order_type in {OrderType.IOC, OrderType.FOK, OrderType.POST_ONLY} and self.market_rules.is_feature_enabled("market_rules_v1"):
            if not self._check_price_limit(order.price):
                order.status = OrderStatus.REJECTED
                order.reason = "price outside limit"
                return False

        if not self._check_t_plus_one(order):
            return False

        if order.order_type == OrderType.POST_ONLY and self._would_cross_book(order):
            order.status = OrderStatus.REJECTED
            order.reason = "post-only order would take liquidity"
            return False
        if order.order_type == OrderType.FOK:
            available = self.estimate_available_qty(order.side, order.price)
            if available < order.quantity:
                order.status = OrderStatus.REJECTED
                order.reason = "insufficient liquidity for FOK"
                return False
        return True

    def add_order(self, order: Order) -> List[Trade]:
        if not self._preprocess_order(order):
            return []

        self.orders[order.order_id] = order
        order.status = OrderStatus.PENDING
        self.submitted_orders += 1

        trades = self._match_incoming_buy(order) if order.side == OrderSide.BUY else self._match_incoming_sell(order)

        resting_allowed = order.order_type in {OrderType.LIMIT, OrderType.POST_ONLY}
        if not order.is_filled and resting_allowed and order.status not in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            self._push_to_heap(order)
        elif not order.is_filled and order.status not in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            order.reason = order.reason or "unfilled remainder canceled"

        return trades

    def _match_incoming_buy(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        while order.remaining_qty > 0 and self.asks:
            self._clean_heap_top(OrderSide.SELL)
            if not self.asks:
                break
            ask_price, _, ask_id = self.asks[0]
            if order.price >= ask_price:
                ask_order = self.orders[ask_id]
                trades.append(self._execute_trade(order, ask_order))
            else:
                break
        return trades

    def _match_incoming_sell(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        while order.remaining_qty > 0 and self.bids:
            self._clean_heap_top(OrderSide.BUY)
            if not self.bids:
                break
            neg_bid_price, _, bid_id = self.bids[0]
            bid_price = -neg_bid_price
            if order.price <= bid_price:
                bid_order = self.orders[bid_id]
                trades.append(self._execute_trade(bid_order, order))
            else:
                break
        return trades

    def _push_to_heap(self, order: Order):
        queue_ts = self.market_rules.normalize_timestamp(order.timestamp) if self.market_rules.is_feature_enabled("strict_queue_timestamps") else order.timestamp
        queue_key = (queue_ts, order.order_id)
        if order.side == OrderSide.BUY:
            heapq.heappush(self.bids, (-order.price, queue_key[0], queue_key[1]))
        else:
            heapq.heappush(self.asks, (order.price, queue_key[0], queue_key[1]))

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            order = self.orders[order_id]
            if not order.is_filled:
                order.status = OrderStatus.CANCELLED
                self.cancelled_orders += 1
                return True
        return False

    def _clean_heap_top(self, side: OrderSide):
        heap = self.bids if side == OrderSide.BUY else self.asks
        while heap:
            _, _, order_id = heap[0]
            if order_id not in self.orders:
                heapq.heappop(heap)
                continue
            order = self.orders[order_id]
            if order.status == OrderStatus.CANCELLED or order.is_filled:
                heapq.heappop(heap)
                if order_id in self.orders:
                    del self.orders[order_id]
            else:
                break

    def _execute_trade(self, bid_order: Order, ask_order: Order) -> Trade:
        if (bid_order.timestamp, bid_order.order_id) < (ask_order.timestamp, ask_order.order_id):
            exec_price = bid_order.price
            maker, taker = bid_order, ask_order
        else:
            exec_price = ask_order.price
            maker, taker = ask_order, bid_order

        exec_qty = min(bid_order.remaining_qty, ask_order.remaining_qty)
        bid_order.filled_qty += exec_qty
        ask_order.filled_qty += exec_qty

        for order in [bid_order, ask_order]:
            order.status = OrderStatus.FILLED if order.is_filled else OrderStatus.PARTIAL

        notional = exec_price * exec_qty
        commission = notional * float(self.market_rules.commission_rate)
        stamp = notional * float(self.market_rules.stamp_duty_rate)
        trade_ts = max(
            self.market_rules.normalize_timestamp(bid_order.timestamp),
            self.market_rules.normalize_timestamp(ask_order.timestamp),
        )
        if not self.market_rules.is_feature_enabled("strict_queue_timestamps"):
            trade_ts = max(trade_ts, time.time())

        trade = Trade(
            trade_id=str(uuid.uuid4()),
            price=exec_price,
            quantity=exec_qty,
            maker_id=maker.order_id,
            taker_id=taker.order_id,
            maker_agent_id=maker.agent_id,
            taker_agent_id=taker.agent_id,
            buyer_agent_id=bid_order.agent_id,
            seller_agent_id=ask_order.agent_id,
            timestamp=trade_ts,
            buyer_fee=commission,
            seller_fee=commission,
            seller_tax=stamp if self.market_rules.seller_only_stamp_duty else 0.0,
        )

        self.last_price = exec_price
        self.total_volume += exec_qty
        self.trades_history.append(trade)
        self._step_trades.append(trade)
        self.trade_count += 1
        return trade

    def get_best_bid(self) -> Optional[float]:
        self._clean_heap_top(OrderSide.BUY)
        return -self.bids[0][0] if self.bids else None

    def get_best_ask(self) -> Optional[float]:
        self._clean_heap_top(OrderSide.SELL)
        return self.asks[0][0] if self.asks else None

    def get_spread(self) -> Optional[float]:
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return float(best_ask - best_bid)

    def get_depth(self, levels: int = 5) -> Dict[str, List[Dict[str, float]]]:
        temp_bids = self.bids[:]
        final_bids = []
        while temp_bids and len(final_bids) < levels:
            p, _, oid = heapq.heappop(temp_bids)
            if oid in self.orders and self.orders[oid].status not in [OrderStatus.CANCELLED, OrderStatus.FILLED]:
                final_bids.append({"price": -p, "qty": self.orders[oid].remaining_qty})

        temp_asks = self.asks[:]
        final_asks = []
        while temp_asks and len(final_asks) < levels:
            p, _, oid = heapq.heappop(temp_asks)
            if oid in self.orders and self.orders[oid].status not in [OrderStatus.CANCELLED, OrderStatus.FILLED]:
                final_asks.append({"price": p, "qty": self.orders[oid].remaining_qty})

        return {"bids": final_bids, "asks": final_asks}

    def flush_step_trades(self) -> List[Trade]:
        trades = self._step_trades[:]
        self._step_trades = []
        return trades

    def update_prev_close(self, close: float):
        self.prev_close = float(close)
        self.market_rules = resolve_market_rule_schema(
            symbol=self.symbol,
            prev_close=self.prev_close,
            overrides=self.market_rules.to_dict(),
            feature_flags=self.market_rules.feature_flags,
        )

    def get_activity_stats(self) -> Dict[str, int]:
        return {
            "submitted_orders": int(self.submitted_orders),
            "cancelled_orders": int(self.cancelled_orders),
            "trade_count": int(self.trade_count),
        }

    def sweep_cost_estimate(
        self,
        side: OrderSide | str,
        quantity: int,
        *,
        levels: int = 5,
    ) -> Dict[str, float]:
        qty = max(0, int(quantity))
        if qty <= 0:
            return {
                "requested_qty": 0.0,
                "filled_qty": 0.0,
                "fill_ratio": 0.0,
                "avg_price": 0.0,
                "reference_price": float(self.last_price),
                "slippage_bps": 0.0,
                "notional": 0.0,
            }

        side_enum = side if isinstance(side, OrderSide) else OrderSide(str(side).strip().lower())
        book_side = self.get_depth(max(1, int(levels)))
        ladder = book_side["asks"] if side_enum == OrderSide.BUY else book_side["bids"]
        remaining = qty
        filled = 0
        notional = 0.0
        for row in ladder:
            level_qty = int(max(0, row.get("qty", 0)))
            if level_qty <= 0:
                continue
            take = min(level_qty, remaining)
            if take <= 0:
                continue
            px = float(row.get("price", 0.0))
            notional += px * take
            filled += take
            remaining -= take
            if remaining <= 0:
                break

        reference_price = self.get_best_ask() if side_enum == OrderSide.BUY else self.get_best_bid()
        if reference_price is None or float(reference_price) <= 0:
            reference_price = float(self.last_price)
        avg_price = (notional / filled) if filled > 0 else 0.0
        fill_ratio = float(filled / qty) if qty > 0 else 0.0
        if filled <= 0 or reference_price <= 0:
            slippage_bps = 0.0
        elif side_enum == OrderSide.BUY:
            slippage_bps = ((avg_price - reference_price) / reference_price) * 10_000.0
        else:
            slippage_bps = ((reference_price - avg_price) / reference_price) * 10_000.0
        return {
            "requested_qty": float(qty),
            "filled_qty": float(filled),
            "fill_ratio": float(fill_ratio),
            "avg_price": float(avg_price),
            "reference_price": float(reference_price),
            "slippage_bps": float(slippage_bps),
            "notional": float(notional),
        }

    def order_flow_imbalance(self, *, levels: int = 5) -> float:
        depth = self.get_depth(max(1, int(levels)))
        bid_qty = float(sum(max(0, int(row.get("qty", 0))) for row in depth.get("bids", [])))
        ask_qty = float(sum(max(0, int(row.get("qty", 0))) for row in depth.get("asks", [])))
        denom = bid_qty + ask_qty
        if denom <= 0:
            return 0.0
        return float((bid_qty - ask_qty) / denom)

    def impact_curve_snapshot(
        self,
        *,
        order_sizes: Optional[List[int]] = None,
        levels: int = 5,
    ) -> Dict[str, List[Dict[str, float]]]:
        sizes = list(order_sizes or [100, 300, 800, 1500])
        buy_curve: List[Dict[str, float]] = []
        sell_curve: List[Dict[str, float]] = []
        for size in sizes:
            qty = int(max(1, size))
            buy_stats = self.sweep_cost_estimate(OrderSide.BUY, qty, levels=levels)
            sell_stats = self.sweep_cost_estimate(OrderSide.SELL, qty, levels=levels)
            buy_curve.append(
                {
                    "size": float(qty),
                    "slippage_bps": float(buy_stats["slippage_bps"]),
                    "fill_ratio": float(buy_stats["fill_ratio"]),
                    "avg_price": float(buy_stats["avg_price"]),
                }
            )
            sell_curve.append(
                {
                    "size": float(qty),
                    "slippage_bps": float(sell_stats["slippage_bps"]),
                    "fill_ratio": float(sell_stats["fill_ratio"]),
                    "avg_price": float(sell_stats["avg_price"]),
                }
            )
        return {"buy": buy_curve, "sell": sell_curve}

    def clear(self):
        self.bids.clear()
        self.asks.clear()
        self.orders.clear()
        self.submitted_orders = 0
        self.cancelled_orders = 0
        self.trade_count = 0
