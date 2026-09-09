"""Execution adapter: convert execution plans into buffered intents."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional

from core.types import ExecutionPlan, Order, OrderSide, OrderType
from simulation_runner import BufferedIntent


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


class ExecutionAdapter:
    """Adapter for plan -> child orders -> SimulationRunner intents."""

    def __init__(self, *, default_symbol: str = "A_SHARE_IDX") -> None:
        self.default_symbol = str(default_symbol)

    def _reference_price(self, plan: ExecutionPlan, market_state: Optional[Mapping[str, Any]]) -> float:
        fallback = 0.0
        if market_state:
            fallback = float(market_state.get("last_price", 0.0) or 0.0)
            prices = market_state.get("prices")
            if isinstance(prices, Mapping):
                symbol_price = prices.get(plan.symbol)
                if symbol_price is not None:
                    fallback = float(symbol_price)
        return max(0.0, float(plan.resolved_reference_price(fallback)))

    def _child_price(self, *, base_price: float, side: OrderSide, max_slippage: float, idx: int, parts: int) -> float:
        if base_price <= 0:
            return 0.0
        normalized = (idx + 1) / max(parts, 1)
        slip = max_slippage * (0.35 + 0.65 * normalized)
        if side == OrderSide.BUY:
            return base_price * (1.0 + slip)
        return base_price * (1.0 - slip)

    def compile(
        self,
        plan: ExecutionPlan,
        market_state: Optional[Mapping[str, Any]],
        step: int,
    ) -> List[BufferedIntent]:
        if plan is None:
            return []

        reference_price = self._reference_price(plan, market_state)
        if reference_price <= 0:
            return []
        schedule = plan.resolved_child_schedule(reference_price)
        if not schedule:
            return []

        side = plan.side if isinstance(plan.side, OrderSide) else OrderSide(str(plan.side).lower())
        order_type = plan.order_type if isinstance(plan.order_type, OrderType) else OrderType(str(plan.order_type).lower())
        intents: List[BufferedIntent] = []
        for idx, qty in enumerate(schedule):
            if int(qty) <= 0:
                continue


            if order_type == OrderType.LIMIT and float(plan.price or 0.0) > 0:
                child_price = float(plan.price)
            else:
                child_price = self._child_price(
                    base_price=reference_price,
                    side=side,
                    max_slippage=float(max(0.0, plan.max_slippage)),
                    idx=idx,
                    parts=len(schedule),
                )
            activate_step = int(step + idx if len(schedule) > 1 else step)
            intents.append(
                BufferedIntent(
                    intent_id=f"{plan.agent_id}-{step}-{idx}-{uuid.uuid4().hex[:8]}",
                    agent_id=str(plan.agent_id),
                    side=side.value,
                    quantity=int(qty),
                    price=max(0.01, float(child_price)),
                    symbol=str(plan.symbol or self.default_symbol),
                    order_type=order_type.value,
                    intent_type="order",
                    activate_step=activate_step,
                    metadata={
                        "source": "execution_adapter",
                        "action": str(plan.action),
                        "plan_config_hash": str(plan.config_hash),
                        "child_index": int(idx),
                        "child_count": int(len(schedule)),
                        "participation_rate": float(plan.participation_rate),
                        "slicing_rule": str(plan.slicing_rule),
                    },
                )
            )
        return intents

    def compile_batch(
        self,
        plans: Iterable[ExecutionPlan],
        market_state: Optional[Mapping[str, Any]],
        step: int,
    ) -> List[BufferedIntent]:
        intents: List[BufferedIntent] = []
        for plan in plans:
            intents.extend(self.compile(plan, market_state=market_state, step=step))
        return intents

    def plan_from_order(self, order: Order, *, action: Optional[str] = None) -> ExecutionPlan:
        direction = action or ("BUY" if order.side == OrderSide.BUY else "SELL")
        return ExecutionPlan(
            symbol=str(order.symbol or self.default_symbol),
            agent_id=str(order.agent_id),
            action=direction,
            side=order.side,
            target_qty=int(order.quantity),
            urgency=0.5,
            order_type=order.order_type,
            max_slippage=0.01,
            participation_rate=0.2,
            slicing_rule="single",
            cancel_replace_policy="none",
            time_horizon=1,
            price=float(order.price),
            timestamp=float(order.timestamp or time.time()),
            snapshot_info={"last_price": float(order.price), "symbol": str(order.symbol or self.default_symbol)},
            metadata={"source": "legacy_order"},
        )

    def plan_from_legacy_action(
        self,
        *,
        agent_id: str,
        symbol: str,
        action: str,
        amount: float,
        target_price: Optional[float],
        step: int,
        market_state: Optional[Mapping[str, Any]] = None,
    ) -> Optional[ExecutionPlan]:
        direction = str(action or "HOLD").strip().upper()
        qty = int(max(0.0, float(amount or 0.0)))
        if direction not in {"BUY", "SELL"} or qty <= 0:
            return None
        side = OrderSide.BUY if direction == "BUY" else OrderSide.SELL
        ref_price = float(target_price or 0.0)
        if ref_price <= 0 and market_state:
            ref_price = float(market_state.get("last_price", 0.0) or 0.0)
            prices = market_state.get("prices")
            if isinstance(prices, Mapping):
                ref_price = float(prices.get(symbol, ref_price) or ref_price)
        if ref_price <= 0:
            ref_price = 1.0
        urgency = _clip(abs(float(amount)) / 5000.0, 0.1, 0.9)
        return ExecutionPlan(
            symbol=str(symbol or self.default_symbol),
            agent_id=str(agent_id),
            action=direction,
            side=side,
            target_qty=qty,
            urgency=urgency,
            order_type=OrderType.LIMIT,
            max_slippage=0.012,
            participation_rate=0.18,
            slicing_rule="single" if qty < 400 else "twap-like",
            cancel_replace_policy="none",
            time_horizon=1 if qty < 400 else 2,
            price=ref_price,
            timestamp=float(step),
            snapshot_info={"last_price": ref_price, "symbol": str(symbol or self.default_symbol)},
            metadata={"source": "legacy_action"},
        )
