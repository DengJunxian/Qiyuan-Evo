"""
统一的核心数据类型定义。
包含订单、成交和K线等结构。
"""

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time
import uuid


class OrderSide(str, Enum):
    """订单方向。"""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """订单类型。"""

    LIMIT = "limit"
    MARKET = "market"
    IOC = "ioc"
    FOK = "fok"
    POST_ONLY = "post-only"


class OrderStatus(str, Enum):
    """订单状态。"""

    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


def _coerce_order_type(order_type: str | OrderType) -> OrderType:
    if isinstance(order_type, OrderType):
        return order_type

    normalized = str(order_type).strip().lower().replace("_", "-")
    aliases = {
        "postonly": OrderType.POST_ONLY,
        "post-only": OrderType.POST_ONLY,
    }
    if normalized in aliases:
        return aliases[normalized]
    return OrderType(normalized)


def _coerce_side(side: str | OrderSide) -> OrderSide:
    if isinstance(side, OrderSide):
        return side
    return OrderSide(str(side).strip().lower())


@dataclass
class Order:
    """通用订单对象。"""

    symbol: str
    price: float
    quantity: int
    side: OrderSide
    order_type: OrderType
    agent_id: str
    timestamp: float
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    reason: str = ""
    filled_qty: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_qty(self) -> int:
        return self.quantity - self.filled_qty

    @property
    def is_filled(self) -> bool:
        return self.filled_qty >= self.quantity

    @property
    def value(self) -> float:
        """订单总价值。"""
        return self.price * self.quantity

    @property
    def action(self) -> str:
        """
        兼容旧字段：部分旧测试与脚本使用 `order.action` 表示买卖方向。
        """
        return self.side.value.upper()

    @classmethod
    def create(
        cls,
        *,
        agent_id: str,
        symbol: str,
        side: str | OrderSide,
        order_type: str | OrderType,
        price: float,
        quantity: int,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Order":
        """
        兼容旧接口的订单工厂方法。
        支持字符串或枚举输入，未传时间戳时使用当前时间。
        """
        side_enum = _coerce_side(side)
        type_enum = _coerce_order_type(order_type)
        return cls(
            symbol=str(symbol),
            price=float(price),
            quantity=int(quantity),
            side=side_enum,
            order_type=type_enum,
            agent_id=str(agent_id),
            timestamp=float(time.time() if timestamp is None else timestamp),
            metadata=dict(metadata or {}),
        )


@dataclass
class ExecutionPlan:
    """Structured execution intent used by the trading layer."""

    symbol: str
    agent_id: str
    action: str
    side: OrderSide | str
    target_qty: Optional[int] = None
    target_notional: Optional[float] = None
    urgency: float = 0.5
    order_type: OrderType | str = OrderType.LIMIT
    max_slippage: float = 0.01
    participation_rate: float = 0.1
    slicing_rule: str = "single"
    cancel_replace_policy: str = "none"
    time_horizon: int = 1
    price: float = 0.0
    timestamp: float = field(default_factory=time.time)
    child_order_schedule: List[int] = field(default_factory=list)
    seed: int = 42
    config_hash: str = ""
    snapshot_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action = str(self.action).strip().upper()
        self.side = _coerce_side(self.side)
        self.order_type = _coerce_order_type(self.order_type)
        self.urgency = float(max(0.0, min(1.0, self.urgency)))
        self.max_slippage = float(max(0.0, self.max_slippage))
        self.participation_rate = float(max(0.0, min(1.0, self.participation_rate)))
        self.time_horizon = max(1, int(self.time_horizon))
        self.timestamp = float(self.timestamp)
        if self.price is not None:
            self.price = float(self.price)
        self.child_order_schedule = [max(0, int(q)) for q in self.child_order_schedule if int(q) > 0]
        if self.snapshot_info and "last_price" in self.snapshot_info:
            self.snapshot_info["last_price"] = float(self.snapshot_info["last_price"])

    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL

    def resolved_reference_price(self, fallback: float = 0.0) -> float:
        if self.price > 0:
            return float(self.price)
        if self.snapshot_info:
            ref_price = self.snapshot_info.get("last_price")
            if ref_price is not None:
                return float(ref_price)
        return float(fallback)

    def resolved_qty(self, reference_price: Optional[float] = None) -> int:
        qty = self.target_qty
        if qty is None and self.target_notional is not None:
            ref = float(reference_price if reference_price is not None else self.resolved_reference_price())
            if ref > 0:
                qty = int(self.target_notional / ref)
        return max(0, int(qty or 0))

    def resolved_child_schedule(self, reference_price: Optional[float] = None) -> List[int]:
        qty = self.resolved_qty(reference_price)
        if qty <= 0:
            return []
        if self.child_order_schedule:
            schedule = [min(qty, max(0, int(x))) for x in self.child_order_schedule if int(x) > 0]
            total = sum(schedule)
            if total < qty:
                schedule.append(qty - total)
            elif total > qty:
                overflow = total - qty
                if schedule:
                    schedule[-1] = max(1, schedule[-1] - overflow)
            return [q for q in schedule if q > 0]

        rule = str(self.slicing_rule).strip().lower()
        if rule in {"twap", "twap-like", "twap_like"}:
            parts = max(1, self.time_horizon)
            base = qty // parts
            remainder = qty % parts
            return [base + (1 if i < remainder else 0) for i in range(parts) if base + (1 if i < remainder else 0) > 0]

        if rule in {"vwap", "vwap-like", "vwap_like"}:
            parts = max(2, self.time_horizon)
            weights = list(range(1, parts + 1))
            if parts > 3:
                mid = parts // 2
                weights = [i + 1 for i in range(mid)] + [mid + 1] + [parts - i for i in range(mid, parts - 1)]
                weights = [max(1, w) for w in weights[:parts]]
            weight_sum = float(sum(weights))
            schedule = [max(1, int(round(qty * (w / weight_sum)))) for w in weights]
            delta = qty - sum(schedule)
            if schedule:
                schedule[-1] += delta
            return [max(1, q) for q in schedule if q > 0]

        return [qty]

    def to_order(
        self,
        *,
        price: Optional[float] = None,
        quantity: Optional[int] = None,
        timestamp: Optional[float] = None,
        child_index: int = 0,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Order:
        ref_price = float(price if price is not None else self.resolved_reference_price())
        qty = int(quantity if quantity is not None else self.resolved_qty(ref_price))
        metadata = dict(self.metadata)
        if self.snapshot_info:
            metadata.setdefault("snapshot_info", dict(self.snapshot_info))
        metadata.update(
            {
                "seed": int(self.seed),
                "config_hash": self.config_hash,
                "execution_action": self.action,
                "execution_side": self.side.value,
                "execution_order_type": self.order_type.value,
                "urgency": float(self.urgency),
                "max_slippage": float(self.max_slippage),
                "participation_rate": float(self.participation_rate),
                "slicing_rule": self.slicing_rule,
                "cancel_replace_policy": self.cancel_replace_policy,
                "time_horizon": int(self.time_horizon),
                "child_index": int(child_index),
            }
        )
        if extra_metadata:
            metadata.update(extra_metadata)
        return Order(
            symbol=self.symbol,
            price=ref_price,
            quantity=qty,
            side=self.side,
            order_type=self.order_type,
            agent_id=self.agent_id,
            timestamp=float(self.timestamp if timestamp is None else timestamp),
            metadata=metadata,
        )

    def fingerprint(self) -> str:
        payload = {
            "symbol": self.symbol,
            "agent_id": self.agent_id,
            "action": self.action,
            "side": self.side.value,
            "target_qty": self.target_qty,
            "target_notional": self.target_notional,
            "urgency": self.urgency,
            "order_type": self.order_type.value,
            "max_slippage": self.max_slippage,
            "participation_rate": self.participation_rate,
            "slicing_rule": self.slicing_rule,
            "cancel_replace_policy": self.cancel_replace_policy,
            "time_horizon": self.time_horizon,
            "price": self.price,
            "seed": self.seed,
            "snapshot_info": self.snapshot_info,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


ExecutionIntent = ExecutionPlan


@dataclass
class AgentDecisionEnvelope:
    """Unified decision envelope for smart and vectorized agents."""

    agent_id: str
    symbol: str
    action: str
    side: OrderSide | str
    target_qty: Optional[int] = None
    target_notional: Optional[float] = None
    urgency: float = 0.5
    confidence: float = 0.5
    order_type: OrderType | str = OrderType.LIMIT
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action = str(self.action).strip().upper()
        self.side = _coerce_side(self.side)
        self.order_type = _coerce_order_type(self.order_type)
        self.urgency = float(max(0.0, min(1.0, self.urgency)))
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        if self.target_qty is not None:
            self.target_qty = max(0, int(self.target_qty))
        if self.target_notional is not None:
            self.target_notional = max(0.0, float(self.target_notional))

    def to_execution_plan(
        self,
        *,
        reference_price: float,
        seed: int = 42,
        config_hash: str = "",
        feature_flags: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        time_horizon = max(1, int(1 + round(self.urgency * 4)))
        plan = ExecutionPlan(
            symbol=self.symbol,
            agent_id=self.agent_id,
            action=self.action,
            side=self.side,
            target_qty=self.target_qty,
            target_notional=self.target_notional,
            urgency=self.urgency,
            order_type=self.order_type,
            max_slippage=float(max(0.0, min(0.04, 0.004 + (1.0 - self.confidence) * 0.02))),
            participation_rate=float(max(0.0, min(1.0, 0.05 + self.urgency * 0.30))),
            slicing_rule="single" if (self.target_qty or 0) < 1000 else "twap-like",
            cancel_replace_policy="none" if self.order_type in {OrderType.MARKET, OrderType.IOC, OrderType.FOK} else "cancel-replace",
            time_horizon=time_horizon,
            price=float(max(reference_price, 0.0)),
            seed=int(seed),
            config_hash=str(config_hash),
            snapshot_info={
                "reference_price": float(reference_price),
                "confidence": float(self.confidence),
                "feature_flags": dict(feature_flags or {}),
            },
            metadata={
                "decision_envelope": True,
                "confidence": float(self.confidence),
                "source_kind": str(self.metadata.get("agent_kind", "")),
                **dict(self.metadata),
            },
        )
        return plan

    @staticmethod
    def from_execution_plan(plan: ExecutionPlan) -> "AgentDecisionEnvelope":
        return AgentDecisionEnvelope(
            agent_id=plan.agent_id,
            symbol=plan.symbol,
            action=plan.action,
            side=plan.side,
            target_qty=plan.target_qty,
            target_notional=plan.target_notional,
            urgency=plan.urgency,
            confidence=float(plan.metadata.get("confidence", 0.5)),
            order_type=plan.order_type,
            metadata=dict(plan.metadata),
        )


@dataclass
class Trade:
    """成交记录。"""

    trade_id: str
    price: float
    quantity: int
    maker_id: str
    taker_id: str
    maker_agent_id: str
    taker_agent_id: str
    buyer_agent_id: str = ""
    seller_agent_id: str = ""
    timestamp: float = field(default_factory=time.time)
    buyer_fee: float = 0.0
    seller_fee: float = 0.0
    seller_tax: float = 0.0

    @property
    def notional(self) -> float:
        return self.price * self.quantity

    @property
    def buyer_total_cost(self) -> float:
        return self.notional + self.buyer_fee

    @property
    def seller_net_proceeds(self) -> float:
        return self.notional - self.seller_fee - self.seller_tax


@dataclass
class Candle:
    """K线数据（OHLCV）。"""

    symbol: str
    step: int
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    amount: float = 0.0
    is_simulated: bool = False
