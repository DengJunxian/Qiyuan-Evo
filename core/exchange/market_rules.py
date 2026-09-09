from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Any, Dict, Mapping, Optional, Tuple

from config import GLOBAL_CONFIG
from core.types import OrderType
from core.utils import PriceQuantizer

SCHEMA_VERSION = "market_rule_schema/v1"


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _dynamic_price_limit(symbol: str, default_limit: float) -> float:
    sym = str(symbol or "")
    if sym.startswith("688") or sym.startswith("300"):
        return 0.20
    if sym.startswith("8"):
        return 0.30
    if "ST" in sym.upper():
        return 0.05
    return float(default_limit)


@dataclass(frozen=True)
class SessionRule:
    """Serializable market session segment.

    The defaults model a China A-share style day: opening call auction,
    continuous trading split by a midday break, and a close phase that can be
    extended later with closing auction logic.
    """

    name: str
    phase: str
    start: str
    end: str
    accepts_orders: bool = True
    matches_immediately: bool = True
    auction_uncross: bool = False
    notes: str = ""

    def contains(self, timestamp: float) -> bool:
        current = datetime.fromtimestamp(float(timestamp)).time()
        return self.start_time <= current < self.end_time

    @property
    def start_time(self) -> dt_time:
        return datetime.strptime(self.start, "%H:%M").time()

    @property
    def end_time(self) -> dt_time:
        return datetime.strptime(self.end, "%H:%M").time()

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class MarketRuleSchema:
    """Single source of truth for microstructure rules.

    The schema is intentionally serializable so Python, C++, tests, and UI
    export layers can all consume the same rule set.
    """

    symbol: str
    prev_close: float
    schema_version: str = SCHEMA_VERSION
    price_limit_pct: float = GLOBAL_CONFIG.PRICE_LIMIT
    min_price_tick: float = 0.01
    min_trade_unit: int = 1
    board_lot: int = 100
    allow_odd_lots: bool = True
    enforce_min_trade_unit: bool = False
    enforce_board_lot: bool = False
    timestamp_precision: str = "microsecond"
    queue_priority: str = "price_time"
    t_plus_one: bool = False
    commission_rate: float = GLOBAL_CONFIG.TAX_RATE_COMMISSION
    stamp_duty_rate: float = GLOBAL_CONFIG.TAX_RATE_STAMP
    seller_only_stamp_duty: bool = True
    supported_order_types: Tuple[str, ...] = (
        OrderType.LIMIT.value,
        OrderType.MARKET.value,
        OrderType.IOC.value,
        OrderType.FOK.value,
        OrderType.POST_ONLY.value,
    )
    sessions: Tuple[SessionRule, ...] = field(default_factory=tuple)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_feature_enabled(self, name: str, default: bool = False) -> bool:
        return _coerce_bool(self.feature_flags.get(name), default=default)

    def get_limit_prices(self) -> Tuple[float, float]:
        return PriceQuantizer.get_limit_prices(self.prev_close, self.price_limit_pct)

    def check_price_limit(self, price: float) -> bool:
        lower, upper = self.get_limit_prices()
        return float(lower) <= float(price) <= float(upper)

    def quantize_price(self, price: float) -> float:
        tick = float(self.min_price_tick or 0.01)
        if tick <= 0:
            return float(price)
        steps = round(float(price) / tick)
        quantized = round(steps * tick, 8)
        return round(quantized, max(2, len(str(tick).split(".")[-1])))

    def normalize_quantity(self, quantity: int) -> int:
        qty = max(0, int(quantity))
        unit = max(1, int(self.min_trade_unit or 1))
        if self.enforce_min_trade_unit:
            qty = (qty // unit) * unit
        if self.enforce_board_lot and qty >= self.board_lot:
            qty = (qty // self.board_lot) * self.board_lot
        return int(qty)

    def normalize_timestamp(self, timestamp: float) -> float:
        precision = str(self.timestamp_precision or "microsecond").strip().lower()
        ts = float(timestamp)
        if precision == "second":
            return round(ts, 0)
        if precision == "millisecond":
            return round(ts, 3)
        if precision == "microsecond":
            return round(ts, 6)
        return ts

    def find_session(self, timestamp: float) -> Optional[SessionRule]:
        for session in self.sessions:
            if session.contains(timestamp):
                return session
        return None

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["supported_order_types"] = list(self.supported_order_types)
        payload["sessions"] = [session.to_dict() for session in self.sessions]
        return payload


def default_sessions(*, call_auction_start: str = "09:15", call_auction_end: str = "09:25", continuous_start: str = "09:30", midday_break_start: str = "11:30", midday_break_end: str = "13:00", continuous_end: str = "15:00") -> Tuple[SessionRule, ...]:
    return (
        SessionRule(
            name="opening_call_auction",
            phase="call_auction",
            start=call_auction_start,
            end=call_auction_end,
            accepts_orders=True,
            matches_immediately=False,
            auction_uncross=True,
            notes="Default China A-share opening auction. Extendable to closing auction or after-hours blocks.",
        ),
        SessionRule(
            name="continuous_am",
            phase="continuous",
            start=continuous_start,
            end=midday_break_start,
            accepts_orders=True,
            matches_immediately=True,
            notes="Continuous matching session before the midday break.",
        ),
        SessionRule(
            name="midday_break",
            phase="midday_break",
            start=midday_break_start,
            end=midday_break_end,
            accepts_orders=False,
            matches_immediately=False,
            notes="Midday pause. Orders can be queued by the kernel and released later.",
        ),
        SessionRule(
            name="continuous_pm",
            phase="continuous",
            start=midday_break_end,
            end=continuous_end,
            accepts_orders=True,
            matches_immediately=True,
            notes="Continuous matching session after the midday break.",
        ),
        SessionRule(
            name="market_close",
            phase="closed",
            start=continuous_end,
            end="23:59",
            accepts_orders=False,
            matches_immediately=False,
            notes="Structured close phase placeholder for future closing auction extensions.",
        ),
    )


def build_market_rule_schema(
    symbol: str = "A_SHARE_IDX",
    prev_close: float = 3000.0,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
    feature_flags: Optional[Mapping[str, Any]] = None,
) -> MarketRuleSchema:
    raw_overrides: Dict[str, Any] = dict(overrides or {})
    nested_flags = dict(raw_overrides.pop("feature_flags", {}) or {})
    merged_flags: Dict[str, bool] = {
        "market_rules_v1": _coerce_bool(raw_overrides.get("market_rules_v1"), False),
        "session_rules_v1": _coerce_bool(raw_overrides.get("session_rules_v1"), False),
        "strict_queue_timestamps": _coerce_bool(raw_overrides.get("strict_queue_timestamps"), False),
    }
    for key, value in nested_flags.items():
        merged_flags[str(key)] = _coerce_bool(value, merged_flags.get(str(key), False))
    for key, value in dict(feature_flags or {}).items():
        merged_flags[str(key)] = _coerce_bool(value, merged_flags.get(str(key), False))

    session_items = raw_overrides.get("sessions")
    if session_items:
        sessions = tuple(
            item if isinstance(item, SessionRule) else SessionRule(**dict(item))
            for item in session_items
        )
    else:
        sessions = default_sessions(
            call_auction_start=str(raw_overrides.get("call_auction_start", "09:15")),
            call_auction_end=str(raw_overrides.get("call_auction_end", "09:25")),
            continuous_start=str(raw_overrides.get("continuous_start", "09:30")),
            midday_break_start=str(raw_overrides.get("midday_break_start", "11:30")),
            midday_break_end=str(raw_overrides.get("midday_break_end", "13:00")),
            continuous_end=str(raw_overrides.get("continuous_end", "15:00")),
        )

    schema = MarketRuleSchema(
        symbol=str(symbol),
        prev_close=float(prev_close),
        price_limit_pct=float(raw_overrides.get("price_limit_pct", _dynamic_price_limit(symbol, GLOBAL_CONFIG.PRICE_LIMIT))),
        min_price_tick=float(raw_overrides.get("min_price_tick", 0.01)),
        min_trade_unit=int(raw_overrides.get("min_trade_unit", 1)),
        board_lot=int(raw_overrides.get("board_lot", 100)),
        allow_odd_lots=_coerce_bool(raw_overrides.get("allow_odd_lots"), True),
        enforce_min_trade_unit=_coerce_bool(raw_overrides.get("enforce_min_trade_unit"), False),
        enforce_board_lot=_coerce_bool(raw_overrides.get("enforce_board_lot"), False),
        timestamp_precision=str(raw_overrides.get("timestamp_precision", "microsecond")),
        queue_priority=str(raw_overrides.get("queue_priority", "price_time")),
        t_plus_one=_coerce_bool(raw_overrides.get("t_plus_one"), False),
        commission_rate=float(raw_overrides.get("commission_rate", GLOBAL_CONFIG.TAX_RATE_COMMISSION)),
        stamp_duty_rate=float(raw_overrides.get("stamp_duty_rate", GLOBAL_CONFIG.TAX_RATE_STAMP)),
        seller_only_stamp_duty=_coerce_bool(raw_overrides.get("seller_only_stamp_duty"), True),
        supported_order_types=tuple(raw_overrides.get("supported_order_types", MarketRuleSchema.supported_order_types)),
        sessions=sessions,
        feature_flags=merged_flags,
        metadata={
            "market": str(raw_overrides.get("market", "CN_A_SHARE")),
            "extensible": True,
            **dict(raw_overrides.get("metadata", {})),
        },
    )
    return schema


def resolve_market_rule_schema(
    *,
    symbol: str,
    prev_close: float,
    overrides: Optional[Mapping[str, Any]] = None,
    feature_flags: Optional[Mapping[str, Any]] = None,
) -> MarketRuleSchema:
    if isinstance(overrides, MarketRuleSchema):
        return overrides
    if overrides and set(overrides.keys()) >= {"symbol", "prev_close", "schema_version"}:
        raw = dict(overrides)
        raw_symbol = str(raw.pop("symbol", symbol))
        raw_prev_close = float(raw.pop("prev_close", prev_close))
        raw.pop("schema_version", None)
        return build_market_rule_schema(raw_symbol, raw_prev_close, overrides=raw, feature_flags=feature_flags)
    return build_market_rule_schema(symbol, prev_close, overrides=overrides, feature_flags=feature_flags)


__all__ = [
    "SCHEMA_VERSION",
    "SessionRule",
    "MarketRuleSchema",
    "build_market_rule_schema",
    "resolve_market_rule_schema",
    "default_sessions",
]
