"""Unified market state for multi-asset policy simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


def _as_float_dict(payload: Mapping[str, Any] | None) -> Dict[str, float]:
    if not payload:
        return {}
    out: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


@dataclass(slots=True)
class MarketState:
    prices: Dict[str, float] = field(default_factory=dict)
    returns: Dict[str, float] = field(default_factory=dict)
    volatility: Dict[str, float] = field(default_factory=dict)
    order_book_stats: Dict[str, float] = field(default_factory=dict)
    sector_heat: Dict[str, float] = field(default_factory=dict)
    risk_appetite: float = 0.5
    sentiment_map: Dict[str, float] = field(default_factory=dict)
    regulatory_flags: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_symbol_price(cls, symbol: str, price: float) -> "MarketState":
        return cls(prices={str(symbol): float(price)})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prices": dict(self.prices),
            "returns": dict(self.returns),
            "volatility": dict(self.volatility),
            "order_book_stats": dict(self.order_book_stats),
            "sector_heat": dict(self.sector_heat),
            "risk_appetite": float(self.risk_appetite),
            "sentiment_map": dict(self.sentiment_map),
            "regulatory_flags": dict(self.regulatory_flags),
            "symbols": list(self.prices.keys()),
            "last_price": float(next(iter(self.prices.values()), 0.0)),
        }

    def apply_snapshot(
        self,
        snapshot: Mapping[str, Any],
        *,
        symbol: str,
        policy_pkg: Mapping[str, Any] | None = None,
    ) -> "MarketState":
        symbol = str(symbol)
        new_prices = dict(self.prices)
        previous_price = float(new_prices.get(symbol, 0.0) or 0.0)
        current_price = float(snapshot.get("last_price", previous_price) or previous_price)
        new_prices[symbol] = current_price

        new_returns = dict(self.returns)
        if previous_price > 0:
            new_returns[symbol] = float((current_price - previous_price) / previous_price)
        else:
            new_returns[symbol] = 0.0

        new_stats = dict(self.order_book_stats)
        activity_stats = snapshot.get("activity_stats")
        if isinstance(activity_stats, Mapping):
            for key, value in activity_stats.items():
                try:
                    new_stats[str(key)] = float(value)
                except Exception:
                    continue
        spread = snapshot.get("spread")
        if spread is not None:
            try:
                new_stats["spread"] = float(spread)
            except Exception:
                pass
        trade_count = snapshot.get("trade_count")
        if trade_count is not None:
            try:
                new_stats["trade_count"] = float(trade_count)
            except Exception:
                pass

        new_sector_heat = dict(self.sector_heat)
        if policy_pkg and isinstance(policy_pkg, Mapping):
            sectors = policy_pkg.get("sector_effects")
            if isinstance(sectors, Mapping):
                new_sector_heat = _as_float_dict(sectors)

        return MarketState(
            prices=new_prices,
            returns=new_returns,
            volatility=dict(self.volatility),
            order_book_stats=new_stats,
            sector_heat=new_sector_heat,
            risk_appetite=float(self.risk_appetite),
            sentiment_map=dict(self.sentiment_map),
            regulatory_flags=dict(self.regulatory_flags),
        )

