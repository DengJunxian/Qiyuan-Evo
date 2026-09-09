"""State-conditioned regime routing for agent policy parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping


def _as_float(payload: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in payload:
            try:
                return float(payload.get(key, default) or default)
            except Exception:
                return float(default)
    return float(default)


def _clip(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, float(value))))


@dataclass(frozen=True)
class RegimeRoute:
    regime: str
    agent_group: str
    policy_sensitivity: float
    leverage_multiplier: float
    risk_budget: float
    execution_style: str
    slow_thinking_allowed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RegimeRouter:
    """Map market state to group-level policy and execution parameters."""

    def classify(self, market_state: Mapping[str, Any] | None = None) -> str:
        state = dict(market_state or {})
        explicit = str(state.get("regime", "") or "").strip().lower()
        if explicit and explicit not in {"neutral", "normal"}:
            if explicit in {"risk_on", "bull", "bullish"}:
                return "risk_on"
            if explicit in {"risk_off", "bear", "bearish"}:
                return "risk_off"
            if explicit in {"stress", "panic", "crash"}:
                return "stress"
            if explicit in {"policy_support", "stabilization"}:
                return "policy_support"
            return explicit

        sentiment = _as_float(state, "sentiment_index", "sentiment", default=0.5)
        volatility = _as_float(state, "volatility", "realized_volatility", default=0.02)
        drawdown = abs(_as_float(state, "drawdown", "index_drawdown", default=0.0))
        liquidity = _as_float(state, "liquidity_index", "liquidity", default=0.5)
        funding_stress = _as_float(state, "funding_stress", "credit_stress", default=0.0)
        if drawdown >= 0.07 or volatility >= 0.055 or funding_stress >= 0.45:
            return "stress"
        if sentiment <= 0.42 or drawdown >= 0.035 or liquidity <= 0.35:
            return "risk_off"
        if sentiment >= 0.60 and volatility <= 0.035:
            return "risk_on"
        if _as_float(state, "policy_support", "stabilization_intensity", default=0.0) > 0.20:
            return "policy_support"
        return "neutral"

    def route(
        self,
        market_state: Mapping[str, Any] | None = None,
        *,
        agent_group: str = "retail",
        persona_prior: Any = None,
    ) -> RegimeRoute:
        state = dict(market_state or {})
        group = str(agent_group or "retail").strip().lower()
        regime = self.classify(state)
        prior_policy = float(getattr(persona_prior, "policy_sensitivity", 0.5))
        prior_risk = float(getattr(persona_prior, "risk_aversion", 0.5))
        base_policy = 0.45 + 0.35 * prior_policy
        base_leverage = 1.0
        base_risk_budget = _clip(0.75 - 0.45 * prior_risk, 0.05, 1.0)
        execution = "balanced"
        slow_allowed = False

        if group in {"retail", "retail_general", "leveraged_flow", "leveraged"} or "retail" in group:
            base_policy += 0.12
            base_leverage += 0.15
            execution = "aggressive" if regime in {"risk_on", "stress"} else "balanced"
        elif any(token in group for token in ("market_maker", "quant")):
            base_policy -= 0.12
            base_leverage *= 0.85
            base_risk_budget += 0.10
            execution = "inventory_aware"
        elif any(token in group for token in ("institution", "fund", "pension", "insurance", "mutual_fund")):
            base_policy += 0.05
            base_leverage *= 0.70
            execution = "patient"
            slow_allowed = True
        elif any(token in group for token in ("policy", "state", "stabilization", "national")):
            base_policy += 0.32
            base_leverage *= 0.55
            base_risk_budget += 0.18
            execution = "stabilizing"
            slow_allowed = True
        elif "foreign" in group or "northbound" in group:
            base_policy -= 0.02
            execution = "macro_allocator"

        if regime == "risk_on":
            leverage = base_leverage * 1.12
            risk_budget = base_risk_budget * 1.10
        elif regime == "risk_off":
            leverage = base_leverage * 0.78
            risk_budget = base_risk_budget * 0.72
        elif regime == "stress":
            leverage = base_leverage * (0.45 if "policy" not in group and "stabilization" not in group else 0.80)
            risk_budget = base_risk_budget * (0.45 if "policy" not in group and "stabilization" not in group else 1.15)
            base_policy += 0.18
        elif regime == "policy_support":
            leverage = base_leverage * 0.95
            risk_budget = base_risk_budget * 1.05
            base_policy += 0.15
        else:
            leverage = base_leverage
            risk_budget = base_risk_budget

        return RegimeRoute(
            regime=regime,
            agent_group=group,
            policy_sensitivity=float(_clip(base_policy, 0.0, 1.5)),
            leverage_multiplier=float(_clip(leverage, 0.0, 2.0)),
            risk_budget=float(_clip(risk_budget, 0.01, 1.5)),
            execution_style=execution,
            slow_thinking_allowed=bool(slow_allowed),
            metadata={
                "router": "regime_router_v1",
                "sentiment_index": _as_float(state, "sentiment_index", "sentiment", default=0.5),
                "volatility": _as_float(state, "volatility", "realized_volatility", default=0.02),
                "drawdown": _as_float(state, "drawdown", "index_drawdown", default=0.0),
            },
        )


__all__ = ["RegimeRoute", "RegimeRouter"]
