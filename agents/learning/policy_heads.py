"""Deterministic policy heads for agent learning smoke paths."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

from agents.learning.persona_prior import PersonaPrior, kl_to_prior
from agents.learning.regime_router import RegimeRoute


def _clip(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, float(value))))


def _num(state: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in state:
            try:
                return float(state.get(key, default) or default)
            except Exception:
                return float(default)
    return float(default)


def _softmax(logits: Mapping[str, float]) -> Dict[str, float]:
    max_logit = max(float(v) for v in logits.values())
    exp = {key: math.exp(float(value) - max_logit) for key, value in logits.items()}
    total = sum(exp.values()) or 1.0
    return {str(key): float(value / total) for key, value in exp.items()}


@dataclass(frozen=True)
class PolicyHeadOutput:
    action: str
    target_qty: int = 0
    target_notional: float = 0.0
    limit_price: float = 0.0
    urgency: float = 0.0
    confidence: float = 0.0
    action_distribution: Dict[str, float] = field(default_factory=dict)
    regularization: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HeuristicPolicyHead:
    """Rule/vector policy head that does not call a slow model."""

    def __init__(self, *, board_lot: int = 100, max_qty: int = 1000) -> None:
        self.board_lot = max(1, int(board_lot))
        self.max_qty = max(self.board_lot, int(max_qty))

    def _lot_round(self, qty: float) -> int:
        if qty <= 0:
            return 0
        lots = max(1, int(round(float(qty) / self.board_lot)))
        return int(min(self.max_qty, lots * self.board_lot))

    def _group_score(
        self,
        state: Mapping[str, Any],
        *,
        agent_group: str,
        persona_prior: PersonaPrior,
        route: Optional[RegimeRoute],
    ) -> float:
        group = str(agent_group or "").lower()
        sentiment = _num(state, "sentiment_index", "sentiment", default=0.5) - 0.5
        news_heat = _num(state, "news_heat", "social_pressure", default=0.0)
        event_shock = _num(state, "event_shock", "shock_strength", default=0.0)
        policy_context = _num(state, "policy_intensity", "policy_support", default=0.0)
        short_return = _num(state, "short_return", "momentum", "last_return", default=0.0)
        drawdown = abs(_num(state, "drawdown", "index_drawdown", default=0.0))
        spread = _num(state, "spread", default=0.0)
        inventory = _num(state, "inventory_ratio", "inventory", default=0.0)
        pnl = _num(state, "pnl", "unrealized_pnl", default=0.0)
        margin_ratio = _num(state, "margin_ratio", default=1.0)
        funding_stress = _num(state, "funding_stress", "credit_stress", default=0.0)
        risk_on = _num(state, "risk_on", "risk_appetite_proxy", default=0.0)
        route_policy = float(route.policy_sensitivity if route is not None else persona_prior.policy_sensitivity)
        route_risk = float(route.risk_budget if route is not None else 0.5)

        score = 0.55 * sentiment + 0.45 * short_return + 0.25 * event_shock
        score += 0.22 * policy_context * route_policy
        score -= 0.28 * funding_stress

        if "retail" in group:
            loss_chasing = 0.08 if pnl < 0 and short_return > 0 else 0.0
            score = score * (1.05 + persona_prior.turnover_preference) + 0.35 * news_heat - 0.10 * persona_prior.loss_aversion * max(-short_return, 0.0) + loss_chasing
        elif "leveraged" in group:
            score = score * 1.25 - 0.85 * max(0.0, 1.0 - margin_ratio) - 0.40 * funding_stress
        elif "market_maker" in group or "quant" in group:
            score = -0.55 * inventory + 0.30 * spread - 0.25 * short_return
        elif any(token in group for token in ("institution", "fund", "pension", "insurance")):
            redemption = _num(state, "redemption_pressure", "fund_flow_pressure", default=0.0)
            tracking = abs(_num(state, "tracking_error", default=0.0))
            score = 0.34 * score + 0.24 * policy_context - 0.30 * redemption - 0.18 * tracking
        elif any(token in group for token in ("policy", "stabilization", "state", "national")):
            liquidity_dry = max(0.0, 0.45 - _num(state, "liquidity_index", "depth", default=0.45))
            panic = _num(state, "panic", "panic_index", default=max(0.0, 0.5 - sentiment))
            score = 0.85 * drawdown + 0.40 * panic + 0.35 * liquidity_dry - 0.06
        elif "foreign" in group or "northbound" in group:
            valuation_gap = _num(state, "valuation_gap", default=0.0)
            fx_stress = _num(state, "fx_stress", "currency_pressure", default=0.0)
            score = 0.45 * risk_on + 0.28 * valuation_gap - 0.35 * fx_stress + 0.20 * sentiment
        elif "regulator" in group:
            score = 0.0

        score *= _clip(route_risk, 0.05, 1.5)
        score += persona_prior.buy_bias - persona_prior.sell_bias
        return float(_clip(score, -1.0, 1.0))

    def decide(
        self,
        state: Mapping[str, Any] | None = None,
        *,
        agent_group: str = "retail",
        persona_prior: PersonaPrior | None = None,
        route: RegimeRoute | None = None,
    ) -> PolicyHeadOutput:
        payload = dict(state or {})
        prior = persona_prior or PersonaPrior.from_persona(agent_group=agent_group)
        score = self._group_score(payload, agent_group=agent_group, persona_prior=prior, route=route)
        if abs(score) < 0.045 or "regulator" in str(agent_group).lower():
            action = "HOLD"
        else:
            action = "BUY" if score > 0 else "SELL"
        confidence = _clip(0.45 + abs(score) * 0.50, 0.0, 1.0)
        logits = {
            "BUY": float(score + prior.buy_bias),
            "HOLD": float(0.20 + prior.hold_bias - abs(score) * 0.35),
            "SELL": float(-score + prior.sell_bias),
        }
        dist = _softmax(logits)
        qty = 0 if action == "HOLD" else self._lot_round(abs(score) * self.max_qty)
        price = _num(payload, "price", "last_price", "current_price", default=0.0)
        urgency = 0.0 if action == "HOLD" else _clip(0.30 + abs(score) * 0.60, 0.0, 1.0)
        if action == "BUY":
            price *= 1.0 + 0.0015 * urgency
        elif action == "SELL":
            price *= 1.0 - 0.0015 * urgency
        kl = kl_to_prior(dist, prior)
        return PolicyHeadOutput(
            action=action,
            target_qty=int(qty),
            target_notional=float(qty * price),
            limit_price=float(max(0.0, price)),
            urgency=float(urgency),
            confidence=float(confidence),
            action_distribution=dist,
            regularization={
                "kl_to_persona_prior": float(kl),
                "risk_penalty": float(max(0.0, abs(score) - _clip(float(route.risk_budget if route else 0.5), 0.0, 1.0))),
                "inventory_penalty": float(abs(_num(payload, "inventory_ratio", "inventory", default=0.0))),
            },
            metadata={
                "policy_head": "heuristic_policy_head_v1",
                "agent_group": str(agent_group),
                "regime": route.regime if route is not None else str(payload.get("regime", "neutral")),
                "slow_model_called": False,
            },
        )


def select_policy_head(agent_group: str) -> HeuristicPolicyHead:
    return HeuristicPolicyHead()


__all__ = ["HeuristicPolicyHead", "PolicyHeadOutput", "select_policy_head"]
