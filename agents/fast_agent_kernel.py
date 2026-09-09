"""Fast cohort-based population agent kernel.

Large retail/population buckets should not call slow per-agent reasoning every
tick. This module produces deterministic, vector-friendly action views that the
existing execution adapter can still consume.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from agents.learning.persona_prior import PersonaPrior, kl_to_prior
from agents.learning.policy_heads import HeuristicPolicyHead
from agents.learning.regime_router import RegimeRouter


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _persona_key(agent: Any) -> str:
    persona = getattr(agent, "persona", None)
    key = str(getattr(persona, "archetype_key", "") or "").strip()
    if key:
        return key
    archetype = getattr(persona, "archetype", None)
    if archetype is not None:
        key = str(getattr(archetype, "key", "") or "").strip()
        if key:
            return key
    profile = getattr(agent, "psychology_profile", None)
    if isinstance(profile, Mapping):
        inst = str(profile.get("institution_type", "") or "").strip()
        if inst:
            return inst
    return "retail_general"


def _base_risk(agent: Any) -> float:
    persona = getattr(agent, "persona", None)
    value = getattr(persona, "risk_tolerance", None)
    if value is not None:
        return float(_clip(float(value), 0.0, 1.0))
    archetype = getattr(persona, "archetype", None)
    if archetype is not None:
        return float(_clip(getattr(archetype, "risk_budget", 0.5), 0.0, 1.0))
    return 0.5


@dataclass(slots=True)
class FastAgentAction:
    action: str
    amount: float
    target_price: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FastCohortState:
    cohort_id: str
    archetype: str
    sentiment_bucket: str
    risk_bucket: str
    multiplicity: int
    aggregate_weight: float
    net_intent: float
    action: str
    quantity_per_agent: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cohort_id": self.cohort_id,
            "archetype": self.archetype,
            "sentiment_bucket": self.sentiment_bucket,
            "risk_bucket": self.risk_bucket,
            "multiplicity": int(self.multiplicity),
            "aggregate_weight": float(self.aggregate_weight),
            "net_intent": float(self.net_intent),
            "action": self.action,
            "quantity_per_agent": float(self.quantity_per_agent),
        }


class FastAgentKernel:
    """Cohort-state machine for large populations."""

    def __init__(self, *, min_quantity: int = 10, max_quantity: int = 650) -> None:
        self.min_quantity = int(min_quantity)
        self.max_quantity = int(max_quantity)
        self.regime_router = RegimeRouter()
        self.policy_head = HeuristicPolicyHead(board_lot=max(1, int(min_quantity)), max_qty=int(max_quantity))

    @staticmethod
    def _sentiment_signal(event_digest: Mapping[str, Any] | None, macro_state: Mapping[str, Any] | None) -> float:
        signal = 0.0
        macro = dict(macro_state or {})
        signal += float(macro.get("sentiment_index", 0.5) or 0.5) - 0.5
        digest = dict(event_digest or {})
        impact = dict(digest.get("aggregate_impact", {}) or {})
        signal += float(impact.get("sentiment_impact", 0.0) or 0.0)
        signal -= float(impact.get("funding_stress", 0.0) or 0.0) * 0.6
        by_type = dict(digest.get("by_type", {}) or {})
        if by_type.get("rumor"):
            signal -= 0.18
        if by_type.get("refute"):
            signal += 0.12
        if by_type.get("major_news"):
            signal += 0.04 if signal >= 0 else -0.04
        return float(_clip(signal, -1.0, 1.0))

    @staticmethod
    def _cohort_id(archetype: str, sentiment_bucket: str, risk_bucket: str) -> str:
        raw = f"{archetype}|{sentiment_bucket}|{risk_bucket}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    def decide(
        self,
        agents: Sequence[Any],
        *,
        current_price: float,
        event_digest: Mapping[str, Any] | None = None,
        macro_state: Mapping[str, Any] | None = None,
        tick: int = 0,
    ) -> tuple[List[FastAgentAction], List[FastCohortState]]:
        if not agents:
            return [], []

        signal = self._sentiment_signal(event_digest, macro_state)
        grouped: Dict[tuple[str, str, str], List[int]] = {}
        risks: Dict[tuple[str, str, str], float] = {}
        archetypes: List[str] = []
        for idx, agent in enumerate(agents):
            archetype = _persona_key(agent)
            risk = _base_risk(agent)
            sentiment_bucket = "bull" if signal > 0.08 else "bear" if signal < -0.08 else "neutral"
            risk_bucket = "high" if risk >= 0.67 else "low" if risk <= 0.34 else "mid"
            key = (archetype, sentiment_bucket, risk_bucket)
            grouped.setdefault(key, []).append(idx)
            risks[key] = risks.get(key, 0.0) + risk
            archetypes.append(archetype)

        actions: List[FastAgentAction] = [
            FastAgentAction(action="HOLD", amount=0.0, target_price=float(current_price), metadata={})
            for _ in agents
        ]
        cohorts: List[FastCohortState] = []
        for key, indices in grouped.items():
            archetype, sentiment_bucket, risk_bucket = key
            multiplicity = len(indices)
            avg_risk = risks[key] / max(multiplicity, 1)
            representative = agents[indices[0]]
            persona_prior = PersonaPrior.from_persona(getattr(representative, "persona", None), agent_group=archetype)
            route = self.regime_router.route(macro_state or {}, agent_group=archetype, persona_prior=persona_prior)
            archetype_bias = 1.0
            if "mean_reverter" in archetype or "market_maker" in archetype:
                archetype_bias = -0.55
            elif "momentum" in archetype or "day_trader" in archetype:
                archetype_bias = 1.25
            elif "fund" in archetype or "institution" in archetype:
                archetype_bias = 0.55
            elif "leveraged" in archetype:
                archetype_bias = 1.55
            net_intent = float(_clip(signal * (0.35 + avg_risk) * archetype_bias, -1.0, 1.0))
            net_intent = float(_clip(net_intent * max(0.05, route.leverage_multiplier) * max(0.05, route.risk_budget), -1.0, 1.0))
            head_state = {
                **dict(macro_state or {}),
                "sentiment_index": float(dict(macro_state or {}).get("sentiment_index", 0.5) or 0.5),
                "event_shock": float(dict(dict(event_digest or {}).get("aggregate_impact", {}) or {}).get("sentiment_impact", 0.0) or 0.0),
                "funding_stress": float(dict(dict(event_digest or {}).get("aggregate_impact", {}) or {}).get("funding_stress", 0.0) or 0.0),
                "current_price": float(current_price),
                "news_heat": abs(float(signal)),
            }
            policy_output = self.policy_head.decide(
                head_state,
                agent_group=archetype,
                persona_prior=persona_prior,
                route=route,
            )
            if abs(net_intent) < 0.035:
                action = "HOLD"
            else:
                action = "BUY" if net_intent > 0 else "SELL"
            qty = 0.0 if action == "HOLD" else _clip(abs(net_intent) * self.max_quantity, self.min_quantity, self.max_quantity)
            price_skew = 1.0 + (0.0015 if action == "BUY" else -0.0015 if action == "SELL" else 0.0)
            for local_idx in indices:
                actions[local_idx] = FastAgentAction(
                    action=action,
                    amount=float(qty),
                    target_price=float(max(0.01, current_price * price_skew)),
                    metadata={
                        "execution_path": "fast_population",
                        "cohort_id": self._cohort_id(archetype, sentiment_bucket, risk_bucket),
                        "multiplicity": int(multiplicity),
                        "net_intent": float(net_intent),
                        "policy_head": "heuristic_policy_head_v1",
                        "policy_head_action": policy_output.action,
                        "policy_head_distribution": dict(policy_output.action_distribution),
                        "persona_prior_kl": float(kl_to_prior(policy_output.action_distribution, persona_prior)),
                        "regime": route.regime,
                        "execution_style": route.execution_style,
                        "slow_model_called": False,
                    },
                )
            cohorts.append(
                FastCohortState(
                    cohort_id=self._cohort_id(archetype, sentiment_bucket, risk_bucket),
                    archetype=archetype,
                    sentiment_bucket=sentiment_bucket,
                    risk_bucket=risk_bucket,
                    multiplicity=multiplicity,
                    aggregate_weight=float(multiplicity),
                    net_intent=float(net_intent),
                    action=action,
                    quantity_per_agent=float(qty),
                )
            )
        return actions, cohorts


__all__ = ["FastAgentAction", "FastAgentKernel", "FastCohortState"]
