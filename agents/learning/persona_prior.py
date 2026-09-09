"""Persona priors used as policy regularizers, not final decisions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Sequence


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, float(value))))


def _softmax(logits: Mapping[str, float]) -> Dict[str, float]:
    if not logits:
        return {"BUY": 1.0 / 3.0, "HOLD": 1.0 / 3.0, "SELL": 1.0 / 3.0}
    max_logit = max(float(v) for v in logits.values())
    exp = {str(k): math.exp(float(v) - max_logit) for k, v in logits.items()}
    total = sum(exp.values()) or 1.0
    return {key: float(value / total) for key, value in exp.items()}


def _persona_dict(persona: Any) -> Dict[str, Any]:
    if persona is None:
        return {}
    if isinstance(persona, Mapping):
        return dict(persona)
    if hasattr(persona, "agent_schema"):
        try:
            schema = persona.agent_schema()
            if isinstance(schema, Mapping):
                return dict(schema)
        except Exception:
            pass
    if hasattr(persona, "to_dict"):
        try:
            payload = persona.to_dict()
            if isinstance(payload, Mapping):
                return dict(payload)
        except Exception:
            pass
    payload: Dict[str, Any] = {}
    for key in (
        "archetype_key",
        "risk_tolerance",
        "risk_appetite",
        "loss_aversion",
        "overconfidence",
        "patience",
        "conformity",
        "policy_channel_sensitivity",
        "rumor_sensitivity",
        "leverage_limit",
        "turnover_target",
    ):
        if hasattr(persona, key):
            payload[key] = getattr(persona, key)
    archetype = getattr(persona, "archetype", None)
    if archetype is not None and hasattr(archetype, "to_dict"):
        try:
            payload.update(dict(archetype.to_dict()))
        except Exception:
            pass
    return payload


def _constraints_from_persona(persona: Any, payload: Mapping[str, Any]) -> Dict[str, Any]:
    constraints = dict(payload.get("constraints", {}) or {}) if isinstance(payload.get("constraints"), Mapping) else {}
    if constraints:
        return constraints
    if persona is not None and hasattr(persona, "agent_schema"):
        try:
            schema = persona.agent_schema()
            raw = schema.get("constraints", {}) if isinstance(schema, Mapping) else {}
            if isinstance(raw, Mapping):
                return dict(raw)
        except Exception:
            pass
    archetype = getattr(persona, "archetype", None)
    if archetype is not None and hasattr(archetype, "constraint_schema"):
        try:
            raw = archetype.constraint_schema()
            if isinstance(raw, Mapping):
                return dict(raw)
        except Exception:
            pass
    return {}


@dataclass(frozen=True)
class PersonaPrior:
    """Compact prior vector used by policy heads and KL regularization."""

    buy_bias: float = 0.0
    sell_bias: float = 0.0
    hold_bias: float = 0.0
    risk_aversion: float = 0.5
    turnover_preference: float = 0.5
    policy_sensitivity: float = 0.5
    rumor_sensitivity: float = 0.5
    leverage_bias: float = 0.0
    loss_aversion: float = 1.5
    inventory_tolerance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_persona(cls, persona: Any = None, *, agent_group: str = "") -> "PersonaPrior":
        payload = _persona_dict(persona)
        constraints = _constraints_from_persona(persona, payload)
        group = str(agent_group or payload.get("participant_type") or payload.get("archetype_key") or "").lower()
        risk_appetite = str(payload.get("risk_appetite", "")).lower()
        if "aggressive" in risk_appetite or "gambler" in risk_appetite:
            risk_aversion = 0.25
        elif "conservative" in risk_appetite:
            risk_aversion = 0.75
        else:
            risk_aversion = 1.0 - _clip(float(payload.get("risk_tolerance", constraints.get("risk_budget", 0.5)) or 0.5))

        turnover = _clip(float(payload.get("turnover_target", constraints.get("turnover_target", 0.5)) or 0.5) / 5.0)
        policy = _clip(float(payload.get("policy_channel_sensitivity", constraints.get("policy_channel_sensitivity", 0.5)) or 0.5))
        rumor = _clip(float(payload.get("rumor_sensitivity", constraints.get("rumor_sensitivity", 0.5)) or 0.5))
        leverage = float(payload.get("leverage_bias", constraints.get("leverage_bias", 0.0)) or 0.0)
        loss_aversion = float(payload.get("loss_aversion", constraints.get("loss_aversion", 1.5)) or 1.5)
        inventory_limit = float(payload.get("inventory_limit", constraints.get("inventory_limit", 0.5)) or 0.5)

        buy_bias = 0.12 * (1.0 - risk_aversion) + 0.08 * turnover + 0.04 * max(leverage, 0.0)
        sell_bias = 0.10 * risk_aversion + 0.04 * rumor
        hold_bias = 0.18 * (1.0 - turnover) + 0.08 * risk_aversion
        if "market_maker" in group or "quant" in group:
            hold_bias += 0.10
        if "policy" in group or "stabilization" in group or "state" in group:
            buy_bias += 0.10
            sell_bias *= 0.60
        if "leveraged" in group:
            buy_bias += 0.08
            sell_bias += 0.06

        return cls(
            buy_bias=float(buy_bias),
            sell_bias=float(sell_bias),
            hold_bias=float(hold_bias),
            risk_aversion=float(_clip(risk_aversion)),
            turnover_preference=float(turnover),
            policy_sensitivity=float(policy),
            rumor_sensitivity=float(rumor),
            leverage_bias=float(leverage),
            loss_aversion=float(loss_aversion),
            inventory_tolerance=float(_clip(inventory_limit)),
            metadata={
                "agent_group": str(agent_group or group or "generic"),
                "source": "persona_prior_v1",
                "persona_fields": sorted(str(k) for k in payload.keys()),
            },
        )

    def logits(self) -> Dict[str, float]:
        return {
            "BUY": float(self.buy_bias),
            "HOLD": float(self.hold_bias),
            "SELL": float(self.sell_bias),
        }

    def distribution(self) -> Dict[str, float]:
        return _softmax(self.logits())

    def to_vector(self) -> Sequence[float]:
        return (
            float(self.buy_bias),
            float(self.sell_bias),
            float(self.hold_bias),
            float(self.risk_aversion),
            float(self.turnover_preference),
            float(self.policy_sensitivity),
            float(self.rumor_sensitivity),
            float(self.leverage_bias),
            float(self.loss_aversion),
            float(self.inventory_tolerance),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def kl_to_prior(policy_distribution: Mapping[str, float], prior: PersonaPrior | Mapping[str, float]) -> float:
    """KL(policy || persona_prior) with clipping for stable smoke tests."""

    prior_dist = prior.distribution() if isinstance(prior, PersonaPrior) else dict(prior)
    p = {str(k).upper(): max(float(v), 1e-12) for k, v in dict(policy_distribution or {}).items()}
    q = {str(k).upper(): max(float(v), 1e-12) for k, v in prior_dist.items()}
    keys = set(p) | set(q) | {"BUY", "HOLD", "SELL"}
    p_total = sum(p.get(key, 0.0) for key in keys) or 1.0
    q_total = sum(q.get(key, 0.0) for key in keys) or 1.0
    return float(sum((p.get(key, 0.0) / p_total) * math.log((p.get(key, 0.0) / p_total) / (q.get(key, 1e-12) / q_total)) for key in keys if p.get(key, 0.0) > 0))


__all__ = ["PersonaPrior", "kl_to_prior"]
