"""Policy optimization objectives shared across search backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


@dataclass(frozen=True)
class PolicyObjectiveVector:
    growth_proxy_score: float
    financial_stability_score: float
    fairness_compliance_score: float
    financing_function_score: float
    confidence_score: float

    def to_dict(self) -> Dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


DEFAULT_OBJECTIVE_WEIGHTS: Dict[str, float] = {
    "growth_proxy_score": 0.18,
    "financial_stability_score": 0.28,
    "fairness_compliance_score": 0.18,
    "financing_function_score": 0.18,
    "confidence_score": 0.18,
}


def extract_policy_objectives(metrics: Mapping[str, Any]) -> PolicyObjectiveVector:
    """Build the five required objective scores from a simulator result row."""

    stability = _safe_float(metrics.get("macro_stability", metrics.get("financial_stability_score", 0.0)))
    crash_risk = _safe_float(metrics.get("crash_risk", metrics.get("max_drawdown", 0.0)))
    volatility = _safe_float(metrics.get("volatility", 0.0))
    avg_reward = _safe_float(metrics.get("avg_reward", metrics.get("episode_reward", 0.0)))
    tracking_rmse = _safe_float(metrics.get("tracking_rmse", 0.0))
    growth = metrics.get("growth_proxy_score")
    if growth is None:
        growth = 0.50 + 0.20 * avg_reward - 0.50 * tracking_rmse + 0.20 * _safe_float(metrics.get("liquidity", 0.0))
    financial = metrics.get("financial_stability_score")
    if financial is None:
        financial = 0.60 * stability + 0.25 * (1.0 - crash_risk) + 0.15 * (1.0 - min(1.0, volatility))
    return PolicyObjectiveVector(
        growth_proxy_score=_clip01(_safe_float(growth)),
        financial_stability_score=_clip01(_safe_float(financial)),
        fairness_compliance_score=_clip01(_safe_float(metrics.get("fairness_compliance_score", metrics.get("fairness_compliance", 0.0)))),
        financing_function_score=_clip01(_safe_float(metrics.get("financing_function_score", metrics.get("financing_function", 0.0)))),
        confidence_score=_clip01(_safe_float(metrics.get("confidence_score", metrics.get("welfare_confidence", metrics.get("market_confidence", 0.0))))),
    )


def composite_objective_score(
    metrics: Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
    cost_penalty_weight: float = 0.08,
) -> float:
    objectives = extract_policy_objectives(metrics).to_dict()
    merged_weights = {**DEFAULT_OBJECTIVE_WEIGHTS, **dict(weights or {})}
    weight_sum = sum(max(0.0, float(value)) for value in merged_weights.values()) or 1.0
    score = sum(float(objectives.get(name, 0.0)) * max(0.0, float(weight)) for name, weight in merged_weights.items()) / weight_sum
    return float(score - float(cost_penalty_weight) * _safe_float(metrics.get("intervention_cost", 0.0)))


def objective_payload(metrics: Mapping[str, Any]) -> Dict[str, float]:
    payload = extract_policy_objectives(metrics).to_dict()
    payload["composite_objective_score"] = composite_objective_score(metrics)
    return payload


__all__ = [
    "DEFAULT_OBJECTIVE_WEIGHTS",
    "PolicyObjectiveVector",
    "composite_objective_score",
    "extract_policy_objectives",
    "objective_payload",
]
