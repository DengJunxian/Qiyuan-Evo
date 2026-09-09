"""Constraint checks for policy optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    metric: str
    op: str
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONSTRAINTS: tuple[ConstraintSpec, ...] = (
    ConstraintSpec("max_drawdown", "max_drawdown", "<=", 0.25),
    ConstraintSpec("panic", "panic", "<=", 0.45),
    ConstraintSpec("intervention_cost", "intervention_cost", "<=", 1.20),
    ConstraintSpec("market_function", "market_function_score", ">=", 0.45),
)


def _metric_value(metrics: Mapping[str, Any], metric: str) -> float:
    aliases = {
        "max_drawdown": ("max_drawdown", "crash_risk", "sim_max_drawdown"),
        "panic": ("panic", "panic_peak", "panic_level"),
        "market_function_score": ("market_function_score", "financing_function", "liquidity"),
    }
    for key in aliases.get(metric, (metric,)):
        if key in metrics:
            return _safe_float(metrics.get(key))
    return 0.0


def evaluate_constraints(
    metrics: Mapping[str, Any],
    constraints: Sequence[ConstraintSpec] | None = None,
) -> Dict[str, Any]:
    specs = list(constraints or DEFAULT_CONSTRAINTS)
    rows = []
    total_violation = 0.0
    for spec in specs:
        value = _metric_value(metrics, spec.metric)
        threshold = float(spec.threshold)
        if spec.op == "<=":
            violated = value > threshold
            amount = max(0.0, value - threshold)
        elif spec.op == ">=":
            violated = value < threshold
            amount = max(0.0, threshold - value)
        else:
            raise ValueError(f"Unsupported constraint operator: {spec.op}")
        total_violation += amount
        rows.append(
            {
                "name": spec.name,
                "metric": spec.metric,
                "op": spec.op,
                "threshold": threshold,
                "value": float(value),
                "violated": bool(violated),
                "violation": float(amount),
            }
        )
    return {
        "feasible": not any(row["violated"] for row in rows),
        "total_violation": float(total_violation),
        "violations": rows,
    }


def constraint_penalty(metrics: Mapping[str, Any], constraints: Sequence[ConstraintSpec] | None = None) -> float:
    return float(evaluate_constraints(metrics, constraints)["total_violation"])


__all__ = ["ConstraintSpec", "DEFAULT_CONSTRAINTS", "constraint_penalty", "evaluate_constraints"]
