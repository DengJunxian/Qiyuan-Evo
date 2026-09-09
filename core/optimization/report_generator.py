"""Optimization report assembly for UI, reports, and defense materials."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

import numpy as np

from core.optimization.constraints import evaluate_constraints


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def sensitivity_analysis(
    *,
    simulator: Callable[[Dict[str, Any]], Mapping[str, Any]],
    best_params: Mapping[str, Any],
    parameter_space: Mapping[str, Any],
    step_fraction: float = 0.10,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    base_metrics = dict(simulator(dict(best_params))) if best_params else {}
    base_score = _safe_float(base_metrics.get("avg_reward", base_metrics.get("composite_objective_score", 0.0)))
    for name, spec in parameter_space.items():
        values = list(spec)
        if len(values) != 2 or not all(isinstance(x, (int, float)) for x in values):
            continue
        low, high = float(values[0]), float(values[1])
        width = high - low
        current = _safe_float(best_params.get(name), (low + high) / 2.0)
        for direction, sign in (("down", -1.0), ("up", 1.0)):
            params = dict(best_params)
            params[name] = float(np.clip(current + sign * width * float(step_fraction), low, high))
            metrics = dict(simulator(params))
            score = _safe_float(metrics.get("avg_reward", metrics.get("composite_objective_score", 0.0)))
            rows.append(
                {
                    "parameter": name,
                    "direction": direction,
                    "value": float(params[name]),
                    "score_delta": float(score - base_score),
                    "macro_stability": _safe_float(metrics.get("macro_stability", 0.0)),
                    "intervention_cost": _safe_float(metrics.get("intervention_cost", 0.0)),
                }
            )
    return rows


def generate_optimization_report(
    *,
    input_scenario: Mapping[str, Any],
    data_snapshot: Mapping[str, Any],
    parameter_space: Mapping[str, Any],
    q_learning_result: Mapping[str, Any] | None,
    bayesian_result: Mapping[str, Any],
    nsga_result: Mapping[str, Any],
    rule_baseline: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
    validation: Mapping[str, Any],
    sensitivity: list[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    bo_best = dict(bayesian_result.get("best", {}) or {})
    best_metrics = dict(bo_best.get("metrics", {}) or {})
    best_params = dict(bo_best.get("params", {}) or {})
    constraints = evaluate_constraints(best_metrics)
    pareto = list(nsga_result.get("pareto_frontier") or bayesian_result.get("pareto_frontier") or [])
    objective_scores = dict(bo_best.get("objectives", {}) or {})
    stable = bool(validation.get("promote_blackbox_default", False))
    recommendation = (
        "Use bayesian_blackbox as the default path for this scenario family."
        if stable
        else "Keep q_learning_baseline or manual rule path as production default; use black-box output as decision support."
    )
    llm_summary = (
        f"Best candidate improves stability to {float(best_metrics.get('macro_stability', 0.0)):.3f} "
        f"with cost {float(best_metrics.get('intervention_cost', 0.0)):.3f}. "
        f"Default promotion guard passed={stable}."
    )
    return {
        "input_scenario": dict(input_scenario),
        "data_snapshot": dict(data_snapshot),
        "parameter_space": {str(k): list(v) if isinstance(v, (list, tuple)) else v for k, v in parameter_space.items()},
        "best_solution": {"params": best_params, "metrics": best_metrics, "score": float(bo_best.get("score", 0.0))},
        "pareto_frontier": pareto,
        "primary_objective_scores": objective_scores,
        "secondary_objective_scores": {
            "tracking_rmse": _safe_float(best_metrics.get("tracking_rmse", 0.0)),
            "avg_reward": _safe_float(best_metrics.get("avg_reward", 0.0)),
            "episode_reward": _safe_float(best_metrics.get("episode_reward", 0.0)),
        },
        "constraint_violations": constraints,
        "counterfactual_controls": dict(counterfactual),
        "sensitivity_analysis": list(sensitivity or []),
        "final_recommendation_text": recommendation,
        "llm_explanation_summary": llm_summary,
        "method_outputs": {
            "q_learning": dict(q_learning_result or {}),
            "bayesian_optimization": dict(bayesian_result),
            "nsga_ii": dict(nsga_result),
            "rule_baseline": dict(rule_baseline),
        },
        "validation": dict(validation),
    }


__all__ = ["generate_optimization_report", "sensitivity_analysis"]
