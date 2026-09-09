"""Lightweight NSGA-II style Pareto search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np

from core.optimization.constraints import evaluate_constraints
from core.optimization.objectives import composite_objective_score, objective_payload


ParameterSpace = Mapping[str, Sequence[float] | tuple[float, float]]
SimulatorFn = Callable[[Dict[str, Any]], Mapping[str, Any]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    objective_names = (
        "growth_proxy_score",
        "financial_stability_score",
        "fairness_compliance_score",
        "financing_function_score",
        "confidence_score",
    )
    left_obj = left.get("objectives", {})
    right_obj = right.get("objectives", {})
    left_constraints = left.get("constraints", {})
    right_constraints = right.get("constraints", {})
    left_violation = _safe_float(left_constraints.get("total_violation", 0.0))
    right_violation = _safe_float(right_constraints.get("total_violation", 0.0))
    if left_violation < right_violation:
        return True
    if left_violation > right_violation:
        return False
    better_or_equal = all(_safe_float(left_obj.get(name)) >= _safe_float(right_obj.get(name)) for name in objective_names)
    strictly_better = any(_safe_float(left_obj.get(name)) > _safe_float(right_obj.get(name)) for name in objective_names)
    lower_cost = _safe_float(left.get("metrics", {}).get("intervention_cost", 0.0)) <= _safe_float(right.get("metrics", {}).get("intervention_cost", 0.0))
    return bool(better_or_equal and strictly_better and lower_cost)


def pareto_frontier(evaluations: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in evaluations]
    frontier: List[Dict[str, Any]] = []
    for row in rows:
        if not any(_dominates(other, row) for other in rows if other is not row):
            frontier.append(row)
    frontier.sort(
        key=lambda item: (
            _safe_float(item.get("constraints", {}).get("total_violation", 0.0)),
            -_safe_float(item.get("score", 0.0)),
            _safe_float(item.get("metrics", {}).get("intervention_cost", 0.0)),
        )
    )
    return frontier


def _sample(space: ParameterSpace, rng: np.random.Generator) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in space.items():
        values = list(spec)
        if len(values) == 2 and all(isinstance(x, (int, float)) for x in values):
            low, high = float(values[0]), float(values[1])
            params[name] = float(rng.uniform(low, high))
        else:
            params[name] = values[int(rng.integers(0, len(values)))]
    return params


def _mutate(params: Mapping[str, Any], space: ParameterSpace, rng: np.random.Generator, scale: float = 0.18) -> Dict[str, Any]:
    out = dict(params)
    for name, spec in space.items():
        values = list(spec)
        if len(values) == 2 and all(isinstance(x, (int, float)) for x in values):
            low, high = float(values[0]), float(values[1])
            width = high - low
            out[name] = float(np.clip(_safe_float(out.get(name), (low + high) / 2.0) + rng.normal(0.0, width * scale), low, high))
        elif rng.random() < scale:
            out[name] = values[int(rng.integers(0, len(values)))]
    return out


@dataclass
class NSGAResult:
    evaluations: List[Dict[str, Any]] = field(default_factory=list)
    pareto_frontier: List[Dict[str, Any]] = field(default_factory=list)
    best: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"evaluations": self.evaluations, "pareto_frontier": self.pareto_frontier, "best": self.best}


def nsga_search(
    *,
    simulator: SimulatorFn,
    parameter_space: ParameterSpace,
    population_size: int = 12,
    generations: int = 3,
    seed: int = 42,
) -> NSGAResult:
    rng = np.random.default_rng(int(seed))
    population = [_sample(parameter_space, rng) for _ in range(max(2, int(population_size)))]
    evaluations: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _evaluate(params: Mapping[str, Any]) -> Dict[str, Any]:
        metrics = dict(simulator(dict(params)))
        objectives = objective_payload(metrics)
        constraints = evaluate_constraints(metrics)
        row = {
            "params": dict(params),
            "metrics": metrics,
            "objectives": objectives,
            "constraints": constraints,
            "score": composite_objective_score(metrics) - 0.25 * _safe_float(constraints.get("total_violation", 0.0)),
        }
        return row

    for _ in range(max(1, int(generations))):
        for params in population:
            key = str(sorted(params.items()))
            if key in seen:
                continue
            seen.add(key)
            evaluations.append(_evaluate(params))
        frontier = pareto_frontier(evaluations)
        parents = [row["params"] for row in frontier[: max(2, min(len(frontier), len(population)))]]
        if not parents:
            parents = population[:2]
        next_population = list(parents)
        while len(next_population) < max(2, int(population_size)):
            parent = parents[int(rng.integers(0, len(parents)))]
            next_population.append(_mutate(parent, parameter_space, rng))
        population = next_population

    frontier = pareto_frontier(evaluations)
    best = max(evaluations, key=lambda row: float(row.get("score", 0.0))) if evaluations else {}
    return NSGAResult(evaluations=evaluations, pareto_frontier=frontier, best=best)


__all__ = ["NSGAResult", "nsga_search", "pareto_frontier"]
