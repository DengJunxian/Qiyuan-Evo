"""Lightweight Bayesian-style black-box policy search.

This is deliberately dependency-light. It uses deterministic initial design
points, random exploration, and local exploitation around the current best
candidate. The public result contract is the important part for the simulator:
every evaluation carries objectives, constraints, Pareto membership candidates,
and enough metadata for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np

from core.optimization.constraints import evaluate_constraints
from core.optimization.nsga_search import pareto_frontier
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


def _space_bounds(spec: Sequence[float] | tuple[float, float]) -> tuple[float, float] | None:
    values = list(spec)
    if len(values) == 2 and all(isinstance(x, (int, float)) for x in values):
        return float(values[0]), float(values[1])
    return None


def _coerce_params(params: Mapping[str, Any], parameter_space: ParameterSpace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, spec in parameter_space.items():
        bounds = _space_bounds(spec)
        if bounds is not None:
            low, high = bounds
            out[name] = float(np.clip(_safe_float(params.get(name), (low + high) / 2.0), low, high))
        else:
            values = list(spec)
            value = params.get(name, values[0])
            out[name] = value if value in values else values[0]
    return out


def _midpoint(parameter_space: ParameterSpace) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in parameter_space.items():
        bounds = _space_bounds(spec)
        if bounds is not None:
            params[name] = (bounds[0] + bounds[1]) / 2.0
        else:
            values = list(spec)
            params[name] = values[len(values) // 2]
    return params


def _sample(parameter_space: ParameterSpace, rng: np.random.Generator) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in parameter_space.items():
        bounds = _space_bounds(spec)
        if bounds is not None:
            params[name] = float(rng.uniform(bounds[0], bounds[1]))
        else:
            values = list(spec)
            params[name] = values[int(rng.integers(0, len(values)))]
    return params


def _known_policy_anchor(parameter_space: ParameterSpace) -> Dict[str, Any]:
    anchor = _midpoint(parameter_space)
    preferred = {
        "stamp_tax_rate": 0.0005,
        "reserve_cut_bps": 25.0,
        "policy_rate_cut_bps": 10.0,
        "rumor_refute_strength": 0.60,
        "stabilization_capital": 0.20,
        "stabilization_timing": 1.0,
    }
    anchor.update({key: value for key, value in preferred.items() if key in parameter_space})
    return _coerce_params(anchor, parameter_space)


def _perturb(best: Mapping[str, Any], parameter_space: ParameterSpace, rng: np.random.Generator, scale: float) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in parameter_space.items():
        bounds = _space_bounds(spec)
        if bounds is not None:
            low, high = bounds
            width = high - low
            params[name] = float(np.clip(_safe_float(best.get(name), (low + high) / 2.0) + rng.normal(0.0, width * scale), low, high))
        else:
            values = list(spec)
            params[name] = values[int(rng.integers(0, len(values)))] if rng.random() < scale else best.get(name, values[0])
    return _coerce_params(params, parameter_space)


@dataclass
class BayesianSearchResult:
    best: Dict[str, Any] = field(default_factory=dict)
    evaluations: List[Dict[str, Any]] = field(default_factory=list)
    pareto_frontier: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "lightweight_bayesian_search"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "best": self.best,
            "evaluations": self.evaluations,
            "pareto_frontier": self.pareto_frontier,
        }


def bayesian_search(
    *,
    simulator: SimulatorFn,
    parameter_space: ParameterSpace,
    n_iter: int = 16,
    seed: int = 42,
    initial_params: Sequence[Mapping[str, Any]] | None = None,
) -> BayesianSearchResult:
    rng = np.random.default_rng(int(seed))
    design: List[Dict[str, Any]] = []
    design.extend([_midpoint(parameter_space), _known_policy_anchor(parameter_space)])
    design.extend([_coerce_params(row, parameter_space) for row in list(initial_params or [])])
    while len(design) < max(3, min(6, int(n_iter))):
        design.append(_sample(parameter_space, rng))

    evaluations: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _evaluate(params: Mapping[str, Any]) -> Dict[str, Any]:
        clean_params = _coerce_params(params, parameter_space)
        metrics = dict(simulator(clean_params))
        objectives = objective_payload(metrics)
        constraints = evaluate_constraints(metrics)
        score = composite_objective_score(metrics) - 0.30 * _safe_float(constraints.get("total_violation", 0.0))
        return {
            "params": clean_params,
            "metrics": metrics,
            "objectives": objectives,
            "constraints": constraints,
            "score": float(score),
        }

    for idx in range(max(1, int(n_iter))):
        if idx < len(design):
            params = design[idx]
        elif evaluations:
            ranked = sorted(evaluations, key=lambda row: float(row.get("score", 0.0)), reverse=True)
            base = ranked[int(rng.integers(0, min(3, len(ranked))))]["params"]
            params = _perturb(base, parameter_space, rng, scale=max(0.04, 0.20 * (1.0 - idx / max(int(n_iter), 1))))
        else:
            params = _sample(parameter_space, rng)
        key = str(sorted(params.items()))
        if key in seen:
            params = _sample(parameter_space, rng)
            key = str(sorted(params.items()))
        seen.add(key)
        evaluations.append(_evaluate(params))

    frontier = pareto_frontier(evaluations)
    best = max(evaluations, key=lambda row: float(row.get("score", 0.0))) if evaluations else {}
    return BayesianSearchResult(best=best, evaluations=evaluations, pareto_frontier=frontier)


__all__ = ["BayesianSearchResult", "ParameterSpace", "bayesian_search"]
