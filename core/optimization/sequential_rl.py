"""Sequential intervention baselines without mandatory RL dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping

import numpy as np

from core.optimization.bayes_search import bayesian_search


Observation = Mapping[str, Any]
ActionDict = Dict[str, Any]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


@dataclass(frozen=True)
class SequentialRulePolicy:
    panic_threshold: float = 0.35
    crash_threshold: float = 0.12
    base_strength: float = 0.35
    max_strength: float = 0.85

    def act(self, observation: Observation) -> ActionDict:
        panic = _safe_float(observation.get("panic_level", observation.get("panic", 0.0)))
        crash = _safe_float(observation.get("crash_loss_rate", observation.get("crash_risk", 0.0)))
        stress = max(
            0.0,
            (panic - self.panic_threshold) / max(1.0 - self.panic_threshold, 1e-9),
            (crash - self.crash_threshold) / max(1.0 - self.crash_threshold, 1e-9),
        )
        strength = float(np.clip(self.base_strength + stress * (self.max_strength - self.base_strength), 0.0, 1.0))
        return {
            "stamp_tax_rate": 0.0005 if stress < 0.4 else 0.0003,
            "reserve_cut_bps": 25.0 if stress > 0.1 else 0.0,
            "policy_rate_cut_bps": 10.0 if stress > 0.25 else 0.0,
            "rumor_refute_strength": strength,
            "stabilization_capital": 0.2 + 0.5 * stress,
            "stabilization_timing": 0.0,
        }


@dataclass
class SequentialOptimizationResult:
    baseline_policy: Dict[str, Any]
    optimized_policy: Dict[str, Any] = field(default_factory=dict)
    search_result: Dict[str, Any] = field(default_factory=dict)
    rl_backend: str = "rule_baseline_plus_blackbox"
    ppo_sac_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_policy": self.baseline_policy,
            "optimized_policy": self.optimized_policy,
            "search_result": self.search_result,
            "rl_backend": self.rl_backend,
            "ppo_sac_enabled": self.ppo_sac_enabled,
        }


def optimize_sequential_rule(
    *,
    simulator: Callable[[Dict[str, Any]], Mapping[str, Any]],
    seed: int = 42,
    n_iter: int = 10,
) -> SequentialOptimizationResult:
    """Optimize rule-policy thresholds with the same black-box interface."""

    parameter_space = {
        "panic_threshold": (0.20, 0.55),
        "crash_threshold": (0.05, 0.25),
        "base_strength": (0.10, 0.55),
        "max_strength": (0.55, 1.00),
    }
    search = bayesian_search(simulator=simulator, parameter_space=parameter_space, n_iter=n_iter, seed=seed)
    best_params = dict(search.best.get("params", {})) if search.best else {}
    return SequentialOptimizationResult(
        baseline_policy=SequentialRulePolicy().act({"panic_level": 0.4, "crash_loss_rate": 0.1}),
        optimized_policy=best_params,
        search_result=search.to_dict(),
    )


__all__ = ["SequentialOptimizationResult", "SequentialRulePolicy", "optimize_sequential_rule"]
