"""Black-box and multi-objective policy optimization primitives."""

from __future__ import annotations

from core.optimization.bayes_search import BayesianSearchResult, ParameterSpace, bayesian_search
from core.optimization.constraints import ConstraintSpec, DEFAULT_CONSTRAINTS, constraint_penalty, evaluate_constraints
from core.optimization.nsga_search import NSGAResult, nsga_search, pareto_frontier
from core.optimization.objectives import (
    DEFAULT_OBJECTIVE_WEIGHTS,
    PolicyObjectiveVector,
    composite_objective_score,
    extract_policy_objectives,
    objective_payload,
)
from core.optimization.report_generator import generate_optimization_report, sensitivity_analysis
from core.optimization.sequential_rl import SequentialRulePolicy, optimize_sequential_rule

__all__ = [
    "BayesianSearchResult",
    "ConstraintSpec",
    "DEFAULT_CONSTRAINTS",
    "DEFAULT_OBJECTIVE_WEIGHTS",
    "NSGAResult",
    "ParameterSpace",
    "PolicyObjectiveVector",
    "SequentialRulePolicy",
    "bayesian_search",
    "composite_objective_score",
    "constraint_penalty",
    "evaluate_constraints",
    "extract_policy_objectives",
    "generate_optimization_report",
    "nsga_search",
    "objective_payload",
    "optimize_sequential_rule",
    "pareto_frontier",
    "sensitivity_analysis",
]
