"""Unified evaluation suite for replay and policy experiments."""

from __future__ import annotations

from core.eval.event_study import abnormal_return_error, event_study, event_window_direction_hit_rate
from core.eval.orderbook_realism import compute_orderbook_realism
from core.eval.replay_scorecard import (
    EvaluationResult,
    ReplayScorecard,
    build_evaluation_result,
    build_replay_scorecard,
    build_statistical_tests,
    compute_risk_metrics,
    compute_shanghai_tracking_metrics,
    scorecard_to_json,
)
from core.eval.stat_tests import (
    ad_test,
    block_bootstrap_excess_loss,
    bootstrap_ci,
    diebold_mariano,
    ks_test,
    ljung_box,
    permutation_test,
)
from core.eval.stylized_facts import compute_behavioral_metrics

__all__ = [
    "EvaluationResult",
    "ReplayScorecard",
    "abnormal_return_error",
    "ad_test",
    "block_bootstrap_excess_loss",
    "bootstrap_ci",
    "build_evaluation_result",
    "build_replay_scorecard",
    "build_statistical_tests",
    "compute_behavioral_metrics",
    "compute_orderbook_realism",
    "compute_risk_metrics",
    "compute_shanghai_tracking_metrics",
    "diebold_mariano",
    "event_study",
    "event_window_direction_hit_rate",
    "ks_test",
    "ljung_box",
    "permutation_test",
    "scorecard_to_json",
]
