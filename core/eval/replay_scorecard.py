"""Unified replay scorecards and evaluation results."""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from core.eval.event_study import event_study
from core.eval.orderbook_realism import compute_orderbook_realism
from core.eval.stat_tests import ad_test, diebold_mariano, ks_test, ljung_box
from core.eval.stylized_facts import compute_behavioral_metrics


def stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _arr(values: Sequence[float] | None) -> np.ndarray:
    arr = np.asarray([] if values is None else list(values), dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    return arr[np.isfinite(arr)]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _numeric_dict(values: Mapping[str, Any] | None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in dict(values or {}).items():
        parsed = _safe_float(value, default=np.nan)
        if np.isfinite(parsed):
            out[str(key)] = float(parsed)
    return out


def rmse(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float:
    a = _arr(left)
    b = _arr(right)
    n = min(a.size, b.size)
    if n <= 0:
        return 0.0
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


def returns(prices: Sequence[float] | np.ndarray) -> np.ndarray:
    p = _arr(prices)
    if p.size < 2:
        return np.asarray([], dtype=float)
    return np.diff(p) / np.maximum(p[:-1], 1e-12)


def realized_volatility(prices: Sequence[float] | np.ndarray) -> float:
    r = returns(prices)
    return float(np.std(r)) if r.size else 0.0


def max_drawdown(prices: Sequence[float] | np.ndarray) -> float:
    p = _arr(prices)
    if p.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(p)
    dd = p / np.maximum(peaks, 1e-12) - 1.0
    return float(abs(np.min(dd)))


def ks_distance(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float:
    a = np.sort(_arr(left))
    b = np.sort(_arr(right))
    if a.size == 0 or b.size == 0:
        return 0.0
    values = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, values, side="right") / a.size
    cdf_b = np.searchsorted(b, values, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def direction_hit_rate(sim_prices: Sequence[float] | np.ndarray, real_prices: Sequence[float] | np.ndarray) -> float:
    sim = returns(sim_prices)
    real = returns(real_prices)
    n = min(sim.size, real.size)
    if n <= 0:
        return 0.0
    return float(np.mean(np.sign(sim[:n]) == np.sign(real[:n])))


def _corr(left: Sequence[float], right: Sequence[float]) -> float:
    a = _arr(left)
    b = _arr(right)
    n = min(a.size, b.size)
    if n < 2 or np.std(a[:n]) < 1e-12 or np.std(b[:n]) < 1e-12:
        return 0.0
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def _normalize_path(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=float)
    base = float(values[0])
    if abs(base) <= 1e-12:
        base = 1.0
    return values / base


def compute_shanghai_tracking_metrics(
    *,
    sim_close: Sequence[float] | None,
    real_close: Sequence[float] | None,
    event_points: Sequence[Any] | None = None,
    dates: Sequence[Any] | None = None,
    benchmark_symbol: str = "sh000001",
) -> Dict[str, float]:
    """First-level index tracking metrics for the Shanghai index benchmark."""

    sim = _arr(sim_close)
    real = _arr(real_close)
    n = min(sim.size, real.size)
    sim = sim[:n]
    real = real[:n]
    sim_ret = returns(sim)
    real_ret = returns(real)
    event_metrics = event_study(
        sim_close=sim,
        real_close=real,
        event_points=[] if event_points is None else list(event_points),
        dates=dates,
    )
    normalized_rmse = rmse(_normalize_path(sim), _normalize_path(real)) if n else 0.0
    return {
        "tracking_rmse": float(normalized_rmse),
        "normalized_rmse": float(normalized_rmse),
        "shanghai_index_path_error": float(normalized_rmse if str(benchmark_symbol) == "sh000001" else normalized_rmse),
        "price_correlation": _corr(sim, real),
        "return_correlation": _corr(sim_ret, real_ret),
        "direction_hit_rate": direction_hit_rate(sim, real),
        "event_window_direction_hit_rate": float(event_metrics["event_window_direction_hit_rate"]),
        "abnormal_return_error": float(event_metrics["abnormal_return_error"]),
        "ks_distance": ks_distance(sim_ret, real_ret),
    }


def compute_risk_metrics(
    *,
    sim_close: Sequence[float] | None,
    real_close: Sequence[float] | None = None,
    panic_series: Sequence[float] | None = None,
    contagion_metrics: Mapping[str, Any] | None = None,
) -> Dict[str, float]:
    """Risk metrics shared by UI, reports, and optimizers."""

    sim = _arr(sim_close)
    real = _arr(real_close)
    sim_ret = returns(sim)
    real_ret = returns(real)
    tail_loss = 0.0
    if sim_ret.size:
        threshold = float(np.quantile(sim_ret, 0.05))
        tail = sim_ret[sim_ret <= threshold]
        tail_loss = float(abs(np.mean(tail))) if tail.size else 0.0
    covar = 0.0
    if sim_ret.size and real_ret.size:
        n = min(sim_ret.size, real_ret.size)
        threshold = float(np.quantile(real_ret[:n], 0.05))
        tail_mask = real_ret[:n] <= threshold
        covar = float(abs(np.mean(sim_ret[:n][tail_mask]))) if np.any(tail_mask) else 0.0
    panic = _arr(panic_series)
    contagion = _numeric_dict(contagion_metrics)
    return {
        "volatility": realized_volatility(sim),
        "sim_volatility": realized_volatility(sim),
        "real_volatility": realized_volatility(real),
        "volatility_gap": abs(realized_volatility(sim) - realized_volatility(real)),
        "max_drawdown": max_drawdown(sim),
        "sim_max_drawdown": max_drawdown(sim),
        "real_max_drawdown": max_drawdown(real),
        "max_drawdown_gap": abs(max_drawdown(sim) - max_drawdown(real)),
        "tail_loss": float(tail_loss),
        "covar": float(covar),
        "network_shock": float(contagion.get("network_shock", contagion.get("debt_rank_systemic_risk", 0.0))),
        "panic_peak": float(np.max(panic)) if panic.size else float(contagion.get("panic_peak", 0.0)),
    }


@dataclass
class ReplayScorecard:
    experiment_id: str
    config_hash: str
    data_snapshot_hash: str
    seed: int
    benchmark_symbol: str
    replay_window: Dict[str, Any]
    path_fit_metrics: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    microstructure_metrics: Dict[str, float] = field(default_factory=dict)
    behavioral_metrics: Dict[str, float] = field(default_factory=dict)
    contagion_metrics: Dict[str, float] = field(default_factory=dict)
    regulatory_metrics: Dict[str, float] = field(default_factory=dict)
    pass_fail_flags: Dict[str, bool] = field(default_factory=dict)
    narrative_summary: str = ""
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "replay_scorecard_v2"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return scorecard_to_json(self)


@dataclass
class EvaluationResult:
    """Envelope for scorecard plus test and comparison outputs."""

    scorecard: ReplayScorecard
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    comparison_tests: Dict[str, Any] = field(default_factory=dict)
    optimizer_metrics: Dict[str, float] = field(default_factory=dict)
    schema_version: str = "evaluation_result_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scorecard": self.scorecard.to_dict(),
            "statistical_tests": dict(self.statistical_tests),
            "comparison_tests": dict(self.comparison_tests),
            "optimizer_metrics": dict(self.optimizer_metrics),
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, default=str)


def build_statistical_tests(
    *,
    sim_close: Sequence[float] | None,
    real_close: Sequence[float] | None,
    candidate_loss: Sequence[float] | None = None,
    baseline_loss: Sequence[float] | None = None,
) -> Dict[str, Any]:
    sim_ret = returns(_arr(sim_close))
    real_ret = returns(_arr(real_close))
    tests: Dict[str, Any] = {
        "return_distribution_ks": ks_test(sim_ret, real_ret),
        "return_distribution_ad": ad_test(sim_ret, real_ret),
        "serial_dependence_ljung_box": ljung_box(sim_ret),
    }
    if candidate_loss is not None and baseline_loss is not None:
        tests["loss_comparison_diebold_mariano"] = diebold_mariano(candidate_loss, baseline_loss)
    return tests


def build_replay_scorecard(
    *,
    sim_close: Sequence[float] | None,
    real_close: Sequence[float] | None,
    benchmark_symbol: str,
    replay_window: Mapping[str, Any],
    seed: int,
    config_hash: str,
    data_snapshot_hash: str,
    microstructure_metrics: Mapping[str, Any] | None = None,
    behavioral_metrics: Mapping[str, Any] | None = None,
    contagion_metrics: Mapping[str, Any] | None = None,
    regulatory_metrics: Mapping[str, Any] | None = None,
    event_points: Sequence[Any] | None = None,
    dates: Sequence[Any] | None = None,
    orderbook_snapshots: Sequence[Mapping[str, Any]] | None = None,
    trade_tape: Sequence[Mapping[str, Any]] | None = None,
    cross_sectional_returns: Sequence[Sequence[float]] | None = None,
    panic_series: Sequence[float] | None = None,
) -> ReplayScorecard:
    """Build a JSON-ready unified scorecard for one replay or policy experiment."""

    sim = _arr(sim_close)
    real = _arr(real_close)
    n = min(sim.size, real.size)
    sim = sim[:n]
    real = real[:n]
    event_point_list = [] if event_points is None else list(event_points)
    path = compute_shanghai_tracking_metrics(
        sim_close=sim,
        real_close=real,
        event_points=event_point_list,
        dates=dates,
        benchmark_symbol=benchmark_symbol,
    )
    risk = compute_risk_metrics(
        sim_close=sim,
        real_close=real,
        panic_series=panic_series,
        contagion_metrics=contagion_metrics,
    )
    micro = _numeric_dict(microstructure_metrics)
    if orderbook_snapshots or trade_tape:
        micro.update(
            compute_orderbook_realism(
                snapshots=orderbook_snapshots,
                trade_tape=trade_tape,
                reference_price=float(sim[-1]) if sim.size else None,
            )
        )
    behavior = _numeric_dict(behavioral_metrics)
    behavior.update(
        {
            key: value
            for key, value in compute_behavioral_metrics(
                prices=sim,
                cross_sectional_returns=cross_sectional_returns,
                trades=trade_tape,
                market_returns=returns(sim),
            ).items()
            if key not in behavior
        }
    )
    contagion = _numeric_dict(contagion_metrics)
    regulatory = _numeric_dict(regulatory_metrics)
    statistical_tests = build_statistical_tests(sim_close=sim, real_close=real)
    flags = {
        "direction_beats_random": path["direction_hit_rate"] >= 0.50,
        "event_direction_beats_random": path["event_window_direction_hit_rate"] >= 0.50 if event_point_list else True,
        "tracking_rmse_acceptable": path["tracking_rmse"] <= 0.08,
        "abnormal_return_error_acceptable": path["abnormal_return_error"] <= 0.02 if event_point_list else True,
        "volatility_not_catastrophic": risk["sim_volatility"] <= max(risk["real_volatility"] * 4.0, 0.08),
        "drawdown_not_catastrophic": risk["sim_max_drawdown"] <= max(risk["real_max_drawdown"] * 3.0, 0.35),
        "has_hashes": bool(config_hash and data_snapshot_hash),
        "stats_distribution_not_rejected_5pct": float(statistical_tests["return_distribution_ks"].get("p_value", 1.0)) >= 0.05,
    }
    experiment_id = "replay_" + stable_hash(
        {
            "benchmark_symbol": benchmark_symbol,
            "window": dict(replay_window),
            "seed": int(seed),
            "config_hash": config_hash,
            "data_snapshot_hash": data_snapshot_hash,
        }
    )[:16]
    narrative = (
        f"{benchmark_symbol} replay {dict(replay_window).get('start', '')} to {dict(replay_window).get('end', '')}: "
        f"tracking RMSE {path['tracking_rmse']:.4f}, direction hit {path['direction_hit_rate']:.2%}, "
        f"abnormal return error {path['abnormal_return_error']:.4f}, max drawdown {risk['sim_max_drawdown']:.2%}."
    )
    return ReplayScorecard(
        experiment_id=experiment_id,
        config_hash=str(config_hash),
        data_snapshot_hash=str(data_snapshot_hash),
        seed=int(seed),
        benchmark_symbol=str(benchmark_symbol),
        replay_window=dict(replay_window),
        path_fit_metrics=path,
        risk_metrics=risk,
        microstructure_metrics=micro,
        behavioral_metrics=behavior,
        contagion_metrics=contagion,
        regulatory_metrics=regulatory,
        pass_fail_flags=flags,
        narrative_summary=narrative,
        statistical_tests=statistical_tests,
    )


def build_evaluation_result(
    *,
    scorecard: ReplayScorecard,
    candidate_loss: Sequence[float] | None = None,
    baseline_loss: Sequence[float] | None = None,
) -> EvaluationResult:
    comparison_tests: Dict[str, Any] = {}
    if candidate_loss is not None and baseline_loss is not None:
        comparison_tests["loss_comparison_diebold_mariano"] = diebold_mariano(candidate_loss, baseline_loss)
    optimizer_metrics = {
        "tracking_rmse": float(scorecard.path_fit_metrics.get("tracking_rmse", 0.0)),
        "financial_stability_score": float(1.0 - min(1.0, scorecard.risk_metrics.get("max_drawdown", 0.0))),
        "fairness_compliance_score": float(scorecard.regulatory_metrics.get("fairness", scorecard.regulatory_metrics.get("fairness_compliance", 0.0))),
        "financing_function_score": float(scorecard.regulatory_metrics.get("financing_function", 0.0)),
        "confidence_score": float(scorecard.regulatory_metrics.get("market_confidence", scorecard.behavioral_metrics.get("confidence", 0.0))),
    }
    return EvaluationResult(
        scorecard=scorecard,
        statistical_tests=dict(scorecard.statistical_tests),
        comparison_tests=comparison_tests,
        optimizer_metrics=optimizer_metrics,
    )


def scorecard_to_json(scorecard: ReplayScorecard) -> str:
    return json.dumps(scorecard.to_dict(), ensure_ascii=False, sort_keys=True, default=str)


__all__ = [
    "EvaluationResult",
    "ReplayScorecard",
    "build_evaluation_result",
    "build_replay_scorecard",
    "build_statistical_tests",
    "compute_risk_metrics",
    "compute_shanghai_tracking_metrics",
    "scorecard_to_json",
]
