"""Explainable objective discovery for policy experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


@dataclass(slots=True)
class DiscoveredMetric:
    name: str
    category: str
    value: float
    direction: str
    policy_sensitivity: float
    robustness: float
    counterfactual_discriminability: float
    historical_replay_alignment: float
    early_warning_utility: float
    composite_weight: float
    rank_score: float
    relation_to_shanghai_index: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "value": float(self.value),
            "direction": self.direction,
            "policy_sensitivity": float(self.policy_sensitivity),
            "robustness": float(self.robustness),
            "counterfactual_discriminability": float(self.counterfactual_discriminability),
            "historical_replay_alignment": float(self.historical_replay_alignment),
            "early_warning_utility": float(self.early_warning_utility),
            "composite_weight": float(self.composite_weight),
            "rank_score": float(self.rank_score),
            "relation_to_shanghai_index": self.relation_to_shanghai_index,
            "explanation": self.explanation,
        }


@dataclass(slots=True)
class ObjectiveDiscoveryResult:
    ranked_metrics: List[DiscoveredMetric] = field(default_factory=list)
    pareto_frontier: List[Dict[str, Any]] = field(default_factory=list)
    composite_score: float = 0.0
    weight_decomposition: Dict[str, float] = field(default_factory=dict)
    stability_heatmap: List[Dict[str, Any]] = field(default_factory=list)
    candidate_pool: List[str] = field(default_factory=list)
    shanghai_index_metric: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ranked_metrics": [item.to_dict() for item in self.ranked_metrics],
            "top_metrics": [item.to_dict() for item in self.ranked_metrics[:8]],
            "pareto_frontier": list(self.pareto_frontier),
            "composite_score": float(self.composite_score),
            "weight_decomposition": dict(self.weight_decomposition),
            "stability_heatmap": list(self.stability_heatmap),
            "candidate_pool": list(self.candidate_pool),
            "shanghai_index_metric": dict(self.shanghai_index_metric),
            "schema_version": "objective_discovery_v1",
        }


class ObjectiveDiscoveryEngine:
    """Interpretable metric selection across market, microstructure, behavior, macro, and ecology."""

    CANDIDATE_CATEGORIES: Dict[str, str] = {
        "shanghai_index_return": "market_price_risk",
        "shanghai_tracking_rmse": "market_price_risk",
        "shanghai_alignment_score": "market_price_risk",
        "event_direction_hit_rate": "market_price_risk",
        "abnormal_return_error": "market_price_risk",
        "index_return": "market_price_risk",
        "max_drawdown": "market_price_risk",
        "realized_volatility": "market_price_risk",
        "tail_risk": "market_price_risk",
        "turnaround_speed": "market_price_risk",
        "spread": "microstructure",
        "depth": "microstructure",
        "slippage": "microstructure",
        "cancel_to_trade_ratio": "microstructure",
        "order_imbalance": "microstructure",
        "liquidity_thinness": "microstructure",
        "microstructure_score": "microstructure",
        "csad_herding": "behavior_sentiment",
        "panic_level": "behavior_sentiment",
        "belief_dispersion": "behavior_sentiment",
        "social_contagion_intensity": "behavior_sentiment",
        "rumor_impact_persistence": "behavior_sentiment",
        "liquidity_index": "macro_financing",
        "policy_rate": "macro_financing",
        "credit_spread": "macro_financing",
        "unemployment": "macro_financing",
        "wage_growth": "macro_financing",
        "inflation": "macro_financing",
        "financing_function": "macro_financing",
        "fairness_compliance": "macro_financing",
        "welfare_confidence": "macro_financing",
        "entropy": "ecology_stability",
        "hhi": "ecology_stability",
        "modularity": "ecology_stability",
        "coalition_persistence": "ecology_stability",
        "phase_change_score": "ecology_stability",
        "abuse_event_rate": "ecology_stability",
        "intervention_cost": "ecology_stability",
    }

    POSITIVE_GOOD = {
        "shanghai_index_return",
        "shanghai_alignment_score",
        "event_direction_hit_rate",
        "index_return",
        "turnaround_speed",
        "depth",
        "microstructure_score",
        "liquidity_index",
        "wage_growth",
        "financing_function",
        "fairness_compliance",
        "welfare_confidence",
        "entropy",
    }

    def _frame_from_path(self, path: Any) -> pd.DataFrame:
        if isinstance(path, pd.DataFrame):
            return path.copy()
        if isinstance(path, Mapping):
            if "frame" in path:
                return self._frame_from_path(path["frame"])
            if "日度结果" in path:
                return pd.DataFrame(path["日度结果"])
            if "rows" in path:
                return pd.DataFrame(path["rows"])
        if isinstance(path, Sequence) and not isinstance(path, (str, bytes, bytearray)):
            return pd.DataFrame(list(path))
        return pd.DataFrame()

    def _extract_candidates(self, frame: pd.DataFrame, reports: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, float]:
        if frame.empty:
            return {name: 0.0 for name in self.CANDIDATE_CATEGORIES}

        def _series_from(*names: str, default: float = 0.0) -> pd.Series:
            for name in names:
                if name in frame.columns:
                    return pd.to_numeric(frame[name], errors="coerce").fillna(float(default)).astype(float)
            return pd.Series([float(default)] * len(frame), index=frame.index, dtype=float)

        close_col = "close" if "close" in frame.columns else "收盘价" if "收盘价" in frame.columns else None
        close = pd.to_numeric(frame[close_col], errors="coerce").ffill().bfill() if close_col else pd.Series(dtype=float)
        returns = close.pct_change().fillna(0.0) if not close.empty else pd.Series(dtype=float)
        high = pd.to_numeric(frame.get("high", frame.get("最高", close)), errors="coerce").ffill().bfill() if close_col else close
        low = pd.to_numeric(frame.get("low", frame.get("最低", close)), errors="coerce").ffill().bfill() if close_col else close
        volume = _series_from("volume", "成交量", "总买量")
        buy = _series_from("买入量", "总买量")
        sell = _series_from("卖出量", "总卖量")
        panic = _series_from("panic_level", "恐慌度")
        csad = _series_from("csad", "羊群度")
        drawdown = close / close.cummax() - 1.0 if not close.empty else pd.Series(dtype=float)
        tail_threshold = float(returns.std() * 1.65) if len(returns) > 2 else 0.0

        report_payloads = list(reports or [])
        micro_values: Dict[str, List[float]] = {
            "spread": [],
            "depth": [],
            "slippage": [],
            "cancel_to_trade_ratio": [],
            "liquidity_thinness": [],
            "microstructure_score": [],
            "belief_dispersion": [],
            "abuse_event_rate": [],
            "entropy": [],
            "hhi": [],
            "modularity": [],
            "coalition_persistence": [],
            "phase_change_score": [],
        }
        macro_values: Dict[str, List[float]] = {
            "liquidity_index": [],
            "policy_rate": [],
            "credit_spread": [],
            "unemployment": [],
            "wage_growth": [],
            "inflation": [],
        }
        path_values: Dict[str, List[float]] = {
            "shanghai_tracking_rmse": [],
            "shanghai_alignment_score": [],
            "event_direction_hit_rate": [],
            "abnormal_return_error": [],
        }
        for report in report_payloads:
            micro = dict(report.get("microstructure_metrics", {}) or {})
            realism = dict(report.get("realism_diagnostics", {}) or {})
            macro = dict(report.get("macro_state", {}) or {})
            ecology = dict(report.get("ecology_metrics", {}) or {})
            abuse = dict(report.get("abuse_detection", {}) or {})
            chain = dict(report.get("policy_transmission_chain", {}) or {})
            scorecard = dict(report.get("replay_scorecard", report.get("scorecard", {})) or {})
            path_fit = dict(scorecard.get("path_fit_metrics", report.get("path_fit_metrics", {})) or {})
            if "tracking_rmse" in path_fit:
                tracking_rmse = _safe_float(path_fit.get("tracking_rmse"))
                path_values["shanghai_tracking_rmse"].append(tracking_rmse)
                path_values["shanghai_alignment_score"].append(_clip(1.0 - tracking_rmse * 8.0, 0.0, 1.0))
            if "event_window_direction_hit_rate" in path_fit:
                path_values["event_direction_hit_rate"].append(_safe_float(path_fit.get("event_window_direction_hit_rate")))
            if "abnormal_return_error" in path_fit:
                path_values["abnormal_return_error"].append(_safe_float(path_fit.get("abnormal_return_error")))
            beliefs = dict(chain.get("agent_beliefs", {}) or {})
            for key, source_key in (
                ("spread", "spread_pct"),
                ("slippage", "slippage_bps"),
                ("cancel_to_trade_ratio", "cancel_to_trade_ratio"),
            ):
                if source_key in micro:
                    micro_values[key].append(_safe_float(micro.get(source_key)))
            depth = _safe_float(micro.get("depth_bid_total")) + _safe_float(micro.get("depth_ask_total"))
            if depth:
                micro_values["depth"].append(depth)
            for key in ("liquidity_thinness", "microstructure_score"):
                if key in realism:
                    micro_values[key].append(_safe_float(realism.get(key)))
            if "dispersion" in beliefs:
                micro_values["belief_dispersion"].append(_safe_float(beliefs.get("dispersion")))
            micro_values["abuse_event_rate"].append(_safe_float(abuse.get("events_detected", 0.0)))
            for key in ("entropy", "hhi", "modularity", "coalition_persistence", "phase_change_score"):
                if key in ecology:
                    micro_values[key].append(_safe_float(ecology.get(key)))
            for key in macro_values:
                if key in macro:
                    macro_values[key].append(_safe_float(macro.get(key)))

        def _avg(series: Iterable[float]) -> float:
            values = [float(v) for v in series if np.isfinite(float(v))]
            return float(np.mean(values)) if values else 0.0

        candidates = {
            "shanghai_index_return": float(close.iloc[-1] / max(close.iloc[0], 1e-12) - 1.0) if len(close) else 0.0,
            "shanghai_tracking_rmse": _avg(path_values["shanghai_tracking_rmse"]),
            "shanghai_alignment_score": _avg(path_values["shanghai_alignment_score"]) if path_values["shanghai_alignment_score"] else 1.0,
            "event_direction_hit_rate": _avg(path_values["event_direction_hit_rate"]),
            "abnormal_return_error": _avg(path_values["abnormal_return_error"]),
            "index_return": float(close.iloc[-1] / max(close.iloc[0], 1e-12) - 1.0) if len(close) else 0.0,
            "max_drawdown": float(abs(drawdown.min())) if len(drawdown) else 0.0,
            "realized_volatility": float(returns.std()) if len(returns) else 0.0,
            "tail_risk": float(np.mean(np.abs(returns) > tail_threshold)) if len(returns) and tail_threshold > 0 else 0.0,
            "turnaround_speed": float(max(0.0, returns.tail(5).sum())) if len(returns) else 0.0,
            "spread": _avg(micro_values["spread"]),
            "depth": _avg(micro_values["depth"]),
            "slippage": _avg(micro_values["slippage"]),
            "cancel_to_trade_ratio": _avg(micro_values["cancel_to_trade_ratio"]),
            "order_imbalance": float(np.mean(np.abs((buy - sell) / np.maximum(np.abs(buy) + np.abs(sell), 1.0)))) if len(buy) else 0.0,
            "liquidity_thinness": _avg(micro_values["liquidity_thinness"]),
            "microstructure_score": _avg(micro_values["microstructure_score"]),
            "csad_herding": float(csad.mean()) if len(csad) else 0.0,
            "panic_level": float(panic.mean()) if len(panic) else 0.0,
            "belief_dispersion": _avg(micro_values["belief_dispersion"]),
            "social_contagion_intensity": float(panic.diff().abs().mean()) if len(panic) > 1 else 0.0,
            "rumor_impact_persistence": float(panic.rolling(3).mean().max()) if len(panic) else 0.0,
            "liquidity_index": _avg(macro_values["liquidity_index"]),
            "policy_rate": _avg(macro_values["policy_rate"]),
            "credit_spread": _avg(macro_values["credit_spread"]),
            "unemployment": _avg(macro_values["unemployment"]),
            "wage_growth": _avg(macro_values["wage_growth"]),
            "inflation": _avg(macro_values["inflation"]),
            "financing_function": float(_clip(1.0 - _avg(macro_values["credit_spread"]) * 8.0 - float(panic.mean() if len(panic) else 0.0) * 0.35, 0.0, 1.0)),
            "fairness_compliance": float(_clip(1.0 - _avg(micro_values["abuse_event_rate"]) * 0.25 - float(panic.mean() if len(panic) else 0.0) * 0.20, 0.0, 1.0)),
            "welfare_confidence": float(_clip(1.0 - float(panic.mean() if len(panic) else 0.0), 0.0, 1.0)),
            "entropy": _avg(micro_values["entropy"]),
            "hhi": _avg(micro_values["hhi"]),
            "modularity": _avg(micro_values["modularity"]),
            "coalition_persistence": _avg(micro_values["coalition_persistence"]),
            "phase_change_score": _avg(micro_values["phase_change_score"]),
            "abuse_event_rate": _avg(micro_values["abuse_event_rate"]),
            "intervention_cost": float(np.clip(_avg(micro_values["abuse_event_rate"]) * 0.4 + float(panic.mean() if len(panic) else 0.0) * 0.15, 0.0, 1.0)),
        }
        return {name: float(candidates.get(name, 0.0)) for name in self.CANDIDATE_CATEGORIES}

    def _score_metric(self, name: str, value: float, all_values: Mapping[str, float], scenario_count: int) -> DiscoveredMetric:
        direction = "maximize" if name in self.POSITIVE_GOOD else "minimize"
        normalized = _clip(value if direction == "maximize" else 1.0 - value, 0.0, 1.0)
        index_abs = abs(float(all_values.get("shanghai_index_return", 0.0)))
        sensitivity = _clip(abs(value) / max(index_abs, 0.02), 0.0, 1.0)
        robustness = _clip(0.55 + 0.08 * scenario_count - min(abs(value), 1.0) * 0.10, 0.0, 1.0)
        discriminability = _clip(abs(normalized - 0.5) * 1.6 + (0.15 if name != "shanghai_index_return" else 0.05), 0.0, 1.0)
        historical_alignment = _clip(
            0.62
            + 0.18 * float(all_values.get("microstructure_score", 0.0))
            - 0.10 * float(all_values.get("liquidity_thinness", 0.0)),
            0.0,
            1.0,
        )
        early_warning = _clip(
            0.20
            + (0.55 if name in {"panic_level", "belief_dispersion", "social_contagion_intensity", "liquidity_thinness", "credit_spread", "phase_change_score"} else 0.20)
            + 0.10 * float(all_values.get("panic_level", 0.0)),
            0.0,
            1.0,
        )
        rank_score = float(
            0.24 * sensitivity
            + 0.20 * robustness
            + 0.22 * discriminability
            + 0.17 * historical_alignment
            + 0.17 * early_warning
        )
        relation = "self" if name in {"shanghai_index_return", "shanghai_tracking_rmse", "shanghai_alignment_score"} else (
            "leading" if name in {"panic_level", "belief_dispersion", "credit_spread", "liquidity_thinness"} else
            "complement" if name in {"microstructure_score", "financing_function", "fairness_compliance"} else
            "lagging" if name in {"max_drawdown", "realized_volatility", "abnormal_return_error"} else
            "alternative"
        )
        explanation = (
            f"{name} captures {self.CANDIDATE_CATEGORIES[name]} with sensitivity={sensitivity:.2f}, "
            f"robustness={robustness:.2f}, and relation_to_index={relation}."
        )
        return DiscoveredMetric(
            name=name,
            category=self.CANDIDATE_CATEGORIES[name],
            value=float(value),
            direction=direction,
            policy_sensitivity=float(sensitivity),
            robustness=float(robustness),
            counterfactual_discriminability=float(discriminability),
            historical_replay_alignment=float(historical_alignment),
            early_warning_utility=float(early_warning),
            composite_weight=0.0,
            rank_score=float(rank_score),
            relation_to_shanghai_index=relation,
            explanation=explanation,
        )

    @staticmethod
    def _pareto(metrics: Sequence[DiscoveredMetric]) -> List[Dict[str, Any]]:
        rows = [item.to_dict() for item in metrics]
        frontier: List[Dict[str, Any]] = []
        for row in rows:
            dominated = False
            for other in rows:
                if other is row:
                    continue
                better_or_equal = (
                    float(other["policy_sensitivity"]) >= float(row["policy_sensitivity"])
                    and float(other["robustness"]) >= float(row["robustness"])
                    and float(other["early_warning_utility"]) >= float(row["early_warning_utility"])
                )
                strictly = (
                    float(other["policy_sensitivity"]) > float(row["policy_sensitivity"])
                    or float(other["robustness"]) > float(row["robustness"])
                    or float(other["early_warning_utility"]) > float(row["early_warning_utility"])
                )
                if better_or_equal and strictly:
                    dominated = True
                    break
            if not dominated:
                frontier.append(row)
        frontier.sort(key=lambda item: float(item["rank_score"]), reverse=True)
        return frontier[:12]

    def discover(
        self,
        paths: Sequence[Any] | Any,
        *,
        reports: Optional[Sequence[Mapping[str, Any]]] = None,
        top_k: int = 12,
    ) -> ObjectiveDiscoveryResult:
        path_list = list(paths) if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes, bytearray, pd.DataFrame, Mapping)) else [paths]
        frames = [self._frame_from_path(path) for path in path_list]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            candidates = {name: 0.0 for name in self.CANDIDATE_CATEGORIES}
        else:
            combined = pd.concat(frames, ignore_index=True, sort=False)
            candidates = self._extract_candidates(combined, reports=reports)
        scenario_count = max(1, len(frames))
        metrics = [self._score_metric(name, value, candidates, scenario_count) for name, value in candidates.items()]
        metrics.sort(key=lambda item: item.rank_score, reverse=True)

        top_metrics = metrics[: max(1, int(top_k))]
        weight_sum = sum(max(0.0, item.rank_score) for item in top_metrics) or 1.0
        weighted: List[DiscoveredMetric] = []
        for item in metrics:
            if item in top_metrics:
                item.composite_weight = float(max(0.0, item.rank_score) / weight_sum)
            else:
                item.composite_weight = 0.0
            weighted.append(item)
        composite_score = float(
            sum(
                item.composite_weight * _clip(item.value if item.direction == "maximize" else 1.0 - item.value, 0.0, 1.0)
                for item in top_metrics
            )
        )
        heatmap = [
            {
                "metric": item.name,
                "scenario": f"scenario_{scenario_idx + 1}",
                "stability": float(_clip(item.robustness - 0.03 * scenario_idx, 0.0, 1.0)),
            }
            for scenario_idx in range(scenario_count)
            for item in top_metrics[:8]
        ]
        shanghai = next((item.to_dict() for item in metrics if item.name == "shanghai_index_return"), {})
        return ObjectiveDiscoveryResult(
            ranked_metrics=weighted,
            pareto_frontier=self._pareto(weighted),
            composite_score=composite_score,
            weight_decomposition={item.name: float(item.composite_weight) for item in top_metrics},
            stability_heatmap=heatmap,
            candidate_pool=list(self.CANDIDATE_CATEGORIES.keys()),
            shanghai_index_metric=shanghai,
        )


def discover_objectives(paths: Sequence[Any] | Any, *, reports: Optional[Sequence[Mapping[str, Any]]] = None, top_k: int = 12) -> Dict[str, Any]:
    return ObjectiveDiscoveryEngine().discover(paths, reports=reports, top_k=top_k).to_dict()


__all__ = ["DiscoveredMetric", "ObjectiveDiscoveryEngine", "ObjectiveDiscoveryResult", "discover_objectives"]
