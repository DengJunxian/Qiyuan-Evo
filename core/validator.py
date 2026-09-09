"""
典型事实验证器 (Stylized Facts Validator)

用于验证仿真市场是否表现出真实市场的统计特征，
包括尖峰厚尾、波动率聚集、量价相关性等。
"""

import numpy as np
from scipy import stats
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import uuid


@dataclass
class ValidationResult:
    """单项验证结果"""
    name: str
    passed: bool
    actual_value: float
    threshold: float
    description: str


ANALYST_EVIDENCE_TYPES = {"news", "price", "macro", "social", "risk"}
ANALYST_TIME_HORIZONS = {"intraday", "swing", "weekly", "macro"}
ANALYST_ACTIONS = {"buy", "sell", "hold", "reduce_risk"}


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_action(action: Any) -> str:
    action_text = str(action or "hold").strip().lower()
    if action_text not in ANALYST_ACTIONS:
        return "hold"
    return action_text


def _normalize_horizon(horizon: Any) -> str:
    horizon_text = str(horizon or "swing").strip().lower()
    if horizon_text not in ANALYST_TIME_HORIZONS:
        return "swing"
    return horizon_text


def _normalize_evidence_type(value: Any) -> str:
    evidence_type = str(value or "risk").strip().lower()
    if evidence_type not in ANALYST_EVIDENCE_TYPES:
        return "risk"
    return evidence_type


def coerce_analyst_card(card: Dict[str, Any], analyst_id: str = "analyst") -> Dict[str, Any]:
    """
    Coerce analyst output into the required structured schema.

    Returns a normalized card with all mandatory fields.
    """
    card = dict(card or {})
    evidence_items: List[Dict[str, Any]] = []
    for raw in card.get("evidence", []) if isinstance(card.get("evidence"), list) else []:
        if not isinstance(raw, dict):
            continue
        evidence_items.append(
            {
                "type": _normalize_evidence_type(raw.get("type")),
                "content": str(raw.get("content", "")).strip()[:400],
                "weight": _clip(_safe_float(raw.get("weight", 0.5), 0.5), 0.0, 1.0),
            }
        )
    if not evidence_items:
        evidence_items = [
            {
                "type": "risk",
                "content": "fallback_evidence",
                "weight": 0.5,
            }
        ]

    risk_tags = card.get("risk_tags", [])
    if not isinstance(risk_tags, list):
        risk_tags = []
    normalized_tags = [str(tag).strip().lower() for tag in risk_tags if str(tag).strip()]

    counterarguments = card.get("counterarguments", [])
    if not isinstance(counterarguments, list):
        counterarguments = []
    normalized_counterarguments = [str(item).strip()[:240] for item in counterarguments if str(item).strip()]

    return {
        "analyst_id": str(card.get("analyst_id", analyst_id)),
        "thesis": str(card.get("thesis", "neutral thesis")).strip()[:400],
        "evidence": evidence_items,
        "time_horizon": _normalize_horizon(card.get("time_horizon")),
        "risk_tags": normalized_tags,
        "confidence": _clip(_safe_float(card.get("confidence", 0.5), 0.5), 0.0, 1.0),
        "counterarguments": normalized_counterarguments,
        "recommended_action": _normalize_action(card.get("recommended_action")),
    }


def validate_analyst_card(card: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate analyst card schema and value ranges.

    Returns: (is_valid, list_of_errors)
    """
    errors: List[str] = []
    required_fields = (
        "thesis",
        "evidence",
        "time_horizon",
        "risk_tags",
        "confidence",
        "counterarguments",
        "recommended_action",
    )
    for field_name in required_fields:
        if field_name not in card:
            errors.append(f"missing_field:{field_name}")

    if not isinstance(card.get("evidence"), list) or len(card.get("evidence", [])) == 0:
        errors.append("invalid_evidence")
    else:
        for idx, item in enumerate(card["evidence"]):
            if not isinstance(item, dict):
                errors.append(f"invalid_evidence_item:{idx}")
                continue
            if _normalize_evidence_type(item.get("type")) != item.get("type"):
                errors.append(f"invalid_evidence_type:{idx}")
            weight = _safe_float(item.get("weight", -1), -1)
            if weight < 0.0 or weight > 1.0:
                errors.append(f"invalid_evidence_weight:{idx}")

    if _normalize_horizon(card.get("time_horizon")) != card.get("time_horizon"):
        errors.append("invalid_time_horizon")
    if _normalize_action(card.get("recommended_action")) != card.get("recommended_action"):
        errors.append("invalid_recommended_action")

    confidence = _safe_float(card.get("confidence", -1), -1)
    if confidence < 0.0 or confidence > 1.0:
        errors.append("invalid_confidence")

    if not isinstance(card.get("risk_tags", []), list):
        errors.append("invalid_risk_tags")
    if not isinstance(card.get("counterarguments", []), list):
        errors.append("invalid_counterarguments")

    return len(errors) == 0, errors


def build_contradiction_matrix(cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build pairwise contradiction matrix between analyst cards."""
    normalized = [coerce_analyst_card(card, analyst_id=f"analyst_{idx}") for idx, card in enumerate(cards)]
    n = len(normalized)
    matrix: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    action_to_score = {"buy": 1.0, "hold": 0.0, "reduce_risk": -0.5, "sell": -1.0}

    for i in range(n):
        for j in range(i + 1, n):
            action_i = action_to_score.get(normalized[i]["recommended_action"], 0.0)
            action_j = action_to_score.get(normalized[j]["recommended_action"], 0.0)
            action_conflict = min(1.0, abs(action_i - action_j))

            tags_i = set(normalized[i].get("risk_tags", []))
            tags_j = set(normalized[j].get("risk_tags", []))
            if not tags_i and not tags_j:
                risk_divergence = 0.0
            else:
                risk_divergence = 1.0 - (len(tags_i & tags_j) / max(1, len(tags_i | tags_j)))

            horizon_conflict = 0.0 if normalized[i]["time_horizon"] == normalized[j]["time_horizon"] else 1.0
            score = _clip(0.6 * action_conflict + 0.25 * risk_divergence + 0.15 * horizon_conflict, 0.0, 1.0)
            matrix[i][j] = score
            matrix[j][i] = score

    upper = [matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    contradiction_index = float(sum(upper) / len(upper)) if upper else 0.0
    return {
        "analysts": [card.get("analyst_id", f"analyst_{idx}") for idx, card in enumerate(normalized)],
        "matrix": matrix,
        "contradiction_index": contradiction_index,
    }


def aggregate_analyst_cards(
    cards: List[Dict[str, Any]],
    risk_alert: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Manager aggregation over structured analyst cards.

    Includes weighted aggregation, contradiction discount and confidence calibration.
    """
    normalized = [coerce_analyst_card(card, analyst_id=f"analyst_{idx}") for idx, card in enumerate(cards)]
    if not normalized:
        normalized = [coerce_analyst_card({}, analyst_id="fallback")]

    action_to_score = {"buy": 1.0, "hold": 0.0, "reduce_risk": -0.5, "sell": -1.0}
    votes: List[Dict[str, Any]] = []
    weighted_signal = 0.0
    total_weight = 0.0

    for card in normalized:
        evidence_weights = [float(item.get("weight", 0.5)) for item in card["evidence"]]
        evidence_strength = float(sum(evidence_weights) / len(evidence_weights))
        weight = float(card["confidence"]) * evidence_strength
        action_score = action_to_score.get(card["recommended_action"], 0.0)
        votes.append(
            {
                "analyst_id": card["analyst_id"],
                "recommended_action": card["recommended_action"],
                "weight": weight,
                "confidence": card["confidence"],
                "evidence_strength": evidence_strength,
            }
        )
        weighted_signal += action_score * weight
        total_weight += weight

    if total_weight <= 1e-9:
        normalized_signal = 0.0
    else:
        normalized_signal = weighted_signal / total_weight

    contradiction = build_contradiction_matrix(normalized)
    contradiction_index = float(contradiction["contradiction_index"])

    raw_confidence = float(sum(card["confidence"] for card in normalized) / len(normalized))
    calibrated_confidence = _clip(raw_confidence * (1.0 - 0.55 * contradiction_index), 0.0, 1.0)

    risk_level = str((risk_alert or {}).get("level", "normal")).lower()
    if risk_level in {"high", "critical"}:
        calibrated_confidence = _clip(calibrated_confidence * 0.85, 0.0, 1.0)

    if risk_level in {"high", "critical"}:
        final_action = "reduce_risk"
    elif normalized_signal >= 0.15:
        final_action = "buy"
    elif normalized_signal <= -0.25:
        final_action = "sell"
    elif normalized_signal <= -0.10:
        final_action = "reduce_risk"
    else:
        final_action = "hold"

    outcome_proxy = 1.0 if abs(normalized_signal) >= 0.25 else 0.0
    brier_like = float((calibrated_confidence - outcome_proxy) ** 2)
    confidence_drift = float(calibrated_confidence - raw_confidence)

    return {
        "recommended_action": final_action,
        "aggregated_signal": float(normalized_signal),
        "weighted_votes": votes,
        "contradiction_matrix": contradiction,
        "raw_confidence": raw_confidence,
        "calibrated_confidence": calibrated_confidence,
        "calibration": {
            "brier_like_score": brier_like,
            "confidence_drift": confidence_drift,
            "outcome_proxy": outcome_proxy,
        },
    }


@dataclass
class RiskAlert:
    """Structured risk alert emitted by the risk committee."""

    level: str
    alerts: List[str]
    metrics: Dict[str, float]
    recommended_action: str
    risk_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "alerts": list(self.alerts),
            "metrics": dict(self.metrics),
            "recommended_action": self.recommended_action,
            "risk_score": self.risk_score,
        }


class RiskCommittee:
    """Risk committee that evaluates CVaR, drawdown, turnover, crowding and vol spike."""

    def assess(self, metrics: Dict[str, Any]) -> RiskAlert:
        normalized_metrics = {
            "cvar": _safe_float(metrics.get("cvar", 0.0)),
            "max_drawdown": _safe_float(metrics.get("max_drawdown", 0.0)),
            "turnover": _safe_float(metrics.get("turnover", 0.0)),
            "crowding": _safe_float(metrics.get("crowding", 0.0)),
            "volatility_spike": _safe_float(metrics.get("volatility_spike", 1.0)),
        }
        alerts: List[str] = []
        score = 0.0

        if normalized_metrics["cvar"] <= -0.08:
            alerts.append("cvar_breach")
            score += 0.30
        if normalized_metrics["max_drawdown"] <= -0.15:
            alerts.append("drawdown_breach")
            score += 0.25
        if normalized_metrics["turnover"] >= 1.5:
            alerts.append("turnover_spike")
            score += 0.15
        if normalized_metrics["crowding"] >= 0.70:
            alerts.append("overcrowding")
            score += 0.15
        if normalized_metrics["volatility_spike"] >= 2.0:
            alerts.append("volatility_spike")
            score += 0.20

        score = _clip(score, 0.0, 1.0)
        if score >= 0.75:
            level = "critical"
            action = "reduce_risk"
        elif score >= 0.45:
            level = "high"
            action = "reduce_risk"
        elif score >= 0.20:
            level = "medium"
            action = "hold"
        else:
            level = "normal"
            action = "hold"

        return RiskAlert(
            level=level,
            alerts=alerts,
            metrics=normalized_metrics,
            recommended_action=action,
            risk_score=score,
        )


class PolicyCommittee:
    """Translate policy events into explicit regulatory condition changes."""

    def translate(self, policy_event: str, baseline: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        conditions = dict(
            baseline
            or {
                "tax_rate": 0.001,
                "circuit_breaker_pct": 0.10,
                "leverage_limit": 2.0,
                "market_making_obligation": 0.5,
            }
        )
        event_text = str(policy_event or "")
        event_lower = event_text.lower()
        applied_rules: List[str] = []

        def _contains_any(tokens: List[str]) -> bool:
            return any((token in event_text) or (token in event_lower) for token in tokens)

        tax_tokens = ["???", "stamp tax", "???", "?????"]
        tax_down_tokens = ["??", "??", "??", "cut", "reduce", "lower", "???", "???", "???"]
        tax_up_tokens = ["??", "??", "raise", "increase", "hike", "???", "???"]
        breaker_tokens = ["??", "????", "?????", "circuit breaker", "trading halt", "??????", "???"]
        leverage_tight_tokens = [
            "????",
            "?????",
            "???",
            "tighten leverage",
            "margin increase",
            "deleveraging",
            "??????",
            "????????",
            "?????",
        ]
        leverage_loose_tokens = ["????", "?????", "leverage easing", "margin cut", "??????", "????????"]
        making_tokens = ["????", "?????", "????", "market making", "liquidity injection", "??????", "????????", "??????"]

        if _contains_any(tax_tokens) and _contains_any(tax_down_tokens):
            conditions["tax_rate"] = _clip(conditions["tax_rate"] - 0.0005, 0.0, 0.01)
            applied_rules.append("tax_rate_down")
        if _contains_any(tax_tokens) and _contains_any(tax_up_tokens):
            conditions["tax_rate"] = _clip(conditions["tax_rate"] + 0.0005, 0.0, 0.01)
            applied_rules.append("tax_rate_up")
        if _contains_any(breaker_tokens):
            conditions["circuit_breaker_pct"] = _clip(conditions["circuit_breaker_pct"] - 0.02, 0.05, 0.20)
            applied_rules.append("circuit_breaker_tighter")
        if _contains_any(leverage_tight_tokens):
            conditions["leverage_limit"] = _clip(conditions["leverage_limit"] - 0.3, 1.0, 5.0)
            applied_rules.append("leverage_down")
        if _contains_any(leverage_loose_tokens):
            conditions["leverage_limit"] = _clip(conditions["leverage_limit"] + 0.3, 1.0, 5.0)
            applied_rules.append("leverage_up")
        if _contains_any(making_tokens):
            conditions["market_making_obligation"] = _clip(
                conditions["market_making_obligation"] + 0.15,
                0.0,
                1.0,
            )
            applied_rules.append("market_making_up")

        return {
            "policy_event": event_text,
            "regulatory_conditions": conditions,
            "applied_rules": applied_rules,
        }
def export_decision_trace(trace: Dict[str, Any], trace_dir: str = "artifacts/decision_trace") -> str:
    """Persist one decision trace JSON and return the file path."""
    path = Path(trace_dir)
    path.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}.json"
    file_path = path / filename
    file_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return str(file_path)


class StylizedFactsValidator:
    """
    典型事实验证器
    
    验证仿真数据是否符合真实金融市场的统计特征：
    1. 尖峰厚尾 (Fat Tails) - 收益率分布峰度 > 3
    2. 波动率聚集 (Volatility Clustering) - 收益率平方的自相关性
    3. 量价相关性 - 价格波动与成交量的相关系数
    4. 收益率均值接近零 - 日收益率均值接近0
    5. 负偏度 - 市场下跌时波动更大
    """
    
    def __init__(self):
        self.results: List[ValidationResult] = []
    
    def validate_fat_tails(self, returns: np.ndarray, threshold: float = 3.0) -> ValidationResult:
        """
        验证尖峰厚尾特征
        
        真实市场的收益率分布具有"尖峰厚尾"特征，即峰度(Kurtosis)显著大于正态分布的3。
        这意味着极端事件（大涨大跌）发生的频率比正态分布预期的要高。
        
        Args:
            returns: 收益率序列
            threshold: 峰度阈值，默认3（正态分布的峰度）
        """
        if len(returns) < 30:
            return ValidationResult(
                name="尖峰厚尾",
                passed=False,
                actual_value=0,
                threshold=threshold,
                description="样本量不足（需要至少30个数据点）"
            )
        
        kurtosis = stats.kurtosis(returns, fisher=False)  # 关闭 Fisher 修正，返回超额峰度+3
        passed = kurtosis > threshold
        
        return ValidationResult(
            name="尖峰厚尾",
            passed=passed,
            actual_value=round(kurtosis, 3),
            threshold=threshold,
            description=f"峰度={kurtosis:.3f}，{'符合' if passed else '不符合'}真实市场特征（阈值>{threshold}）"
        )
    
    def validate_volatility_clustering(self, returns: np.ndarray, lag: int = 1, threshold: float = 0.1) -> ValidationResult:
        """
        验证波动率聚集特征
        
        真实市场中，大波动后往往跟着大波动，小波动后往往跟着小波动。
        通过检验收益率平方的自相关性来验证。
        
        Args:
            returns: 收益率序列
            lag: 滞后期数
            threshold: 自相关系数阈值
        """
        if len(returns) < 50:
            return ValidationResult(
                name="波动率聚集",
                passed=False,
                actual_value=0,
                threshold=threshold,
                description="样本量不足（需要至少50个数据点）"
            )
        
        squared_returns = returns ** 2
        # 计算滞后自相关
        autocorr = np.corrcoef(squared_returns[:-lag], squared_returns[lag:])[0, 1]
        passed = autocorr > threshold
        
        return ValidationResult(
            name="波动率聚集",
            passed=passed,
            actual_value=round(autocorr, 3),
            threshold=threshold,
            description=f"收益率平方自相关={autocorr:.3f}，{'符合' if passed else '不符合'}聚集特征（阈值>{threshold}）"
        )
    
    def validate_volume_price_correlation(
        self, 
        price_changes: np.ndarray, 
        volumes: np.ndarray, 
        threshold: float = 0.2
    ) -> ValidationResult:
        """
        验证量价相关性
        
        真实市场中，价格剧烈波动时往往伴随着成交量放大。
        
        Args:
            price_changes: 价格变化率的绝对值
            volumes: 成交量序列
            threshold: 相关系数阈值
        """
        if len(price_changes) != len(volumes) or len(price_changes) < 30:
            return ValidationResult(
                name="量价相关性",
                passed=False,
                actual_value=0,
                threshold=threshold,
                description="数据不足或长度不匹配"
            )
        
        # 使用价格变化绝对值与成交量的相关性
        abs_changes = np.abs(price_changes)
        correlation = np.corrcoef(abs_changes, volumes)[0, 1]
        
        # 处理 NaN（成交量全为0时可能出现）
        if np.isnan(correlation):
            correlation = 0.0
            
        passed = correlation > threshold
        
        return ValidationResult(
            name="量价相关性",
            passed=passed,
            actual_value=round(correlation, 3),
            threshold=threshold,
            description=f"|价格变化|与成交量相关系数={correlation:.3f}，{'符合' if passed else '不符合'}量价关系（阈值>{threshold}）"
        )
    
    def validate_negative_skewness(self, returns: np.ndarray, threshold: float = 0.0) -> ValidationResult:
        """
        验证负偏度特征
        
        真实市场通常呈现负偏度，即下跌时的波动往往比上涨时更剧烈。
        
        Args:
            returns: 收益率序列
            threshold: 偏度阈值（负偏度应小于此值）
        """
        if len(returns) < 30:
            return ValidationResult(
                name="负偏度",
                passed=False,
                actual_value=0,
                threshold=threshold,
                description="样本量不足"
            )
        
        skewness = stats.skew(returns)
        passed = skewness < threshold
        
        return ValidationResult(
            name="负偏度",
            passed=passed,
            actual_value=round(skewness, 3),
            threshold=threshold,
            description=f"偏度={skewness:.3f}，{'符合' if passed else '不符合'}负偏度特征（阈值<{threshold}）"
        )
    
    def validate_mean_reversion(self, returns: np.ndarray, threshold: float = 0.001) -> ValidationResult:
        """
        验证收益率均值接近零
        
        有效市场假说下，日收益率均值应接近零（扣除无风险利率后）。
        
        Args:
            returns: 收益率序列
            threshold: 均值绝对值阈值
        """
        if len(returns) < 30:
            return ValidationResult(
                name="均值回归",
                passed=False,
                actual_value=0,
                threshold=threshold,
                description="样本量不足"
            )
        
        mean_return = np.mean(returns)
        passed = abs(mean_return) < threshold
        
        return ValidationResult(
            name="均值回归",
            passed=passed,
            actual_value=round(mean_return, 5),
            threshold=threshold,
            description=f"日均收益率={mean_return:.5f}，{'符合' if passed else '不符合'}均值回归（阈值<{threshold}）"
        )
    
    def run_full_validation(
        self, 
        prices: List[float], 
        volumes: Optional[List[int]] = None
    ) -> Dict[str, any]:
        """
        执行完整验证
        
        Args:
            prices: 价格序列（收盘价）
            volumes: 成交量序列（可选）
            
        Returns:
            包含所有验证结果的字典
        """
        self.results = []
        
        # 计算收益率
        prices_arr = np.array(prices)
        returns = np.diff(prices_arr) / prices_arr[:-1]
        
        # 1. 尖峰厚尾
        self.results.append(self.validate_fat_tails(returns))
        
        # 2. 波动率聚集
        self.results.append(self.validate_volatility_clustering(returns))
        
        # 3. 负偏度
        self.results.append(self.validate_negative_skewness(returns))
        
        # 4. 均值回归
        self.results.append(self.validate_mean_reversion(returns))
        
        # 5. 量价相关性（如果有成交量数据）
        if volumes is not None and len(volumes) == len(prices):
            volumes_arr = np.array(volumes[1:])  # 与收益率对齐
            self.results.append(self.validate_volume_price_correlation(returns, volumes_arr))
        
        # 统计结果
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        
        return {
            "passed": passed_count,
            "total": total_count,
            "pass_rate": passed_count / total_count if total_count > 0 else 0,
            "results": self.results,
            "summary": self.generate_summary()
        }
    
    def generate_summary(self) -> str:
        """生成验证报告摘要"""
        lines = ["=" * 50, "📊 典型事实验证报告", "=" * 50, ""]
        
        for result in self.results:
            status = "✅" if result.passed else "❌"
            lines.append(f"{status} {result.name}")
            lines.append(f"   {result.description}")
            lines.append("")
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        lines.append("-" * 50)
        lines.append(f"通过率: {passed}/{total} ({passed/total*100:.1f}%)")
        
        if passed == total:
            lines.append("🎉 仿真市场表现出所有典型事实特征！")
        elif passed >= total * 0.6:
            lines.append("⚠️ 仿真市场基本符合真实市场特征。")
        else:
            lines.append("❗ 仿真市场与真实市场存在较大差异，需调整参数。")
        
        return "\n".join(lines)


# 使用示例
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    
    # 模拟具有厚尾特征的收益率（t分布）
    n_days = 200
    returns = np.random.standard_t(df=3, size=n_days) * 0.02
    prices = [3000.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    
    volumes = [int(1e8 * (1 + abs(r) * 10)) for r in returns]  # 量价正相关
    volumes.insert(0, int(1e8))
    
    # 验证
    validator = StylizedFactsValidator()
    result = validator.run_full_validation(prices, volumes)
    
    print(result["summary"])
