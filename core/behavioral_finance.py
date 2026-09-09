"""
行为金融量化库

将人性代码化 - 实现：
1. 前景理论（Prospect Theory）效用计算
2. 羊群效应（Herding Effect）增强检测
3. 心理账户（Mental Accounting）
4. 过度自信（Overconfidence）偏差
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np


# ========== 前景理论 ==========

@dataclass
class ProspectTheoryParams:
    """前景理论参数"""
    alpha: float = 0.88       # 收益区域曲率
    beta: float = 0.88        # 损失区域曲率
    lambda_: float = 2.25     # 损失厌恶系数 (Kahneman-Tversky 经典值)
    gamma: float = 0.61       # 权重函数参数（正面）
    delta: float = 0.69       # 权重函数参数（负面）


def prospect_value(
    outcome: float, 
    reference: float = 0,
    params: Optional[ProspectTheoryParams] = None
) -> float:
    """
    前景理论价值函数 V(x)
    
    Kahneman-Tversky (1992) 累积前景理论价值函数
    
    Args:
        outcome: 实际结果（如盈利金额或收益率）
        reference: 参考点（通常为0或成本价）
        params: 前景理论参数
        
    Returns:
        心理效用值
    """
    if params is None:
        params = ProspectTheoryParams()
    
    x = outcome - reference
    
    if x >= 0:
        # 收益区域：V = x^α
        return np.power(x, params.alpha)
    else:
        # 损失区域：V = -λ *^β
        return -params.lambda_ * np.power(-x, params.beta)


def prospect_utility(
    gain_pct: float,  # 盈亏百分比，如 0.05 = 5%
    reference: float = 0,
    loss_aversion: float = 2.25
) -> float:
    """
    简化版前景理论效用计算
    
    Args:
        gain_pct: 盈亏百分比
        reference: 参考点
        loss_aversion: 损失厌恶系数
        
    Returns:
        心理效用值（-100 ~ +100 范围）
    """
    x = gain_pct - reference
    
    if x >= 0:
        # 收益带来的快乐
        utility = 100 * np.power(x, 0.88)
    else:
        # 损失带来的痛苦（放大 λ 倍）
        utility = -100 * loss_aversion * np.power(-x, 0.88)
    
    return np.clip(utility, -300, 100)


def probability_weight(p: float, is_gain: bool = True) -> float:
    """
    前景理论概率权重函数 w(p)
    
    人们倾向于高估小概率事件，低估大概率事件
    
    Args:
        p: 客观概率 (0-1)
        is_gain: 是否为收益情景
        
    Returns:
        主观决策权重
    """
    if is_gain:
        gamma = 0.61
        return np.power(p, gamma) / np.power(
            np.power(p, gamma) + np.power(1-p, gamma), 
            1/gamma
        )
    else:
        delta = 0.69
        return np.power(p, delta) / np.power(
            np.power(p, delta) + np.power(1-p, delta), 
            1/delta
        )


def disposition_effect_score(
    current_pnl: float,
    action: str,
    holding_days: int = 1
) -> float:
    """
    处置效应评分
    
    处置效应：投资者倾向于过早卖出盈利股票，过久持有亏损股票
    
    Args:
        current_pnl: 当前盈亏百分比
        action: 采取的动作 (BUY/SELL/HOLD)
        holding_days: 持有天数
        
    Returns:
        处置效应得分 (0-1，越高越受处置效应影响)
    """
    if current_pnl > 0 and action == 'SELL':
        # 盈利就卖 - 典型处置效应
        return min(1.0, 0.5 + current_pnl * 2)
    elif current_pnl < 0 and action == 'HOLD':
        # 亏损死扛 - 典型处置效应
        return min(1.0, 0.5 + abs(current_pnl) * 2 + holding_days * 0.01)
    elif current_pnl < 0 and action == 'BUY':
        # 亏损加仓 - 极端处置效应
        return min(1.0, 0.7 + abs(current_pnl) * 3)
    else:
        return 0.0


# ========== 羊群效应 ==========

def calculate_csad(returns: np.ndarray, market_return: float) -> float:
    """
    横截面绝对偏差 (Cross-Sectional Absolute Deviation)
    
    CSAD 衡量个股收益率与市场收益率的分散程度
    CSAD 下降 + 市场大涨/大跌 = 羊群效应
    
    Args:
        returns: 个股收益率数组
        market_return: 市场收益率
        
    Returns:
        CSAD 值
    """
    if len(returns) == 0:
        return 0.0
    
    return np.mean(np.abs(returns - market_return))


def calculate_lsv_herding(
    buy_counts: np.ndarray,
    sell_counts: np.ndarray
) -> Tuple[float, str]:
    """
    Lakonishok-Shleifer-Vishny (LSV) 羊群指标
    
    衡量机构投资者在同一方向交易的程度
    
    Args:
        buy_counts: 各股票的买入人数
        sell_counts: 各股票的卖出人数
        
    Returns:
        (LSV 羊群指标, 羊群方向 'buy'/'sell'/'none')
    """
    total = buy_counts + sell_counts
    valid_mask = total > 0
    
    if not np.any(valid_mask):
        return 0.0, 'none'
    
    buy_ratio = buy_counts[valid_mask] / total[valid_mask]


    
    # 简化：直接返回买入比例的偏差
    herding_score = np.mean(np.abs(buy_ratio - 0.5))
    
    direction = 'buy' if np.mean(buy_ratio) > 0.5 else 'sell'
    
    return herding_score, direction


def csad_regression(
    returns_history: np.ndarray,  # Shape: T期，N只股票
    market_returns: np.ndarray,
    window: int = 20
) -> Dict[str, float]:
    """
    CSAD 非线性回归检测羊群效应
    
    模型: CSAD_t = α + γ1 * |R_m,t| + γ2 * R_m,t^2 + ε
    
    如果 γ2 显著为负，说明存在羊群效应
    
    Args:
        returns_history: 历史收益率矩阵
        market_returns: 市场收益率序列
        window: 滚动窗口大小
        
    Returns:
        回归系数字典
    """
    T = len(market_returns)
    
    if T < window:
        return {'gamma1': 0.0, 'gamma2': 0.0, 'herding_detected': False}
    
    # 计算每期的 CSAD
    csad_series = np.array([
        calculate_csad(returns_history[t], market_returns[t])
        for t in range(T)
    ])
    
    # 取最近 window 期数据
    csad_recent = csad_series[-window:]
    rm_recent = market_returns[-window:]
    rm_abs = np.abs(rm_recent)
    rm_sq = rm_recent ** 2
    
    # 简化 OLS 回归
    X = np.column_stack([np.ones(window), rm_abs, rm_sq])
    y = csad_recent
    
    try:

        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        
        gamma1 = beta[1]
        gamma2 = beta[2]
        
        # γ2 < -0.01 视为存在羊群效应
        herding_detected = gamma2 < -0.01
        
        return {
            'gamma1': float(gamma1),
            'gamma2': float(gamma2),
            'herding_detected': bool(herding_detected),
            'strength': float(abs(gamma2) if herding_detected else 0.0)
        }
        
    except np.linalg.LinAlgError:
        return {'gamma1': 0.0, 'gamma2': 0.0, 'herding_detected': False}


def herding_intensity(
    csad: float,
    market_return: float,
    baseline_csad: float = 0.02
) -> float:
    """
    羊群效应强度指标
    
    当市场大涨/大跌时，CSAD 应该上升
    如果 CSAD 反而下降，说明存在羊群效应
    
    Args:
        csad: 当前 CSAD 值
        market_return: 市场收益率
        baseline_csad: 基准 CSAD (平静期)
        
    Returns:
        羊群强度 (0-1)
    """
    # 预期 CSAD（市场波动越大，预期 CSAD 越高）
    expected_csad = baseline_csad * (1 + 3 * abs(market_return))
    
    if expected_csad == 0:
        return 0.0
    
    # 实际 CSAD 低于预期 → 羊群效应
    deviation_ratio = (expected_csad - csad) / expected_csad
    
    return np.clip(deviation_ratio, 0, 1)


# ========== 过度自信 ==========

def overconfidence_score(
    predicted_returns: List[float],
    actual_returns: List[float]
) -> float:
    """
    过度自信评分
    
    基于预测精度衡量过度自信程度
    
    Args:
        predicted_returns: 预测收益率列表
        actual_returns: 实际收益率列表
        
    Returns:
        过度自信得分 (0-1)
    """
    if len(predicted_returns) < 2:
        return 0.5  # 数据不足，返回中性
    
    predicted = np.array(predicted_returns)
    actual = np.array(actual_returns)
    
    # 预测误差
    errors = np.abs(predicted - actual)
    mean_error = np.mean(errors)
    
    # 预测的绝对值（代表信心程度）
    confidence = np.mean(np.abs(predicted))
    
    if confidence < 0.001:
        return 0.5
    
    # 过度自信 = 高信心 + 高误差
    overconfidence = confidence * mean_error
    
    return np.clip(overconfidence * 10, 0, 1)


def predict_next_return(
    prices: List[float],
    short_window: int = 5,
    long_window: int = 20
) -> Optional[float]:
    """
    行为金融风洞预测：基于历史价格序列预测次日收益。
    结合动量与均值回归的简化模型，用于“内部模拟风洞”拦截。
    """
    if prices is None or len(prices) < max(short_window, long_window) + 1:
        return None
    series = np.array([float(x) for x in prices if x is not None])
    if len(series) < max(short_window, long_window) + 1:
        return None

    short_ma = np.mean(series[-short_window:])
    long_ma = np.mean(series[-long_window:])
    momentum = (short_ma - long_ma) / long_ma if long_ma != 0 else 0.0
    reversion = (series[-1] - long_ma) / long_ma if long_ma != 0 else 0.0
    predicted = 0.6 * momentum - 0.4 * reversion
    return float(np.clip(predicted, -0.05, 0.05))


# ========== 心理账户 ==========

@dataclass
class MentalAccount:
    """心理账户"""
    name: str
    balance: float
    pnl: float
    importance: float  # 0-1 重要性权重


def mental_accounting_bias(
    accounts: List[MentalAccount]
) -> float:
    """
    心理账户偏差评分
    
    投资者倾向于将资金分割到不同"心理账户"
    并对不同账户采取不同的风险态度
    
    Args:
        accounts: 心理账户列表
        
    Returns:
        心理账户偏差得分
    """
    if len(accounts) < 2:
        return 0.0
    
    # 计算各账户盈亏的方差
    pnl_values = [a.pnl for a in accounts]
    pnl_variance = np.var(pnl_values)
    
    # 方差越大 → 心理账户偏差越明显
    return np.clip(pnl_variance * 100, 0, 1)


# ========== 综合行为评估 ==========

@dataclass
class BehavioralProfile:
    """行为画像"""
    loss_aversion: float      # 损失厌恶程度
    herding_tendency: float   # 羊群倾向
    overconfidence: float     # 过度自信
    disposition_effect: float  # 处置效应
    mental_utility: float     # 心理效用
    
    def get_bias_summary(self) -> str:
        """获取偏差摘要"""
        biases = []
        
        if self.loss_aversion > 2.5:
            biases.append("🔴 高度损失厌恶")
        if self.herding_tendency > 0.6:
            biases.append("🐑 易受羊群效应")
        if self.overconfidence > 0.7:
            biases.append("😎 过度自信")
        if self.disposition_effect > 0.5:
            biases.append("💰 处置效应明显")
        
        return " | ".join(biases) if biases else "✅ 行为相对理性"


@dataclass
class InstitutionConstraintProfile:
    """Institution-level behavioral constraints used by layered memory."""

    institution_type: str
    mandate: str
    benchmark: str
    holding_period: int
    turnover_target: float
    max_drawdown: float
    liquidity_preference: float
    leverage_limit: float
    policy_channel_sensitivity: float
    rumor_sensitivity: float
    benchmark_tracking_pressure: float
    redemption_pressure: float
    inventory_limit: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_institution_constraint_profile(institution_type: str) -> InstitutionConstraintProfile:
    """Return a reproducible constraint profile for the requested institution class."""

    key = str(institution_type or "retail_swing").strip().lower().replace(" ", "_")
    profiles = {
        "retail_day_trader": dict(
            mandate="intraday tactical alpha",
            benchmark="intraday_index",
            holding_period=1,
            turnover_target=0.85,
            max_drawdown=0.18,
            liquidity_preference=0.35,
            leverage_limit=1.2,
            policy_channel_sensitivity=0.45,
            rumor_sensitivity=0.75,
            benchmark_tracking_pressure=0.15,
            redemption_pressure=0.0,
            inventory_limit=0.10,
        ),
        "retail_swing": dict(
            mandate="medium-horizon relative return",
            benchmark="broad_market_index",
            holding_period=5,
            turnover_target=0.30,
            max_drawdown=0.16,
            liquidity_preference=0.45,
            leverage_limit=1.0,
            policy_channel_sensitivity=0.40,
            rumor_sensitivity=0.55,
            benchmark_tracking_pressure=0.20,
            redemption_pressure=0.0,
            inventory_limit=0.15,
        ),
        "mutual_fund": dict(
            mandate="beat benchmark with controlled tracking error",
            benchmark="broad_market_index",
            holding_period=20,
            turnover_target=0.10,
            max_drawdown=0.12,
            liquidity_preference=0.70,
            leverage_limit=1.0,
            policy_channel_sensitivity=0.60,
            rumor_sensitivity=0.25,
            benchmark_tracking_pressure=0.85,
            redemption_pressure=0.70,
            inventory_limit=0.08,
        ),
        "pension_fund": dict(
            mandate="long-term capital preservation and liability matching",
            benchmark="policy_adjusted_long_horizon_index",
            holding_period=60,
            turnover_target=0.04,
            max_drawdown=0.08,
            liquidity_preference=0.85,
            leverage_limit=1.0,
            policy_channel_sensitivity=0.35,
            rumor_sensitivity=0.10,
            benchmark_tracking_pressure=0.90,
            redemption_pressure=0.15,
            inventory_limit=0.05,
        ),
        "insurer": dict(
            mandate="capital efficiency with solvency protection",
            benchmark="liability_matching_index",
            holding_period=45,
            turnover_target=0.05,
            max_drawdown=0.07,
            liquidity_preference=0.90,
            leverage_limit=1.0,
            policy_channel_sensitivity=0.30,
            rumor_sensitivity=0.10,
            benchmark_tracking_pressure=0.75,
            redemption_pressure=0.20,
            inventory_limit=0.05,
        ),
        "prop_desk": dict(
            mandate="short-horizon alpha with risk budget discipline",
            benchmark="risk_neutral_pnl",
            holding_period=3,
            turnover_target=0.60,
            max_drawdown=0.22,
            liquidity_preference=0.40,
            leverage_limit=2.5,
            policy_channel_sensitivity=0.45,
            rumor_sensitivity=0.35,
            benchmark_tracking_pressure=0.10,
            redemption_pressure=0.0,
            inventory_limit=0.30,
        ),
        "market_maker": dict(
            mandate="spread capture and inventory control",
            benchmark="spread_capture",
            holding_period=1,
            turnover_target=1.00,
            max_drawdown=0.06,
            liquidity_preference=1.00,
            leverage_limit=1.0,
            policy_channel_sensitivity=0.20,
            rumor_sensitivity=0.15,
            benchmark_tracking_pressure=0.05,
            redemption_pressure=0.0,
            inventory_limit=0.02,
        ),
        "state_stabilization_fund": dict(
            mandate="market stabilization over profit maximization",
            benchmark="stability_index",
            holding_period=30,
            turnover_target=0.12,
            max_drawdown=0.10,
            liquidity_preference=0.95,
            leverage_limit=1.0,
            policy_channel_sensitivity=0.95,
            rumor_sensitivity=0.55,
            benchmark_tracking_pressure=0.05,
            redemption_pressure=0.05,
            inventory_limit=0.20,
        ),
        "rumor_trader": dict(
            mandate="exploit narrative dislocations",
            benchmark="event_driven_pnl",
            holding_period=2,
            turnover_target=0.90,
            max_drawdown=0.30,
            liquidity_preference=0.25,
            leverage_limit=1.5,
            policy_channel_sensitivity=0.25,
            rumor_sensitivity=1.00,
            benchmark_tracking_pressure=0.02,
            redemption_pressure=0.0,
            inventory_limit=0.12,
        ),
        "etf_arbitrageur": dict(
            mandate="tight tracking between ETF and basket",
            benchmark="etf_basket_spread",
            holding_period=1,
            turnover_target=0.75,
            max_drawdown=0.05,
            liquidity_preference=0.98,
            leverage_limit=1.2,
            policy_channel_sensitivity=0.40,
            rumor_sensitivity=0.18,
            benchmark_tracking_pressure=1.00,
            redemption_pressure=0.85,
            inventory_limit=0.03,
        ),
    }
    default_profile = profiles["retail_swing"]
    payload = dict(profiles.get(key, default_profile))
    payload["institution_type"] = key if key in profiles else "retail_swing"
    return InstitutionConstraintProfile(**payload)


def create_behavioral_profile(
    pnl_pct: float,
    action: str,
    market_csad: float,
    market_return: float,
    predictions: List[float] = None,
    actuals: List[float] = None,
    loss_aversion: float = 2.25
) -> BehavioralProfile:
    """
    创建行为画像
    
    综合评估投资者的行为偏差
    """
    # 计算各项指标
    mental_utility = prospect_utility(pnl_pct, loss_aversion=loss_aversion)
    herding = herding_intensity(market_csad, market_return)
    disposition = disposition_effect_score(pnl_pct, action)
    
    if predictions and actuals:
        overconf = overconfidence_score(predictions, actuals)
    else:
        overconf = 0.5
    
    return BehavioralProfile(
        loss_aversion=loss_aversion,
        herding_tendency=herding,
        overconfidence=overconf,
        disposition_effect=disposition,
        mental_utility=mental_utility
    )




@dataclass
class CPTAgentProfile:
    """Standardized CPT profile for different agent archetypes."""

    name: str
    params: ProspectTheoryParams
    reference_return: float = 0.0


_CPT_PROFILE_MAP: Dict[str, ProspectTheoryParams] = {
    "retail": ProspectTheoryParams(alpha=0.86, beta=0.90, lambda_=2.8, gamma=0.60, delta=0.70),
    "institution": ProspectTheoryParams(alpha=0.92, beta=0.92, lambda_=1.6, gamma=0.72, delta=0.78),
    "quant": ProspectTheoryParams(alpha=0.95, beta=0.95, lambda_=1.2, gamma=0.80, delta=0.82),
}


def get_cpt_profile(agent_type: Literal["retail", "institution", "quant"] | str) -> CPTAgentProfile:
    key = str(agent_type).strip().lower()
    params = _CPT_PROFILE_MAP.get(key, ProspectTheoryParams())
    return CPTAgentProfile(name=key or "custom", params=params)


def cpt_decision_utility(
    outcome: float,
    probability: float,
    *,
    profile: Optional[CPTAgentProfile] = None,
    reference: Optional[float] = None,
) -> float:
    """CPT subjective utility for a single outcome-probability pair."""
    prof = profile or CPTAgentProfile(name="default", params=ProspectTheoryParams())
    ref = prof.reference_return if reference is None else float(reference)
    subjective_value = prospect_value(outcome=outcome, reference=ref, params=prof.params)
    weighted_prob = probability_weight(
        p=float(np.clip(probability, 1e-8, 1 - 1e-8)),
        is_gain=(outcome - ref) >= 0,
    )
    return float(subjective_value * weighted_prob)


def cpt_expected_utility(
    outcomes: List[float],
    probabilities: List[float],
    *,
    profile: Optional[CPTAgentProfile] = None,
    reference: Optional[float] = None,
) -> float:
    """Aggregate CPT utility over a distribution of outcomes."""
    if not outcomes or not probabilities:
        return 0.0
    n = min(len(outcomes), len(probabilities))
    if n == 0:
        return 0.0
    utils = [
        cpt_decision_utility(
            outcome=float(outcomes[i]),
            probability=float(probabilities[i]),
            profile=profile,
            reference=reference,
        )
        for i in range(n)
    ]
    return float(np.sum(utils))





def _safe_price(value: float, fallback: float = 1.0) -> float:
    price = float(value)
    if price <= 0:
        return float(max(fallback, 1e-6))
    return price


@dataclass
class ReferencePoints:
    """
    Agent-level reference points used by behavioral finance dynamics.

    All anchors are price-like levels (must be > 0).
    """

    purchase_anchor: float
    recent_high_anchor: float
    peer_anchor: float
    policy_anchor: float

    def normalized_returns(self, current_price: float) -> Dict[str, float]:
        """Return anchor-relative returns at current price."""
        price = _safe_price(current_price)
        return {
            "purchase_anchor": (price - _safe_price(self.purchase_anchor, price)) / _safe_price(self.purchase_anchor, price),
            "recent_high_anchor": (price - _safe_price(self.recent_high_anchor, price)) / _safe_price(self.recent_high_anchor, price),
            "peer_anchor": (price - _safe_price(self.peer_anchor, price)) / _safe_price(self.peer_anchor, price),
            "policy_anchor": (price - _safe_price(self.policy_anchor, price)) / _safe_price(self.policy_anchor, price),
        }


@dataclass
class ReferenceShiftConfig:
    """Smoothing parameters for reference-point shift."""

    purchase_decay: float = 0.03
    recent_high_decay: float = 0.12
    peer_decay: float = 0.18
    policy_decay: float = 0.20
    policy_sensitivity: float = 0.20


@dataclass
class BehavioralStepState:
    """Output state from one behavioral update step."""

    sentiment: float
    reference_points: ReferencePoints
    prospect_direction: float
    risk_appetite: float
    trading_intent: float
    loss_aversion_intensity: float
    weighted_reference_return: float


def initialize_reference_points(
    current_price: float,
    *,
    peer_anchor: Optional[float] = None,
    policy_anchor: Optional[float] = None,
) -> ReferencePoints:
    """Initialize four anchors with sensible defaults."""
    price = _safe_price(current_price)
    return ReferencePoints(
        purchase_anchor=price,
        recent_high_anchor=price,
        peer_anchor=_safe_price(peer_anchor if peer_anchor is not None else price, price),
        policy_anchor=_safe_price(policy_anchor if policy_anchor is not None else price, price),
    )


def update_reference_points(
    reference_points: ReferencePoints,
    *,
    current_price: float,
    peer_anchor: float,
    policy_anchor: float,
    policy_shock: float = 0.0,
    config: Optional[ReferenceShiftConfig] = None,
) -> ReferencePoints:
    """
    Update reference points every step.

    Order requirement mapping:
    sentiment -> reference point shift -> risk appetite -> trading intent
    """
    cfg = config or ReferenceShiftConfig()
    price = _safe_price(current_price)
    old = reference_points

    purchase_target = old.purchase_anchor * (1.0 - cfg.purchase_decay) + price * cfg.purchase_decay
    recent_high_target = max(old.recent_high_anchor, price)
    recent_high = old.recent_high_anchor * (1.0 - cfg.recent_high_decay) + recent_high_target * cfg.recent_high_decay

    peer = old.peer_anchor * (1.0 - cfg.peer_decay) + _safe_price(peer_anchor, price) * cfg.peer_decay
    policy_target = _safe_price(policy_anchor, price) * (1.0 + cfg.policy_sensitivity * float(policy_shock))
    policy = old.policy_anchor * (1.0 - cfg.policy_decay) + policy_target * cfg.policy_decay

    return ReferencePoints(
        purchase_anchor=_safe_price(purchase_target, price),
        recent_high_anchor=_safe_price(recent_high, price),
        peer_anchor=_safe_price(peer, price),
        policy_anchor=_safe_price(policy, price),
    )


def prospect_direction_preference(
    current_price: float,
    reference_points: ReferencePoints,
    *,
    loss_aversion: float = 2.25,
    reference_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Prospect-theory style direction preference from four anchors."""
    returns = reference_points.normalized_returns(current_price)
    weights = {
        "purchase_anchor": 0.34,
        "recent_high_anchor": 0.26,
        "peer_anchor": 0.20,
        "policy_anchor": 0.20,
    }
    if reference_weights:
        weights.update(reference_weights)

    weighted_return = 0.0
    weighted_utility = 0.0
    underwater = 0.0
    total_w = 0.0

    for key, value in returns.items():
        w = float(max(0.0, weights.get(key, 0.0)))
        total_w += w
        weighted_return += w * value
        utility = prospect_utility(value, reference=0.0, loss_aversion=loss_aversion)
        weighted_utility += w * utility
        if value < 0:
            underwater += w

    if total_w <= 0:
        total_w = 1.0
    weighted_return /= total_w
    weighted_utility /= total_w
    underwater_ratio = underwater / total_w


    direction = float(np.tanh(weighted_utility / 55.0))

    loss_intensity = float(np.clip(loss_aversion * (1.0 + 0.8 * underwater_ratio), 0.5, 6.0))

    return {
        "direction": direction,
        "weighted_return": float(weighted_return),
        "weighted_utility": float(weighted_utility),
        "underwater_ratio": float(underwater_ratio),
        "loss_aversion_intensity": loss_intensity,
    }


def update_risk_appetite(
    base_risk_appetite: float,
    sentiment: float,
    direction_preference: float,
    loss_aversion_intensity: float,
) -> float:
    """Update risk appetite in [0, 1] using sentiment + prospect signals."""
    base = float(np.clip(base_risk_appetite, 0.0, 1.0))
    senti = float(np.clip(sentiment, -1.0, 1.0))
    direction = float(np.clip(direction_preference, -1.0, 1.0))
    loss_penalty = (float(loss_aversion_intensity) - 1.0) / 5.0
    updated = base + 0.22 * senti + 0.30 * direction - 0.18 * max(0.0, loss_penalty)
    return float(np.clip(updated, 0.0, 1.0))


def build_trading_intent(
    risk_appetite: float,
    direction_preference: float,
    sentiment: float,
) -> float:
    """
    Trading intent score in [-1, 1].
    Positive means buy-side intent, negative means sell-side intent.
    """
    risk = float(np.clip(risk_appetite, 0.0, 1.0))
    direction = float(np.clip(direction_preference, -1.0, 1.0))
    senti = float(np.clip(sentiment, -1.0, 1.0))
    intent = 0.65 * direction + 0.25 * senti + 0.10 * (risk - 0.5) * 2.0
    return float(np.clip(intent, -1.0, 1.0))


def behavioral_update_step(
    *,
    sentiment: float,
    current_price: float,
    reference_points: ReferencePoints,
    base_risk_appetite: float,
    peer_anchor: float,
    policy_anchor: float,
    policy_shock: float,
    loss_aversion: float = 2.25,
    shift_config: Optional[ReferenceShiftConfig] = None,
    reference_weights: Optional[Dict[str, float]] = None,
) -> BehavioralStepState:
    """
    One-step behavioral pipeline:
    sentiment -> reference point shift -> risk appetite -> trading intent
    """
    shifted = update_reference_points(
        reference_points,
        current_price=current_price,
        peer_anchor=peer_anchor,
        policy_anchor=policy_anchor,
        policy_shock=policy_shock,
        config=shift_config,
    )
    prospect = prospect_direction_preference(
        current_price=current_price,
        reference_points=shifted,
        loss_aversion=loss_aversion,
        reference_weights=reference_weights,
    )
    risk = update_risk_appetite(
        base_risk_appetite=base_risk_appetite,
        sentiment=sentiment,
        direction_preference=prospect["direction"],
        loss_aversion_intensity=prospect["loss_aversion_intensity"],
    )
    intent = build_trading_intent(risk, prospect["direction"], sentiment)
    return BehavioralStepState(
        sentiment=float(np.clip(sentiment, -1.0, 1.0)),
        reference_points=shifted,
        prospect_direction=prospect["direction"],
        risk_appetite=risk,
        trading_intent=intent,
        loss_aversion_intensity=prospect["loss_aversion_intensity"],
        weighted_reference_return=prospect["weighted_return"],
    )





@dataclass
class DispositionCounter:
    """Counters for computing PGR / PLR."""

    realized_gains: int = 0
    realized_losses: int = 0
    paper_gains: int = 0
    paper_losses: int = 0

    def record_realized(self, pnl: float) -> None:
        if pnl >= 0:
            self.realized_gains += 1
        else:
            self.realized_losses += 1

    def record_paper(self, pnl: float) -> None:
        if pnl >= 0:
            self.paper_gains += 1
        else:
            self.paper_losses += 1


def calculate_pgr_plr(counter: DispositionCounter) -> Dict[str, float]:
    """Compute realization rates for gains and losses."""
    pgr_den = counter.realized_gains + counter.paper_gains
    plr_den = counter.realized_losses + counter.paper_losses
    pgr = counter.realized_gains / pgr_den if pgr_den > 0 else 0.0
    plr = counter.realized_losses / plr_den if plr_den > 0 else 0.0
    return {
        "pgr": float(pgr),
        "plr": float(plr),
        "disposition_gap": float(pgr - plr),
        "realized_gains": float(counter.realized_gains),
        "realized_losses": float(counter.realized_losses),
        "paper_gains": float(counter.paper_gains),
        "paper_losses": float(counter.paper_losses),
    }





def all_time_high_effect(returns: Sequence[float], ath_flags: Sequence[bool]) -> Dict[str, float]:
    """Measure return anomaly around all-time-high (ATH) states."""
    if not returns:
        return {
            "ath_hit_rate": 0.0,
            "mean_return_on_ath": 0.0,
            "mean_return_off_ath": 0.0,
            "ath_outperformance": 0.0,
        }
    n = min(len(returns), len(ath_flags))
    if n <= 0:
        return {
            "ath_hit_rate": 0.0,
            "mean_return_on_ath": 0.0,
            "mean_return_off_ath": 0.0,
            "ath_outperformance": 0.0,
        }

    arr = np.asarray([float(x) for x in returns[:n]], dtype=float)
    flags = np.asarray([bool(x) for x in ath_flags[:n]], dtype=bool)
    on = arr[flags]
    off = arr[~flags]
    mean_on = float(np.mean(on)) if on.size else 0.0
    mean_off = float(np.mean(off)) if off.size else 0.0
    return {
        "ath_hit_rate": float(np.mean(flags)),
        "mean_return_on_ath": mean_on,
        "mean_return_off_ath": mean_off,
        "ath_outperformance": float(mean_on - mean_off),
    }


def volatility_clustering_metrics(returns: Sequence[float]) -> Dict[str, float]:
    """Estimate volatility clustering via lag autocorrelation of |r| and r^2."""
    if len(returns) < 3:
        return {
            "abs_return_lag1_autocorr": 0.0,
            "squared_return_lag1_autocorr": 0.0,
            "volatility_level": 0.0,
        }

    arr = np.asarray([float(x) for x in returns], dtype=float)
    abs_r = np.abs(arr)
    sq_r = arr ** 2

    def _lag1_autocorr(series: np.ndarray) -> float:
        x0 = series[:-1]
        x1 = series[1:]
        if x0.size < 2 or float(np.std(x0)) < 1e-12 or float(np.std(x1)) < 1e-12:
            return 0.0
        return float(np.corrcoef(x0, x1)[0, 1])

    return {
        "abs_return_lag1_autocorr": _lag1_autocorr(abs_r),
        "squared_return_lag1_autocorr": _lag1_autocorr(sq_r),
        "volatility_level": float(np.std(arr)),
    }


def drawdown_distribution(prices: Sequence[float]) -> Dict[str, float]:
    """Build drawdown distribution summary."""
    if len(prices) < 2:
        return {
            "mean_drawdown": 0.0,
            "max_drawdown": 0.0,
            "median_drawdown": 0.0,
            "q90_drawdown": 0.0,
            "count": 0.0,
        }
    arr = np.asarray([_safe_price(x) for x in prices], dtype=float)
    running_max = np.maximum.accumulate(arr)
    drawdowns = arr / running_max - 1.0
    dd = drawdowns[drawdowns < 0]
    if dd.size == 0:
        return {
            "mean_drawdown": 0.0,
            "max_drawdown": 0.0,
            "median_drawdown": 0.0,
            "q90_drawdown": 0.0,
            "count": 0.0,
        }
    return {
        "mean_drawdown": float(np.mean(dd)),
        "max_drawdown": float(np.min(dd)),
        "median_drawdown": float(np.median(dd)),
        "q90_drawdown": float(np.quantile(dd, 0.9)),
        "count": float(dd.size),
    }


@dataclass
class StylizedFactsTracker:
    """
    Collect per-step market/behavior traces and export stylized facts.
    """

    prices: List[float] = field(default_factory=list)
    market_returns: List[float] = field(default_factory=list)
    csad_series: List[float] = field(default_factory=list)
    cross_sectional_returns: List[List[float]] = field(default_factory=list)
    ath_flags: List[bool] = field(default_factory=list)
    loss_aversion_intensity: List[float] = field(default_factory=list)
    disposition: DispositionCounter = field(default_factory=DispositionCounter)

    def record_step(
        self,
        *,
        price: float,
        market_return: float,
        csad: float,
        cross_returns: Sequence[float],
        is_all_time_high: bool,
        loss_aversion_intensity: float,
    ) -> None:
        self.prices.append(float(price))
        self.market_returns.append(float(market_return))
        self.csad_series.append(float(csad))
        self.cross_sectional_returns.append([float(x) for x in cross_returns])
        self.ath_flags.append(bool(is_all_time_high))
        self.loss_aversion_intensity.append(float(loss_aversion_intensity))

    def report(self) -> Dict[str, object]:
        csad_mean = float(np.mean(self.csad_series)) if self.csad_series else 0.0
        csad_std = float(np.std(self.csad_series)) if self.csad_series else 0.0
        matrix = np.zeros((0, 0), dtype=float)
        if self.cross_sectional_returns:
            valid_rows = [row for row in self.cross_sectional_returns if row]
            if valid_rows:
                width = min(len(row) for row in valid_rows)
                if width > 0:
                    matrix = np.asarray([row[:width] for row in valid_rows], dtype=float)
        aligned_market_returns = np.asarray(self.market_returns, dtype=float) if self.market_returns else np.zeros(0, dtype=float)
        if matrix.shape[0] > 0 and aligned_market_returns.shape[0] != matrix.shape[0]:
            aligned_market_returns = aligned_market_returns[-matrix.shape[0]:]
        herding = csad_regression(
            returns_history=matrix,
            market_returns=aligned_market_returns,
            window=min(20, int(aligned_market_returns.shape[0])) if aligned_market_returns.size else 20,
        )
        ath = all_time_high_effect(self.market_returns, self.ath_flags)
        vol_cluster = volatility_clustering_metrics(self.market_returns)
        drawdowns = drawdown_distribution(self.prices)
        pgr_plr = calculate_pgr_plr(self.disposition)
        loss_avg = float(np.mean(self.loss_aversion_intensity)) if self.loss_aversion_intensity else 0.0
        loss_p90 = float(np.quantile(self.loss_aversion_intensity, 0.9)) if self.loss_aversion_intensity else 0.0
        payload = {
            "csad": {
                "mean": csad_mean,
                "std": csad_std,
                "herding_regression": herding,
            },
            "pgr_plr": pgr_plr,
            "volatility_clustering": vol_cluster,
            "drawdown_distribution": drawdowns,
            "all_time_high_effect": ath,
            "loss_aversion_intensity": {
                "mean": loss_avg,
                "p90": loss_p90,
                "n_obs": len(self.loss_aversion_intensity),
            },
        }

        payload["volatility clustering"] = payload["volatility_clustering"]
        payload["drawdown distribution"] = payload["drawdown_distribution"]
        payload["all_time_high effect"] = payload["all_time_high_effect"]
        return payload

    def save_json(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.report()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _to_float_array(values: Optional[Sequence[float]]) -> np.ndarray:
    if values is None:
        return np.zeros(0, dtype=float)
    try:
        arr = np.asarray([float(x) for x in values if x is not None], dtype=float)
    except Exception:
        return np.zeros(0, dtype=float)
    return arr.astype(float, copy=False)


def _safe_corrcoef(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = _to_float_array(x)
    y_arr = _to_float_array(y)
    n = min(x_arr.size, y_arr.size)
    if n < 2:
        return 0.0
    x0 = x_arr[:n]
    y0 = y_arr[:n]
    if float(np.std(x0)) < 1e-12 or float(np.std(y0)) < 1e-12:
        return 0.0
    corr = float(np.corrcoef(x0, y0)[0, 1])
    return 0.0 if np.isnan(corr) else corr


def _lag_autocorr(values: Sequence[float], lag: int = 1) -> float:
    arr = _to_float_array(values)
    if arr.size <= lag:
        return 0.0
    left = arr[:-lag]
    right = arr[lag:]
    return _safe_corrcoef(left, right)


def _excess_kurtosis(values: Sequence[float]) -> float:
    arr = _to_float_array(values)
    if arr.size < 4:
        return 0.0
    centered = arr - float(np.mean(arr))
    m2 = float(np.mean(centered ** 2))
    if m2 < 1e-12:
        return 0.0
    m4 = float(np.mean(centered ** 4))
    return float(m4 / (m2 ** 2) - 3.0)


def _price_returns(values: Sequence[float]) -> np.ndarray:
    arr = _to_float_array(values)
    if arr.size < 2:
        return np.zeros(0, dtype=float)
    prev = np.maximum(arr[:-1], 1e-12)
    return np.diff(arr) / prev


def _normalize_prices(values: Sequence[float]) -> np.ndarray:
    arr = _to_float_array(values)
    if arr.size == 0:
        return arr
    base = float(arr[0]) if float(arr[0]) != 0 else 1.0
    return arr / base


def _turning_point_mask(values: Sequence[float]) -> np.ndarray:
    arr = _to_float_array(values)
    if arr.size < 3:
        return np.zeros(arr.size, dtype=bool)
    diffs = np.diff(arr)
    signs = np.sign(diffs)
    mask = np.zeros(arr.size, dtype=bool)
    for idx in range(1, signs.size):
        if signs[idx - 1] == 0 or signs[idx] == 0:
            continue
        if signs[idx - 1] != signs[idx]:
            mask[idx] = True
    return mask


def _binary_f1(predicted: np.ndarray, truth: np.ndarray) -> float:
    if predicted.size == 0 or truth.size == 0:
        return 0.0
    n = min(predicted.size, truth.size)
    pred = predicted[:n].astype(bool, copy=False)
    real = truth[:n].astype(bool, copy=False)
    tp = float(np.sum(pred & real))
    fp = float(np.sum(pred & ~real))
    fn = float(np.sum(~pred & real))
    if tp <= 0.0:
        return 0.0
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    if precision + recall <= 1e-12:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _drawdown_shape(values: Sequence[float]) -> Dict[str, float]:
    prices = _to_float_array(values)
    if prices.size < 2:
        return {
            "max_drawdown": 0.0,
            "average_drawdown": 0.0,
            "drawdown_duration": 0.0,
            "recovery_speed": 0.0,
            "shape_score": 0.0,
        }

    running_max = np.maximum.accumulate(prices)
    drawdowns = prices / np.maximum(running_max, 1e-12) - 1.0
    underwater = drawdowns < 0
    if not np.any(underwater):
        return {
            "max_drawdown": 0.0,
            "average_drawdown": 0.0,
            "drawdown_duration": 0.0,
            "recovery_speed": 1.0,
            "shape_score": 1.0,
        }

    max_drawdown = float(abs(np.min(drawdowns)))
    average_drawdown = float(abs(np.mean(drawdowns[underwater])))
    longest_duration = 0
    current = 0
    for flag in underwater:
        if flag:
            current += 1
            longest_duration = max(longest_duration, current)
        else:
            current = 0
    recovery_speed = float(1.0 / (1.0 + longest_duration))
    shape_score = float(np.clip(1.0 - (0.55 * max_drawdown + 0.30 * average_drawdown + 0.15 * longest_duration / max(prices.size, 1)), 0.0, 1.0))
    return {
        "max_drawdown": max_drawdown,
        "average_drawdown": average_drawdown,
        "drawdown_duration": float(longest_duration),
        "recovery_speed": recovery_speed,
        "shape_score": shape_score,
    }


def _volume_volatility_correlation(volumes: Sequence[float], prices: Sequence[float]) -> float:
    vol_arr = _to_float_array(volumes)
    ret_arr = np.abs(_price_returns(prices))
    n = min(vol_arr.size, ret_arr.size)
    if n < 2:
        return 0.0
    return _safe_corrcoef(vol_arr[:n], ret_arr[:n])


def _order_sign_autocorrelation(order_signs: Sequence[float]) -> float:
    signs = _to_float_array(order_signs)
    if signs.size < 3:
        return 0.0
    signs = np.sign(signs)
    return _safe_corrcoef(signs[:-1], signs[1:])


def _price_impact_curve(
    *,
    order_signs: Sequence[float],
    trade_sizes: Optional[Sequence[float]],
    forward_returns: Sequence[float],
) -> Dict[str, Any]:
    signs = _to_float_array(order_signs)
    sizes = _to_float_array(trade_sizes) if trade_sizes is not None else np.ones_like(signs)
    future = _to_float_array(forward_returns)
    n = min(signs.size, sizes.size, future.size)
    if n < 2:
        return {
            "slope": 0.0,
            "correlation": 0.0,
            "bucketed_curve": [],
        }

    signed_volume = np.sign(signs[:n]) * np.abs(sizes[:n])
    response = future[:n]
    corr = _safe_corrcoef(signed_volume, response)
    var = float(np.var(signed_volume))
    slope = float(np.cov(signed_volume, response)[0, 1] / max(var, 1e-12)) if var > 1e-12 else 0.0

    order = np.argsort(signed_volume)
    chunks = np.array_split(order, 4)
    bucketed_curve: List[Dict[str, float]] = []
    for idx, chunk in enumerate(chunks, start=1):
        if chunk.size == 0:
            continue
        bucketed_curve.append(
            {
                "bucket": float(idx),
                "signed_volume_mean": float(np.mean(signed_volume[chunk])),
                "future_return_mean": float(np.mean(response[chunk])),
            }
        )

    return {
        "slope": slope,
        "correlation": corr,
        "bucketed_curve": bucketed_curve,
    }


def _csad_summary(
    *,
    market_returns: Sequence[float],
    cross_sectional_returns: Optional[Sequence[Sequence[float]]],
) -> Dict[str, Any]:
    market = _to_float_array(market_returns)
    if cross_sectional_returns is None:
        return {
            "mean_csad": 0.0,
            "gamma1": 0.0,
            "gamma2": 0.0,
            "herding_detected": False,
            "strength": 0.0,
        }

    matrix_rows = [np.asarray([float(x) for x in row], dtype=float) for row in cross_sectional_returns if row]
    if not matrix_rows or market.size == 0:
        return {
            "mean_csad": 0.0,
            "gamma1": 0.0,
            "gamma2": 0.0,
            "herding_detected": False,
            "strength": 0.0,
        }

    width = min(len(row) for row in matrix_rows)
    if width <= 0:
        return {
            "mean_csad": 0.0,
            "gamma1": 0.0,
            "gamma2": 0.0,
            "herding_detected": False,
            "strength": 0.0,
        }

    matrix = np.asarray([row[:width] for row in matrix_rows], dtype=float)
    n = min(matrix.shape[0], market.size)
    matrix = matrix[:n]
    market = market[:n]
    if n < 3:
        return {
            "mean_csad": float(np.mean(np.abs(matrix - market.reshape(-1, 1)))) if matrix.size else 0.0,
            "gamma1": 0.0,
            "gamma2": 0.0,
            "herding_detected": False,
            "strength": 0.0,
        }

    csad_series = np.asarray([calculate_csad(matrix[i], market[i]) for i in range(n)], dtype=float)
    reg = csad_regression(matrix, market, window=min(20, n))
    return {
        "mean_csad": float(np.mean(csad_series)),
        "gamma1": float(reg.get("gamma1", 0.0)),
        "gamma2": float(reg.get("gamma2", 0.0)),
        "herding_detected": bool(reg.get("herding_detected", False)),
        "strength": float(reg.get("strength", 0.0)),
    }


def _config_hash(seed: int, config: Mapping[str, Any], feature_flag: bool, version: str) -> str:
    payload = {
        "seed": int(seed),
        "feature_flag": bool(feature_flag),
        "version": str(version),
        "config": json.loads(json.dumps(dict(config), sort_keys=True, default=str)),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class StylizedFactsReport:
    """Structured realism evaluation payload."""

    feature_flag: bool
    seed: int
    config_hash: str
    snapshot_info: Dict[str, Any]
    path_fit: Dict[str, Any]
    microstructure_fit: Dict[str, Any]
    behavioral_fit: Dict[str, Any]
    charts: List[Dict[str, Any]]
    credibility_score: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    report_meta: Dict[str, Any] = field(default_factory=dict)
    reproducibility: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StylizedFactsEvaluator:
    """
    Reproducible stylized-facts evaluator with feature flag gating.

    When feature_flag=False, the evaluator only emits path-fit metrics and
    minimal charts so legacy consumers can safely ignore the new framework.
    """

    feature_flag: bool = False
    seed: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    version: str = "stylized_facts_v1"

    def _build_snapshot_info(
        self,
        *,
        real_prices: Sequence[float],
        simulated_prices: Sequence[float],
        real_volumes: Optional[Sequence[float]],
        simulated_volumes: Optional[Sequence[float]],
        timestamps: Optional[Sequence[Any]],
    ) -> Dict[str, Any]:
        real_arr = _to_float_array(real_prices)
        sim_arr = _to_float_array(simulated_prices)
        ts_list = [str(x) for x in timestamps] if timestamps is not None else []
        snapshot = {
            "real_points": int(real_arr.size),
            "simulated_points": int(sim_arr.size),
            "timestamp_count": int(len(ts_list)),
            "real_price_min": float(np.min(real_arr)) if real_arr.size else 0.0,
            "real_price_max": float(np.max(real_arr)) if real_arr.size else 0.0,
            "sim_price_min": float(np.min(sim_arr)) if sim_arr.size else 0.0,
            "sim_price_max": float(np.max(sim_arr)) if sim_arr.size else 0.0,
            "real_volume_sum": float(np.sum(_to_float_array(real_volumes))) if real_volumes is not None else 0.0,
            "sim_volume_sum": float(np.sum(_to_float_array(simulated_volumes))) if simulated_volumes is not None else 0.0,
        }
        if ts_list:
            snapshot["window_start"] = ts_list[0]
            snapshot["window_end"] = ts_list[-1]
        return snapshot

    def evaluate(
        self,
        *,
        real_prices: Sequence[float],
        simulated_prices: Sequence[float],
        real_volumes: Optional[Sequence[float]] = None,
        simulated_volumes: Optional[Sequence[float]] = None,
        order_signs: Optional[Sequence[float]] = None,
        trade_sizes: Optional[Sequence[float]] = None,
        market_returns: Optional[Sequence[float]] = None,
        cross_sectional_returns: Optional[Sequence[Sequence[float]]] = None,
        timestamps: Optional[Sequence[Any]] = None,
        legacy_metrics: Optional[Mapping[str, Any]] = None,
        snapshot_info: Optional[Mapping[str, Any]] = None,
    ) -> StylizedFactsReport:
        real_arr = _to_float_array(real_prices)
        sim_arr = _to_float_array(simulated_prices)
        real_returns = _price_returns(real_arr)
        sim_returns = _price_returns(sim_arr)
        n = min(real_arr.size, sim_arr.size)
        aligned_real = real_arr[:n]
        aligned_sim = sim_arr[:n]
        aligned_real_returns = real_returns[: max(0, n - 1)]
        aligned_sim_returns = sim_returns[: max(0, n - 1)]

        legacy = dict(legacy_metrics or {})
        price_correlation = float(legacy.get("price_correlation", _safe_corrcoef(aligned_real, aligned_sim)))
        real_vol = float(np.std(aligned_real_returns)) if aligned_real_returns.size else 0.0
        sim_vol = float(np.std(aligned_sim_returns)) if aligned_sim_returns.size else 0.0
        volatility_correlation = float(
            legacy.get(
                "volatility_correlation",
                min(real_vol, sim_vol) / max(real_vol, sim_vol, 1e-12) if (real_vol > 0.0 or sim_vol > 0.0) else 0.0,
            )
        )
        normalized_real = _normalize_prices(aligned_real)
        normalized_sim = _normalize_prices(aligned_sim)
        if normalized_real.size and normalized_sim.size:
            delta = normalized_sim[: min(normalized_real.size, normalized_sim.size)] - normalized_real[: min(normalized_real.size, normalized_sim.size)]
            price_rmse = float(legacy.get("price_rmse", np.sqrt(np.mean(delta ** 2))))
            price_mae = float(legacy.get("price_mae", np.mean(np.abs(delta))))
        else:
            price_rmse = float(legacy.get("price_rmse", 0.0))
            price_mae = float(legacy.get("price_mae", 0.0))

        base_credibility = float(legacy.get("credibility_score", 0.0))
        drawdown = {
            "real": _drawdown_shape(aligned_real),
            "simulated": _drawdown_shape(aligned_sim),
        }

        return_autocorr = {
            "real": _lag_autocorr(aligned_real_returns),
            "simulated": _lag_autocorr(aligned_sim_returns),
        }
        abs_return_autocorr = {
            "real": _lag_autocorr(np.abs(aligned_real_returns)),
            "simulated": _lag_autocorr(np.abs(aligned_sim_returns)),
        }
        volatility_clustering = {
            "real": _lag_autocorr(aligned_real_returns ** 2),
            "simulated": _lag_autocorr(aligned_sim_returns ** 2),
        }
        tail_heaviness = {
            "real": _excess_kurtosis(aligned_real_returns),
            "simulated": _excess_kurtosis(aligned_sim_returns),
        }
        direction_accuracy = float(_safe_corrcoef(np.sign(aligned_real_returns), np.sign(aligned_sim_returns)))
        if aligned_real_returns.size and aligned_sim_returns.size:
            direction_accuracy = float(np.mean(np.sign(aligned_real_returns) == np.sign(aligned_sim_returns)))
        turning_point_f1 = float(_binary_f1(_turning_point_mask(aligned_sim), _turning_point_mask(aligned_real)))
        real_volume_series = real_volumes if real_volumes is not None else []
        simulated_volume_series = simulated_volumes if simulated_volumes is not None else []
        order_sign_series = order_signs if order_signs is not None else []
        market_return_series = market_returns if market_returns is not None else aligned_real_returns
        cross_sectional_series = cross_sectional_returns if cross_sectional_returns is not None else None

        volume_volatility_corr = {
            "real": _volume_volatility_correlation(real_volume_series, aligned_real),
            "simulated": _volume_volatility_correlation(simulated_volume_series, aligned_sim),
        }
        order_sign_autocorr = float(_order_sign_autocorrelation(order_sign_series))
        price_impact = _price_impact_curve(
            order_signs=order_sign_series,
            trade_sizes=trade_sizes,
            forward_returns=aligned_real_returns if aligned_real_returns.size else aligned_sim_returns,
        )
        csad = _csad_summary(
            market_returns=market_return_series,
            cross_sectional_returns=cross_sectional_series,
        )

        path_score_inputs = [
            (price_correlation + 1.0) / 2.0,
            max(0.0, 1.0 - price_rmse),
            max(0.0, 1.0 - min(price_mae, 1.0)),
            drawdown["simulated"]["shape_score"],
        ]
        path_fit_score = float(np.clip(np.mean(path_score_inputs), 0.0, 1.0))

        micro_score_inputs = [
            (volume_volatility_corr["simulated"] + 1.0) / 2.0,
            (order_sign_autocorr + 1.0) / 2.0,
            (price_impact["correlation"] + 1.0) / 2.0,
            turning_point_f1,
        ]
        behavioral_score_inputs = [
            (return_autocorr["simulated"] + 1.0) / 2.0,
            (abs_return_autocorr["simulated"] + 1.0) / 2.0,
            (volatility_clustering["simulated"] + 1.0) / 2.0,
            float(np.clip(1.0 / (1.0 + abs(tail_heaviness["simulated"])), 0.0, 1.0)),
            float(np.clip(1.0 / (1.0 + abs(csad["gamma2"])), 0.0, 1.0)),
            direction_accuracy,
        ]

        if self.feature_flag:
            microstructure_fit = {
                "enabled": True,
                "volume_volatility_correlation": volume_volatility_corr,
                "order_sign_autocorrelation": order_sign_autocorr,
                "price_impact_curve": price_impact,
                "turning_point_f1": turning_point_f1,
                "score": float(np.clip(np.mean(micro_score_inputs), 0.0, 1.0)),
            }
            behavioral_fit = {
                "enabled": True,
                "return_autocorrelation": return_autocorr,
                "abs_return_autocorrelation": abs_return_autocorr,
                "volatility_clustering": volatility_clustering,
                "tail_heaviness_kurtosis": tail_heaviness,
                "csad_herding": csad,
                "direction_accuracy": direction_accuracy,
                "score": float(np.clip(np.mean(behavioral_score_inputs), 0.0, 1.0)),
            }
            credibility_score = float(np.clip(np.mean([path_fit_score, microstructure_fit["score"], behavioral_fit["score"]]), 0.0, 1.0))
            chart_specs = [
                {
                    "name": "price_overlay",
                    "kind": "line",
                    "series": [
                        {"label": "real", "values": aligned_real.tolist()},
                        {"label": "simulated", "values": aligned_sim.tolist()},
                    ],
                },
                {
                    "name": "return_autocorrelation",
                    "kind": "paired",
                    "series": {
                        "real": return_autocorr["real"],
                        "simulated": return_autocorr["simulated"],
                        "abs_real": abs_return_autocorr["real"],
                        "abs_simulated": abs_return_autocorr["simulated"],
                    },
                },
                {
                    "name": "price_impact_curve",
                    "kind": "structured",
                    "series": price_impact,
                },
            ]
        else:
            microstructure_fit = {
                "enabled": False,
                "feature_flag": "stylized_facts_v2",
                "score": 0.0,
            }
            behavioral_fit = {
                "enabled": False,
                "feature_flag": "stylized_facts_v2",
                "score": 0.0,
            }
            credibility_score = path_fit_score if base_credibility <= 0.0 else float(np.clip(base_credibility, 0.0, 1.0))
            chart_specs = [
                {
                    "name": "price_overlay",
                    "kind": "line",
                    "series": [
                        {"label": "real", "values": aligned_real.tolist()},
                        {"label": "simulated", "values": aligned_sim.tolist()},
                    ],
                }
            ]

        snapshot = self._build_snapshot_info(
            real_prices=aligned_real.tolist(),
            simulated_prices=aligned_sim.tolist(),
            real_volumes=real_volumes,
            simulated_volumes=simulated_volumes,
            timestamps=timestamps,
        )
        if snapshot_info:
            snapshot.update(dict(snapshot_info))

        report_metrics = {
            "price_correlation": price_correlation,
            "volatility_correlation": volatility_correlation,
            "price_rmse": price_rmse,
            "price_mae": price_mae,
            "credibility_score": credibility_score,
            "return_autocorrelation": return_autocorr,
            "abs_return_autocorrelation": abs_return_autocorr,
            "volatility_clustering": volatility_clustering,
            "tail_heaviness_kurtosis": tail_heaviness,
            "drawdown_shape": drawdown,
            "volume_volatility_correlation": volume_volatility_corr,
            "csad_herding": csad,
            "order_sign_autocorrelation": order_sign_autocorr,
            "price_impact_curve": price_impact,
            "turning_point_f1": turning_point_f1,
            "direction_accuracy": direction_accuracy,
        }

        return StylizedFactsReport(
            feature_flag=bool(self.feature_flag),
            seed=int(self.seed),
            config_hash=_config_hash(self.seed, self.config, self.feature_flag, self.version),
            snapshot_info=snapshot,
            path_fit={
                "enabled": True,
                "score": path_fit_score,
                "price_correlation": price_correlation,
                "volatility_correlation": volatility_correlation,
                "price_rmse": price_rmse,
                "price_mae": price_mae,
                "drawdown_shape": drawdown,
            },
            microstructure_fit=microstructure_fit,
            behavioral_fit=behavioral_fit,
            charts=chart_specs,
            credibility_score=credibility_score,
            metrics=report_metrics,
            reproducibility={
                "seed": int(self.seed),
                "config_hash": _config_hash(self.seed, self.config, self.feature_flag, self.version),
                "feature_flag": bool(self.feature_flag),
                "snapshot_info": snapshot,
            },
            notes=[
                "feature_flag=off emits only path-fit diagnostics for legacy compatibility"
                if not self.feature_flag
                else "feature_flag=on emits the full stylized-facts stack",
            ],
        )
