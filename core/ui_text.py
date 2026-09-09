"""UI-facing text helpers for Chinese labels and display-only localization."""

from __future__ import annotations

from typing import Any

import pandas as pd


SCENARIO_DISPLAY_NAMES = {
    "tax_cut_liquidity_boost": "减税与流动性提振",
    "rumor_panic_selloff": "传言冲击与恐慌抛售",
    "regulator_stabilization_intervention": "监管维稳与市场修复",
}

MODE_DISPLAY_NAMES = {
    "LIVE_MODE": "实时联机",
    "DEMO_MODE": "场景推演",
    "COMPETITION_DEMO_MODE": "综合展示",
}

RISK_ALERT_DISPLAY_NAMES = {
    "GREEN": "绿色",
    "YELLOW": "黄色",
    "RED": "红色",
}

KEY_TRANSLATIONS = {
    "Research workbench": "研究工作台",
    "research_workbench": "研究工作台",
    "benchmark selector": "基准指数",
    "Scorecard": "评估卡",
    "scorecard": "评估卡",
    "Replay Scorecard": "回放评估卡",
    "Reproducibility": "可复现信息",
    "Experiment registry": "实验登记信息",
    "Provider versions": "依赖版本",
    "Runtime chain": "运行链路",
    "Composite Policy Score": "政策综合评分",
    "composite_policy_score": "政策综合评分",
    "composite_score": "综合得分",
    "Pareto": "帕累托",
    "pareto": "帕累托",
    "pareto_frontier": "帕累托前沿",
    "policy_sensitivity": "政策敏感性",
    "robustness": "跨场景稳健性",
    "early_warning_utility": "预警价值",
    "relation_to_shanghai_index": "与上证指数关系",
    "rank_score": "综合排序分",
    "composite_weight": "综合权重",
    "candidate_pool": "候选指标池",
    "name": "指标名称",
    "category": "类别",
    "metric": "指标",
    "value": "数值",
    "section": "模块",
    "flag": "校验项",
    "passed": "是否通过",
    "date": "日期",
    "time": "时间",
    "step": "交易日序号",
    "open": "开盘价",
    "high": "最高价",
    "low": "最低价",
    "close": "收盘价",
    "real": "真实走势",
    "simulated": "仿真走势",
    "baseline": "基线走势",
    "volume": "成交量",
    "amount": "成交额",
    "source": "来源",
    "count": "数量",
    "topic": "主题",
    "impact": "影响",
    "panic_level": "恐慌度",
    "csad": "羊群度（CSAD）",
    "spread": "买卖价差",
    "depth_imbalance": "盘口深度失衡",
    "sentiment_index": "情绪指数",
    "MACD signal": "MACD 信号线",
    "macd_signal": "MACD 信号线",
    "BOLL upper": "布林线上轨",
    "boll_upper": "布林线上轨",
    "BOLL mid": "布林中轨",
    "boll_mid": "布林中轨",
    "BOLL lower": "布林下轨",
    "boll_lower": "布林下轨",
    "trend_alignment": "趋势一致性",
    "turning_point_match": "拐点匹配度",
    "drawdown_gap": "回撤差距",
    "vol_similarity": "波动相似度",
    "response_gap": "响应滞后",
    "strict_authenticity_score": "基准拟真评分",
    "baseline_authenticity_score": "基准拟真评分",
    "demo_authenticity_score": "综合拟真评分",
    "raw metrics": "原始指标",
    "raw_metrics": "原始指标",
    "display metrics": "展示指标",
    "display_metrics": "展示指标",
    "intervention_cost": "干预成本",
    "macro_stability": "宏观稳定性",
    "liquidity": "流动性",
    "avg_reward": "平均收益",
    "avg_episode_reward": "平均回合收益",
    "financing_function": "融资功能",
    "fairness_compliance": "公平合规",
    "default_production_path": "默认优化路径",
    "q_learning_baseline": "强化学习基线",
    "bayesian_optimization": "贝叶斯优化",
    "nsga_ii": "多目标进化搜索",
    "action_description": "动作说明",
    "action_signature": "动作签名",
    "tradeoff_summary": "权衡摘要",
    "side_effects": "潜在副作用",
    "delta": "差分变化",
    "world": "世界线",
    "return_pct": "区间收益率",
    "max_drawdown": "最大回撤",
    "avg_panic": "平均恐慌度",
    "max_panic": "最大恐慌度",
    "volatility": "波动率",
    "role": "主体角色",
    "agent": "智能体",
    "Agent": "智能体",
    "net_flow": "净买卖量",
    "score": "得分",
    "status": "状态",
    "decision": "决策",
    "decision_label": "决策摘要",
    "confidence": "置信度",
    "ticker": "标的",
    "price": "价格",
    "qty": "数量",
    "thought": "判断依据",
    "history": "历史轨迹",
    "scenario": "场景",
    "analyst_outputs": "分析师输出",
    "analyst_cards": "分析师卡片",
    "analyst_id": "分析师",
    "headline": "标题",
    "sentiment_score": "情绪得分",
    "key_event": "关键事件",
    "momentum": "动量",
    "herding_intensity": "羊群强度",
    "signal": "信号",
    "cvar": "条件风险价值",
    "risk_level": "风险等级",
    "manager_decision": "经理决策",
    "manager_final_card": "经理最终卡",
    "stance": "策略立场",
    "allocation": "仓位分配",
    "execution_plan": "执行计划",
    "equity": "股票",
    "cash": "现金",
    "hedge": "对冲",
    "contradiction_matrix": "矛盾矩阵",
    "contradiction_index": "矛盾指数",
    "analysts": "分析师列表",
    "matrix": "矩阵",
    "risk_alert": "风险预警",
    "risk_alerts": "风险告警",
    "calibration": "校准指标",
    "brier_like_score": "Brier 类分数",
    "confidence_drift": "置信漂移",
    "outcome_proxy": "结果代理值",
    "raw_confidence": "原始置信度",
    "calibrated_confidence": "校准后置信度",
    "time_horizon": "时间跨度",
    "counterarguments": "反方观点",
    "recommended_action": "建议动作",
    "risk_tags": "风险标签",
    "evidence": "证据",
    "type": "类型",
    "content": "内容",
    "weight": "权重",
    "thesis": "核心判断",
    "speaker": "角色",
    "text": "内容",
    "event": "事件",
    "level": "等级",
}

METRIC_TRANSLATIONS = {
    **KEY_TRANSLATIONS,
    "shanghai_index_return": "上证指数收益",
    "microstructure_score": "微观结构评分",
    "liquidity_pressure": "流动性压力",
    "policy_sentiment_beta": "政策情绪敏感度",
    "market_confidence": "市场信心",
    "direction_hit_rate": "方向命中率",
    "tracking_rmse": "路径跟踪误差",
    "normalized_rmse": "标准化误差",
    "price_correlation": "价格相关性",
    "return_correlation": "收益相关性",
    "sim_volatility": "仿真波动率",
    "real_volatility": "真实波动率",
    "volatility_gap": "波动差距",
    "sim_max_drawdown": "仿真最大回撤",
    "real_max_drawdown": "真实最大回撤",
    "max_drawdown_gap": "最大回撤差距",
    "trade_count": "成交笔数",
    "csad_mean": "CSAD 均值",
    "herd_intensity": "羊群强度",
}

VALUE_TRANSLATIONS = {
    "self": "核心基准",
    "complement": "补充指标",
    "leading": "前导指标",
    "lagging": "滞后指标",
    "strict": "严格验证",
    "demo": "展示增强",
    "factor": "因子回测",
    "agent": "新闻驱动政策仿真",
    "trade tape aggregation": "逐笔成交聚合",
    "synthetic tape fallback -> trade tape aggregation": "仿真逐笔成交回退后聚合",
    "1w": "近 1 周",
    "1m": "近 1 月",
    "BUY": "买入",
    "SELL": "卖出",
    "HOLD": "观望",
    "buy": "买入",
    "sell": "卖出",
    "hold": "观望",
    "Risk-On": "风险偏好提升",
    "Risk-Off": "风险偏好收缩",
    "Observe": "观察",
    "risk_on": "风险偏好提升",
    "risk-off": "风险偏好收缩",
    "risk_on_controlled": "进攻但受控",
    "panic_sell": "恐慌抛售",
    "stabilizing": "趋于稳定",
    "moderate": "中等",
    "high": "高",
    "controlled": "可控",
    "RISK_ON_CONTROLLED": "进攻但受控",
    "DEFENSIVE_DELEVERAGING": "防御去杠杆",
    "STABILIZE_AND_REBALANCE": "稳市再平衡",
    "swing": "波段",
    "expert_replay": "专家复盘",
    "demo_mode": "场景推演",
    "news_analyst": "新闻分析师",
    "quant_analyst": "量化分析师",
    "risk_analyst": "风险分析师",
    "GREEN": "绿色",
    "YELLOW": "黄色",
    "RED": "红色",
    "retail_day_trader": "散户短线",
    "retail_swing": "散户波段",
    "retail_momentum_chaser": "散户趋势",
    "retail_general": "散户资金",
    "mutual_fund": "公募基金",
    "quant_arbitrage": "量化机构",
    "quant_timing": "量化择时",
    "market_maker": "做市资金",
    "state_stabilization_fund": "稳定资金",
    "committee_01": "智能体一号",
    "committee_02": "智能体二号",
    "committee_03": "智能体三号",
    "no_intervention": "不介入",
    "early_intervention": "提前介入",
    "late_intervention": "延后介入",
    "default_production_path": "默认优化路径",
    "q_learning_baseline": "强化学习基线",
    "bayesian_optimization": "贝叶斯优化",
    "nsga_ii": "多目标进化搜索",
    "offline_fallback": "离线兜底",
    "regulator_q_learning": "监管强化学习",
    "DeepSeek": "DeepSeek",
    "GLM-4-flashx": "GLM-4-flashx",
    "Python OrderBook fallback": "Python 撮合回退",
    "C++ _civitas_lob": "C++ 撮合扩展",
    "旁白": "旁白",
    "Tax cut + liquidity support announced": "减税与流动性支持政策公布",
    "Policy package reduces transaction friction": "政策组合降低了交易摩擦",
    "Unverified restructuring rumor spreads": "未经证实的重组传言迅速扩散",
    "Negative social amplification accelerates risk-off": "负面舆情扩散加速风险偏好收缩",
    "Regulator launches stabilization package": "监管部门推出市场稳定方案",
    "Market receives coordinated policy backstop": "市场获得协同政策托底",
    "Increase index exposure in three tranches": "分三批提高指数仓位",
    "Keep liquidity buffer to avoid chasing spikes": "保留流动性缓冲，避免追高",
    "Switch to defensive plan if regulation reverses": "若监管预期逆转，切换至防御方案",
    "Cut high-beta positions first": "优先削减高 Beta 仓位",
    "Increase cash and wait for volatility compression": "提高现金比例，等待波动收敛",
    "Activate emergency hedge against gap risk": "启动应急对冲，应对跳空风险",
    "Rebalance exposure toward liquid large caps": "将仓位再平衡至高流动性大盘蓝筹",
    "Reduce tail-risk hedge gradually as panic cools": "随着恐慌缓解，逐步降低尾部风险对冲",
    "Track policy persistence before adding risk": "确认政策持续性后再增加风险敞口",
}

WORLD_TRANSLATIONS = {
    "no_intervention": "不介入",
    "early_intervention": "提前介入",
    "late_intervention": "延后介入",
    "baseline": "基线方案",
    "policy_a": "方案 A",
    "policy_b": "方案 B",
    "optimized_policy": "推荐方案",
}

ACTION_TRANSLATIONS = {
    "BUY": "买入",
    "SELL": "卖出",
    "HOLD": "观望",
    "buy": "买入",
    "sell": "卖出",
    "hold": "观望",
    "no_intervention": "不介入",
    "early_intervention": "提前介入",
    "late_intervention": "延后介入",
    "q_learning_baseline": "强化学习基线",
    "bayesian_optimization": "贝叶斯优化",
    "nsga_ii": "多目标进化搜索",
}

PROVIDER_TRANSLATIONS = {
    "q_learning_baseline": "强化学习基线",
    "offline_fallback": "离线兜底",
    "regulator_q_learning": "监管强化学习",
    "bayesian_optimization": "贝叶斯优化",
    "nsga_ii": "多目标进化搜索",
    "GLM-4-flashx": "GLM-4-flashx",
    "DeepSeek": "DeepSeek",
    "deepseek-chat": "DeepSeek Chat",
}

TEXT_REPLACEMENTS = (
    ("A/B World Compare", "甲乙世界对照"),
    ("A/B Compare", "甲乙对照"),
    ("A/B", "甲乙对照"),
    ("Research workbench", "研究工作台"),
    ("benchmark selector", "基准指数"),
    ("Scorecard", "评估卡"),
    ("Reproducibility", "可复现信息"),
    ("Composite Policy Score", "政策综合评分"),
    ("MACD signal", "MACD 信号线"),
    ("BOLL upper", "布林线上轨"),
    ("BOLL mid", "布林中轨"),
    ("BOLL lower", "布林下轨"),
    ("synthetic tape fallback -> trade tape aggregation", "仿真逐笔成交回退后聚合"),
    ("trade tape aggregation", "逐笔成交聚合"),
    ("risk-on", "风险偏好提升"),
    ("risk-off", "风险偏好收缩"),
    ("legacy output", "历史输出"),
    ("large caps", "大盘蓝筹"),
    ("tail-risk hedge", "尾部风险对冲"),
    ("gap risk", "跳空风险"),
)


def _lookup(mapping: dict[str, str], key: str) -> str:
    text = str(key)
    return mapping.get(text, mapping.get(text.strip(), text))


def zh_label(key: str) -> str:
    return _lookup(KEY_TRANSLATIONS, key)


def zh_value(value: str) -> str:
    text = _lookup(VALUE_TRANSLATIONS, value)
    for source, target in TEXT_REPLACEMENTS:
        text = text.replace(source, target)
    return text


def zh_metric_name(name: str) -> str:
    return _lookup(METRIC_TRANSLATIONS, name)


def zh_world_name(name: str) -> str:
    return _lookup(WORLD_TRANSLATIONS, name)


def zh_action_name(name: str) -> str:
    return _lookup(ACTION_TRANSLATIONS, name)


def zh_provider_name(name: str) -> str:
    return _lookup(PROVIDER_TRANSLATIONS, name)


def localize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-only copy with Chinese column labels and common values."""

    if df is None:
        return df
    display = df.copy()
    display = display.rename(columns={col: zh_label(str(col)) for col in display.columns})
    for col in display.columns:
        if str(col) in {"指标", "指标名称"}:
            display[col] = display[col].map(lambda value: zh_metric_name(value) if isinstance(value, str) else value)
        elif str(col) == "世界线":
            display[col] = display[col].map(lambda value: zh_world_name(value) if isinstance(value, str) else value)
        else:
            display[col] = display[col].map(lambda value: zh_value(value) if isinstance(value, str) else value)
    return display


def display_scenario_name(name: str) -> str:
    return SCENARIO_DISPLAY_NAMES.get(name, name)


def display_runtime_mode(mode: str) -> str:
    return MODE_DISPLAY_NAMES.get(mode, mode)


def display_risk_alert(level: str) -> str:
    return RISK_ALERT_DISPLAY_NAMES.get(level, level)


def translate_display_text(text: str) -> str:
    translated = SCENARIO_DISPLAY_NAMES.get(text, VALUE_TRANSLATIONS.get(text, KEY_TRANSLATIONS.get(text, text)))
    for source, target in TEXT_REPLACEMENTS:
        translated = translated.replace(source, target)
    return translated


def translate_ui_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {zh_label(str(key)): translate_ui_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [translate_ui_payload(item) for item in value]
    if isinstance(value, str):
        return translate_display_text(value)
    return value


__all__ = [
    "KEY_TRANSLATIONS",
    "MODE_DISPLAY_NAMES",
    "SCENARIO_DISPLAY_NAMES",
    "VALUE_TRANSLATIONS",
    "display_risk_alert",
    "display_runtime_mode",
    "display_scenario_name",
    "localize_dataframe_columns",
    "translate_display_text",
    "translate_ui_payload",
    "zh_action_name",
    "zh_label",
    "zh_metric_name",
    "zh_provider_name",
    "zh_value",
    "zh_world_name",
]
