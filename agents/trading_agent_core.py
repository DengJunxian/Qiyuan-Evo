"""
Civitas-Economica 核心交易智能体模块 (Trading Agent Module)

按照架构的严格面向对象设计 (Strict OOD) 原则：
隔离各模块的职责，如 engine/ (撮合), agents/ (LLM 包装、画像、记忆), policy/ (宏观事件), utils/。

此模块包含基于 Pydantic 的 Persona 数据模型、集成了 ChromaDB 与 LangChain 的 TradingAgent 类，
以及负责人群生成的 AgentFactory 类。
所有代码符合 Python 3.11+, asyncio 标准以及 PEP 8 与 Google-style Docstrings。
"""

import asyncio
import json
import logging
import random
import statistics
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from core.macro.state import MacroContextDTO
from core.validator import (
    PolicyCommittee,
    RiskCommittee,
    aggregate_analyst_cards,
    coerce_analyst_card,
    export_decision_trace,
    validate_analyst_card,
)

# 配置模块级别的 Logger (按照要求提供清晰的日志追踪)
logger = logging.getLogger("civitas.agents.trading_agent")
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
if not logger.handlers:
    logger.addHandler(sh)


# ----------------------------------------------------------------------------
# 1. 配置文件加载，避免魔法数字
# ----------------------------------------------------------------------------
def _load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载基于 YAML 的全局配置参数文件。

    Args:
        path (str): 配置文件路径，默认到项目根目录下的 config.yaml

    Returns:
        Dict[str, Any]: 解析后的配置字典
    """
    config_path = Path(__file__).parent.parent / path
    if not config_path.exists():
        logger.warning(f"Configuration file {config_path} not found. Using safe fallbacks.")
        return {
            "llm_temperature": 0.7,
            "market_slippage": 0.001,
            "transaction_costs": 0.0005,
            "ebbinghaus_base_decay_rate": 0.1
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

GLOBAL_CONFIG = _load_config()


# ----------------------------------------------------------------------------
# 2. Pydantic 数据模型与枚举类型定义
# ----------------------------------------------------------------------------
class CognitiveBias(str, Enum):
    Loss_Aversion = "Loss_Aversion"
    Herding = "Herding"
    Overconfidence = "Overconfidence"
    Rational = "Rational"


class InvestmentHorizon(str, Enum):
    Short_term = "Short_term"
    Medium_term = "Medium_term"
    Long_term = "Long_term"


class Persona(BaseModel):
    """
    用于定义交易智能体心理和策略特征的严格数据模型。
    
    采用 Pydantic 进行基于类型的字段验证。
    """
    model_config = ConfigDict(strict=True)

    risk_tolerance: float = Field(
        ..., ge=0.0, le=1.0, 
        description="代理的风险承受能力 (0.0=极度厌恶风险, 1.0=极端赌徒)"
    )
    cognitive_bias: CognitiveBias = Field(
        ..., 
        description="占主导地位的认知偏差，影响 LLM 对市场信息的解释"
    )
    investment_horizon: InvestmentHorizon = Field(
        ..., 
        description="投资的时间跨度偏好"
    )
    policy_sensitivity: float = Field(
        ..., ge=0.0, le=1.0,
        description="对宏观经济政策和新闻的敏感度"
    )

    def describe(self) -> str:
        """生成供 LLM 提示词 (Prompt) 使用的人格自述文本。"""
        return (f"您的风险承受度为 {self.risk_tolerance} (0-1)。"
                f"您的核心认知偏差是 {self.cognitive_bias.value}。"
                f"您的投资周期专注于 {self.investment_horizon.value}。"
                f"您对宏观政策事件敏感度指标为 {self.policy_sensitivity}。")


class TradeAction(BaseModel):
    """
    LLM 计划输出的标准化交易决策抽象类。
    """
    action: str = Field(..., description="如 BUY, SELL 或 HOLD")
    amount: float = Field(..., description="拟交易的资产数量或资金等价物")
    target_price: Optional[float] = Field(None, description="限价单的目标价格（空为市价）")
    reasoning_chain: str = Field(..., description="基于上下文进行决策的思考过程摘要")


# ----------------------------------------------------------------------------
# 3. 核心实体类：Trading智能体
# ----------------------------------------------------------------------------
class TradeIntent(BaseModel):
    """
    LLM 仅输出的交易意图结构（意图与执行解耦）。
    """
    direction: str = Field(..., description="BUY / SELL / HOLD")
    urgency: str = Field(..., description="Low / Medium / High")
    reason: str = Field(..., description="意图产生的原因")


class TradingAgent:
    """
    Civitas 经济体中的核心交易智能体。

    不仅具有基础的聊天功能，还是一个具有心理和战略约束的专业金融实体。
    通过 LangChain/原生 LLM API 和 ChromaDB 进行市场分析与记忆存储。
    """

    def __init__(self, agent_id: str, persona: Persona):
        """
        初始化交易代理。

        Args:
            agent_id (str): 系统中独一无二的智能体标识符。
            persona (Persona): 限定该智能体行为逻辑的心理与策略约束模型。
        """
        self.agent_id = agent_id
        self.persona = persona

        # 用于持有与市场交互所需的资金/资产等内部状态
        self.portfolio: Dict[str, float] = {}
        self.cash: float = GLOBAL_CONFIG.get("default_starting_cash", 100000.0)

        # 初始化基于 ChromaDB 包装的 Ebbinghaus 认知记忆银行
        # 延迟到这里局部导入以避免潜在的循环依赖引用
        from agents.ebbinghaus_memory import EbbinghausMemoryBank
        self.memory_bank = EbbinghausMemoryBank(agent_id=self.agent_id)
        self.risk_committee = RiskCommittee()
        self.policy_committee = PolicyCommittee()
        self.last_decision_artifacts: Dict[str, Any] = {}

        logger.info(f"Initialized TradingAgent {self.agent_id} with Persona: "
                    f"[{self.persona.cognitive_bias.value}, Risk: {self.persona.risk_tolerance}]")

    async def retrieve_context(self, current_market_data: Any) -> str:
        """
        异步从 ChromaDB 或本地长期记忆中根据上下文获取最相关的历史信息。

        Args:
            current_market_data (Any): 当期市场行情及宏观情绪。
            
        Returns:
            str: 用于注入给 Prompt 的上下文关联记忆字符串。
        """
        # 后续接入：与 ChromaDB 向量库联通
        # 此处使用简单的模拟返回
        await asyncio.sleep(0.01)  # 模拟异步 IO
        decay_rate = GLOBAL_CONFIG.get("ebbinghaus_base_decay_rate", 0.1)
        return f"基于艾宾浩斯认知衰减 ({decay_rate})，之前记忆最深的损失：股票近期发生过大幅回撤。"

    def _normalize_urgency(self, urgency: str) -> str:
        """统一意图紧急度标签，降低 LLM 输出噪声对执行层的影响。"""
        text = str(urgency or "").strip().lower()
        if text in {"high", "urgent", "h"}:
            return "High"
        if text in {"medium", "mid", "m"}:
            return "Medium"
        if text in {"low", "l"}:
            return "Low"
        return "Medium"

    def _coerce_legacy_intent(self, data: Dict[str, Any]) -> TradeIntent:
        """
        兼容旧版 LLM 输出（action/amount/target_price），转换为 TradeIntent。
        """
        direction = str(data.get("direction") or data.get("action", "HOLD")).upper()
        urgency = str(data.get("urgency", "Medium"))
        reason = str(data.get("reason") or data.get("reasoning_chain", "legacy output"))
        return TradeIntent(direction=direction, urgency=urgency, reason=reason)

    def _compute_reference_price(self, price_history: List[float], current_price: float, window: int = 20) -> float:
        """
        计算 FCN 模型中的“基本面价格”。
        默认使用移动均线作为稳态基准，兼容 TwinMarket 的稳态锚定思想。
        """
        if not price_history:
            return float(current_price)
        history = [float(x) for x in price_history if x is not None]
        if not history:
            return float(current_price)
        if len(history) >= window:
            return float(statistics.mean(history[-window:]))
        return float(statistics.mean(history))

    def _intent_to_order(self, intent: TradeIntent, market_data: Dict[str, Any]) -> TradeAction:
        """
        FCN 规则模型：将“交易意图”映射为具体限价单。
        论文映射（FCLAgent / TwinMarket）：
        - 意图由 LLM 输出，执行由规则模型完成，避免数值计算误差。
        """
        direction = str(intent.direction or "HOLD").upper()
        urgency = self._normalize_urgency(intent.urgency)
        reason = intent.reason or "LLM intent"

        current_price = float(market_data.get("current_price", 0.0) or 0.0)
        price_history = market_data.get("price_history", []) or []
        reference_price = self._compute_reference_price(price_history, current_price)

        # Chartist 动量项（短期趋势）
        momentum = 0.0
        if len(price_history) >= 6:
            recent = [float(x) for x in price_history[-6:] if x is not None]
            if len(recent) >= 6 and statistics.mean(recent[:-1]) > 0:
                momentum = (recent[-1] - statistics.mean(recent[:-1])) / statistics.mean(recent[:-1])

        # Urgency 影响溢价与仓位比例
        urgency_premium = {"High": 0.006, "Medium": 0.003, "Low": 0.001}.get(urgency, 0.003)
        urgency_size = {"High": 1.5, "Medium": 1.0, "Low": 0.6}.get(urgency, 1.0)

        # 微小噪声用于“噪声交易者”扰动
        noise = random.normalvariate(0, max(0.01, reference_price * 0.002))

        if direction == "BUY":
            base_price = reference_price * (1 + urgency_premium + 0.5 * max(momentum, 0.0))
            target_price = max(0.01, base_price + noise)
            budget = self.cash * (0.05 + 0.2 * self.persona.risk_tolerance) * urgency_size
            qty = int(budget / target_price) if target_price > 0 else 0
        elif direction == "SELL":
            base_price = reference_price * (1 - urgency_premium - 0.5 * max(-momentum, 0.0))
            target_price = max(0.01, base_price + noise)
            holding = int(self.portfolio.get(market_data.get("symbol", ""), 0) or 0)
            qty = int(holding * (0.25 + 0.5 * self.persona.risk_tolerance) * urgency_size)
        else:
            target_price = 0.0
            qty = 0

        if qty <= 0:
            return TradeAction(
                action="HOLD",
                amount=0.0,
                target_price=None,
                reasoning_chain=f"意图={direction}/{urgency}，但执行层未形成有效数量，降级 HOLD。原因：{reason}"
            )

        return TradeAction(
            action=direction,
            amount=float(qty),
            target_price=float(target_price),
            reasoning_chain=f"意图={direction}/{urgency}，FCN 映射执行。原因：{reason}"
        )

    def _build_structured_analyst_cards(
        self,
        market_data: Dict[str, Any],
        macro_context: Optional[MacroContextDTO],
        retrieved_context: str,
    ) -> List[Dict[str, Any]]:
        """Build analyst cards in required JSON schema."""
        current_price = float(market_data.get("current_price", 0.0) or 0.0)
        panic_level = float(market_data.get("panic_level", 0.0) or 0.0)
        pnl_pct = float(market_data.get("pnl_pct", 0.0) or 0.0)
        news_text = str(market_data.get("news", ""))
        news_lower = news_text.lower()
        policy_text = str(market_data.get("policy_description", ""))
        macro_state = macro_context.macro_state if macro_context is not None else None

        price_history = [float(x) for x in (market_data.get("price_history", []) or []) if x is not None]
        momentum = 0.0
        if len(price_history) >= 3 and abs(price_history[-3]) > 1e-9:
            momentum = (price_history[-1] - price_history[-3]) / abs(price_history[-3])

        news_sentiment = 0.0
        if any(token in news_lower for token in ("beat", "support", "stimulus", "upside", "bull")):
            news_sentiment += 0.35
        if any(token in news_lower for token in ("panic", "default", "selloff", "downgrade", "risk")):
            news_sentiment -= 0.35
        news_sentiment -= 0.4 * panic_level

        liquidity = float(macro_state.liquidity_index) if macro_state is not None else 1.0
        unemployment = float(macro_state.unemployment) if macro_state is not None else 0.05
        credit_spread = float(macro_state.credit_spread) if macro_state is not None else 0.02
        sentiment_index = float(macro_state.sentiment_index) if macro_state is not None else 0.5

        news_action = "buy" if news_sentiment > 0.08 else "sell" if news_sentiment < -0.08 else "hold"
        macro_action = "buy" if (liquidity > 1.05 and credit_spread < 0.03) else "reduce_risk" if (unemployment > 0.08 or credit_spread > 0.05) else "hold"
        risk_action = "reduce_risk" if (panic_level > 0.55 or pnl_pct < -0.08) else "hold"

        cards = [
            {
                "analyst_id": f"{self.agent_id}_news_analyst",
                "thesis": f"News-price thesis around current_price={current_price:.2f}",
                "evidence": [
                    {"type": "news", "content": news_text[:280] or "no_news", "weight": 0.70},
                    {"type": "price", "content": f"momentum={momentum:.4f}", "weight": 0.58},
                ],
                "time_horizon": "intraday",
                "risk_tags": ["panic"] if panic_level > 0.45 else [],
                "confidence": min(0.92, 0.45 + abs(news_sentiment)),
                "counterarguments": ["news impact may fade quickly", "headline may be partially priced in"],
                "recommended_action": news_action,
            },
            {
                "analyst_id": f"{self.agent_id}_macro_analyst",
                "thesis": "Macro-policy pass-through assessment",
                "evidence": [
                    {"type": "macro", "content": f"liquidity={liquidity:.3f}, credit_spread={credit_spread:.3f}", "weight": 0.72},
                    {"type": "macro", "content": f"unemployment={unemployment:.3f}, sentiment_index={sentiment_index:.3f}", "weight": 0.64},
                ],
                "time_horizon": "weekly",
                "risk_tags": ["liquidity"] if liquidity < 0.9 else [],
                "confidence": min(0.88, 0.48 + 0.22 * abs(liquidity - 1.0) + 0.20 * abs(credit_spread - 0.02)),
                "counterarguments": ["macro effect has lag", "policy execution uncertainty"],
                "recommended_action": macro_action,
            },
            {
                "analyst_id": f"{self.agent_id}_risk_analyst",
                "thesis": "Portfolio preservation and market micro risk assessment",
                "evidence": [
                    {"type": "risk", "content": f"panic_level={panic_level:.3f}, pnl_pct={pnl_pct:.3f}", "weight": 0.78},
                    {"type": "social", "content": str(market_data.get("social_signal", "neutral"))[:220], "weight": 0.52},
                    {"type": "risk", "content": str(retrieved_context or "no_memory_context")[:220], "weight": 0.46},
                ],
                "time_horizon": "intraday",
                "risk_tags": ["liquidity", "panic", "overcrowding"] if panic_level > 0.50 else ["liquidity"] if panic_level > 0.30 else [],
                "confidence": min(0.93, 0.52 + 0.40 * panic_level),
                "counterarguments": ["defensive stance may miss rebound"],
                "recommended_action": risk_action,
            },
        ]
        return [coerce_analyst_card(card, analyst_id=card.get("analyst_id", "analyst")) for card in cards]

    def _build_risk_metrics(self, market_data: Dict[str, Any], panic_level: float, pnl_pct: float) -> Dict[str, float]:
        """Build committee-ready risk metrics payload."""
        return {
            "cvar": float(market_data.get("cvar", market_data.get("risk_cvar", pnl_pct - 0.02))),
            "max_drawdown": float(market_data.get("max_drawdown", pnl_pct)),
            "turnover": float(market_data.get("turnover", 1.0 + panic_level * 1.4)),
            "crowding": float(market_data.get("crowding", min(1.0, panic_level + abs(pnl_pct) * 0.5))),
            "volatility_spike": float(market_data.get("volatility_spike", 1.0 + panic_level * 2.2)),
        }

    def _manager_card_to_intent(
        self,
        manager_final_card: Dict[str, Any],
        risk_alert: Dict[str, Any],
    ) -> TradeIntent:
        """Convert manager final card into execution intent."""
        action = str(manager_final_card.get("recommended_action", "hold")).lower()
        calibrated_conf = float(manager_final_card.get("calibrated_confidence", 0.5))
        risk_level = str(risk_alert.get("level", "normal")).lower()

        if action == "buy":
            direction = "BUY"
        elif action in {"sell", "reduce_risk"}:
            direction = "SELL"
        else:
            direction = "HOLD"

        if risk_level in {"high", "critical"} and direction == "SELL":
            urgency = "High"
        elif calibrated_conf >= 0.75:
            urgency = "High"
        elif calibrated_conf >= 0.45:
            urgency = "Medium"
        else:
            urgency = "Low"

        reason = (
            f"manager_action={action}; signal={float(manager_final_card.get('aggregated_signal', 0.0)):.3f}; "
            f"calibrated_confidence={calibrated_conf:.3f}; risk_level={risk_level}"
        )
        return TradeIntent(direction=direction, urgency=urgency, reason=reason)

    async def generate_trading_decision(self, market_data: Dict[str, Any], retrieved_context: str) -> TradeAction:
        """Generate orders from structured evidence flow with committee governance."""
        logger.debug(f"[{self.agent_id}] running structured evidence flow")

        macro_context = MacroContextDTO.coerce(market_data.get("macro_context"))
        market_payload = dict(market_data)
        if macro_context is not None:
            market_payload["macro_context"] = macro_context.to_payload()

        panic_level = float(market_payload.get("panic_level", 0.0) or 0.0)
        pnl_pct = float(market_payload.get("pnl_pct", 0.0) or 0.0)

        analyst_cards = self._build_structured_analyst_cards(market_payload, macro_context, retrieved_context)
        valid_cards: List[Dict[str, Any]] = []
        for idx, card in enumerate(analyst_cards):
            is_valid, _ = validate_analyst_card(card)
            if is_valid:
                valid_cards.append(card)
            else:
                valid_cards.append(coerce_analyst_card({}, analyst_id=f"{self.agent_id}_fallback_{idx}"))

        risk_metrics = self._build_risk_metrics(market_payload, panic_level, pnl_pct)
        risk_alert = self.risk_committee.assess(risk_metrics).to_dict()

        policy_text = str(
            market_payload.get("policy_description")
            or (macro_context.policy_summary if macro_context is not None else "")
            or market_payload.get("news", "")
        )
        policy_conditions = self.policy_committee.translate(policy_text)

        manager_final_card = aggregate_analyst_cards(valid_cards, risk_alert=risk_alert)
        manager_intent = self._manager_card_to_intent(manager_final_card, risk_alert)
        action_decision = self._intent_to_order(manager_intent, market_payload)

        calibration = manager_final_card.get("calibration", {})
        contradiction = manager_final_card.get("contradiction_matrix", {})
        contradiction_idx = float(contradiction.get("contradiction_index", 0.0)) if isinstance(contradiction, dict) else 0.0

        trace_payload = {
            "agent_id": self.agent_id,
            "timestamp": time.time(),
            "analyst_cards": valid_cards,
            "manager_final_card": manager_final_card,
            "risk_alerts": risk_alert,
            "policy_conditions": policy_conditions,
            "market_data": market_payload,
            "retrieved_context": str(retrieved_context),
            "decision": action_decision.model_dump(),
        }
        trace_path = export_decision_trace(trace_payload)

        self.last_decision_artifacts = {
            "analyst_cards": valid_cards,
            "manager_final_card": manager_final_card,
            "risk_alerts": risk_alert,
            "policy_conditions": policy_conditions,
            "decision_trace_path": trace_path,
        }

        structured_summary = (
            f"manager={manager_final_card.get('recommended_action', 'hold')}; "
            f"conf={float(manager_final_card.get('calibrated_confidence', 0.0)):.3f}; "
            f"contradiction={contradiction_idx:.3f}; "
            f"brier_like={float(calibration.get('brier_like_score', 0.0)):.4f}; "
            f"confidence_drift={float(calibration.get('confidence_drift', 0.0)):.4f}; "
            f"trace={trace_path}"
        )
        action_decision.reasoning_chain = f"{action_decision.reasoning_chain} | {structured_summary}"
        return action_decision


# ----------------------------------------------------------------------------
# 4. 工厂模式设计：智能体工厂
# ----------------------------------------------------------------------------
class AgentFactory:
    """
    负责批量、可控、具统计学意义地创建多种类型交易代理的工厂类。
    
    在启动 Civitas 时，工厂通过调整 Persona 参数的联合分布，生成具备社会分工差异化的人口基础。
    """

    @staticmethod
    def create_diverse_population(num_agents: int) -> List[TradingAgent]:
        """
        生成在统计上逼真的群体。
        例如设定为:
        - 60% 高神经质的噪音交易者 (High-neuroticism noise traders)
        - 30% 低神经质的基本面分析师 (Low-neuroticism fundamental analysts)
        - 10% 宏观对冲基金 (Macro hedge funds)

        Args:
            num_agents (int): 被生成并注入系统仿真空间的总人数。

        Returns:
            List[TradingAgent]: 多样化的并配备不同有效 Persona 心理的 Agent 列表。
        """
        population: List[TradingAgent] = []
        
        count_noise_traders = int(num_agents * 0.6)
        count_fundamental = int(num_agents * 0.3)
        count_hedge = num_agents - count_noise_traders - count_fundamental  # 剩下的10%

        idx = 1
        
        # 1. 产生 60% 的噪音交易员 (高风险倾向，容易跟风或过度自信，短视，受政策影响敏感)
        for _ in range(count_noise_traders):
            persona = Persona(
                risk_tolerance=0.8,
                cognitive_bias=CognitiveBias.Herding,  # 或 Overconfidence
                investment_horizon=InvestmentHorizon.Short_term,
                policy_sensitivity=0.9
            )
            population.append(TradingAgent(agent_id=f"NoiseTrader_{idx}", persona=persona))
            idx += 1
            
        # 2. 产生 30% 的基本面分析师 (低风险/适度风险倾向，理性分析为主，长视，对短期政策敏感较低)
        for _ in range(count_fundamental):
            persona = Persona(
                risk_tolerance=0.3,
                cognitive_bias=CognitiveBias.Rational,
                investment_horizon=InvestmentHorizon.Long_term,
                policy_sensitivity=0.3
            )
            population.append(TradingAgent(agent_id=f"FundamentalAnalyst_{idx}", persona=persona))
            idx += 1
            
        # 3. 产生 10% 宏观对冲基金 (高风险，极其关注政策面，损失厌恶用于风控保护)
        for _ in range(count_hedge):
            persona = Persona(
                risk_tolerance=0.7,
                cognitive_bias=CognitiveBias.Loss_Aversion,
                investment_horizon=InvestmentHorizon.Medium_term,
                policy_sensitivity=0.95
            )
            population.append(TradingAgent(agent_id=f"MacroHedgeFund_{idx}", persona=persona))
            idx += 1

        logger.info(f"成功生成了 {len(population)} 名在统计学上呈特化分布的智能体 (Agents)!")
        return population
