"""
Mesa Agent 适配器

将 TraderAgent (BaseAgent) 包装为 Mesa Agent，实现与 Mesa 调度器的兼容。
"""

from __future__ import annotations

from mesa import Agent
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING

from agents.base_agent import BaseAgent, MarketSnapshot
from agents.trader_agent import TraderAgent
from agents.cognition.utility import InvestorType
from agents.persona import Persona
from config import GLOBAL_CONFIG
from core.types import Order

if TYPE_CHECKING:
    from core.mesa.civitas_model import CivitasModel

class CivitasAgent(Agent):
    """
    Civitas Mesa Agent (Adapter)
    
    作为 BaseAgent/TraderAgent 的 Mesa 包装器。
    
    Composition:
        self.core: BaseAgent (实际的智能体)
    """

    def __init__(
        self,
        model: "CivitasModel",
        investor_type: InvestorType = InvestorType.NORMAL,
        initial_cash: float = 100000.0,
        use_local_reasoner: bool = True,
        persona: Optional[Persona] = None,
        core_agent: Optional[BaseAgent] = None,
        model_router: Optional[Any] = None,
        use_llm: bool = True,
        model_priority: Optional[List[str]] = None
    ):
        """
        初始化 Mesa Agent
        
        Args:
            model: Mesa Model 实例
            investor_type: 投资者类型 (决定心理画像)
            initial_cash: 初始资金
            use_local_reasoner: 是否使用本地推理器 (已废弃，由 TraderAgent 内部处理)
            persona: 智能体人格 (可选)
            core_agent: 预先创建的核心 Agent 实例 (可选)
            model_router: 模型路由器 (用于 LLM 调用)
        """
        super().__init__(model)
        self.investor_type = investor_type
        
        if core_agent:
            self.core = core_agent
        else:
            # 根据投资者类型生成心理画像
            profile = self._get_profile_by_type(investor_type)
            
            # 初始化内核 智能体
            # Mesa 的 unique_id 可能是 int
            self.core: TraderAgent = TraderAgent(
                agent_id=str(self.unique_id),
                cash_balance=initial_cash,
                portfolio={},
                psychology_profile=profile,
                persona=persona,
                model_router=model_router,
                use_llm=use_llm,
                model_priority=model_priority
            )
        
        # 待执行动作（用于情绪推断和 DataCollector）
        self.pending_action: Optional[Dict[str, Any]] = None
        self._portfolio_override: Optional[Any] = None
        
        # 恢复成本跟踪 (用于计算 PnL)
        self.avg_cost: float = 0.0

    def _get_profile_by_type(self, investor_type: InvestorType) -> Dict[str, float]:
        """根据类型获取心理画像"""
        if investor_type == InvestorType.PANIC_RETAIL:
            return {
                "risk_aversion": 0.8,
                "confidence_level": 0.3,
                "attention_span": 2.0,
                "loss_sensitivity": 2.5
            }
        elif investor_type == InvestorType.DISCIPLINED_QUANT:
            return {
                "risk_aversion": 0.2,
                "confidence_level": 0.8,
                "attention_span": 10.0,
                "loss_sensitivity": 1.0
            }
        else:
            return {
                "risk_aversion": 0.5,
                "confidence_level": 0.5,
                "attention_span": 3.0,
                "loss_sensitivity": 1.5
            }

    @property
    def cash(self) -> float:
        if self._portfolio_override is not None and hasattr(self._portfolio_override, "available_cash"):
            return float(getattr(self._portfolio_override, "available_cash", 0.0) or 0.0)
        return self.core.cash_balance
        
    @property
    def portfolio(self) -> Any:
        if self._portfolio_override is not None:
            return self._portfolio_override
        return self.core.portfolio

    @portfolio.setter
    def portfolio(self, value: Any) -> None:
        self._portfolio_override = value

    @property
    def position(self) -> int:
        portfolio_obj = self.portfolio
        if hasattr(portfolio_obj, "positions"):
            positions = getattr(portfolio_obj, "positions", {}) or {}
            if isinstance(positions, dict):
                return int(sum(positions.values()))
        if isinstance(portfolio_obj, dict):
            return int(sum(portfolio_obj.values()))
        return 0

    @property
    def sentiment(self) -> float:
        """获取 Agent 情绪值 (-1.0 ~ 1.0)，基于最近决策推断"""
        if self.pending_action is None:
            return 0.0
        action = self.pending_action.get("action", "HOLD")
        if action == "BUY":
            return 0.5
        elif action == "SELL":
            return -0.5
        return 0.0

    @property
    def confidence(self) -> float:
        profile = getattr(self.core, "profile", getattr(self.core, "psychology_profile", {}))
        return profile.get("confidence_level", 0.5)

    @property
    def wealth(self) -> float:

        price = self.model.current_price if hasattr(self.model, "current_price") else 3000.0
        return self.core.cash_balance + (self.position * price)
        
    @property
    def pnl_pct(self) -> float:
        """计算持仓盈亏比例"""
        if self.position == 0 or self.avg_cost == 0:
            return 0.0
        
        current_price = self.model.current_price if hasattr(self.model, "current_price") else 3000.0
        return (current_price - self.avg_cost) / self.avg_cost

    @property
    def id(self) -> str:
        return self.core.agent_id



    async def async_act(self, market_snapshot: MarketSnapshot, news: List[str]) -> Optional[Order]:
        """
        异步与内核交互 (委托给 TraderAgent.act)
        """
        order = await self.core.act(market_snapshot, news)
        
        # 兼容性: 设置 pending_action 供 Model 计算 sentiment
        if order:
            self.pending_action = {
                "action": "BUY" if order.side == "buy" else "SELL",
                "quantity": order.quantity,
                "price": order.price
            }
        else:
            self.pending_action = None
            
        return order



    async def prepare_decision(self, market_state: Dict) -> Tuple[Optional[List[Dict]], Dict]:
        account_state = self._build_account_state(market_state)
        cognitive = getattr(self, "cognitive_agent", None)
        if cognitive and hasattr(cognitive, "prepare_decision_prompt_async"):
            prompt = await cognitive.prepare_decision_prompt_async(market_state, account_state)
            return prompt, account_state
        return None, account_state

    def _build_account_state(self, market_state: Dict) -> Dict[str, Any]:
        portfolio_obj = self.portfolio
        cash = self.cash
        position = 0
        avg_cost = self.avg_cost

        if hasattr(portfolio_obj, "positions"):
            positions = getattr(portfolio_obj, "positions", {}) or {}
            if isinstance(positions, dict):
                position = int(sum(positions.values()))
            avg_cost = float(getattr(portfolio_obj, "avg_cost", avg_cost) or avg_cost)
        elif isinstance(portfolio_obj, dict):
            position = int(sum(portfolio_obj.values()))

        current_price = float(market_state.get("price", getattr(self.model, "current_price", 0.0)) or 0.0)
        market_value = float(max(0.0, position * current_price))
        pnl_pct = (current_price - avg_cost) / avg_cost if avg_cost and position > 0 else 0.0

        return {
            "cash": cash,
            "position": position,
            "market_value": market_value,
            "avg_cost": float(avg_cost or 0.0),
            "pnl_pct": float(pnl_pct),
        }

    def finalize_decision(self, result: Any, market_state: Dict, account_state: Dict) -> None:
        pass

    def step(self):
        pass

    def advance(self):
        pass 

    def _process_trades(self, trades: list):
        """处理成交记录 (回调)"""
        if not trades:
            return
            
        for trade in trades:
            symbol = "SH000001"
            qty = trade.quantity
            price = trade.price
            
            # 检查 Trade 对象属性 (使用 vars() 或 dir() 确认，这里基于 logs 修正)

            
            # 使用 getattr 安全获取，防止 AttributeError
            buyer = getattr(trade, 'buyer_agent_id', getattr(trade, 'buy_agent_id', None))
            seller = getattr(trade, 'seller_agent_id', getattr(trade, 'sell_agent_id', None))
            
            if buyer == self.id:

                 cost = price * qty * (1 + GLOBAL_CONFIG.TAX_RATE_COMMISSION)
                 
                 # 更新成本价 (加权平均)
                 old_pos = self.core.portfolio.get(symbol, 0)
                 new_pos = old_pos + qty
                 if new_pos > 0:
                     total_cost = (old_pos * self.avg_cost) + (qty * price * (1 + GLOBAL_CONFIG.TAX_RATE_COMMISSION))
                     self.avg_cost = total_cost / new_pos
                 
                 self.core.cash_balance -= cost
                 self.core.portfolio[symbol] = new_pos
                 
            elif seller == self.id:

                 revenue = price * qty * (1 - (GLOBAL_CONFIG.TAX_RATE_COMMISSION + GLOBAL_CONFIG.TAX_RATE_STAMP))
                 
                 # 卖出不影响平均成本，只减少持仓
                 old_pos = self.core.portfolio.get(symbol, 0)
                 new_pos = max(0, old_pos - qty)
                 
                 self.core.cash_balance += revenue
                 self.core.portfolio[symbol] = new_pos
                 
                 if new_pos == 0:
                     self.avg_cost = 0.0

            # 记录记忆 (简单模拟 outcomes)
            outcome = {"status": "FILLED", "price": price, "qty": qty}
            self.core.memory.append({
                "timestamp": 0,
                "decision": "EXECUTED",
                "outcome": outcome
            })

    def get_state(self) -> Dict[str, Any]:
        """
        获取可序列化的状态 (用于 DataCollector)
        """
        return {
            "AgentID": self.id,
            "Cash": self.core.cash_balance,
            "Position": self.position,
            "Wealth": self.wealth, 
            "RiskAversion": self.core.profile.get("risk_aversion", 0.5),
            "Confidence": self.core.profile.get("confidence_level", 0.5),
            "Sentiment": self.sentiment
        }
