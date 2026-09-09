"""
投资团队编排模块

灵感来源:
- CrewAI: 多 Agent 协作框架
- TradingAgents: 投资决策工作流

实现:
1. InvestmentTeam: 团队编排器
2. RiskManager: 风险管理 Agent
3. Trader: 交易执行 Agent
4. Debate 机制: 信号冲突解决

工作流:
Analyst Agents → Signal → Risk Manager → Trader → Order

作者: Civitas Economica Team
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from agents.roles.analyst import (
    Analyst,
    FundamentalAnalyst,
    TechnicalAnalyst,
    SentimentAnalyst,
    Signal,
    SignalType,
    AnalystType,
)


# ==========================================
# 数据结构
# ==========================================

class RiskCheckResult(str, Enum):
    """风控检查结果"""
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


@dataclass
class RiskConstraints:
    """风控约束条件"""
    max_position_pct: float = 0.3      # 单只股票最大仓位
    max_leverage: float = 1.0          # 最大杠杆 (1.0 = 无杠杆)
    max_daily_loss_pct: float = 0.05   # 单日最大亏损
    min_cash_pct: float = 0.1          # 最低现金比例
    max_order_value: float = 100000.0  # 单笔订单最大金额


@dataclass
class Order:
    """交易订单"""
    symbol: str
    action: str
    quantity: int
    price: float
    order_type: str = "LIMIT"
    status: str = "pending"
    
    @property
    def value(self) -> float:
        return self.quantity * self.price


@dataclass
class TeamDecision:
    """
    团队决策结果
    
    包含各分析师信号、冲突解决过程、风控结果和最终订单。
    """
    # 分析师信号
    analyst_signals: List[Signal] = field(default_factory=list)
    
    # 共识信号
    consensus_action: SignalType = SignalType.HOLD
    consensus_confidence: float = 0.5
    
    # 冲突解决
    had_conflict: bool = False
    debate_log: List[str] = field(default_factory=list)
    
    # 风控结果
    risk_check: RiskCheckResult = RiskCheckResult.APPROVED
    risk_message: str = ""
    
    # 最终订单
    final_order: Optional[Order] = None
    
    # 元数据
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "signals": [s.to_dict() for s in self.analyst_signals],
            "consensus": {
                "action": self.consensus_action.value,
                "confidence": self.consensus_confidence
            },
            "conflict": {
                "had_conflict": self.had_conflict,
                "debate_log": self.debate_log
            },
            "risk": {
                "result": self.risk_check.value,
                "message": self.risk_message
            },
            "order": self.final_order.__dict__ if self.final_order else None
        }


# ==========================================
# 风险管理 智能体
# ==========================================

class RiskManager:
    """
    风险管理 Agent
    
    职责:
    1. 检查交易是否符合风控约束
    2. 计算合适的仓位大小
    3. 拒绝或修改高风险交易
    """
    
    def __init__(
        self,
        constraints: RiskConstraints = None,
        agent_id: str = "risk_manager"
    ):
        self.agent_id = agent_id
        self.constraints = constraints or RiskConstraints()
    
    def check(
        self,
        signal: Signal,
        account_state: Dict,
        current_price: float
    ) -> Tuple[RiskCheckResult, str, Optional[int]]:
        """
        风控检查
        
        Args:
            signal: 交易信号
            account_state: 账户状态
            current_price: 当前价格
            
        Returns:
            (result, message, adjusted_quantity)
        """
        cash = account_state.get('cash', 0)
        holdings = account_state.get('holdings', 0)
        total_value = cash + holdings * current_price
        daily_pnl = account_state.get('daily_pnl_pct', 0)
        
        # 检查 1: 单日亏损限制
        if daily_pnl < -self.constraints.max_daily_loss_pct:
            return (
                RiskCheckResult.REJECTED,
                f"单日亏损 {daily_pnl:.2%} 已超过限制 {self.constraints.max_daily_loss_pct:.2%}，禁止交易",
                None
            )
        
        # 检查 2: 买入时的资金检查
        if signal.action == SignalType.BUY:
            # 计算最大可买数量
            available_cash = cash - total_value * self.constraints.min_cash_pct
            max_position_value = total_value * self.constraints.max_position_pct
            current_position_value = holdings * current_price
            
            # 限制 1: 可用资金
            max_by_cash = int(available_cash / current_price / 100) * 100
            
            # 限制 2: 仓位上限
            remaining_allocation = max_position_value - current_position_value
            max_by_position = int(remaining_allocation / current_price / 100) * 100
            
            # 限制 3: 单笔金额
            max_by_order = int(self.constraints.max_order_value / current_price / 100) * 100
            
            # 取最小值
            max_quantity = max(0, min(max_by_cash, max_by_position, max_by_order))
            
            if max_quantity == 0:
                return (
                    RiskCheckResult.REJECTED,
                    "资金不足或仓位已满，无法买入",
                    None
                )
            
            return (
                RiskCheckResult.APPROVED,
                f"买入通过风控，建议数量 {max_quantity}",
                max_quantity
            )
        
        # 检查 3: 卖出时的持仓检查
        elif signal.action == SignalType.SELL:
            if holdings <= 0:
                return (
                    RiskCheckResult.REJECTED,
                    "无持仓可卖出",
                    None
                )
            
            # 分批卖出 (最多卖 50%)
            suggested_qty = min(holdings, int(holdings * 0.5 / 100) * 100)
            suggested_qty = max(100, suggested_qty)
            
            return (
                RiskCheckResult.APPROVED,
                f"卖出通过风控，建议数量 {suggested_qty}",
                suggested_qty
            )
        

        return (
            RiskCheckResult.APPROVED,
            "观望，无需交易",
            0
        )


# ==========================================
# 交易执行 智能体
# ==========================================

class Trader:
    """
    交易执行 Agent
    
    职责:
    1. 将信号转换为订单
    2. 执行订单 (与交易所对接)
    3. 记录交易日志
    """
    
    def __init__(self, agent_id: str = "trader"):
        self.agent_id = agent_id
        self.order_history: List[Order] = []
    
    def create_order(
        self,
        signal: Signal,
        quantity: int,
        price: float
    ) -> Order:
        """创建订单"""
        if signal.action == SignalType.HOLD or quantity <= 0:
            return None
        
        order = Order(
            symbol=signal.symbol,
            action=signal.action.value,
            quantity=quantity,
            price=price
        )
        
        self.order_history.append(order)
        return order
    
    def execute(self, order: Order) -> bool:
        """
        执行订单 (模拟)
        
        在实际系统中，这里应该与交易所 API 对接
        """
        if order is None:
            return False
        
        order.status = "executed"
        return True


# ==========================================
# 投资团队编排器
# ==========================================

class InvestmentTeam:
    """
    投资团队编排器 (CrewAI 风格)
    
    整合分析师、风险管理和交易执行，实现完整决策流程。
    
    工作流:
    1. 各分析师生成信号
    2. 检测信号冲突，必要时启动 Debate
    3. 风险管理检查约束
    4. 交易员执行订单
    
    Attributes:
        analysts: 分析师列表
        risk_manager: 风险管理 Agent
        trader: 交易执行 Agent
    """
    
    def __init__(
        self,
        analysts: List[Analyst] = None,
        risk_manager: RiskManager = None,
        trader: Trader = None
    ):
        """
        初始化投资团队
        
        Args:
            analysts: 分析师列表 (默认创建三大分析师)
            risk_manager: 风险管理 (默认使用默认配置)
            trader: 交易员
        """
        self.analysts = analysts or [
            FundamentalAnalyst(),
            TechnicalAnalyst(),
            SentimentAnalyst()
        ]
        
        self.risk_manager = risk_manager or RiskManager()
        self.trader = trader or Trader()
        
        # 决策历史
        self.decision_history: List[TeamDecision] = []
    
    def decide(
        self,
        symbol: str,
        market_data: Dict,
        account_state: Dict
    ) -> TeamDecision:
        """
        执行团队决策流程
        
        Args:
            symbol: 交易标的
            market_data: 市场数据
            account_state: 账户状态
            
        Returns:
            TeamDecision: 完整决策结果
        """
        decision = TeamDecision()
        current_price = market_data.get('price', 100)
        
        # ========== 阶段 1: 收集分析师信号 ==========
        for analyst in self.analysts:
            signal = analyst.analyze(symbol, market_data)
            decision.analyst_signals.append(signal)
        
        # ========== 阶段 2: 检测冲突并解决 ==========
        actions = [s.action for s in decision.analyst_signals]
        
        if self._has_conflict(actions):
            decision.had_conflict = True
            consensus, debate_log = self._resolve_conflict(decision.analyst_signals)
            decision.consensus_action = consensus
            decision.debate_log = debate_log
        else:
            # 无冲突，取多数
            decision.consensus_action = self._majority_vote(actions)
        
        # 计算共识信心
        matching_signals = [s for s in decision.analyst_signals
                          if s.action == decision.consensus_action]
        if matching_signals:
            decision.consensus_confidence = sum(s.confidence for s in matching_signals) / len(matching_signals)
        
        # ========== 阶段 3: 风控检查 ==========
        consensus_signal = Signal(
            action=decision.consensus_action,
            confidence=decision.consensus_confidence,
            analyst_type=AnalystType.FUNDAMENTAL,
            symbol=symbol
        )
        
        risk_result, risk_msg, quantity = self.risk_manager.check(
            consensus_signal, account_state, current_price
        )
        
        decision.risk_check = risk_result
        decision.risk_message = risk_msg
        
        # ========== 阶段 4: 创建订单 ==========
        if risk_result == RiskCheckResult.APPROVED and quantity and quantity > 0:
            decision.final_order = self.trader.create_order(
                consensus_signal, quantity, current_price
            )
        
        # 记录历史
        self.decision_history.append(decision)
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)
        
        return decision
    
    def _has_conflict(self, actions: List[SignalType]) -> bool:
        """检测是否存在信号冲突"""
        buy_count = sum(1 for a in actions if a == SignalType.BUY)
        sell_count = sum(1 for a in actions if a == SignalType.SELL)
        
        # 买入 和 卖出 同时存在 = 冲突
        return buy_count > 0 and sell_count > 0
    
    def _majority_vote(self, actions: List[SignalType]) -> SignalType:
        """多数投票"""
        buy_count = sum(1 for a in actions if a == SignalType.BUY)
        sell_count = sum(1 for a in actions if a == SignalType.SELL)
        hold_count = sum(1 for a in actions if a == SignalType.HOLD)
        
        if buy_count > sell_count and buy_count > hold_count:
            return SignalType.BUY
        elif sell_count > buy_count and sell_count > hold_count:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _resolve_conflict(
        self,
        signals: List[Signal]
    ) -> Tuple[SignalType, List[str]]:
        """
        Debate 机制: 解决信号冲突
        
        当分析师意见分歧时，进行多轮辩论直至达成共识。
        
        Args:
            signals: 各分析师信号
            
        Returns:
            (consensus_action, debate_log)
        """
        debate_log = []
        debate_log.append("=== 启动 Debate 机制 ===")
        
        # 分组
        buy_signals = [s for s in signals if s.action == SignalType.BUY]
        sell_signals = [s for s in signals if s.action == SignalType.SELL]
        
        # 记录各方观点
        for s in buy_signals:
            debate_log.append(f"[{s.analyst_type.value}] 看涨: {s.reasoning}")
        for s in sell_signals:
            debate_log.append(f"[{s.analyst_type.value}] 看跌: {s.reasoning}")
        
        # 第一轮: 加权投票 (按信心)
        buy_weight = sum(s.confidence for s in buy_signals)
        sell_weight = sum(s.confidence for s in sell_signals)
        
        debate_log.append(f"[Round 1] 加权投票: 看涨 {buy_weight:.2f} vs 看跌 {sell_weight:.2f}")
        
        if abs(buy_weight - sell_weight) > 0.3:
            # 差距明显，直接决定
            if buy_weight > sell_weight:
                debate_log.append("结论: 看涨信号更强，决定 BUY")
                return SignalType.BUY, debate_log
            else:
                debate_log.append("结论: 看跌信号更强，决定 SELL")
                return SignalType.SELL, debate_log
        
        # 第二轮: 优先级仲裁
        debate_log.append("[Round 2] 信号接近，启动优先级仲裁")
        
        # 优先级: 技术面 > 情绪面 > 基本面 (短线视角)
        priority_order = [AnalystType.TECHNICAL, AnalystType.SENTIMENT, AnalystType.FUNDAMENTAL]
        
        for priority_type in priority_order:
            for s in signals:
                if s.analyst_type == priority_type and s.action != SignalType.HOLD:
                    debate_log.append(f"仲裁: {priority_type.value} 优先级最高，采纳其信号 {s.action.value}")
                    return s.action, debate_log
        
        # 第三轮: 僵局，保守处理
        debate_log.append("[Round 3] 无法达成共识，保守选择 HOLD")
        return SignalType.HOLD, debate_log
    
    def query_analyst(
        self,
        analyst_type: AnalystType,
        question: str
    ) -> str:
        """
        Trader 向特定分析师提问
        
        用于 Debate 过程中的迭代询问。
        """
        for analyst in self.analysts:
            if analyst.analyst_type == analyst_type:
                # 简化实现：返回最近的分析理由
                return f"[{analyst_type.value}] 回复: 请参考我的分析报告"
        return "分析师未找到"
    
    def get_team_status(self) -> Dict:
        """获取团队状态"""
        return {
            "analysts": [a.__repr__() for a in self.analysts],
            "risk_constraints": self.risk_manager.constraints.__dict__,
            "total_decisions": len(self.decision_history),
            "recent_orders": len(self.trader.order_history)
        }


# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    import numpy as np
    
    print("=" * 60)
    print("投资团队编排测试")
    print("=" * 60)
    
    # 创建团队
    team = InvestmentTeam()
    
    # 模拟市场数据
    market = {
        "price": 100.0,
        "prices": [100 + np.sin(i/5) * 5 for i in range(60)],
        "volatility": 0.03,
        "volume_ratio": 1.5,
        "momentum": -0.1,
        "panic_level": 0.65,
        "news": ["央行降准利好股市", "外资持续流入"]
    }
    
    # 模拟账户状态
    account = {
        "cash": 100000.0,
        "holdings": 500,
        "daily_pnl_pct": -0.02
    }
    
    # 执行决策
    decision = team.decide("000001", market, account)
    
    print("\n[分析师信号]")
    for signal in decision.analyst_signals:
        print(f"  {signal.analyst_type.value}: {signal.action.value} (信心 {signal.confidence:.2f})")
    
    print(f"\n[冲突检测] 存在冲突: {decision.had_conflict}")
    if decision.debate_log:
        print("[Debate 日志]")
        for log in decision.debate_log:
            print(f"  {log}")
    
    print(f"\n[共识结果] {decision.consensus_action.value} (信心 {decision.consensus_confidence:.2f})")
    
    print(f"\n[风控检查] {decision.risk_check.value}")
    print(f"  消息: {decision.risk_message}")
    
    if decision.final_order:
        print(f"\n[最终订单]")
        print(f"  {decision.final_order.action} {decision.final_order.quantity} @ {decision.final_order.price}")
    else:
        print("\n[最终订单] 无")
    
    print("\n[团队状态]")
    status = team.get_team_status()
    print(f"  分析师: {len(status['analysts'])}")
    print(f"  历史决策: {status['total_decisions']}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
