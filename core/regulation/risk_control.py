"""
风险控制引擎

实现监管压力测试核心机制：
1. 高频交易监控 (Order-to-Trade Ratio)
2. 杠杆账户与强制平仓 (Margin Call)
3. 交易税和费用计算

参考:
- 《证券市场程序化交易管理规定》(2024)
- 融资融券业务管理办法

作者: Civitas Economica Team
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from enum import Enum

import pandas as pd
import numpy as np

from core.account import Portfolio, PositionBatch
from config import GLOBAL_CONFIG
from core.types import Order, OrderSide, OrderType


# ==========================================
# 数据结构
# ==========================================

class ViolationType(str, Enum):
    """违规类型"""
    HIGH_OTR = "high_order_to_trade_ratio"
    EXCESSIVE_CANCEL = "excessive_cancellation"
    SPOOFING = "spoofing"
    LAYERING = "layering"


@dataclass
class HFTViolation:
    """高频交易违规记录"""
    agent_id: str
    violation_type: ViolationType
    otr_value: float
    penalty_rate: float
    timestamp: float = field(default_factory=time.time)
    blocked: bool = False


@dataclass
class MarginCallEvent:
    """保证金追缴事件"""
    agent_id: str
    ticker: str
    equity: float
    borrowed: float
    margin_level: float
    required_margin: float
    shortfall: float
    liquidation_qty: int
    liquidation_price: float
    timestamp: float = field(default_factory=time.time)






# ==========================================
# 高频交易监控器
# ==========================================

class HighFrequencyMonitor:
    """
    高频交易监控器
    
    追踪每个 Agent 的订单成交比 (Order-to-Trade Ratio)，
    超过阈值时施加惩罚或阻止订单。
    
    监管依据:
    - OTR > 10: 警告
    - OTR > 20: 加收流动性费用
    - OTR > 50: 暂停交易资格
    
    Attributes:
        otr_threshold: OTR 阈值
        penalty_base_rate: 基础惩罚费率
        block_threshold: 阻止订单的 OTR 阈值
    """
    
    def __init__(
        self,
        otr_threshold: float = 10.0,
        penalty_base_rate: float = 0.001,
        block_threshold: float = 50.0
    ):
        self.otr_threshold = otr_threshold
        self.penalty_base_rate = penalty_base_rate
        self.block_threshold = block_threshold
        
        # 统计数据
        self.agent_orders: Dict[str, int] = defaultdict(int)
        self.agent_trades: Dict[str, int] = defaultdict(int)
        self.agent_cancels: Dict[str, int] = defaultdict(int)
        
        # 违规记录
        self.violations: List[HFTViolation] = []
        
        # 累计惩罚
        self.total_penalties: Dict[str, float] = defaultdict(float)
    
    def register_order(self, agent_id: str) -> None:
        """注册订单"""
        self.agent_orders[agent_id] += 1
    
    def register_trade(self, agent_id: str) -> None:
        """注册成交"""
        self.agent_trades[agent_id] += 1
    
    def register_cancel(self, agent_id: str) -> None:
        """注册撤单"""
        self.agent_cancels[agent_id] += 1
    
    def get_otr(self, agent_id: str) -> float:
        """计算 Order-to-Trade Ratio"""
        orders = self.agent_orders.get(agent_id, 0)
        trades = self.agent_trades.get(agent_id, 0)
        
        if trades == 0:
            return float('inf') if orders > 0 else 0.0
        
        return orders / trades
    
    def check_order(
        self,
        agent_id: str,
        order_value: float = 0.0
    ) -> Tuple[bool, float, Optional[str]]:
        """
        检查订单合规性
        
        Args:
            agent_id: Agent ID
            order_value: 订单金额 (用于计算惩罚)
            
        Returns:
            (是否允许, 惩罚费用, 拒绝原因)
        """
        otr = self.get_otr(agent_id)
        
        # 阻止订单
        if otr > self.block_threshold:
            violation = HFTViolation(
                agent_id=agent_id,
                violation_type=ViolationType.HIGH_OTR,
                otr_value=otr,
                penalty_rate=0.0,
                blocked=True
            )
            self.violations.append(violation)
            return False, 0.0, f"OTR {otr:.1f} 超过阻止阈值 {self.block_threshold}"
        
        # 施加惩罚
        if otr > self.otr_threshold:
            # 惩罚率随 OTR 增加而增加
            penalty_rate = self.penalty_base_rate * (otr / self.otr_threshold)
            penalty = order_value * penalty_rate
            
            violation = HFTViolation(
                agent_id=agent_id,
                violation_type=ViolationType.HIGH_OTR,
                otr_value=otr,
                penalty_rate=penalty_rate,
                blocked=False
            )
            self.violations.append(violation)
            
            self.total_penalties[agent_id] += penalty
            return True, penalty, None
        
        return True, 0.0, None
    
    def reset_daily(self) -> None:
        """每日重置统计"""
        self.agent_orders.clear()
        self.agent_trades.clear()
        self.agent_cancels.clear()
    
    def get_stats(self, agent_id: str) -> Dict:
        """获取 Agent 统计"""
        return {
            "orders": self.agent_orders.get(agent_id, 0),
            "trades": self.agent_trades.get(agent_id, 0),
            "cancels": self.agent_cancels.get(agent_id, 0),
            "otr": self.get_otr(agent_id),
            "total_penalty": self.total_penalties.get(agent_id, 0.0)
        }
    
    def get_violation_report(self) -> Dict:
        """获取违规报告"""
        return {
            "total_violations": len(self.violations),
            "blocked_count": sum(1 for v in self.violations if v.blocked),
            "total_penalties": sum(self.total_penalties.values()),
            "worst_offenders": sorted(
                [(k, self.get_otr(k)) for k in self.agent_orders.keys()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


# ==========================================
# 杠杆账户
# ==========================================

class MarginAccount(Portfolio):
    """
    保证金账户 - 支持杠杆交易
    
    继承自 Portfolio，增加融资融券功能。
    
    关键指标:
    - Equity = Cash + Market Value - Borrowed
    - Margin Level = Equity / (Borrowed + Market Value)
    - 维持保证金率 = 25% (A股融资融券最低标准)
    
    强制平仓机制:
    当 Margin Level < Maintenance Margin 时:
    1. 发出保证金追缴通知
    2. 若未在 T+1 内补足，强制市价卖出
    
    Attributes:
        leverage_limit: 最大杠杆倍数
        borrowed: 借入金额
        maintenance_margin: 维持保证金率
        initial_margin: 初始保证金率
    """
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        leverage_limit: float = 2.0,
        maintenance_margin: float = 0.25,
        initial_margin: float = 0.50
    ):
        """
        初始化保证金账户
        
        Args:
            initial_cash: 初始资金
            leverage_limit: 最大杠杆 (2.0 = 2倍杠杆)
            maintenance_margin: 维持保证金率 (0.25 = 25%)
            initial_margin: 初始保证金率 (0.50 = 50%)
        """
        super().__init__(initial_cash)
        
        self.leverage_limit = leverage_limit
        self.borrowed = 0.0
        self.maintenance_margin = maintenance_margin
        self.initial_margin = initial_margin
        
        # 记录初始权益 (用于计算最大借款限额)
        self._initial_equity = float(initial_cash)
        
        # 保证金追缴历史
        self.margin_call_history: List[MarginCallEvent] = []
        
        # 利息 (年化)
        self.interest_rate = 0.065  # 6.5%
        self.accrued_interest = 0.0
    
    def get_equity(self, current_prices: Dict[str, float]) -> float:
        """
        计算权益
        
        Equity = Cash + Market Value - Borrowed - Accrued Interest
        """
        market_value = self.get_market_value(current_prices)
        return self.available_cash + market_value - self.borrowed - self.accrued_interest
    
    def get_margin_level(self, current_prices: Dict[str, float]) -> float:
        """
        计算保证金水平
        
        Margin Level = Equity / Total Exposure
        其中 Total Exposure = Market Value (多头) 或 Borrowed (无持仓时)
        """
        market_value = self.get_market_value(current_prices)
        equity = self.get_equity(current_prices)
        
        total_exposure = max(market_value, self.borrowed)
        
        if total_exposure == 0:
            return 1.0  # 无杠杆，安全
        
        return equity / total_exposure
    
    def get_buying_power(self) -> float:
        """
        计算购买力
        
        Buying Power = (Cash - Min Reserve) * Leverage
        """
        min_reserve = self.available_cash * 0.1  # 保留 10% 作为缓冲
        return (self.available_cash - min_reserve) * self.leverage_limit
    
    def borrow(self, amount: float) -> bool:
        """
        借入资金
        
        最大借款额 = 初始权益 × (杠杆倍数 - 1)
        例如：10万初始资金，2倍杠杆，最多借10万
        
        Args:
            amount: 借入金额
            
        Returns:
            是否成功
        """
        # 基于初始权益计算最大借款 (防止循环借款)
        max_borrow = self._initial_equity * (self.leverage_limit - 1)
        
        if self.borrowed + amount > max_borrow:
            return False
        
        self.borrowed += amount
        self.available_cash += amount
        return True
    
    def repay(self, amount: float) -> float:
        """
        偿还借款
        
        Returns:
            实际偿还金额
        """
        actual_repay = min(amount, self.borrowed, self.available_cash)
        
        self.borrowed -= actual_repay
        self.available_cash -= actual_repay
        
        return actual_repay
    
    def accrue_interest(self, days: int = 1) -> float:
        """
        计算利息
        
        Args:
            days: 计息天数
            
        Returns:
            本次计提的利息
        """
        daily_rate = self.interest_rate / 365
        interest = self.borrowed * daily_rate * days
        self.accrued_interest += interest
        return interest
    
    def check_margin_level(self, current_prices: Dict[str, float]) -> bool:
        """
        检查保证金水平
        
        Returns:
            True = 安全, False = 需要追缴
        """
        if self.borrowed == 0:
            return True
        
        margin_level = self.get_margin_level(current_prices)
        return margin_level >= self.maintenance_margin
    
    def calculate_shortfall(self, current_prices: Dict[str, float]) -> float:
        """
        计算保证金缺口
        
        Returns:
            需要补充的金额 (正数表示缺口)
        """
        market_value = self.get_market_value(current_prices)
        equity = self.get_equity(current_prices)
        
        required_equity = (market_value + self.borrowed) * self.maintenance_margin
        shortfall = required_equity - equity
        
        return max(0, shortfall)
    
    def force_liquidate(
        self,
        ticker: str,
        current_price: float,
        agent_id: str = ""
    ) -> Optional[Order]:
        """
        强制平仓
        
        当保证金不足时，生成市价卖单清仓。
        
        Args:
            ticker: 股票代码
            current_price: 当前价格
            agent_id: Agent ID
            
        Returns:
            强制平仓订单 (或 None 如果无持仓)
        """
        qty = self.get_total_holdings_qty(ticker)
        
        if qty <= 0:
            return None
        
        # 计算当前保证金状态
        current_prices = {ticker: current_price}
        equity = self.get_equity(current_prices)
        margin_level = self.get_margin_level(current_prices)
        shortfall = self.calculate_shortfall(current_prices)
        
        # 记录保证金追缴事件
        event = MarginCallEvent(
            agent_id=agent_id,
            ticker=ticker,
            equity=equity,
            borrowed=self.borrowed,
            margin_level=margin_level,
            required_margin=self.maintenance_margin,
            shortfall=shortfall,
            liquidation_qty=qty,
            liquidation_price=current_price
        )
        self.margin_call_history.append(event)
        
        # 生成强制平仓订单
        # 生成强制平仓订单
        return Order(
            symbol=ticker,
            side=OrderSide.SELL,
            quantity=qty,
            price=current_price,
            order_type=OrderType.MARKET,
            reason="MARGIN_CALL",
            agent_id=agent_id,
            timestamp=time.time() # 使用当前系统时间作为回退，理想情况下应传入 simulation clock
        )
    
    def get_account_summary(self, current_prices: Dict[str, float]) -> Dict:
        """获取账户摘要"""
        market_value = self.get_market_value(current_prices)
        equity = self.get_equity(current_prices)
        
        return {
            "cash": self.available_cash,
            "market_value": market_value,
            "borrowed": self.borrowed,
            "accrued_interest": self.accrued_interest,
            "equity": equity,
            "margin_level": self.get_margin_level(current_prices),
            "maintenance_margin": self.maintenance_margin,
            "buying_power": self.get_buying_power(),
            "leverage_used": (market_value / equity) if equity > 0 else 0,
            "is_margin_safe": self.check_margin_level(current_prices),
            "margin_calls": len(self.margin_call_history)
        }


# ==========================================
# 风险引擎 (整合)
# ==========================================

class RiskEngine:
    """
    风险控制引擎
    
    整合高频监控、保证金管理和交易税费计算。
    
    Attributes:
        hft_monitor: 高频交易监控器
        stamp_duty_rate: 印花税率
        commission_rate: 佣金率
    """
    
    def __init__(
        self,
        stamp_duty_rate: float = 0.001,
        commission_rate: float = 0.0003,
        otr_threshold: float = 10.0
    ):
        self.hft_monitor = HighFrequencyMonitor(otr_threshold=otr_threshold)
        self.stamp_duty_rate = stamp_duty_rate
        self.commission_rate = commission_rate
        
        # 保证金账户映射
        self.margin_accounts: Dict[str, MarginAccount] = {}
        
        # 强制平仓订单队列
        self.liquidation_queue: List[Order] = []
    
    def register_margin_account(self, agent_id: str, account: MarginAccount) -> None:
        """注册保证金账户"""
        self.margin_accounts[agent_id] = account
    
    def calculate_transaction_cost(
        self,
        side: OrderSide,
        price: float,
        quantity: int
    ) -> float:
        """
        计算交易成本
        
        A股规则:
        - 印花税: 仅卖出收取
        - 佣金: 双向收取，最低 5 元
        """
        value = price * quantity
        
        # 佣金
        commission = max(5.0, value * self.commission_rate)
        
        # 印花税 (仅卖出)
        # 兼容处理: side 可能是枚举或字符串
        side_str = str(side).upper()
        is_sell = "SELL" in side_str
        
        stamp_duty = value * self.stamp_duty_rate if is_sell else 0.0
        
        return commission + stamp_duty
    
    def check_order_compliance(
        self,
        agent_id: str,
        order: Order,
        market_data: Optional[Dict] = None
    ) -> Tuple[bool, float, Optional[str]]:
        """
        检查订单合规性
        
        Returns:
            (是否允许, 额外费用, 拒绝原因)
        """
        # 1. 高频监控检查
        allowed, hft_penalty, reject_reason = self.hft_monitor.check_order(
            agent_id, order.value
        )
        
        if not allowed:
            return False, 0.0, reject_reason
        
        # 2. 保证金检查 (如适用)
        # 兼容处理: OrderSide 可能是枚举或字符串
        side_str = str(order.side).upper()
        is_buy = "BUY" in side_str
        
        if agent_id in self.margin_accounts:
            account = self.margin_accounts[agent_id]
            
            if is_buy:
                # 检查购买力
                if order.value > account.get_buying_power():
                    return False, 0.0, f"超出购买力限制 (需: {order.value:.2f}, 有: {account.get_buying_power():.2f})"
        
        # 3. 计算总费用
        # 确保传入正确的 OrderSide 枚举
        calc_side = OrderSide.BUY if is_buy else OrderSide.SELL
        
        tx_cost = self.calculate_transaction_cost(
            calc_side, order.price, order.quantity
        )
        total_cost = tx_cost + hft_penalty
        
        return True, total_cost, None
    
    def check_all_margin_accounts(
        self,
        current_prices: Dict[str, float]
    ) -> List[Order]:
        """
        检查所有保证金账户，触发强制平仓
        
        Returns:
            强制平仓订单列表
        """
        liquidation_orders = []
        
        for agent_id, account in self.margin_accounts.items():
            if not account.check_margin_level(current_prices):
                # 遍历所有持仓，生成平仓订单
                for ticker in list(account.positions.keys()):
                    price = current_prices.get(ticker, 0)
                    if price > 0:
                        order = account.force_liquidate(ticker, price, agent_id)
                        if order:
                            liquidation_orders.append(order)
        
        self.liquidation_queue.extend(liquidation_orders)
        return liquidation_orders
    
    def get_risk_report(self, current_prices: Dict[str, float] = None) -> Dict:
        """获取风险报告"""
        current_prices = current_prices or {}
        
        margin_summaries = {}
        for agent_id, account in self.margin_accounts.items():
            margin_summaries[agent_id] = account.get_account_summary(current_prices)
        
        return {
            "hft_violations": self.hft_monitor.get_violation_report(),
            "margin_accounts": margin_summaries,
            "pending_liquidations": len(self.liquidation_queue),
            "stamp_duty_rate": self.stamp_duty_rate,
            "commission_rate": self.commission_rate
        }


# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("风险控制引擎测试")
    print("=" * 60)
    
    # 创建风险引擎
    engine = RiskEngine(stamp_duty_rate=0.001, otr_threshold=5.0)
    
    # 创建保证金账户
    margin_account = MarginAccount(
        initial_cash=100000,
        leverage_limit=2.0,
        maintenance_margin=0.25
    )
    engine.register_margin_account("trader_001", margin_account)
    
    # 模拟借款并买入
    print("\n[借款并买入]")
    margin_account.borrow(100000)  # 借 10 万
    print(f"  借入: 100000, 总可用: {margin_account.available_cash}")
    
    # 模拟买入
    margin_account.buy("000001", 10.0, 10000, pd.Timestamp.now())
    print(f"  买入 000001 x 10000 @ 10.0")
    
    # 检查保证金
    prices = {"000001": 10.0}
    summary = margin_account.get_account_summary(prices)
    print(f"\n[账户摘要]")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    # 模拟价格下跌
    print("\n[模拟价格下跌]")
    for drop_price in [9.0, 8.0, 7.0, 6.0]:
        prices = {"000001": drop_price}
        is_safe = margin_account.check_margin_level(prices)
        margin_level = margin_account.get_margin_level(prices)
        print(f"  价格 {drop_price}: 保证金率 {margin_level:.2%}, 安全: {is_safe}")
        
        if not is_safe:
            print(f"  触发保证金追缴!")
            order = margin_account.force_liquidate("000001", drop_price, "trader_001")
            if order:
                print(f"  强制平仓订单: {order.side} {order.quantity} @ {order.price}")
            break
    
    # 高频监控测试
    print("\n[高频监控测试]")
    hft = engine.hft_monitor
    
    # 模拟大量订单但很少成交
    for i in range(50):
        hft.register_order("hft_bot")
        if i % 10 == 0:
            hft.register_trade("hft_bot")
    
    otr = hft.get_otr("hft_bot")
    allowed, penalty, reason = hft.check_order("hft_bot", 100000)
    print(f"  OTR: {otr:.1f}")
    print(f"  允许交易: {allowed}")
    print(f"  惩罚费用: {penalty:.2f}")
    if reason:
        print(f"  拒绝原因: {reason}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
