
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from config import GLOBAL_CONFIG

@dataclass
class PositionBatch:
    """
    持仓批次
    记录每一次买入的具体细节，用于 T+1 可卖性检查和税务/收益计算(FIFO)。
    """
    ticker: str
    quantity: int
    cost_basis: float      # 单股成本 (含税费)
    acquired_time: pd.Timestamp
    
    @property
    def acquired_date(self) -> pd.Timestamp:
        """获取标准化的日期 (00:00:00)"""
        return self.acquired_time.normalize()

class Portfolio:
    """
    支持 A 股 T+1 制度的高保真账户模型。
    
    Attributes:
        available_cash (float): 可用资金 (Buying Power)。当日卖出所得立即进入此池，可用于买入。
        withdrawable_cash (float): 可取资金。当日卖出所得需结算后(T+1)才进入此池。
        positions (Dict[str, List[PositionBatch]]): 持仓清单，Key为代码，Value为批次列表。
    """
    
    def __init__(self, initial_cash: float = GLOBAL_CONFIG.DEFAULT_CASH):
        self.available_cash = float(initial_cash)
        self.withdrawable_cash = float(initial_cash)
        # 使用 list 存储批次，以便进行 先进先出 管理和 T+1 过滤
        self.positions: Dict[str, List[PositionBatch]] = {} 
        self.frozen_cash = 0.0 # 预留给挂单冻结的资金 (虽然提示词未显式要求，但为撮合必备)

    def deposit(self, amount: float):
        """入金"""
        if amount > 0:
            self.available_cash += amount
            self.withdrawable_cash += amount

    def withdraw(self, amount: float) -> bool:
        """出金"""
        if amount <= self.withdrawable_cash:
            self.available_cash -= amount
            self.withdrawable_cash -= amount
            return True
        return False

    def get_total_holdings_qty(self, ticker: str) -> int:
        """获取某股票的总持仓量 (包含冻结/不可卖部分)"""
        if ticker not in self.positions:
            return 0
        return sum(batch.quantity for batch in self.positions[ticker])

    def get_sellable_qty(self, ticker: str, current_time: pd.Timestamp) -> int:
        """
        [核心逻辑] 获取当前时间点可卖出的数量。
        遵循 A 股 T+1 规则：
        - 如果 Config.T_PLUS_1 为 True: 仅统计 acquired_date < current_date 的批次。
        - 否则: 返回总持仓。
        """
        if ticker not in self.positions:
            return 0
        
        # 强制转换为 pandas Timestamp 确保比较正确
        if not isinstance(current_time, pd.Timestamp):
            current_time = pd.Timestamp(current_time)
            
        batches = self.positions[ticker]
        
        if not GLOBAL_CONFIG.T_PLUS_1:
            return sum(b.quantity for b in batches)
            
        current_date = current_time.normalize()
        
        sellable = 0
        for b in batches:
            # 只有 买入日期 < 当前日期 才是"昨仓"，可卖
            if b.acquired_date < current_date:
                sellable += b.quantity
        return sellable

    def buy(self, ticker: str, price: float, qty: int, time: pd.Timestamp, transaction_cost: float = 0.0):
        """
        买入操作 (成交后调用)
        """
        total_cost = (price * qty) + transaction_cost
        
        if total_cost > self.available_cash:
            raise ValueError(f"资金不足: 需要 {total_cost:.2f}, 可用 {self.available_cash:.2f}")

        # 1. 扣款
        self.available_cash -= total_cost
        
        # 逻辑说明：如果通过卖出今日股票获得了现金，买入时优先消耗这部分"未结算资金"。
        # withdrawable_cash 只有在消耗完今日收益并开始侵蚀老本时才会减少。
        # 公式：可取资金 = min(新的可用资金, 原可取资金)
        self.withdrawable_cash = min(self.withdrawable_cash, self.available_cash)

        # 2. 增加持仓批次
        if not isinstance(time, pd.Timestamp):
            time = pd.Timestamp(time)
            
        new_batch = PositionBatch(
            ticker=ticker,
            quantity=qty,
            cost_basis=total_cost / qty, # 摊薄成本
            acquired_time=time
        )
        
        if ticker not in self.positions:
            self.positions[ticker] = []
        self.positions[ticker].append(new_batch)

    def sell(self, ticker: str, price: float, qty: int, time: pd.Timestamp, transaction_cost: float = 0.0):
        """
        卖出操作 (成交后调用)
        遵循 FIFO 原则优先平掉老仓位。
        """
        sellable = self.get_sellable_qty(ticker, time)
        if qty > sellable:
            raise ValueError(f"可卖股数不足 (T+1限制): 申请 {qty}, 可卖 {sellable}")

        # 1. 先进先出 扣减持仓
        remaining_qty_to_sell = qty
        kept_batches = []
        
        # 按买入时间排序 (通常 append 已经是顺序的，但为了保险)
        current_batches = sorted(self.positions[ticker], key=lambda x: x.acquired_time)
        
        for batch in current_batches:
            if remaining_qty_to_sell <= 0:
                kept_batches.append(batch)
                continue
            
            # 只有老仓位能被卖出
            is_sellable_batch = True
            if GLOBAL_CONFIG.T_PLUS_1:
                is_sellable_batch = batch.acquired_date < pd.Timestamp(time).normalize()
            
            if not is_sellable_batch:
                kept_batches.append(batch)
                continue
                
            if batch.quantity > remaining_qty_to_sell:
                # 这一批次只需卖出一部分
                batch.quantity -= remaining_qty_to_sell
                remaining_qty_to_sell = 0
                kept_batches.append(batch)
            else:
                # 这一批次全部卖出
                remaining_qty_to_sell -= batch.quantity
                # 不 append 到 kept_batches，即删除
        
        self.positions[ticker] = kept_batches
        
        # 2. 资金结算
        revenue = (price * qty) - transaction_cost
        
        # A股规则：卖出资金当日可用(买股)，次日可取
        self.available_cash += revenue
        # 注意：revenue 不加到 withdrawable_cash，直到 settle()

    def settle(self):
        """
        日终结算 (End of Day Settlement)
        将当日的所有"未结算资金"转为"可取资金"。
        在 T+1 制度下，这通常发生在收盘清算后。
        """
        # 简单逻辑：只要到了第二天，所有的 available 都变成了可取资金
        # (假设没有冻结资金的情况)
        self.withdrawable_cash = self.available_cash

    def get_market_value(self, current_prices: Dict[str, float]) -> float:
        """计算持仓市值"""
        val = 0.0
        for ticker, batches in self.positions.items():
            qty = sum(b.quantity for b in batches)
            price = current_prices.get(ticker, 0.0)
            val += qty * price
        return val

    @property
    def total_assets(self) -> float:
        """
        注意：这只是估算值，准确的总资产需要传入当前行情计算市值。
        此处仅返回 现金部分 + 成本市值 (仅供参考)
        """
        cost_val = 0.0
        for batches in self.positions.values():
            for b in batches:
                cost_val += b.quantity * b.cost_basis
        return self.available_cash + cost_val