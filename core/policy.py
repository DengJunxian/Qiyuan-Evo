"""
Market Policy & Regulatory Engine

Implements market interventions such as:
1. Circuit Breakers (Price Limits)
2. Transaction Taxes (Tobin Tax)
3. Trading Halts
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import time

from core.types import Order, Trade, OrderStatus

class PolicyType(str, Enum):
    CIRCUIT_BREAKER = "circuit_breaker"
    TRANSACTION_TAX = "transaction_tax"

@dataclass
class PolicyResult:
    is_allowed: bool
    reason: str = ""
    modified_order: Optional[Order] = None
    tax_amount: float = 0.0

class Policy:
    """Base class for all market policies."""
    
    def __init__(self, name: str, active: bool = True):
        self.name = name
        self.active = active

    def check_order(self, order: Order, market_state: Dict[str, Any]) -> PolicyResult:
        """Check if an order is allowed under this policy."""
        return PolicyResult(True)

    def on_trade(self, trade: Trade) -> Trade:
        """Apply policy effects on a executed trade (e.g. tax)."""
        return trade

class CircuitBreaker(Policy):
    """
    Dynamic Circuit Breaker
    
    Halts trading if price deviates significantly from reference price (e.g. Prev Close).
    """
    def __init__(self, threshold_pct: float = 0.10, halt_duration_sec: int = 300):
        super().__init__("Circuit Breaker")
        self.threshold_pct = threshold_pct
        self.halt_duration = halt_duration_sec
        
        self.is_halted = False
        self.halt_start_time = 0.0
        self.reference_price = 0.0

    def update_reference_price(self, price: float):
        self.reference_price = price

    def check_market_status(self, current_price: float, timestamp: float) -> bool:
        """
        Check if market should be halted or resumed.
        Returns: True if market is OPEN, False if HALTED.
        """

        if self.is_halted:
            if timestamp - self.halt_start_time > self.halt_duration:
                self.is_halted = False
                return True
            return False


        if self.reference_price > 0:
            deviation = abs(current_price - self.reference_price) / self.reference_price
            if deviation > self.threshold_pct:
                self.is_halted = True
                self.halt_start_time = timestamp
                return False
                
        return True

    def check_order(self, order: Order, market_state: Dict[str, Any]) -> PolicyResult:
        if not self.active:
            return PolicyResult(True)
            
        timestamp = order.timestamp if order.timestamp else time.time()
        current_price = market_state.get("last_price", self.reference_price)


        is_open = self.check_market_status(current_price, timestamp)
        
        if not is_open:
            return PolicyResult(False, f"Market Halted (Circuit Breaker Triggered)")
            


        return PolicyResult(True)

class TransactionTax(Policy):
    """
    Stamp Duty / Transaction Tax
    """
    def __init__(self, rate: float = 0.001):
        super().__init__("Transaction Tax")
        self.rate = rate

    def on_trade(self, trade: Trade) -> Trade:
        if not self.active:
            return trade
            




        


        return trade
    
    def calculate_tax(self, trade: Trade) -> float:
        if not self.active: return 0.0
        return trade.price * trade.quantity * self.rate

class PolicyManager:
    """
    Manages all active policies.
    """
    def __init__(self):
        self.policies: Dict[str, Policy] = {}
        

        self.policies["circuit_breaker"] = CircuitBreaker(threshold_pct=0.10)
        self.policies["tax"] = TransactionTax(rate=0.001)

    def set_policy_param(self, name: str, param: str, value: Any):
        if name in self.policies:
            if hasattr(self.policies[name], param):
                setattr(self.policies[name], param, value)

    def check_order(self, order: Order, market_state: Dict[str, Any]) -> PolicyResult:
        for policy in self.policies.values():
            res = policy.check_order(order, market_state)
            if not res.is_allowed:
                return res
        return PolicyResult(True)

    def calculate_total_tax(self, trade: Trade) -> float:
        total = 0.0
        for policy in self.policies.values():
            if isinstance(policy, TransactionTax):
                total += policy.calculate_tax(trade)
        return total
    
    def is_market_halted(self) -> bool:
        cb = self.policies.get("circuit_breaker")
        if cb and isinstance(cb, CircuitBreaker):
            return cb.is_halted
        return False
