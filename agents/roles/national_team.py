"""
国家队 (National Team) / 平准基金 Agent

负责市场维稳，防止非理性暴跌。
核心逻辑：
1. 资金无限 (或极大)
2. 监听 panic_level
3. 当恐慌触及阈值时，大单买入
4. 在社交网络广播信心
"""

from typing import Dict, List, Optional, Any
from agents.base_agent import BaseAgent, MarketSnapshot
from core.types import Order, OrderSide, OrderType
from core.society.network import SocialGraph, SentimentState

class NationalTeamAgent(BaseAgent):
    """
    国家队 Agent — 市场维稳定海神针
    """
    
    def __init__(
        self,
        agent_id: str,
        cash_balance: float = 10_000_000_000.0, # 100亿初始资金
    ):
        # 传入一个特殊的 profile，极度理性
        profile = {
            "risk_aversion": 0.0,
            "confidence_level": 1.0, # 绝对自信
            "attention_span": 10
        }
        super().__init__(agent_id, cash_balance, portfolio={}, psychology_profile=profile)
        
        self.is_intervening = False
        self.social_graph: Optional[SocialGraph] = None
        self.social_node_id: Optional[int] = None
        
        # 干预参数
        self.panic_threshold = 0.8       # 恐慌阈值
        self.drop_threshold = -0.05      # 3 tick 累计跌幅阈值
        self.intervention_budget_per_tick = 500_000_000 # 每次干预最多 5亿
        
        self.price_history = []

    def bind_social_node(self, node_id: int, graph: SocialGraph):
        """Bind to social graph (Top Influencer)"""
        self.social_node_id = node_id
        self.social_graph = graph
        
        # 强制设定为超级影响者
        if node_id in graph.agents:
            node = graph.agents[node_id]
            node.agent_id = self.agent_id
            node.influence = 0.99
            node.conformity = 0.0
            node.sentiment_state = SentimentState.BULLISH

    async def act(
        self,
        market_snapshot: MarketSnapshot,
        public_news: List[str],
    ) -> Optional[Order]:
        """
        国家队行动逻辑
        """
        # 1. 记录价格历史用于判断急跌
        self.price_history.append(market_snapshot.last_price)
        if len(self.price_history) > 4:
            self.price_history.pop(0)
            
        drop_pct = 0.0
        if len(self.price_history) >= 3:
            drop_pct = (self.price_history[-1] - self.price_history[0]) / self.price_history[0]
            
        # 2. 判断是否触发干预
        panic_val = getattr(market_snapshot, "panic_level", 0.0)
        
        should_intervene = False
        reason = ""
        
        if panic_val > self.panic_threshold:
            should_intervene = True
            reason = f"panic_level {panic_val:.2f} > {self.panic_threshold}"
        elif drop_pct < self.drop_threshold:
            should_intervene = True
            reason = f"price drop {drop_pct:.1%} < {self.drop_threshold}"
            
        # 3. 执行干预
        if should_intervene:
            self.is_intervening = True
            # 广播信心
            self._broadcast_confidence()
            
            # 生成买单 (托单)
            # 在当前价格下方 0.5%, 1.0%, 1.5% 挂单，或者直接扫货?
            # 策略：以当前价格 + 0.5% 限价扫货，提供流动性
            target_price = market_snapshot.last_price * 1.005 
            qty = self.intervention_budget_per_tick // target_price // 100 * 100
            
            print(f"[国家队] 触发维稳! 原因: {reason}. 买入 {qty} 股 @ {target_price}")
            
            return Order(
                symbol=market_snapshot.symbol,
                price=target_price,
                quantity=int(qty),
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                agent_id=self.agent_id,
                timestamp=market_snapshot.timestamp
            )
            
        else:
            self.is_intervening = False
            return None


    async def perceive(self, market_snapshot: MarketSnapshot, public_news: List[str]) -> Dict[str, Any]:
        return {}
        
    async def reason(self, perceived_data: Dict[str, Any]) -> Dict[str, Any]:
        return {}
        
    async def decide(self, reasoning_output: Dict[str, Any]) -> Optional[Order]:
        return None
        
    async def update_memory(self, decision: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        pass

    def _broadcast_confidence(self):
        """向社交网络广播信心 (治愈邻居)"""
        if self.social_graph and self.social_node_id is not None:
            # 治愈所有邻居
            neighbors = self.social_graph.get_neighbors(self.social_node_id)
            cured_count = 0
            for nid in neighbors:
                node = self.social_graph.agents.get(nid)
                if node and node.sentiment_state in (SentimentState.INFECTED, SentimentState.SUSCEPTIBLE):
                    node.sentiment_state = SentimentState.RECOVERED
                    cured_count += 1
            
            # 同时更新自己为 BULLISH
            self.social_graph.agents[self.social_node_id].sentiment_state = SentimentState.BULLISH
            # print(f"[国家队] 信心广播完成，治愈 {cured_count} 个邻居")

