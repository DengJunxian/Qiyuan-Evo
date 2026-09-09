"""
增强型 RAG 记忆模块

功能:
1. 向量化记忆存储与检索
2. 创伤事件记忆管理
3. 波动率触发的记忆回溯
4. "过去崩盘"记忆注入

作者: Civitas Economica Team
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple


# ==========================================
# 数据结构
# ==========================================

@dataclass
class MemoryFragment:
    """
    记忆碎片
    
    存储单次市场事件的上下文与结果。
    """
    content: str              # 记忆内容
    vector: np.ndarray        # 向量表示
    timestamp: float          # 时间戳
    outcome_score: float      # 结果评分: -1.0 (惨败) ~ 1.0 (大胜)
    event_type: str = "normal"  # 事件类型: "normal" | "trauma" | "success"
    volatility: float = 0.0   # 当时的市场波动率
    
    def is_trauma(self) -> bool:
        """是否为创伤记忆"""
        return self.event_type == "trauma" or self.outcome_score < -0.5


@dataclass
class TraumaEvent:
    """
    创伤事件记录
    
    用于模拟投资者的"心理阴影"，在类似情况下触发回忆。
    """
    date: str                 # 事件日期
    description: str          # 事件描述
    loss_pct: float          # 损失百分比
    market_conditions: Dict   # 当时的市场状况
    emotional_impact: float   # 情绪冲击强度 [0, 1]


# ==========================================
# 预设的历史崩盘记忆
# ==========================================

HISTORICAL_CRASHES: List[TraumaEvent] = [
    TraumaEvent(
        date="2015-06-15",
        description="2015年股灾：杠杆牛市崩盘，千股跌停，账户三周缩水60%。融资盘爆仓，流动性枯竭。",
        loss_pct=-0.60,
        market_conditions={"trend": "暴跌", "panic_level": 0.95, "trigger": "去杠杆"},
        emotional_impact=0.95
    ),
    TraumaEvent(
        date="2016-01-04",
        description="2016年熔断：新年首个交易日触发熔断，全市场恐慌性抛售。开盘即跌停。",
        loss_pct=-0.07,
        market_conditions={"trend": "暴跌", "panic_level": 0.90, "trigger": "熔断机制"},
        emotional_impact=0.85
    ),
    TraumaEvent(
        date="2018-02-09",
        description="2018年初闪崩：全球股市联动下跌，A股单周跌幅超10%。VIX飙升至历史高位。",
        loss_pct=-0.12,
        market_conditions={"trend": "暴跌", "panic_level": 0.80, "trigger": "全球联动"},
        emotional_impact=0.75
    ),
    TraumaEvent(
        date="2020-03-09",
        description="2020年新冠暴跌：疫情恐慌叠加油价战争，全球熔断。恐慌指数创历史新高。",
        loss_pct=-0.15,
        market_conditions={"trend": "暴跌", "panic_level": 0.92, "trigger": "疫情+油价"},
        emotional_impact=0.88
    ),
    TraumaEvent(
        date="2024-02-05",
        description="2024年雪球敲入：大量雪球产品集中敲入，引发程序化抛售潮。流动性危机。",
        loss_pct=-0.08,
        market_conditions={"trend": "暴跌", "panic_level": 0.78, "trigger": "衍生品"},
        emotional_impact=0.70
    ),
]


# ==========================================
# 增强型向量记忆
# ==========================================

class TraumaMemory:
    """
    增强型 RAG 记忆库
    
    特点:
    1. 基于向量的语义检索
    2. 创伤事件特殊处理
    3. 波动率触发的自动回忆
    4. 历史崩盘记忆注入
    """
    
    def __init__(
        self,
        dimension: int = 64,
        max_fragments: int = 100,
        trauma_threshold: float = -0.3
    ):
        """
        初始化记忆库
        
        Args:
            dimension: 向量维度
            max_fragments: 最大记忆片段数
            trauma_threshold: 创伤阈值 (outcome_score 低于此值视为创伤)
        """
        self.dimension = dimension
        self.max_fragments = max_fragments
        self.trauma_threshold = trauma_threshold
        
        self.fragments: List[MemoryFragment] = []
        self.trauma_events: List[TraumaEvent] = HISTORICAL_CRASHES.copy()
    
    def _mock_embedding(self, text: str) -> np.ndarray:
        """
        生成确定性的伪向量
        
        注意: 在生产环境中应替换为真实的 Embedding API 调用。
        这里使用文本哈希生成可复现的向量。
        """
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.rand(self.dimension)
        return vec / np.linalg.norm(vec)  # 归一化
    
    def add_memory(
        self,
        content: str,
        outcome: float,
        volatility: float = 0.0
    ) -> None:
        """
        添加记忆
        
        Args:
            content: 记忆内容
            outcome: 结果评分 (-1 ~ 1)
            volatility: 当时的波动率
        """
        vector = self._mock_embedding(content)
        
        # 判断事件类型
        if outcome < self.trauma_threshold:
            event_type = "trauma"
        elif outcome > 0.5:
            event_type = "success"
        else:
            event_type = "normal"
        
        fragment = MemoryFragment(
            content=content,
            vector=vector,
            timestamp=time.time(),
            outcome_score=outcome,
            event_type=event_type,
            volatility=volatility
        )
        
        self.fragments.append(fragment)
        
        # 保持记忆库大小
        if len(self.fragments) > self.max_fragments:
            # 优先保留创伤记忆
            non_trauma = [f for f in self.fragments if not f.is_trauma()]
            if non_trauma:
                self.fragments.remove(non_trauma[0])
            else:
                self.fragments.pop(0)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> List[str]:
        """
        语义检索相关记忆
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            相关记忆内容列表
        """
        if not self.fragments:
            return []
        
        query_vec = self._mock_embedding(query)
        scores = []
        
        for frag in self.fragments:
            cosine_sim = np.dot(query_vec, frag.vector)
            scores.append((cosine_sim, frag.content))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scores[:top_k]]
    
    def retrieve_trauma(
        self,
        volatility: float,
        threshold: float = 0.03,
        top_k: int = 3
    ) -> List[str]:
        """
        波动率触发的创伤记忆检索
        
        当市场波动率超过阈值时，自动检索"过去的崩盘记忆"。
        这模拟了投资者在高波动市场中的 PTSD 反应。
        
        Args:
            volatility: 当前市场波动率 (日波动率)
            threshold: 触发阈值 (默认 3%)
            top_k: 返回数量
            
        Returns:
            创伤记忆列表
        """
        if volatility < threshold:
            return []
        
        memories = []
        
        # 检索历史崩盘事件
        for event in self.trauma_events:
            if event.emotional_impact > 0.5:  # 只返回冲击较大的
                memories.append(
                    f"[{event.date}] {event.description} "
                    f"(当时损失 {event.loss_pct:.1%})"
                )
        
        # 检索个人创伤记忆
        trauma_frags = [f for f in self.fragments if f.is_trauma()]
        for frag in sorted(trauma_frags, key=lambda x: x.outcome_score)[:top_k]:
            memories.append(frag.content)
        
        return memories[:top_k]
    
    def inject_crash_memory(
        self,
        event: TraumaEvent
    ) -> None:
        """
        注入崩盘记忆
        
        用于在仿真开始时为 Agent 注入"历史包袱"。
        
        Args:
            event: 崩盘事件
        """
        self.trauma_events.append(event)
        
        # 同时作为记忆片段添加
        self.add_memory(
            content=event.description,
            outcome=event.loss_pct,
            volatility=0.1  # 高波动
        )
    
    def get_emotional_baggage(self) -> Tuple[float, str]:
        """
        获取情绪包袱
        
        基于历史创伤计算 Agent 的"心理阴影"程度。
        
        Returns:
            (包袱程度, 描述): 0 = 无包袱, 1 = 沉重包袱
        """
        if not self.fragments:
            return 0.0, "初入市场，没有历史包袱"
        
        trauma_count = sum(1 for f in self.fragments if f.is_trauma())
        total_count = len(self.fragments)
        
        trauma_ratio = trauma_count / total_count
        avg_trauma_score = np.mean([
            f.outcome_score for f in self.fragments if f.is_trauma()
        ]) if trauma_count > 0 else 0
        
        # 综合评分
        baggage = min(1.0, trauma_ratio * 2 + abs(avg_trauma_score) * 0.5)
        
        if baggage > 0.7:
            desc = "心理阴影严重，经历过多次重大亏损"
        elif baggage > 0.4:
            desc = "有一定心理包袱，曾经历过市场震荡"
        elif baggage > 0.1:
            desc = "轻微心理影响，整体经历较为平稳"
        else:
            desc = "心态健康，没有明显的投资创伤"
        
        return baggage, desc
    
    def get_context_for_decision(
        self,
        market_state: Dict,
        top_k: int = 3
    ) -> str:
        """
        获取用于决策的记忆上下文
        
        自动判断是否需要触发创伤记忆。
        
        Args:
            market_state: 当前市场状态
            top_k: 返回记忆数量
            
        Returns:
            格式化的记忆上下文字符串
        """
        volatility = market_state.get('volatility', 0.01)
        trend = market_state.get('trend', '震荡')
        panic = market_state.get('panic_level', 0.5)
        
        memories = []
        
        # 高波动时检索创伤记忆
        if volatility > 0.03 or panic > 0.7:
            trauma_memories = self.retrieve_trauma(volatility, threshold=0.02)
            memories.extend(trauma_memories)
        
        # 普通语义检索
        query = f"市场{trend}，恐慌指数{panic:.2f}"
        normal_memories = self.retrieve(query, top_k=top_k - len(memories))
        memories.extend(normal_memories)
        
        if not memories:
            return "无相关历史记忆"
        
        return "\n".join([f"- {m}" for m in memories[:top_k]])
    
    async def get_context_for_decision_async(
        self,
        market_state: Dict,
        top_k: int = 3
    ) -> str:
        """
        [异步] 获取用于决策的记忆上下文
        """
        # 目前 memory 操作主要是本地向量计算，非 IO 密集
        # 但为了未来扩展(如远程向量库)，预留 async 接口
        # 简单 wrap sync method
        return self.get_context_for_decision(market_state, top_k)

    def __len__(self) -> int:
        return len(self.fragments)



