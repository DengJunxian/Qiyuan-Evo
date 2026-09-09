"""
记忆流与反思引擎

受斯坦福 Generative Agents 项目启发，实现三阶段记忆架构：
1. 记忆流（Memory Stream）- 原始事件记录
2. 反思（Reflection）- 高层认知提炼
3. 规划（Planning）- 基于反思的行动计划
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime

from openai import OpenAI
from config import GLOBAL_CONFIG


@dataclass
class MemoryEvent:
    """记忆事件"""
    timestamp: float
    event_type: str
    content: str
    importance: float = 0.5  # 0-1 重要性评分
    embedding: Optional[List[float]] = None


@dataclass
class Insight:
    """反思洞见"""
    timestamp: float
    content: str
    source_events: List[int]  # 来源事件索引
    importance: float = 0.7


@dataclass
class InvestmentDiary:
    """投资日记条目"""
    date: str
    summary: str  # 当日总结
    trades: List[str]  # 交易记录
    emotions: str  # 情绪描述
    lessons: str  # 经验教训
    plan: str  # 明日计划


class MemoryStream:
    """
    记忆流
    
    按时间顺序记录 Agent 的所有感知和行为
    """
    
    def __init__(self, max_size: int = 1000):
        self.events: List[MemoryEvent] = []
        self.max_size = max_size
    
    def add(
        self, 
        event_type: str, 
        content: str, 
        importance: float = 0.5
    ):
        """添加新事件"""
        event = MemoryEvent(
            timestamp=time.time(),
            event_type=event_type,
            content=content,
            importance=importance
        )
        self.events.append(event)
        
        # 超出容量时移除最旧的事件
        if len(self.events) > self.max_size:
            self.events.pop(0)
    
    def add_trade(self, action: str, qty: int, price: float, pnl: float = 0):
        """记录交易事件"""
        content = f"执行{action}操作: {qty}股 @ ¥{price:.2f}"
        if pnl != 0:
            content += f", 盈亏: {pnl:+.2f}%"
        
        importance = min(1.0, 0.5 + abs(pnl) / 20)  # 盈亏越大越重要
        self.add("trade", content, importance)
    
    def add_observation(self, content: str, importance: float = 0.3):
        """记录市场观察"""
        self.add("observation", content, importance)
    
    def add_emotion(self, emotion: str, intensity: float):
        """记录情绪状态"""
        content = f"情绪状态: {emotion} (强度: {intensity:.1f})"
        self.add("emotion", content, importance=abs(intensity))
    
    def get_recent(self, n: int = 20) -> List[MemoryEvent]:
        """获取最近N条记忆"""
        return self.events[-n:]
    
    def get_by_importance(self, threshold: float = 0.6) -> List[MemoryEvent]:
        """获取重要记忆"""
        return [e for e in self.events if e.importance >= threshold]
    
    def get_by_type(self, event_type: str) -> List[MemoryEvent]:
        """按类型获取记忆"""
        return [e for e in self.events if e.event_type == event_type]


class ReflectionEngine:
    """
    反思引擎
    
    定期触发反思过程，从记忆流中提炼高层洞见
    """
    
    # 类级别存储
    agent_reflections: Dict[str, List[Insight]] = {}
    agent_diaries: Dict[str, List[InvestmentDiary]] = {}
    
    def __init__(
        self, 
        agent_id: str, 
        api_key: Optional[str] = None,
        reflection_interval: int = 20  # 每20个tick反思一次
    ):
        self.agent_id = agent_id
        self.memory_stream = MemoryStream()
        self.reflection_interval = reflection_interval
        self.tick_count = 0
        self.last_reflection_tick = 0
        
        # 初始化 接口 客户端路由器
        self._api_key = api_key or GLOBAL_CONFIG.DEEPSEEK_API_KEY
        if self._api_key:
            from core.model_router import ModelRouter
            self.model_router = ModelRouter(
                deepseek_key=self._api_key,
                zhipu_key=GLOBAL_CONFIG.ZHIPU_API_KEY
            )
        else:
            self.model_router = None
        
        # 初始化存储
        if agent_id not in ReflectionEngine.agent_reflections:
            ReflectionEngine.agent_reflections[agent_id] = []
        if agent_id not in ReflectionEngine.agent_diaries:
            ReflectionEngine.agent_diaries[agent_id] = []
    
    def tick(self):
        """时间推进"""
        self.tick_count += 1
    
    def should_reflect(self) -> bool:
        """判断是否应该反思"""
        return (self.tick_count - self.last_reflection_tick) >= self.reflection_interval
    
    def reflect(self) -> Optional[Insight]:
        """
        执行反思
        
        从最近记忆中提炼洞见
        """
        if not self.model_router:
            return None
        
        self.last_reflection_tick = self.tick_count
        
        # 获取最近记忆
        recent_events = self.memory_stream.get_recent(20)
        important_events = self.memory_stream.get_by_importance(0.6)
        
        # 合并去重
        all_events = list({id(e): e for e in recent_events + important_events}.values())
        all_events.sort(key=lambda x: x.timestamp)
        
        if not all_events:
            return None
        
        # 构建反思 提示词
        events_text = "\n".join([
            f"- [{e.event_type}] {e.content}" 
            for e in all_events[-15:]  # 最多15条
        ])
        
        reflection_prompt = f"""
作为一位正在反思的投资者，请阅读以下近期记忆，并提炼出一个重要的投资洞见或自我认知。

## 近期记忆

{events_text}

## 反思要求

1. 识别重复出现的行为模式
2. 发现可能存在的认知偏差
3. 总结经验教训
4. 提出改进建议

请用第一人称写一段反思（100-200字），像是在写投资日记。
"""
        
        try:
            content, _, _ = self.model_router.sync_call_with_fallback(
                messages=[
                    {"role": "system", "content": "你是一位善于自我反思的投资者。"},
                    {"role": "user", "content": reflection_prompt}
                ],
                priority_models=["glm-4-flashx"],
                timeout_budget=30.0
            )
            
            insight_content = content
            
            insight = Insight(
                timestamp=time.time(),
                content=insight_content,
                source_events=list(range(len(all_events))),
                importance=0.8
            )
            
            ReflectionEngine.agent_reflections[self.agent_id].append(insight)
            
            # 保留最近20条洞见
            if len(ReflectionEngine.agent_reflections[self.agent_id]) > 20:
                ReflectionEngine.agent_reflections[self.agent_id].pop(0)
            
            # 同时添加到记忆流
            self.memory_stream.add("reflection", insight_content, importance=0.9)
            
            return insight
            
        except Exception as e:
            print(f"[Reflection Error] Agent {self.agent_id}: {e}")
            return None
    
    def generate_daily_diary(self) -> Optional[InvestmentDiary]:
        """
        生成投资日记
        
        综合当日记忆生成完整的日记条目
        """
        if not self.model_router:
            return None
        
        # 获取当日记忆
        all_events = self.memory_stream.events
        if not all_events:
            return None
        
        trades = [e.content for e in self.memory_stream.get_by_type("trade")]
        emotions = [e.content for e in self.memory_stream.get_by_type("emotion")]
        reflections = ReflectionEngine.agent_reflections.get(self.agent_id, [])
        
        diary_prompt = f"""
请根据以下信息，为这位投资者生成一篇投资日记。

## 今日交易
{chr(10).join(trades) if trades else "无交易"}

## 情绪变化
{chr(10).join(emotions[-5:]) if emotions else "情绪平稳"}

## 近期反思
{reflections[-1].content if reflections else "尚未进行反思"}

## 日记格式

请用第一人称写一篇简短的投资日记（200-300字），包含：
1. 今日操作总结
2. 心理状态描述
3. 经验教训
4. 明日计划

风格要求：像真实投资者的日记，有情感、有反思、有计划。
"""
        
        try:
            content, _, _ = self.model_router.sync_call_with_fallback(
                messages=[
                    {"role": "system", "content": "你是一位专业的金融日记作者。"},
                    {"role": "user", "content": diary_prompt}
                ],
                priority_models=["glm-4-flashx"],
                timeout_budget=40.0
            )
            
            diary_content = content
            
            diary = InvestmentDiary(
                date=datetime.now().strftime("%Y-%m-%d"),
                summary=diary_content,
                trades=trades,
                emotions=emotions[-1] if emotions else "平稳",
                lessons=reflections[-1].content if reflections else "",
                plan="根据反思调整策略"
            )
            
            ReflectionEngine.agent_diaries[self.agent_id].append(diary)
            
            return diary
            
        except Exception as e:
            print(f"[Diary Error] Agent {self.agent_id}: {e}")
            return None
    
    def get_insights_summary(self) -> str:
        """获取洞见摘要"""
        insights = ReflectionEngine.agent_reflections.get(self.agent_id, [])
        if not insights:
            return "尚无投资洞见"
        
        return "\n\n".join([
            f"📝 {datetime.fromtimestamp(i.timestamp).strftime('%m-%d %H:%M')}\n{i.content}"
            for i in insights[-5:]
        ])
    
    def get_diary_entries(self, n: int = 5) -> List[InvestmentDiary]:
        """获取最近日记"""
        return ReflectionEngine.agent_diaries.get(self.agent_id, [])[-n:]


class ReflectiveAgent:
    """
    具备反思能力的智能体包装器
    
    将反思引擎与现有 Brain 集成
    """
    
    def __init__(
        self, 
        agent_id: str, 
        brain,  # DeepSeekBrain 或 DebateBrain
        api_key: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.brain = brain
        self.reflection_engine = ReflectionEngine(agent_id, api_key)
    
    def observe(self, market_state: Dict):
        """观察市场"""
        # 记录市场观察
        price = market_state.get('last_price', 0)
        change = market_state.get('change_pct', 0)
        news = market_state.get('news', '')
        
        observation = f"市场价格 ¥{price:.2f} ({change:+.2%})"
        if news:
            observation += f", 新闻: {news[:50]}"
        
        importance = min(1.0, 0.3 + abs(change) * 3)
        self.reflection_engine.memory_stream.add_observation(observation, importance)
    
    def record_emotion(self, emotion_score: float):
        """记录情绪"""
        if emotion_score > 0.5:
            emotion = "极度贪婪"
        elif emotion_score > 0.2:
            emotion = "乐观"
        elif emotion_score > -0.2:
            emotion = "中性"
        elif emotion_score > -0.5:
            emotion = "担忧"
        else:
            emotion = "极度恐惧"
        
        self.reflection_engine.memory_stream.add_emotion(emotion, emotion_score)
    
    def record_trade(self, action: str, qty: int, price: float, pnl: float = 0):
        """记录交易"""
        self.reflection_engine.memory_stream.add_trade(action, qty, price, pnl)
    
    def think_and_reflect(
        self, 
        market_state: Dict, 
        account_state: Dict
    ) -> Dict:
        """思考并可能触发反思"""
        
        # 1. 观察市场
        self.observe(market_state)
        
        # 2. 调用 brain 进行决策
        result = self.brain.think(market_state, account_state)
        
        # 3. 记录情绪
        emotion_score = result.get('emotion_score', 0)
        self.record_emotion(emotion_score)
        
        # 4. 时间推进
        self.reflection_engine.tick()
        
        # 5. 检查是否需要反思
        if self.reflection_engine.should_reflect():
            insight = self.reflection_engine.reflect()
            if insight:
                result['reflection'] = insight.content
        
        return result
    
    def get_diary(self) -> Optional[str]:
        """获取最新日记"""
        diary = self.reflection_engine.generate_daily_diary()
        if diary:
            return diary.summary
        return None
