"""
Agent 抽象基类 — Agentic Architecture 核心契约

定义所有仿真 Agent 的标准生命周期接口:
    Perceive（感知市场）→ Reason（深度推理）→ Decide（生成指令）→ Act（闭环执行）

设计原则:
1. Agent 是拥有生命周期的持久化进程，而非单次 API 调用
2. 所有核心方法为 async，原生兼容 asyncio 事件循环
3. 通过 MarketSnapshot 封装订单簿状态，解耦 Agent 与 OrderBook 实现
4. 记忆系统支持 RAG 检索，实现经验驱动的决策演化

作者: Civitas Economica Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ==========================================
# 市场快照数据结构
# ==========================================

@dataclass
class MarketSnapshot:
    """
    订单簿市场快照 — Agent 的感知输入

    封装 LOB (Limit Order Book) 的关键状态指标，
    使 Agent 无需直接操作 OrderBook 内部数据结构。

    Attributes:
        symbol: 交易标的代码
        last_price: 最近成交价
        best_bid: 最优买价 (买一)
        best_ask: 最优卖价 (卖一)
        bid_ask_spread: 买卖价差
        mid_price: 中间价 (best_bid + best_ask) / 2
        depth: L5 行情深度 {"bids": [...], "asks": [...]}
        total_volume: 当日累计成交量
        volatility: 当前波动率估计
        market_trend: 市场趋势信号 (-1.0 ~ 1.0)
        panic_level: 市场恐慌指数 (0.0 ~ 1.0)
        timestamp: 快照时间戳 (仿真逻辑时间)
    """
    symbol: str
    last_price: float
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    bid_ask_spread: Optional[float] = None
    mid_price: Optional[float] = None
    depth: Dict[str, List[Dict]] = field(default_factory=lambda: {"bids": [], "asks": []})
    total_volume: int = 0
    volatility: float = 0.0
    market_trend: float = 0.0
    panic_level: float = 0.0
    panic_level: float = 0.0
    timestamp: float = 0.0
    
    # 政策感知字段
    policy_description: str = ""
    policy_tax_rate: float = 0.0
    policy_news: str = ""

    text_dominant_topic: str = "uncategorized"
    text_sentiment_score: float = 0.0
    text_panic_score: float = 0.0
    text_greed_score: float = 0.0
    text_policy_shock: float = 0.0
    text_regime_bias: str = "neutral"
    text_impact_paths: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，用于 Prompt 构建"""
        return {
            "symbol": self.symbol,
            "last_price": self.last_price,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.bid_ask_spread,
            "mid_price": self.mid_price,
            "depth": self.depth,
            "volume": self.total_volume,
            "volatility": self.volatility,
            "trend": self.market_trend,
            "panic_level": self.panic_level,
            "policy_description": self.policy_description,
            "policy_tax_rate": self.policy_tax_rate,
            "policy_news": self.policy_news,
            "text_dominant_topic": self.text_dominant_topic,
            "text_sentiment_score": self.text_sentiment_score,
            "text_panic_score": self.text_panic_score,
            "text_greed_score": self.text_greed_score,
            "text_policy_shock": self.text_policy_shock,
            "text_regime_bias": self.text_regime_bias,
            "text_impact_paths": self.text_impact_paths,
        }


# ==========================================
# 智能体 抽象基类
# ==========================================

class BaseAgent(ABC):
    """
    Agent 抽象基类 — 所有仿真 Agent 的统一契约

    生命周期:
        initialize() → [act() 循环] → shutdown()

    认知闭环 (act 方法内部):
        perceive() → reason() → decide()
        └── update_memory() (异步回调)

    子类必须实现:
        - perceive(): 感知环境，过滤信息
        - reason(): 深度推理，生成思维链
        - decide(): 输出结构化交易指令
        - update_memory(): 记录决策结果

    Usage:
        class TraderAgent(BaseAgent):
            async def perceive(self, market_snapshot, public_news):
                ...
            async def reason(self, perceived_data):
                ...
            async def decide(self, reasoning_output):
                ...
    """

    def __init__(
        self,
        agent_id: str,
        cash_balance: float = 100_000.0,
        portfolio: Optional[Dict[str, int]] = None,
        psychology_profile: Optional[Dict[str, float]] = None,
    ):
        """
        初始化 Agent 基础状态

        Args:
            agent_id: Agent 唯一标识符
            cash_balance: 初始现金余额
            portfolio: 初始持仓 {symbol: quantity}
            psychology_profile: 心理画像参数
        """
        self.agent_id = agent_id
        self.cash_balance = cash_balance
        self.portfolio: Dict[str, int] = portfolio or {}
        self.memory: List[Dict[str, Any]] = []
        self.psychology_profile: Dict[str, float] = psychology_profile or {}

        # 生命周期状态
        self._initialized: bool = False
        self._step_count: int = 0
        self._total_pnl: float = 0.0

    # ------------------------------------------
    # 生命周期方法
    # ------------------------------------------

    async def initialize(self) -> None:
        """
        Agent 初始化 — 在仿真开始前调用

        子类可覆写以执行:
        - 加载历史记忆
        - 校准心理参数
        - 预热 LLM 连接
        """
        self._initialized = True

    async def shutdown(self) -> None:
        """
        Agent 关闭 — 在仿真结束时调用

        子类可覆写以执行:
        - 持久化记忆到磁盘
        - 释放 API 连接
        - 生成 Agent 生涯报告
        """
        self._initialized = False

    # ------------------------------------------
    # 认知闭环 — 子类必须实现
    # ------------------------------------------

    @abstractmethod
    async def perceive(
        self,
        market_snapshot: MarketSnapshot,
        public_news: List[str],
    ) -> Dict[str, Any]:
        """
        感知阶段 (Perception)

        Agent 选择性地关注环境信息。
        不同 Agent 有不同的 "attention span"，
        决定了它们能处理的信息量。

        Args:
            market_snapshot: 订单簿市场快照
            public_news: 公共新闻列表

        Returns:
            filtered_data: 过滤后的感知数据
        """
        ...

    @abstractmethod
    async def reason(
        self,
        perceived_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        推理阶段 (Reasoning)

        基于感知数据进行深度推理，生成 Chain of Thought。
        可调用 LLM API 获取类人思维链。

        Args:
            perceived_data: perceive() 的输出

        Returns:
            reasoning_output: 推理结果，包含思维链和决策建议
        """
        ...

    @abstractmethod
    async def decide(
        self,
        reasoning_output: Dict[str, Any],
    ) -> Optional[Any]:
        """
        决策阶段 (Decision)

        将推理结果转化为具体的交易指令 (Order)。

        Args:
            reasoning_output: reason() 的输出

        Returns:
            order: 结构化订单对象，或 None (HOLD)
        """
        ...

    @abstractmethod
    async def update_memory(
        self,
        decision: Dict[str, Any],
        outcome: Dict[str, Any],
    ) -> None:
        """
        记忆更新阶段 (Memory Update)

        将本轮决策与市场反馈写入记忆库，
        供未来决策的 RAG 检索使用。

        Args:
            decision: 本轮决策记录
            outcome: 市场执行反馈 (成交结果、盈亏等)
        """
        ...

    # ------------------------------------------
    # 闭环执行 (公共方法 — 编排子类实现)
    # ------------------------------------------

    async def act(
        self,
        market_snapshot: MarketSnapshot,
        public_news: List[str],
    ) -> Optional[Any]:
        """
        认知闭环执行 — Perceive → Reason → Decide

        这是 Agent 对外暴露的核心方法。
        仿真引擎在每个 tick/step 调用此方法获取交易指令。

        Args:
            market_snapshot: 当前订单簿快照
            public_news: 当前公共新闻

        Returns:
            order: 结构化订单对象，或 None (HOLD)
        """
        self._step_count += 1

        # 阶段 1: 感知
        perceived = await self.perceive(market_snapshot, public_news)

        # 阶段 2: 推理
        reasoning = await self.reason(perceived)

        # 阶段 3: 决策
        order = await self.decide(reasoning)

        return order

    # ------------------------------------------
    # 辅助方法
    # ------------------------------------------

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        计算 Agent 总资产 (现金 + 持仓市值)

        Args:
            current_prices: 各标的当前价格 {symbol: price}

        Returns:
            总资产价值
        """
        holdings_value = sum(
            qty * current_prices.get(symbol, 0.0)
            for symbol, qty in self.portfolio.items()
        )
        return self.cash_balance + holdings_value

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} id={self.agent_id} "
            f"cash={self.cash_balance:.2f} "
            f"positions={len(self.portfolio)} "
            f"memories={len(self.memory)}>"
        )
