from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class MarketStatePacket(BaseModel):
    """
    服务端 (Engine) 发送给客户端 (Agent Node) 的市场快照广播
    """
    step: int = Field(..., description="当前仿真步数")
    timestamp: str = Field(..., description="当前仿真时间戳")
    price: float = Field(..., description="最新成交价")
    trend: str = Field(..., description="市场趋势")
    panic_level: float = Field(..., description="恐慌指数")
    csad: float = Field(..., description="羊群效应指标 CSAD")
    volatility: float = Field(..., description="波动率")
    recent_news: List[str] = Field(default_factory=list, description="近期新闻快讯")
    order_book_depth: Dict[str, Any] = Field(default_factory=dict, description="五档盘口快照")

class OrderPayload(BaseModel):
    """交易订单明细"""
    symbol: str = "000001"
    price: float
    quantity: int
    side: str
    order_type: str = "LIMIT"

class AgentActionPacket(BaseModel):
    """
    客户端 (Agent Node) 思考完毕后，向服务端 (Engine) 提交的动作反馈
    """
    agent_id: str = Field(..., description="发起决策的 Agent ID")
    step: int = Field(..., description="响应的是哪一个仿真步数")
    has_order: bool = Field(default=False, description="本次决策是否产生并下达了实际订单")
    order: Optional[OrderPayload] = Field(default=None, description="订单明细")
    sentiment: float = Field(default=0.0, description="当前该 Agent 的多空情绪值 (-1.0 到 1.0)")
    confidence: float = Field(default=50.0, description="智能体的当前信心指数")
