# agents/roles 模块
"""
分析师角色模块

包含:
- Analyst 基类
- FundamentalAnalyst 基本面分析
- TechnicalAnalyst 技术分析
- SentimentAnalyst 情绪分析
- Signal 信号数据类
"""

from agents.roles.analyst import (
    Analyst,
    FundamentalAnalyst,
    TechnicalAnalyst,
    SentimentAnalyst,
    Signal,
    SignalType,
    tool,
)
from agents.roles.news_analyst import NewsAnalyst
from agents.roles.quant_analyst import QuantAnalyst
from agents.roles.risk_analyst import RiskAnalyst
from agents.roles.national_team import NationalTeamAgent

__all__ = [
    "Analyst",
    "FundamentalAnalyst",
    "TechnicalAnalyst",
    "SentimentAnalyst",
    "Signal",
    "SignalType",
    "tool",
    "NationalTeamAgent",
    "NewsAnalyst",
    "QuantAnalyst",
    "RiskAnalyst",
]
