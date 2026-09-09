"""
金融工具模块 - FinRobot 集成

提供金融语义理解和 CoT 推理模板，支持：
1. FinRobot 完整功能 (可选依赖)
2. 内置 Fallback 实现

FinRobot: https://github.com/AI4Finance-Foundation/FinRobot
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# ==========================================
# FinRobot 条件导入
# ==========================================

try:
    from finrobot.agents import FinRobotAgent
    from finrobot.data_sources import FMPDataSource
    from finrobot.llm_client import LLMClient
    FINROBOT_AVAILABLE = True
except ImportError:
    FINROBOT_AVAILABLE = False
    FinRobotAgent = None
    FMPDataSource = None
    LLMClient = None


# ==========================================
# 数据结构
# ==========================================

@dataclass
class NewsSentiment:
    """新闻情绪分析结果"""
    headline: str
    sentiment: str
    confidence: float
    entities: List[str]  # 提及的公司/概念


@dataclass
class EarningsAnalysis:
    """财报分析结果"""
    ticker: str
    quarter: str
    revenue_surprise: float  # 收入超预期比例
    eps_surprise: float      # EPS 超预期比例
    guidance: str
    key_insights: List[str]


# ==========================================
# CoT 推理模板 (FinRobot 风格)
# ==========================================

COT_INVESTMENT_PROMPT = """
You are a professional investment analyst. Analyze the following information using Chain-of-Thought reasoning.

【Step 1: Macro Assessment】
Evaluate the overall market environment:
- Market trend: {trend}
- Panic index: {panic_level:.2f}
- CSAD (Herding indicator): {csad:.4f}

【Step 2: News Sentiment】
{news_summary}

【Step 3: Technical Analysis】
Current price: {price:.2f}
Your reference point: {reference_point:.2f}
Price deviation: {price_deviation:.1%}

【Step 4: Psychological State】
- Loss aversion coefficient (λ): {lambda_coeff}
- Confidence level: {confidence_desc}
- Anchoring bias: {anchor_desc}

【Step 5: Risk Assessment】
Based on the above, assess:
1. What is the probability of further decline?
2. What is the potential upside?
3. What is your risk tolerance given current PnL ({pnl_pct:.1%})?

【Step 6: Decision】
Synthesize all factors and make a decision.
Output your decision in JSON format:
```json
{{
    "action": "BUY" | "SELL" | "HOLD",
    "ticker": "{symbol}",
    "quantity": <int>,
    "confidence": <0.0-1.0>,
    "reasoning": "<brief reasoning>"
}}
```
"""

COT_NEWS_ANALYSIS_PROMPT = """
Analyze the following financial news and extract:
1. Overall sentiment (positive/negative/neutral)
2. Key entities mentioned (companies, sectors)
3. Potential market impact

News:
{news_text}

Output in JSON format:
```json
{{
    "sentiment": "positive" | "negative" | "neutral",
    "confidence": <0.0-1.0>,
    "entities": ["entity1", "entity2"],
    "impact_summary": "<brief summary>"
}}
```
"""

COT_EARNINGS_PROMPT = """
Analyze the following earnings report summary:

Company: {ticker}
Quarter: {quarter}
Revenue: {revenue} (Est: {revenue_est})
EPS: {eps} (Est: {eps_est})
Guidance: {guidance_text}

Provide:
1. Revenue surprise percentage
2. EPS surprise percentage
3. Outlook assessment
4. Key takeaways for investors

Output in JSON format:
```json
{{
    "revenue_surprise_pct": <float>,
    "eps_surprise_pct": <float>,
    "outlook": "bullish" | "bearish" | "neutral",
    "key_insights": ["insight1", "insight2"]
}}
```
"""


# ==========================================
# FinancialTools 类
# ==========================================

class FinancialTools:
    """
    金融工具集
    
    提供 FinRobot 风格的金融分析能力：
    - 新闻情绪分析
    - 财报解读
    - CoT 投资推理模板
    
    如果 FinRobot 可用，使用其完整功能；
    否则使用内置的简化实现。
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化金融工具
        
        Args:
            api_key: LLM API Key (用于 FinRobot 或内置分析)
        """
        self.api_key = api_key
        self._finrobot_agent = None
        
        if FINROBOT_AVAILABLE and api_key:
            try:
                self._finrobot_agent = FinRobotAgent(api_key=api_key)
            except Exception:
                pass
    
    @property
    def is_finrobot_available(self) -> bool:
        return self._finrobot_agent is not None
    
    def get_cot_investment_prompt(
        self,
        market_state: Dict,
        account_state: Dict,
        symbol: str = "000001",
        confidence_desc: str = "中性",
        anchor_desc: str = "",
        news_summary: str = "无重大新闻"
    ) -> str:
        """
        获取 CoT 投资推理 Prompt
        
        使用 FinRobot 风格的多步推理模板。
        
        Args:
            market_state: 市场状态
            account_state: 账户状态
            symbol: 标的代码
            confidence_desc: 自信程度描述
            anchor_desc: 锚定偏差描述
            news_summary: 新闻摘要
            
        Returns:
            格式化的 Prompt 字符串
        """
        price = market_state.get("price", 0)
        avg_cost = account_state.get("avg_cost", price)
        
        reference_point = avg_cost if avg_cost > 0 else price
        price_deviation = (price - reference_point) / reference_point if reference_point > 0 else 0
        
        return COT_INVESTMENT_PROMPT.format(
            trend=market_state.get("trend", "未知"),
            panic_level=market_state.get("panic_level", 0.5),
            csad=market_state.get("csad", 0),
            news_summary=news_summary,
            price=price,
            reference_point=reference_point,
            price_deviation=price_deviation,
            lambda_coeff=2.25,
            confidence_desc=confidence_desc,
            anchor_desc=anchor_desc,
            pnl_pct=account_state.get("pnl_pct", 0),
            symbol=symbol
        )
    
    def parse_financial_news(self, news_text: str) -> NewsSentiment:
        """
        解析金融新闻情绪
        
        Args:
            news_text: 新闻文本
            
        Returns:
            NewsSentiment 分析结果
        """
        if self._finrobot_agent:
            # 使用 FinRobot 的完整分析
            try:
                result = self._finrobot_agent.analyze_news(news_text)
                return NewsSentiment(
                    headline=news_text[:100],
                    sentiment=result.get("sentiment", "neutral"),
                    confidence=result.get("confidence", 0.5),
                    entities=result.get("entities", [])
                )
            except Exception:
                pass
        
        # 回退: 简单关键词分析
        return self._fallback_news_analysis(news_text)
    
    def _fallback_news_analysis(self, news_text: str) -> NewsSentiment:
        """Fallback 新闻分析 (无 FinRobot)"""
        text_lower = news_text.lower()
        
        positive_words = ["上涨", "突破", "利好", "增长", "超预期", "买入", 
                         "rise", "surge", "bullish", "growth", "beat"]
        negative_words = ["下跌", "跌破", "利空", "下降", "不及预期", "卖出",
                         "fall", "crash", "bearish", "decline", "miss"]
        
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        if pos_count > neg_count:
            sentiment = "positive"
            confidence = min(0.9, 0.5 + 0.1 * pos_count)
        elif neg_count > pos_count:
            sentiment = "negative"
            confidence = min(0.9, 0.5 + 0.1 * neg_count)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        # 简单实体提取
        entities = re.findall(r'[A-Z]{2,5}|[0-9]{6}', news_text)
        
        return NewsSentiment(
            headline=news_text[:100],
            sentiment=sentiment,
            confidence=confidence,
            entities=entities[:5]
        )
    
    def analyze_earnings(
        self,
        ticker: str,
        quarter: str,
        revenue: float,
        revenue_est: float,
        eps: float,
        eps_est: float,
        guidance_text: str = ""
    ) -> EarningsAnalysis:
        """
        分析财报数据
        
        Args:
            ticker: 股票代码
            quarter: 财报季度 (如 "Q3 2024")
            revenue: 实际收入
            revenue_est: 预期收入
            eps: 实际 EPS
            eps_est: 预期 EPS
            guidance_text: 指引文本
            
        Returns:
            EarningsAnalysis 结果
        """
        revenue_surprise = (revenue - revenue_est) / revenue_est if revenue_est > 0 else 0
        eps_surprise = (eps - eps_est) / eps_est if eps_est > 0 else 0
        
        # 判断指引
        guidance_lower = guidance_text.lower()
        if any(w in guidance_lower for w in ["raised", "上调", "增加"]):
            guidance = "raised"
        elif any(w in guidance_lower for w in ["lowered", "下调", "减少"]):
            guidance = "lowered"
        else:
            guidance = "maintained"
        
        # 生成洞察
        insights = []
        if revenue_surprise > 0.05:
            insights.append(f"收入超预期 {revenue_surprise:.1%}")
        elif revenue_surprise < -0.05:
            insights.append(f"收入不及预期 {revenue_surprise:.1%}")
        
        if eps_surprise > 0.10:
            insights.append(f"EPS 大幅超预期 {eps_surprise:.1%}")
        elif eps_surprise < -0.10:
            insights.append(f"EPS 大幅不及预期 {eps_surprise:.1%}")
        
        if guidance == "raised":
            insights.append("管理层上调全年指引，态度乐观")
        elif guidance == "lowered":
            insights.append("管理层下调指引，需警惕")
        
        return EarningsAnalysis(
            ticker=ticker,
            quarter=quarter,
            revenue_surprise=revenue_surprise,
            eps_surprise=eps_surprise,
            guidance=guidance,
            key_insights=insights or ["财报表现符合预期"]
        )
    
    def get_news_analysis_prompt(self, news_text: str) -> str:
        """获取新闻分析 Prompt"""
        return COT_NEWS_ANALYSIS_PROMPT.format(news_text=news_text)
    
    def get_earnings_prompt(
        self,
        ticker: str,
        quarter: str,
        revenue: float,
        revenue_est: float,
        eps: float,
        eps_est: float,
        guidance_text: str
    ) -> str:
        """获取财报分析 Prompt"""
        return COT_EARNINGS_PROMPT.format(
            ticker=ticker,
            quarter=quarter,
            revenue=revenue,
            revenue_est=revenue_est,
            eps=eps,
            eps_est=eps_est,
            guidance_text=guidance_text
        )


# ==========================================
# 便捷函数
# ==========================================

def get_finrobot_status() -> Dict[str, Any]:
    """获取 FinRobot 可用状态"""
    return {
        "finrobot_available": FINROBOT_AVAILABLE,
        "version": getattr(__import__("finrobot", fromlist=[""]), "__version__", "N/A") if FINROBOT_AVAILABLE else "N/A"
    }



