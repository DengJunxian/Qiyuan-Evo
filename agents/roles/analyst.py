"""
分析师角色模块

灵感来源:
- TradingAgents: 多角色分析师架构
- FinRobot: 金融工具集成
- CrewAI/LangChain: @tool 装饰器模式

实现:
1. Analyst 基类定义通用接口
2. FundamentalAnalyst: 财报、PE/PB 分析
3. TechnicalAnalyst: MACD、RSI、形态识别
4. SentimentAnalyst: 市场情绪、恐贪指数

作者: Civitas Economica Team
"""

import time
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum

import numpy as np


# ==========================================
# 工具装饰器 (兼容 CrewAI/LangChain)
# ==========================================

def tool(func: Callable = None, *, name: str = None, description: str = None):
    """
    @tool 装饰器
    
    兼容 CrewAI/LangChain 的工具注册模式。
    将方法标记为可被 Agent 调用的工具。
    
    Usage:
        @tool
        def fetch_data(self, symbol: str) -> dict:
            '''获取数据'''
            pass
            
        @tool(name="pe_analyzer", description="分析市盈率")
        def analyze_pe(self, symbol: str) -> float:
            pass
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        
        # 添加工具元数据
        wrapper._is_tool = True
        wrapper._tool_name = name or fn.__name__
        wrapper._tool_description = description or fn.__doc__ or ""
        
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


# ==========================================
# 数据结构
# ==========================================

class SignalType(str, Enum):
    """信号类型"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class AnalystType(str, Enum):
    """分析师类型"""
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"


@dataclass
class Signal:
    """
    交易信号
    
    由分析师生成，供风控和交易员使用。
    """
    action: SignalType          # 动作
    confidence: float           # 信心 [0, 1]
    analyst_type: AnalystType   # 来源分析师类型
    symbol: str = ""            # 标的
    target_price: Optional[float] = None  # 目标价
    stop_loss: Optional[float] = None     # 止损价
    reasoning: str = ""         # 分析逻辑
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "analyst_type": self.analyst_type.value,
            "symbol": self.symbol,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "reasoning": self.reasoning
        }


@dataclass
class FinancialReport:
    """财务报告数据"""
    symbol: str
    pe_ratio: float = 0.0       # 市盈率
    pb_ratio: float = 0.0       # 市净率
    roe: float = 0.0            # 净资产收益率
    debt_ratio: float = 0.0     # 资产负债率
    revenue_growth: float = 0.0  # 营收增长率
    profit_growth: float = 0.0   # 利润增长率


@dataclass
class TechnicalIndicators:
    """技术指标数据"""
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    rsi: float = 50.0
    ma_5: float = 0.0
    ma_20: float = 0.0
    ma_60: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    volume_ratio: float = 1.0


# ==========================================
# Analyst 基类
# ==========================================

class Analyst(ABC):
    """
    分析师基类
    
    定义所有分析师的通用接口。
    子类需实现 analyze() 方法。
    """
    
    def __init__(
        self,
        agent_id: str,
        specialty: str = "general"
    ):
        """
        初始化分析师
        
        Args:
            agent_id: 分析师 ID
            specialty: 专业领域
        """
        self.agent_id = agent_id
        self.specialty = specialty
        self._tools: Dict[str, Callable] = {}
        self._register_tools()
    
    def _register_tools(self) -> None:
        """自动注册所有 @tool 装饰的方法"""
        for name in dir(self):
            method = getattr(self, name)
            if callable(method) and getattr(method, '_is_tool', False):
                tool_name = getattr(method, '_tool_name', name)
                self._tools[tool_name] = method
    
    @property
    def tools(self) -> Dict[str, Callable]:
        """获取注册的工具列表"""
        return self._tools
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """获取指定工具"""
        return self._tools.get(name)
    
    def list_tools(self) -> List[Dict]:
        """列出所有工具的元信息"""
        return [
            {
                "name": getattr(tool, '_tool_name', name),
                "description": getattr(tool, '_tool_description', '')
            }
            for name, tool in self._tools.items()
        ]
    
    @abstractmethod
    def analyze(
        self,
        symbol: str,
        market_data: Dict,
        **kwargs
    ) -> Signal:
        """
        执行分析并生成信号
        
        Args:
            symbol: 交易标的
            market_data: 市场数据
            
        Returns:
            Signal: 交易信号
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, specialty={self.specialty})"


# ==========================================
# 基本面分析师
# ==========================================

class FundamentalAnalyst(Analyst):
    """
    基本面分析师
    
    分析公司财务指标、估值水平、成长性等。
    """
    
    def __init__(self, agent_id: str = "fundamental_analyst"):
        super().__init__(agent_id, specialty="fundamental")
        self.analyst_type = AnalystType.FUNDAMENTAL
    
    @tool(name="fetch_financial_report", description="获取公司财务报告")
    def fetch_financial_report(self, symbol: str) -> FinancialReport:
        """
        获取公司财务报告 (FinRobot 包装)
        
        注意: 生产环境应调用真实 API
        这里使用模拟数据
        """
        # 模拟财务数据
        np.random.seed(abs(hash(symbol)) % (2**32))
        
        return FinancialReport(
            symbol=symbol,
            pe_ratio=10 + np.random.rand() * 30,
            pb_ratio=0.5 + np.random.rand() * 3,
            roe=0.05 + np.random.rand() * 0.25,
            debt_ratio=0.2 + np.random.rand() * 0.5,
            revenue_growth=-0.1 + np.random.rand() * 0.4,
            profit_growth=-0.2 + np.random.rand() * 0.5
        )
    
    @tool(name="calculate_intrinsic_value", description="计算内在价值")
    def calculate_intrinsic_value(
        self,
        current_price: float,
        report: FinancialReport
    ) -> Dict:
        """
        计算内在价值
        
        使用简化的 DCF 模型估算
        """
        # 简化估值: 基于 PE 和成长性
        fair_pe = 15  # 合理 PE
        growth_premium = max(0, report.profit_growth) * 10
        
        target_pe = fair_pe + growth_premium
        
        # 如果当前 PE 高于目标 PE，说明高估
        if report.pe_ratio > target_pe * 1.2:
            valuation = "overvalued"
            upside = -0.1
        elif report.pe_ratio < target_pe * 0.8:
            valuation = "undervalued"
            upside = 0.2
        else:
            valuation = "fair"
            upside = 0.0
        
        return {
            "current_pe": report.pe_ratio,
            "target_pe": target_pe,
            "valuation": valuation,
            "expected_upside": upside
        }
    
    def analyze(
        self,
        symbol: str,
        market_data: Dict,
        **kwargs
    ) -> Signal:
        """基本面分析"""
        current_price = market_data.get('price', 100)
        
        # 获取财务数据
        report = self.fetch_financial_report(symbol)
        valuation = self.calculate_intrinsic_value(current_price, report)
        
        # 生成信号
        if valuation['valuation'] == 'undervalued':
            action = SignalType.BUY
            confidence = 0.7 + valuation['expected_upside']
            reasoning = f"股票被低估，PE {report.pe_ratio:.1f} 低于目标 PE {valuation['target_pe']:.1f}"
        elif valuation['valuation'] == 'overvalued':
            action = SignalType.SELL
            confidence = 0.6
            reasoning = f"股票被高估，PE {report.pe_ratio:.1f} 高于目标 PE {valuation['target_pe']:.1f}"
        else:
            action = SignalType.HOLD
            confidence = 0.5
            reasoning = f"股票估值合理，PE {report.pe_ratio:.1f} 接近目标 PE {valuation['target_pe']:.1f}"
        
        return Signal(
            action=action,
            confidence=min(1.0, confidence),
            analyst_type=self.analyst_type,
            symbol=symbol,
            target_price=current_price * (1 + valuation['expected_upside']),
            reasoning=reasoning
        )


# ==========================================
# 技术分析师
# ==========================================

class TechnicalAnalyst(Analyst):
    """
    技术分析师
    
    分析价格走势、技术指标、形态识别。
    """
    
    def __init__(self, agent_id: str = "technical_analyst"):
        super().__init__(agent_id, specialty="technical")
        self.analyst_type = AnalystType.TECHNICAL
    
    @tool(name="calculate_macd", description="计算 MACD 指标")
    def calculate_macd(
        self,
        prices: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict:
        """计算 MACD 指标"""
        if len(prices) < slow:
            return {"macd": 0, "signal": 0, "histogram": 0}
        
        prices = np.array(prices)
        
        # EMA 计算
        def ema(data, period):
            alpha = 2 / (period + 1)
            result = np.zeros_like(data)
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        ema_fast = ema(prices, fast)
        ema_slow = ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            "macd": float(macd_line[-1]),
            "signal": float(signal_line[-1]),
            "histogram": float(histogram[-1])
        }
    
    @tool(name="calculate_rsi", description="计算 RSI 指标")
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """计算 RSI 相对强弱指数"""
        if len(prices) < period + 1:
            return 50.0
        
        prices = np.array(prices)
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    @tool(name="detect_pattern", description="识别价格形态")
    def detect_pattern(self, prices: List[float]) -> str:
        """识别价格形态"""
        if len(prices) < 10:
            return "insufficient_data"
        
        recent = np.array(prices[-10:])
        
        # 简单形态识别
        ma_5 = np.mean(prices[-5:])
        ma_10 = np.mean(prices[-10:])
        
        if recent[-1] > ma_5 > ma_10:
            return "uptrend"
        elif recent[-1] < ma_5 < ma_10:
            return "downtrend"
        else:
            return "consolidation"
    
    def analyze(
        self,
        symbol: str,
        market_data: Dict,
        **kwargs
    ) -> Signal:
        """技术分析"""
        prices = market_data.get('prices', [])
        current_price = market_data.get('price', 100)
        
        if len(prices) < 30:
            # 数据不足，生成模拟数据
            prices = [current_price * (1 + np.random.randn() * 0.02) for _ in range(60)]
            prices[-1] = current_price
        
        # 计算指标
        macd = self.calculate_macd(prices)
        rsi = self.calculate_rsi(prices)
        pattern = self.detect_pattern(prices)
        
        # 综合判断
        signals = []
        reasoning_parts = []
        
        # MACD 信号
        if macd['histogram'] > 0 and macd['macd'] > macd['signal']:
            signals.append(1)
            reasoning_parts.append("MACD 金叉，看涨")
        elif macd['histogram'] < 0 and macd['macd'] < macd['signal']:
            signals.append(-1)
            reasoning_parts.append("MACD 死叉，看跌")
        else:
            signals.append(0)
        
        # RSI 信号
        if rsi < 30:
            signals.append(1)
            reasoning_parts.append(f"RSI {rsi:.1f} 超卖，看涨")
        elif rsi > 70:
            signals.append(-1)
            reasoning_parts.append(f"RSI {rsi:.1f} 超买，看跌")
        else:
            signals.append(0)
        
        # 趋势信号
        if pattern == "uptrend":
            signals.append(1)
            reasoning_parts.append("价格处于上升趋势")
        elif pattern == "downtrend":
            signals.append(-1)
            reasoning_parts.append("价格处于下降趋势")
        
        # 汇总
        avg_signal = np.mean(signals)
        
        if avg_signal > 0.3:
            action = SignalType.BUY
            confidence = 0.5 + avg_signal * 0.3
        elif avg_signal < -0.3:
            action = SignalType.SELL
            confidence = 0.5 + abs(avg_signal) * 0.3
        else:
            action = SignalType.HOLD
            confidence = 0.5
        
        return Signal(
            action=action,
            confidence=min(1.0, confidence),
            analyst_type=self.analyst_type,
            symbol=symbol,
            reasoning="；".join(reasoning_parts) if reasoning_parts else "指标中性"
        )


# ==========================================
# 情绪分析师
# ==========================================

class SentimentAnalyst(Analyst):
    """
    情绪分析师
    
    分析市场情绪、新闻舆情、恐贪指数。
    """
    
    def __init__(self, agent_id: str = "sentiment_analyst"):
        super().__init__(agent_id, specialty="sentiment")
        self.analyst_type = AnalystType.SENTIMENT
    
    @tool(name="calculate_fear_greed", description="计算恐贪指数")
    def calculate_fear_greed(
        self,
        volatility: float,
        volume_ratio: float,
        price_momentum: float
    ) -> Dict:
        """
        计算恐贪指数
        
        Args:
            volatility: 波动率
            volume_ratio: 成交量比率 (当前/平均)
            price_momentum: 价格动量 (-1 ~ 1)
        
        Returns:
            恐贪指数 (0=极度恐惧, 100=极度贪婪)
        """

        vol_score = max(0, 50 - volatility * 1000)
        
        # 成交量成分 (放量 = 情绪化)
        vol_ratio_score = 50 if volume_ratio < 1.5 else (70 if price_momentum > 0 else 30)
        
        # 动量成分
        momentum_score = 50 + price_momentum * 30
        
        # 综合
        fear_greed = (vol_score + vol_ratio_score + momentum_score) / 3
        fear_greed = max(0, min(100, fear_greed))
        
        if fear_greed < 25:
            status = "extreme_fear"
        elif fear_greed < 45:
            status = "fear"
        elif fear_greed < 55:
            status = "neutral"
        elif fear_greed < 75:
            status = "greed"
        else:
            status = "extreme_greed"
        
        return {
            "index": fear_greed,
            "status": status,
            "components": {
                "volatility": vol_score,
                "volume": vol_ratio_score,
                "momentum": momentum_score
            }
        }
    
    @tool(name="analyze_news_sentiment", description="分析新闻情绪")
    def analyze_news_sentiment(self, news: List[str]) -> float:
        """
        分析新闻情绪
        
        Returns:
            情绪分数 (-1 = 极度悲观, +1 = 极度乐观)
        """
        if not news:
            return 0.0
        
        # 关键词匹配
        positive_words = ['利好', '上涨', '突破', '增长', '盈利', '牛市', '反弹', '创新高']
        negative_words = ['利空', '下跌', '暴跌', '亏损', '熊市', '崩盘', '下调', '风险']
        
        positive_count = 0
        negative_count = 0
        
        for text in news:
            positive_count += sum(1 for w in positive_words if w in text)
            negative_count += sum(1 for w in negative_words if w in text)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total
    
    def analyze(
        self,
        symbol: str,
        market_data: Dict,
        **kwargs
    ) -> Signal:
        """情绪分析"""
        volatility = market_data.get('volatility', 0.02)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        price_momentum = market_data.get('momentum', 0.0)
        news = market_data.get('news', [])
        panic_level = market_data.get('panic_level', 0.5)
        
        # 计算恐贪指数
        fear_greed = self.calculate_fear_greed(volatility, volume_ratio, price_momentum)
        
        # 分析新闻情绪
        if isinstance(news, str):
            news = [news]
        news_sentiment = self.analyze_news_sentiment(news)
        
        # 综合判断
        reasoning_parts = []
        
        # 恐贪指数信号 (逆向思维)
        if fear_greed['status'] == 'extreme_fear':
            fg_signal = 1
            reasoning_parts.append(f"恐贪指数 {fear_greed['index']:.0f} (极度恐惧)，逆向看涨")
        elif fear_greed['status'] == 'extreme_greed':
            fg_signal = -1
            reasoning_parts.append(f"恐贪指数 {fear_greed['index']:.0f} (极度贪婪)，逆向看跌")
        else:
            fg_signal = 0
            reasoning_parts.append(f"恐贪指数 {fear_greed['index']:.0f} ({fear_greed['status']})")
        
        # 新闻情绪信号
        if news_sentiment > 0.3:
            news_signal = 1
            reasoning_parts.append("新闻情绪偏正面")
        elif news_sentiment < -0.3:
            news_signal = -1
            reasoning_parts.append("新闻情绪偏负面")
        else:
            news_signal = 0
        
        # 市场恐慌信号
        if panic_level > 0.7:
            panic_signal = 1  # 逆向
            reasoning_parts.append(f"市场恐慌度 {panic_level:.2f}，可能超跌")
        elif panic_level < 0.3:
            panic_signal = -1  # 逆向
            reasoning_parts.append("市场乐观度高，警惕回调")
        else:
            panic_signal = 0
        
        # 汇总
        avg_signal = (fg_signal + news_signal + panic_signal) / 3
        
        if avg_signal > 0.2:
            action = SignalType.BUY
            confidence = 0.5 + avg_signal * 0.3
        elif avg_signal < -0.2:
            action = SignalType.SELL
            confidence = 0.5 + abs(avg_signal) * 0.3
        else:
            action = SignalType.HOLD
            confidence = 0.5
        
        return Signal(
            action=action,
            confidence=min(1.0, confidence),
            analyst_type=self.analyst_type,
            symbol=symbol,
            reasoning="；".join(reasoning_parts)
        )



