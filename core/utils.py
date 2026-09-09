"""
通用工具函数
"""

import math
from typing import Optional

class PriceQuantizer:
    """
    价格量化器
    
    处理 A 股价格规则：
    1. 最小变动单位 (Tick Size) = 0.01
    2. 涨跌停价格计算 (四舍五入到分)
    """
    
    @staticmethod
    def quantize(price: float) -> float:
        """量化价格到 0.01"""
        return round(price, 2)
    
    @staticmethod
    def get_limit_prices(prev_close: float, limit_ratio: float = 0.10) -> tuple[float, float]:
        """
        计算涨跌停价格
        
        A 股规则：
        涨停价 = round(前收盘 * (1 + 涨跌幅比例), 2)
        跌停价 = round(前收盘 * (1 - 涨跌幅比例), 2)
        """
        upper = round(prev_close * (1 + limit_ratio), 2)
        lower = round(prev_close * (1 - limit_ratio), 2)
        return lower, upper

def truncate_text(text: str, max_length: int = 500) -> str:
    """
    截断长文本，保留关键信息
    """
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "...(truncated)"
