"""
中央限价订单簿 (CLOB) 撮合引擎 或 简化价格冲击模型
"""

import logging
from agents.trading_agent_core import GLOBAL_CONFIG

logger = logging.getLogger("civitas.engine.market_match")
logger.setLevel(logging.INFO)

def calculate_new_price(buy_volume: float, sell_volume: float, current_price: float) -> float:
    """
    简化的价格冲击模型 (Price Impact Model)。
    通过衡量买卖力量的不平衡来决定新价格。

    Args:
        buy_volume (float): 总买入量或买入资金
        sell_volume (float): 总卖出量或卖出资金
        current_price (float): 当前市场价格

    Returns:
        float: 冲击后的新价格
    """
    total_volume = buy_volume + sell_volume
    if total_volume <= 0:
        return current_price
        
    # 不平衡度: [-1.0, 1.0]
    imbalance = (buy_volume - sell_volume) / total_volume
    
    # 价格冲击系数
    impact_factor = GLOBAL_CONFIG.get("price_impact_factor", 0.05)
    
    # 根据不平衡度和冲击系数计算价格变化率
    price_change_pct = imbalance * impact_factor
    new_price = current_price * (1.0 + price_change_pct)
    
    # 限制最小价格，防止跌穿
    new_price = max(0.01, new_price)
    
    logger.debug(f"Matching Engine: BuyVol={buy_volume:.2f}, SellVol={sell_volume:.2f}, "
                 f"Imbalance={imbalance:.3f} -> New Price: {new_price:.4f}")
                 
    return round(new_price, 4)
