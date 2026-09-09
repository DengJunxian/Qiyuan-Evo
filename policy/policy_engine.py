"""
宏观政策引擎生成模块 (Policy Event Generation)
"""

import random
from typing import Optional

class PolicyEngine:
    def __init__(self):
        # 一些预设的宏观事件库
        self.policies = [
            "中央银行出乎意料地将基准利率上调了75个基点。(Central Bank hikes interest rates unexpectedly by 75 bps)",
            "财报季开启，市场整体盈利预期向好。(Earnings season starts with positive broad market outlook)",
            "地缘政治冲突升级，导致全球能源价格飙升。(Geopolitical conflict escalates, energy prices soar)",
            "政府发布了针对新能源行业的巨额补贴计划。(Government announces massive subsidies for the renewable energy sector)",
            "监管机构宣布将大幅提高短期交易的印花税以抑制投机。(Regulators announce sharp increase in stamp duty for short-term trading to curb speculation)"
        ]

    def emit_policy(self, tick: int) -> Optional[str]:
        """
        在特定的 Tick 或者随机发射宏观政策事件。
        
        Args:
            tick (int): 当前仿真步数
            
        Returns:
            Optional[str]: 政策文本或 None
        """
        # 确定性事件：比如 tick 5 必然加息
        if tick == 5:
            return self.policies[0]
        if tick == 10:
            return self.policies[3]
        
        # 随机事件：5%概率发生
        if random.random() < 0.05:
            return random.choice(self.policies)
        
        return None
