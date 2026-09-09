"""
QuantAnalyst —— 量化指标分析师。

论文思想映射（FINCON / QuantAgents）：
1) 量化分工：将 CSAD、动量等指标从主决策中剥离为独立角色。
2) 结构化协作：以 JSON 报告输出，供 ManagerAgent 聚合。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from agents.base_agent import MarketSnapshot
from core.behavioral_finance import calculate_csad, herding_intensity


class QuantAnalyst:
    """
    量化指标分析师：计算 CSAD（羊群效应）、动量等量化指标。
    """

    def analyze(
        self,
        snapshot: MarketSnapshot,
        *,
        returns: Optional[List[float]] = None,
        price_series: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Args:
            snapshot: 当前市场快照
            returns: 可选的收益率序列
            price_series: 可选的价格序列

        Returns:
            JSON 风格字典：CSAD、动量、羊群强度等
        """
        try:
            rts = self._ensure_returns(returns, price_series)
            market_return = float(np.mean(rts)) if rts else float(getattr(snapshot, "market_trend", 0.0))
            csad = float(calculate_csad(np.array(rts), market_return)) if rts else 0.0
            herding = float(herding_intensity(csad, market_return)) if rts else 0.0
            momentum = float(self._calculate_momentum(price_series)) if price_series else float(getattr(snapshot, "market_trend", 0.0))
            return {
                "analyst": "QuantAnalyst",
                "status": "ok",
                "csad": csad,
                "market_return": market_return,
                "herding_intensity": herding,
                "momentum": momentum,
            }
        except Exception as exc:
            return {
                "analyst": "QuantAnalyst",
                "status": "error",
                "error": str(exc),
                "csad": 0.0,
                "market_return": 0.0,
                "herding_intensity": 0.0,
                "momentum": 0.0,
            }

    @staticmethod
    def _ensure_returns(
        returns: Optional[List[float]],
        price_series: Optional[List[float]]
    ) -> List[float]:
        if returns:
            return [float(x) for x in returns if x is not None]
        if price_series and len(price_series) > 1:
            prices = [float(x) for x in price_series if x is not None]
            return [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
        return []

    @staticmethod
    def _calculate_momentum(price_series: Optional[List[float]], window: int = 5) -> float:
        if not price_series or len(price_series) < window + 1:
            return 0.0
        recent = price_series[-(window + 1):]
        start = float(recent[0])
        end = float(recent[-1])
        if start == 0:
            return 0.0
        return (end - start) / start
