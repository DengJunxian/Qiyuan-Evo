"""
RiskAnalyst —— 风险分析师。

论文思想映射（FINCON / QuantAgents）：
1) 风险角色分离：独立估计 CVaR 与最大回撤；
2) JSON 输出：供 ManagerAgent 进行风控会议触发与策略切换。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class RiskAnalyst:
    """
    风险分析师：计算 CVaR 与最大回撤。
    """

    def analyze(self, portfolio_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """
        Args:
            portfolio_values: 资产净值序列
            alpha: CVaR 置信水平（默认 5%）

        Returns:
            JSON 风格字典：cvar、max_drawdown 等
        """
        try:
            if portfolio_values is None or len(portfolio_values) < 2:
                return {
                    "analyst": "RiskAnalyst",
                    "status": "ok",
                    "cvar": 0.0,
                    "max_drawdown": 0.0,
                    "sample_size": len(portfolio_values) if portfolio_values else 0,
                }

            values = np.array(portfolio_values, dtype=float)
            returns = values[1:] / values[:-1] - 1.0
            cvar = float(self._cvar(returns, alpha))
            mdd = float(self._max_drawdown(values))
            return {
                "analyst": "RiskAnalyst",
                "status": "ok",
                "cvar": cvar,
                "max_drawdown": mdd,
                "sample_size": int(len(values)),
            }
        except Exception as exc:
            return {
                "analyst": "RiskAnalyst",
                "status": "error",
                "error": str(exc),
                "cvar": 0.0,
                "max_drawdown": 0.0,
                "sample_size": len(portfolio_values) if portfolio_values else 0,
            }

    @staticmethod
    def _cvar(returns: np.ndarray, alpha: float) -> float:
        if returns.size == 0:
            return 0.0
        sorted_returns = np.sort(returns)
        cutoff_index = max(1, int(np.floor(alpha * len(sorted_returns))))
        tail = sorted_returns[:cutoff_index]
        return float(np.mean(tail))

    @staticmethod
    def _max_drawdown(values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (v - peak) / peak if peak > 0 else 0.0
            if dd < max_dd:
                max_dd = dd
        return abs(max_dd)
