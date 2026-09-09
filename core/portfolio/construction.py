"""Unified portfolio construction layer for Civitas.

Design goals:
- decouple portfolio construction from strategy logic
- keep optional compatibility with PyPortfolioOpt
- inject policy/sentiment risk into optimization pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd


@dataclass
class PortfolioConstraints:
    long_only: bool = True
    fully_invested: bool = True
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_turnover: Optional[float] = None
    industry_caps: Dict[str, float] = field(default_factory=dict)
    sentiment_penalty: float = 0.0
    policy_penalty: float = 0.0
    target_max_drawdown: Optional[float] = None


@dataclass
class PortfolioInput:
    expected_returns: pd.Series
    cov_matrix: pd.DataFrame
    current_weights: Optional[pd.Series] = None
    sentiment_risk: Optional[pd.Series] = None
    policy_risk: Optional[pd.Series] = None
    asset_to_industry: Dict[str, str] = field(default_factory=dict)
    returns_history: Optional[pd.DataFrame] = None


class PortfolioConstructionLayer:
    """Composable optimizer interface (equal/inv-vol/mean-variance/hrp)."""

    def __init__(
        self,
        *,
        method: Literal["equal_weight", "inverse_vol", "mean_variance", "hrp"] = "inverse_vol",
        constraints: Optional[PortfolioConstraints] = None,
        risk_aversion: float = 1.0,
    ):
        self.method = method
        self.constraints = constraints or PortfolioConstraints()
        self.risk_aversion = max(float(risk_aversion), 1e-8)

    @staticmethod
    def _align_input(data: PortfolioInput) -> tuple[pd.Index, pd.Series, pd.DataFrame]:
        mu = data.expected_returns.astype(float).copy()
        cov = data.cov_matrix.astype(float).copy()
        common = mu.index.intersection(cov.index).intersection(cov.columns)
        mu = mu.loc[common]
        cov = cov.loc[common, common]
        return common, mu, cov

    def _adjust_expected_returns(self, assets: pd.Index, mu: pd.Series, data: PortfolioInput) -> pd.Series:
        adjusted = mu.copy()
        if data.sentiment_risk is not None and self.constraints.sentiment_penalty > 0:
            sentiment = data.sentiment_risk.reindex(assets).fillna(0.0).abs()
            adjusted -= self.constraints.sentiment_penalty * sentiment
        if data.policy_risk is not None and self.constraints.policy_penalty > 0:
            policy = data.policy_risk.reindex(assets).fillna(0.0).abs()
            adjusted -= self.constraints.policy_penalty * policy
        return adjusted

    def _equal_weight(self, assets: pd.Index) -> pd.Series:
        n = len(assets)
        if n == 0:
            return pd.Series(dtype=float)
        w = np.ones(n, dtype=float) / n
        return pd.Series(w, index=assets)

    def _inverse_vol(self, assets: pd.Index, cov: pd.DataFrame) -> pd.Series:
        n = len(assets)
        if n == 0:
            return pd.Series(dtype=float)
        vol = np.sqrt(np.maximum(np.diag(cov.values), 1e-12))
        inv = 1.0 / vol
        inv_sum = float(np.sum(inv))
        if inv_sum < 1e-12:
            return self._equal_weight(assets)
        return pd.Series(inv / inv_sum, index=assets)

    def _mean_variance(self, assets: pd.Index, mu: pd.Series, cov: pd.DataFrame) -> pd.Series:

        try:  # pragma: no cover - optional dependency
            from pypfopt import EfficientFrontier

            bounds = (self.constraints.min_weight, self.constraints.max_weight)
            ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
            ef.max_quadratic_utility(risk_aversion=self.risk_aversion)
            raw = ef.clean_weights()
            return pd.Series(raw, index=assets, dtype=float)
        except Exception:
            pass


        cov_values = cov.values + np.eye(len(assets)) * 1e-6
        try:
            inv_cov = np.linalg.inv(cov_values)
        except np.linalg.LinAlgError:
            return self._inverse_vol(assets, cov)
        w = inv_cov @ mu.values
        if self.constraints.long_only:
            w = np.clip(w, 0.0, None)
        if float(np.sum(np.abs(w))) < 1e-12:
            return self._equal_weight(assets)
        w = w / max(float(np.sum(np.abs(w))), 1e-12)
        return pd.Series(w, index=assets, dtype=float)

    def _hrp(self, assets: pd.Index, data: PortfolioInput, cov: pd.DataFrame) -> pd.Series:

        if data.returns_history is None or data.returns_history.empty:
            return self._inverse_vol(assets, cov)
        try:  # pragma: no cover - optional dependency
            from pypfopt import HRPOpt

            returns = data.returns_history.loc[:, assets].dropna(how="all").fillna(0.0)
            if returns.empty:
                return self._inverse_vol(assets, cov)
            hrp = HRPOpt(returns=returns)
            raw = hrp.optimize()
            return pd.Series(raw, index=assets, dtype=float)
        except Exception:
            return self._inverse_vol(assets, cov)

    def _apply_industry_caps(self, weights: pd.Series, data: PortfolioInput) -> pd.Series:
        if not self.constraints.industry_caps:
            return weights
        adjusted = weights.copy()
        capped_assets = set()
        for industry, cap in self.constraints.industry_caps.items():
            members = [a for a in adjusted.index if data.asset_to_industry.get(a) == industry]
            if not members:
                continue
            total = float(adjusted.loc[members].sum())
            if total <= cap + 1e-12:
                continue
            scale = cap / max(total, 1e-12)
            adjusted.loc[members] = adjusted.loc[members] * scale
            capped_assets.update(members)


        if self.constraints.fully_invested and self.constraints.long_only:
            residual = 1.0 - float(adjusted.sum())
            if residual > 1e-12:
                eligible = [a for a in adjusted.index if a not in capped_assets]
                if not eligible:
                    eligible = list(adjusted.index)
                base = adjusted.loc[eligible].clip(lower=0.0)
                if float(base.sum()) < 1e-12:
                    base[:] = 1.0
                adjusted.loc[eligible] = adjusted.loc[eligible] + residual * (base / float(base.sum()))
        return adjusted

    def _apply_turnover_limit(self, weights: pd.Series, data: PortfolioInput) -> pd.Series:
        if self.constraints.max_turnover is None or data.current_weights is None:
            return weights
        current = data.current_weights.reindex(weights.index).fillna(0.0).astype(float)
        diff = weights - current
        turnover = float(np.sum(np.abs(diff)))
        limit = max(float(self.constraints.max_turnover), 0.0)
        if turnover <= limit + 1e-12:
            return weights
        scale = limit / max(turnover, 1e-12)
        return current + diff * scale

    def _apply_drawdown_guard(self, weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
        target = self.constraints.target_max_drawdown
        if target is None or target <= 0:
            return weights
        vol = float(np.sqrt(np.maximum(weights.values @ cov.values @ weights.values, 0.0)))
        estimated_drawdown = 2.0 * vol
        if estimated_drawdown <= target:
            return weights

        ratio = float(np.clip(target / max(estimated_drawdown, 1e-12), 0.0, 1.0))
        if self.constraints.fully_invested:

            defensive = self._inverse_vol(weights.index, cov)
            return ratio * weights + (1.0 - ratio) * defensive
        return weights * ratio

    def _finalize_weights(self, weights: pd.Series) -> pd.Series:
        if weights.empty:
            return weights

        w = weights.astype(float).copy()
        min_w = self.constraints.min_weight if self.constraints.long_only else -self.constraints.max_weight
        max_w = self.constraints.max_weight
        w = w.clip(lower=min_w, upper=max_w)

        if self.constraints.long_only:
            w = w.clip(lower=0.0)
            total = float(w.sum())
            if self.constraints.fully_invested:
                if total < 1e-12:
                    w[:] = 1.0 / len(w)
                else:
                    w = w / total
            else:
                if total > 1.0:
                    w = w / total
            return w


        gross = float(np.sum(np.abs(w.values)))
        if gross < 1e-12:
            return w
        return w / gross

    def optimize(self, data: PortfolioInput) -> pd.Series:
        assets, mu, cov = self._align_input(data)
        if len(assets) == 0:
            return pd.Series(dtype=float)

        mu_adj = self._adjust_expected_returns(assets, mu, data)
        if self.method == "equal_weight":
            raw = self._equal_weight(assets)
        elif self.method == "mean_variance":
            raw = self._mean_variance(assets, mu_adj, cov)
        elif self.method == "hrp":
            raw = self._hrp(assets, data, cov)
        else:
            raw = self._inverse_vol(assets, cov)

        constrained = self._apply_industry_caps(raw, data)
        constrained = self._apply_turnover_limit(constrained, data)
        constrained = self._apply_drawdown_guard(constrained, cov)
        return self._finalize_weights(constrained)
