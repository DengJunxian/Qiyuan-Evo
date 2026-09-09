"""Reusable risk/performance analytics utilities.

This module is intentionally decoupled from the execution engine so the same
metrics can be reused by simulation, backtest, and counterfactual pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

try:
    import empyrical as ep
except Exception:  # pragma: no cover - optional package
    ep = None


def _to_numpy(values: Optional[Sequence[float] | np.ndarray]) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.asarray([float(arr)], dtype=float)
    return arr[np.isfinite(arr)]


def _annualized_return(total_return: float, periods: int, periods_per_year: int = 252) -> float:
    if periods <= 0:
        return 0.0
    growth = 1.0 + float(total_return)
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / periods) - 1.0)


def _annualized_sharpe(returns: np.ndarray, risk_free_rate: float, periods_per_year: int) -> float:
    if returns.size < 2:
        return 0.0
    excess = returns - risk_free_rate / max(periods_per_year, 1)
    std = float(np.std(excess))
    if std < 1e-12:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def _annualized_sortino(returns: np.ndarray, risk_free_rate: float, periods_per_year: int) -> float:
    if returns.size < 2:
        return 0.0
    excess = returns - risk_free_rate / max(periods_per_year, 1)
    downside = excess[excess < 0]
    downside_std = float(np.std(downside)) if downside.size > 0 else 0.0
    if downside_std < 1e-12:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = equity / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(drawdowns)) if drawdowns.size > 0 else 0.0


def _alpha_beta(strategy_returns: np.ndarray, benchmark_returns: np.ndarray, risk_free_rate: float) -> tuple[float, float]:
    if strategy_returns.size < 2 or benchmark_returns.size < 2:
        return 0.0, 0.0
    n = min(strategy_returns.size, benchmark_returns.size)
    sr = strategy_returns[:n]
    br = benchmark_returns[:n]

    var_b = float(np.var(br))
    if var_b < 1e-12:
        return 0.0, 0.0
    cov = float(np.cov(sr, br)[0, 1])
    beta = cov / var_b
    alpha_daily = float(np.mean(sr - risk_free_rate / 252.0) - beta * np.mean(br - risk_free_rate / 252.0))
    return alpha_daily * 252.0, float(beta)


def _information_ratio(excess_returns: np.ndarray, periods_per_year: int = 252) -> float:
    if excess_returns.size < 2:
        return 0.0
    std = float(np.std(excess_returns))
    if std < 1e-12:
        return 0.0
    return float(np.mean(excess_returns) / std * np.sqrt(periods_per_year))


def _omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    if returns.size == 0:
        return 0.0
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    denom = float(np.sum(losses))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(gains) / denom)


def _tail_ratio(returns: np.ndarray) -> float:
    if returns.size < 5:
        return 0.0
    p95 = float(np.quantile(returns, 0.95))
    p05 = float(np.quantile(returns, 0.05))
    if abs(p05) < 1e-12:
        return 0.0
    return float(p95 / abs(p05))


def _stability(returns: np.ndarray) -> float:
    if returns.size < 6:
        return 0.0
    cum = np.cumsum(returns)
    x = np.arange(cum.size, dtype=float)
    if float(np.std(cum)) < 1e-12:
        return 0.0
    corr = np.corrcoef(x, cum)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(max(0.0, min(1.0, corr**2)))


def compute_backtest_credibility(
    metrics: Mapping[str, Any],
    *,
    sample_count: Optional[int] = None,
    avg_turnover: Optional[float] = None,
) -> float:
    """Compute a compact credibility score in [0, 1].

    The score rewards stable risk-adjusted returns and penalizes sparse data,
    deep drawdowns, and unrealistic turnover.
    """

    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    information_ratio = float(metrics.get("information_ratio", 0.0))
    max_drawdown = abs(float(metrics.get("max_drawdown", 0.0)))
    win_rate = float(metrics.get("win_rate", 0.0))
    annual_volatility = float(metrics.get("annual_volatility", 0.0))

    n_obs = int(sample_count if sample_count is not None else metrics.get("n_obs", 0))
    sample_factor = min(1.0, n_obs / 252.0) if n_obs > 0 else 0.0

    risk_quality = (
        0.35 * np.clip((sharpe + 1.0) / 3.0, 0.0, 1.0)
        + 0.20 * np.clip((information_ratio + 1.0) / 3.0, 0.0, 1.0)
        + 0.20 * np.clip(win_rate, 0.0, 1.0)
        + 0.25 * np.clip(1.0 - max_drawdown / 0.5, 0.0, 1.0)
    )

    vol_penalty = np.clip(annual_volatility / 1.0, 0.0, 1.0) * 0.15

    turnover_penalty = 0.0
    if avg_turnover is not None:
        turnover_penalty = 0.10 * np.clip((float(avg_turnover) - 0.6) / 0.8, 0.0, 1.0)

    score = risk_quality * sample_factor - vol_penalty - turnover_penalty
    return float(np.clip(score, 0.0, 1.0))


def compute_performance_metrics(
    returns: Sequence[float] | np.ndarray,
    benchmark_returns: Optional[Sequence[float] | np.ndarray] = None,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Compute risk/performance metrics with empyrical-compatible semantics."""

    r = _to_numpy(returns)
    b = _to_numpy(benchmark_returns)
    if r.size <= 1:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "annual_volatility": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "information_ratio": 0.0,
            "win_rate": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "omega_ratio": 0.0,
            "tail_ratio": 0.0,
            "stability": 0.0,
            "n_obs": float(r.size),
        }

    total_return = float(np.prod(1.0 + r) - 1.0)

    if ep is not None:
        try:
            annual_return = float(ep.annual_return(r, annualization=periods_per_year))
        except Exception:
            annual_return = _annualized_return(total_return, r.size, periods_per_year=periods_per_year)

        try:
            sharpe_ratio = float(ep.sharpe_ratio(r, risk_free=risk_free_rate, annualization=periods_per_year) or 0.0)
        except Exception:
            sharpe_ratio = _annualized_sharpe(r, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)

        try:
            sortino_ratio = float(ep.sortino_ratio(r, required_return=risk_free_rate, annualization=periods_per_year) or 0.0)
        except Exception:
            sortino_ratio = _annualized_sortino(r, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)

        try:
            max_drawdown = float(ep.max_drawdown(r))
        except Exception:
            max_drawdown = _max_drawdown_from_returns(r)

        try:
            annual_volatility = float(ep.annual_volatility(r, annualization=periods_per_year))
        except Exception:
            annual_volatility = float(np.std(r) * np.sqrt(periods_per_year))
    else:
        annual_return = _annualized_return(total_return, r.size, periods_per_year=periods_per_year)
        sharpe_ratio = _annualized_sharpe(r, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
        sortino_ratio = _annualized_sortino(r, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
        max_drawdown = _max_drawdown_from_returns(r)
        annual_volatility = float(np.std(r) * np.sqrt(periods_per_year))

    calmar_ratio = float(annual_return / abs(max_drawdown)) if abs(max_drawdown) > 1e-12 else 0.0

    alpha = 0.0
    beta = 0.0
    information_ratio = 0.0
    if b.size > 1:
        n = min(r.size, b.size)
        r_n = r[:n]
        b_n = b[:n]
        if ep is not None:
            try:
                alpha, beta = ep.alpha_beta(r_n, b_n, risk_free=risk_free_rate, annualization=periods_per_year)
                alpha = float(alpha or 0.0)
                beta = float(beta or 0.0)
            except Exception:
                alpha, beta = _alpha_beta(r_n, b_n, risk_free_rate)
        else:
            alpha, beta = _alpha_beta(r_n, b_n, risk_free_rate)
        information_ratio = _information_ratio(r_n - b_n, periods_per_year=periods_per_year)

    var_95 = float(np.quantile(r, 0.05))
    cvar_slice = r[r <= var_95]
    cvar_95 = float(np.mean(cvar_slice)) if cvar_slice.size > 0 else var_95

    metrics = {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "annual_volatility": annual_volatility,
        "alpha": alpha,
        "beta": beta,
        "information_ratio": information_ratio,
        "win_rate": float(np.mean(r > 0)),
        "var_95": var_95,
        "cvar_95": cvar_95,
        "omega_ratio": _omega_ratio(r),
        "tail_ratio": _tail_ratio(r),
        "stability": _stability(r),
        "n_obs": float(r.size),
    }
    return metrics

