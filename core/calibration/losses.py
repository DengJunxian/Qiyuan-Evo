"""Loss functions for replay calibration."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _arr(values: Sequence[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def rmse(left: Sequence[float], right: Sequence[float]) -> float:
    a = _arr(left)
    b = _arr(right)
    n = min(a.size, b.size)
    if n <= 0:
        return 0.0
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


def returns(prices: Sequence[float]) -> np.ndarray:
    p = _arr(prices)
    if p.size < 2:
        return np.asarray([], dtype=float)
    return np.diff(p) / np.maximum(p[:-1], 1e-12)


def realized_volatility(prices: Sequence[float]) -> float:
    r = returns(prices)
    return float(np.std(r)) if r.size else 0.0


def max_drawdown(prices: Sequence[float]) -> float:
    p = _arr(prices)
    if p.size <= 0:
        return 0.0
    peaks = np.maximum.accumulate(p)
    dd = p / np.maximum(peaks, 1e-12) - 1.0
    return float(abs(np.min(dd)))


def ks_distance(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.sort(_arr(left))
    b = np.sort(_arr(right))
    if a.size == 0 or b.size == 0:
        return 0.0
    values = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, values, side="right") / a.size
    cdf_b = np.searchsorted(b, values, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def direction_hit_rate(sim_prices: Sequence[float], real_prices: Sequence[float]) -> float:
    sim = returns(sim_prices)
    real = returns(real_prices)
    n = min(sim.size, real.size)
    if n <= 0:
        return 0.0
    return float(np.mean(np.sign(sim[:n]) == np.sign(real[:n])))


def calibration_loss(
    *,
    sim_close: Sequence[float],
    real_close: Sequence[float],
    sim_csad: Sequence[float] | None = None,
    target_csad: float = 0.0,
    weights: Mapping[str, float] | None = None,
) -> float:
    weights = {
        "path": 1.0,
        "volatility": 1.0,
        "ks": 1.0,
        "csad": 0.5,
        "drawdown": 1.0,
        **dict(weights or {}),
    }
    sim = _arr(sim_close)
    real = _arr(real_close)
    n = min(sim.size, real.size)
    if n <= 0:
        return 0.0
    sim_norm = sim[:n] / max(sim[0], 1e-12)
    real_norm = real[:n] / max(real[0], 1e-12)
    sim_ret = returns(sim[:n])
    real_ret = returns(real[:n])
    csad_mean = float(np.mean(_arr(sim_csad or []))) if sim_csad else 0.0
    return float(
        weights["path"] * rmse(sim_norm, real_norm)
        + weights["volatility"] * abs(realized_volatility(sim[:n]) - realized_volatility(real[:n]))
        + weights["ks"] * ks_distance(sim_ret, real_ret)
        + weights["csad"] * abs(csad_mean - float(target_csad))
        + weights["drawdown"] * abs(max_drawdown(sim[:n]) - max_drawdown(real[:n]))
    )


__all__ = [
    "calibration_loss",
    "direction_hit_rate",
    "ks_distance",
    "max_drawdown",
    "realized_volatility",
    "returns",
    "rmse",
]
