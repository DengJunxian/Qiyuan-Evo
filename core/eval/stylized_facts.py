"""Behavioral-finance metrics used by the unified evaluation suite."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from core.behavioral_finance import calculate_csad, herding_intensity


def _clean(values: Sequence[float] | None) -> np.ndarray:
    arr = np.asarray([] if values is None else list(values), dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    return arr[np.isfinite(arr)]


def _returns(prices: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(prices), dtype=float)
    if arr.size < 2:
        return np.asarray([], dtype=float)
    return np.diff(arr) / np.maximum(arr[:-1], 1e-12)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _pgr_plr_from_trades(trades: Sequence[Mapping[str, Any]] | None) -> tuple[float, float]:
    if not trades:
        return 0.0, 0.0
    gain_realized = 0.0
    gain_opportunities = 0.0
    loss_realized = 0.0
    loss_opportunities = 0.0
    for trade in trades:
        pnl = _safe_float(trade.get("pnl", trade.get("return", trade.get("realized_return", 0.0))))
        action = str(trade.get("action", trade.get("side", ""))).lower()
        is_sell = action in {"sell", "short", "close", "reduce"}
        if pnl >= 0:
            gain_opportunities += 1.0
            gain_realized += 1.0 if is_sell else 0.0
        else:
            loss_opportunities += 1.0
            loss_realized += 1.0 if is_sell else 0.0
    pgr = gain_realized / gain_opportunities if gain_opportunities > 0 else 0.0
    plr = loss_realized / loss_opportunities if loss_opportunities > 0 else 0.0
    return float(pgr), float(plr)


def _pgr_plr_from_returns(ret: np.ndarray) -> tuple[float, float]:
    if ret.size == 0:
        return 0.0, 0.0
    gains = int(np.sum(ret > 0))
    losses = int(np.sum(ret < 0))
    pgr = gains / ret.size
    plr = losses / ret.size
    return float(pgr), float(plr)


def _ath_effect(prices: np.ndarray) -> float:
    if prices.size < 3:
        return 0.0
    ret = _returns(prices)
    prior_high = np.maximum.accumulate(prices[:-1])
    is_ath = prices[:-1] >= prior_high - 1e-12
    if not np.any(is_ath) or not np.any(~is_ath):
        return 0.0
    return float(np.mean(ret[is_ath]) - np.mean(ret[~is_ath]))


def compute_behavioral_metrics(
    *,
    prices: Sequence[float] | None = None,
    returns_series: Sequence[float] | None = None,
    cross_sectional_returns: Sequence[Sequence[float]] | None = None,
    trades: Sequence[Mapping[str, Any]] | None = None,
    loss_aversion_series: Sequence[float] | None = None,
    market_returns: Sequence[float] | None = None,
) -> Dict[str, float]:
    """Compute stable behavioral metrics from optional replay artifacts."""

    price_arr = _clean(prices)
    ret = _clean(returns_series)
    if ret.size == 0 and price_arr.size >= 2:
        ret = _returns(price_arr)
    mkt_ret = _clean(market_returns)
    if mkt_ret.size == 0:
        mkt_ret = ret

    csad_values: list[float] = []
    if cross_sectional_returns is not None:
        for idx, row in enumerate(cross_sectional_returns):
            row_arr = _clean(row)
            market_return = float(mkt_ret[idx]) if idx < mkt_ret.size else float(np.mean(row_arr)) if row_arr.size else 0.0
            csad_values.append(float(calculate_csad(row_arr, market_return)))
    csad_mean = float(np.mean(csad_values)) if csad_values else 0.0
    market_abs_return = float(np.mean(np.abs(mkt_ret))) if mkt_ret.size else 0.0
    herd = float(herding_intensity(csad_mean, market_abs_return, baseline_csad=max(0.02, csad_mean or 0.02)))

    pgr, plr = _pgr_plr_from_trades(trades)
    if pgr == 0.0 and plr == 0.0:
        pgr, plr = _pgr_plr_from_returns(ret)

    loss_aversion = _clean(loss_aversion_series)
    loss_intensity = float(np.mean(loss_aversion)) if loss_aversion.size else 0.0
    disposition_gap = float(pgr - plr)
    return {
        "csad": float(csad_mean),
        "csad_mean": float(csad_mean),
        "pgr": float(pgr),
        "plr": float(plr),
        "disposition_gap": disposition_gap,
        "ath_effect": _ath_effect(price_arr),
        "loss_aversion_intensity": loss_intensity,
        "herd_intensity": herd,
    }


__all__ = ["compute_behavioral_metrics"]
