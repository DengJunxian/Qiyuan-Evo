"""Order-book and execution realism metrics for replay scorecards."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _depth_qty(row: Any) -> float:
    if not isinstance(row, Mapping):
        return 0.0
    for key in ("qty", "quantity", "size", "volume"):
        if key in row:
            return _safe_float(row.get(key))
    return 0.0


def _levels(snapshot: Mapping[str, Any], side: str) -> list[Mapping[str, Any]]:
    depth = snapshot.get("depth", {}) if isinstance(snapshot, Mapping) else {}
    if isinstance(depth, Mapping):
        rows = depth.get(side, []) or []
        return [row for row in rows if isinstance(row, Mapping)]
    return []


def compute_orderbook_realism(
    *,
    snapshots: Sequence[Mapping[str, Any]] | None = None,
    snapshot: Mapping[str, Any] | None = None,
    trade_tape: Sequence[Mapping[str, Any]] | None = None,
    reference_price: float | None = None,
) -> Dict[str, float]:
    """Aggregate spread, depth, impact, cancel/trade and VWAP realism metrics."""

    rows = list(snapshots or ([] if snapshot is None else [snapshot]))
    if not rows:
        rows = [{}]
    trade_rows = [dict(row) for row in list(trade_tape or []) if isinstance(row, Mapping)]

    spreads: list[float] = []
    spread_pcts: list[float] = []
    depth_totals: list[float] = []
    imbalances: list[float] = []
    cancel_trade: list[float] = []
    impact_costs: list[float] = []
    for row in rows:
        best_bid = _safe_float(row.get("best_bid"))
        best_ask = _safe_float(row.get("best_ask"))
        mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else _safe_float(row.get("last_price"))
        spread = max(0.0, best_ask - best_bid) if best_bid > 0 and best_ask > 0 else _safe_float(row.get("spread"))
        bids = _levels(row, "bids")
        asks = _levels(row, "asks")
        bid_depth = float(sum(_depth_qty(level) for level in bids))
        ask_depth = float(sum(_depth_qty(level) for level in asks))
        depth_total = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / max(depth_total, 1.0)
        trade_count = _safe_float(row.get("trade_count", len(trade_rows)))
        cancel_count = _safe_float(row.get("cancel_count", 0.0))
        impact = _safe_float(row.get("impact_cost", row.get("slippage_bps", 0.0)))
        spreads.append(spread)
        spread_pcts.append(spread / mid if mid > 0 else 0.0)
        depth_totals.append(depth_total)
        imbalances.append(imbalance)
        cancel_trade.append(cancel_count / max(trade_count, 1.0))
        impact_costs.append(impact)

    total_notional = 0.0
    total_qty = 0.0
    for trade in trade_rows:
        px = _safe_float(trade.get("price"))
        qty = _safe_float(trade.get("quantity", trade.get("qty", trade.get("volume", 0.0))))
        total_notional += px * qty
        total_qty += qty
    vwap = total_notional / total_qty if total_qty > 0 else _safe_float(reference_price)
    ref = _safe_float(reference_price, vwap)
    vwap_deviation = abs(vwap - ref) / ref if ref > 0 else 0.0

    return {
        "spread": float(np.mean(spreads)) if spreads else 0.0,
        "spread_pct": float(np.mean(spread_pcts)) if spread_pcts else 0.0,
        "depth": float(np.mean(depth_totals)) if depth_totals else 0.0,
        "impact_cost": float(np.mean(impact_costs)) if impact_costs else 0.0,
        "cancel_trade_ratio": float(np.mean(cancel_trade)) if cancel_trade else 0.0,
        "cancel_to_trade_ratio": float(np.mean(cancel_trade)) if cancel_trade else 0.0,
        "vwap": float(vwap),
        "vwap_deviation": float(vwap_deviation),
        "depth_imbalance": float(np.mean(imbalances)) if imbalances else 0.0,
    }


__all__ = ["compute_orderbook_realism"]
