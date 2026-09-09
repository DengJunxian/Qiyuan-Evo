"""Market realism metrics and compact realism diagnostics for pipeline-v2 reporting."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


class MarketMetrics:
    """Compute compact microstructure + behavior metrics."""

    @staticmethod
    def _depth_row_qty(row: Any) -> float:
        if not isinstance(row, Mapping):
            return 0.0
        if "qty" in row:
            return _safe_float(row.get("qty"))
        if "quantity" in row:
            return _safe_float(row.get("quantity"))
        if "size" in row:
            return _safe_float(row.get("size"))
        return 0.0

    @staticmethod
    def compute(
        *,
        snapshot: Mapping[str, Any],
        trade_tape: list[Mapping[str, Any]] | None = None,
        role_flows: Mapping[str, float] | None = None,
    ) -> Dict[str, float]:
        depth = snapshot.get("depth")
        total_bid = 0.0
        total_ask = 0.0
        top_bid_depth = 0.0
        top_ask_depth = 0.0
        if isinstance(depth, Mapping):
            bid_rows = list(depth.get("bids", []) or [])
            ask_rows = list(depth.get("asks", []) or [])
            if bid_rows:
                top_bid_depth = MarketMetrics._depth_row_qty(bid_rows[0])
            if ask_rows:
                top_ask_depth = MarketMetrics._depth_row_qty(ask_rows[0])
            for row in bid_rows:
                if isinstance(row, Mapping):
                    total_bid += MarketMetrics._depth_row_qty(row)
            for row in ask_rows:
                if isinstance(row, Mapping):
                    total_ask += MarketMetrics._depth_row_qty(row)
        denom = max(total_bid + total_ask, 1.0)
        depth_imbalance = (total_bid - total_ask) / denom

        best_bid = _safe_float(snapshot.get("best_bid"))
        best_ask = _safe_float(snapshot.get("best_ask"))
        mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
        spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0.0
        spread_pct = spread / mid if mid > 0 else 0.0

        tape = trade_tape or []
        total_notional = 0.0
        total_qty = 0.0
        for trade in tape:
            if not isinstance(trade, Mapping):
                continue
            px = _safe_float(trade.get("price"))
            qty = _safe_float(trade.get("quantity"))
            total_notional += px * qty
            total_qty += qty
        vwap = total_notional / total_qty if total_qty > 0 else _safe_float(snapshot.get("last_price"))
        avg_trade_size = total_qty / max(len(tape), 1)
        slippage_bps = (abs(vwap - mid) / mid) * 10_000.0 if mid > 0 else 0.0

        cancel_count = _safe_float(snapshot.get("cancel_count", 0.0))
        trade_count = _safe_float(snapshot.get("trade_count", len(tape)))
        cancel_to_trade_ratio = cancel_count / max(trade_count, 1.0)

        flows = role_flows or {}
        herding_proxy = 0.0
        if flows:
            gross = sum(abs(_safe_float(v)) for v in flows.values())
            directional = abs(sum(_safe_float(v) for v in flows.values()))
            herding_proxy = directional / gross if gross > 0 else 0.0

        return {
            "spread": float(spread),
            "spread_pct": float(spread_pct),
            "depth_bid_total": float(total_bid),
            "depth_ask_total": float(total_ask),
            "top_bid_depth": float(top_bid_depth),
            "top_ask_depth": float(top_ask_depth),
            "depth_imbalance": float(depth_imbalance),
            "trade_count": float(trade_count),
            "cancel_count": float(cancel_count),
            "cancel_to_trade_ratio": float(cancel_to_trade_ratio),
            "traded_quantity": float(total_qty),
            "avg_trade_size": float(avg_trade_size),
            "vwap": float(vwap),
            "slippage_bps": float(slippage_bps),
            "herding_proxy": float(herding_proxy),
        }

    @staticmethod
    def evaluation_microstructure_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
        """Map local market metrics to the unified evaluation scorecard schema."""

        depth = _safe_float(metrics.get("depth_bid_total")) + _safe_float(metrics.get("depth_ask_total"))
        vwap = _safe_float(metrics.get("vwap"))
        mid = _safe_float(metrics.get("mid_price", metrics.get("last_price", vwap)))
        return {
            "spread": _safe_float(metrics.get("spread")),
            "spread_pct": _safe_float(metrics.get("spread_pct")),
            "depth": float(depth),
            "impact_cost": _safe_float(metrics.get("slippage_bps")),
            "cancel_trade_ratio": _safe_float(metrics.get("cancel_to_trade_ratio")),
            "cancel_to_trade_ratio": _safe_float(metrics.get("cancel_to_trade_ratio")),
            "vwap": float(vwap),
            "vwap_deviation": abs(vwap - mid) / mid if mid > 0 else 0.0,
            "depth_imbalance": _safe_float(metrics.get("depth_imbalance")),
        }

    @staticmethod
    def scorecard(
        *,
        snapshot: Mapping[str, Any],
        metrics: Mapping[str, Any],
        policy_input: Mapping[str, Any] | None = None,
        policy_chain: Mapping[str, Any] | None = None,
        belief_returns: Sequence[float] | None = None,
        belief_latencies: Sequence[float] | None = None,
    ) -> Dict[str, Any]:
        policy_input = policy_input or {}
        policy_chain = policy_chain or {}
        policy_text = str(policy_input.get("policy_text", "") or "").strip()
        has_policy = bool(policy_text)

        best_bid = _safe_float(snapshot.get("best_bid"))
        best_ask = _safe_float(snapshot.get("best_ask"))
        two_sided_quote = bool(best_bid > 0 and best_ask > 0 and best_ask >= best_bid)

        depth_total = _safe_float(metrics.get("depth_bid_total")) + _safe_float(metrics.get("depth_ask_total"))
        trade_count = _safe_float(metrics.get("trade_count"))
        spread_pct = _safe_float(metrics.get("spread_pct"))
        cancel_to_trade_ratio = _safe_float(metrics.get("cancel_to_trade_ratio"))
        slippage_bps = _safe_float(metrics.get("slippage_bps"))
        herding_proxy = _safe_float(metrics.get("herding_proxy"))

        macro_variables = policy_chain.get("macro_variables", {}) if isinstance(policy_chain, Mapping) else {}
        liquidity_index = _safe_float(getattr(macro_variables, "get", lambda *_: 1.0)("liquidity_index", 1.0), 1.0)
        sentiment_index = _safe_float(getattr(macro_variables, "get", lambda *_: 0.5)("sentiment_index", 0.5), 0.5)

        liquidity_shift = abs(liquidity_index - 1.0)
        sentiment_shift = abs(sentiment_index - 0.5)
        flow_total = _safe_float(
            (policy_chain.get("market_microstructure", {}) if isinstance(policy_chain, Mapping) else {}).get("buy_volume", 0.0)
        ) + _safe_float(
            (policy_chain.get("market_microstructure", {}) if isinstance(policy_chain, Mapping) else {}).get("sell_volume", 0.0)
        )
        flow_imbalance = abs(
            _safe_float((policy_chain.get("market_microstructure", {}) if isinstance(policy_chain, Mapping) else {}).get("buy_volume", 0.0))
            - _safe_float((policy_chain.get("market_microstructure", {}) if isinstance(policy_chain, Mapping) else {}).get("sell_volume", 0.0))
        ) / max(flow_total, 1.0)
        policy_pass_through_ratio = 0.0
        if has_policy:
            policy_pass_through_ratio = min(1.0, (liquidity_shift * 2.5 + sentiment_shift * 2.0 + flow_imbalance) / 3.0)

        returns = [abs(_safe_float(item)) for item in (belief_returns or [])]
        latencies = [_safe_float(item) for item in (belief_latencies or [])]
        belief_dispersion = 0.0
        if len(returns) >= 2:
            mean_return = sum(returns) / len(returns)
            variance = sum((item - mean_return) ** 2 for item in returns) / max(len(returns), 1)
            belief_dispersion = variance ** 0.5
        heterogeneity_score = min(1.0, belief_dispersion / 0.02) if belief_dispersion > 0 else max(0.0, 1.0 - herding_proxy)

        liquidity_thinness = 1.0 / (1.0 + depth_total / 1000.0)
        execution_friction_score = min(1.0, ((slippage_bps / 8.0) + min(cancel_to_trade_ratio, 3.0) + (1.0 if spread_pct > 0 else 0.0)) / 3.0)
        microstructure_score = (
            (1.0 if two_sided_quote else 0.0)
            + min(1.0, depth_total / 1000.0)
            + min(1.0, trade_count)
        ) / 3.0
        transmission_detected = bool(has_policy and policy_pass_through_ratio > 0.0)

        score_components = [microstructure_score, execution_friction_score, heterogeneity_score]
        if has_policy:
            score_components.append(policy_pass_through_ratio)
        realism_score = sum(score_components) / max(len(score_components), 1)

        return {
            "realism_score": float(realism_score),
            "transmission_detected": bool(transmission_detected),
            "policy_pass_through_ratio": float(policy_pass_through_ratio),
            "microstructure_present": bool(two_sided_quote or trade_count > 0 or depth_total > 0),
            "microstructure_score": float(microstructure_score),
            "execution_friction_score": float(execution_friction_score),
            "heterogeneity_score": float(heterogeneity_score),
            "belief_dispersion": float(belief_dispersion),
            "policy_lag_bars": float(sum(latencies) / len(latencies)) if latencies else 0.0,
            "liquidity_thinness": float(liquidity_thinness),
            "flags": {
                "has_policy": bool(has_policy),
                "two_sided_quote": bool(two_sided_quote),
                "stress_spread": bool(spread_pct >= 0.005),
                "stress_cancels": bool(cancel_to_trade_ratio >= 1.0),
                "stress_thin_depth": bool(liquidity_thinness >= 0.5),
            },
        }
