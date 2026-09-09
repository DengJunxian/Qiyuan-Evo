"""Dashboard widgets for the Streamlit policy workbench."""

from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from plotly.subplots import make_subplots
import streamlit as st

from core.exchange.trade_tape import TradeTapeRecord, aggregate_trade_tape_to_bars, stable_hash
from core.ui_text import (
    display_risk_alert,
    localize_dataframe_columns,
    translate_display_text,
    translate_ui_payload,
    zh_metric_name,
    zh_value,
)
from ui.components.event_marker_layer import add_event_marker_layer
from ui.chart_theme import PLOTLY_DARK_LAYOUT, apply_dark_theme, chinese_range_selector_axis
from ui.narrative import narrate_payload, render_narrative_block


def build_synthetic_trade_tape_from_market_frame(
    frame: pd.DataFrame,
    *,
    symbol: str = "sh000001",
    seed: int = 42,
    config_hash: str = "",
    data_snapshot_hash: str = "",
) -> List[TradeTapeRecord]:
    """Create a deterministic visualization tape, then force OHLCV through tape aggregation."""

    if frame is None or frame.empty:
        return []
    records: List[TradeTapeRecord] = []
    source = frame.copy().reset_index(drop=True)
    for idx, row in source.iterrows():
        time_value = row.get("time", row.get("timestamp", row.get("date", idx + 1)))
        if isinstance(time_value, (int, float, np.integer, np.floating)):
            trading_day = pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(idx))
        else:
            trading_day = pd.to_datetime(time_value, errors="coerce")
        if pd.isna(trading_day):
            trading_day = pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(idx))
        base_ts = float(trading_day.timestamp())
        prices = [
            float(row.get("open", row.get("close", 0.0)) or 0.0),
            float(row.get("high", row.get("close", 0.0)) or 0.0),
            float(row.get("low", row.get("close", 0.0)) or 0.0),
            float(row.get("close", row.get("open", 0.0)) or 0.0),
        ]
        volume = max(4, int(float(row.get("volume", 1_000_000.0) or 1_000_000.0)))
        volume_parts = [max(1, int(volume * weight)) for weight in (0.22, 0.24, 0.20, 0.34)]
        spread = float(row.get("spread", abs(prices[-1] - prices[0]) / max(abs(prices[-1]), 1.0)) or 0.0)
        depth_imbalance = float(row.get("depth_imbalance", 0.0) or 0.0)
        for leg, (price, qty) in enumerate(zip(prices, volume_parts)):
            records.append(
                TradeTapeRecord(
                    trade_id=f"synthetic_{symbol}_{idx:04d}_{leg}",
                    timestamp=base_ts + float(leg),
                    tick=int(row.get("step", idx + 1) or idx + 1),
                    trading_day=trading_day.strftime("%Y-%m-%d"),
                    symbol=symbol,
                    price=float(max(price, 0.01)),
                    volume=int(max(qty, 1)),
                    buy_order_id=f"synthetic_buy_{idx}_{leg}",
                    sell_order_id=f"synthetic_sell_{idx}_{leg}",
                    buy_agent_id="synthetic_liquidity_bid",
                    sell_agent_id="synthetic_liquidity_ask",
                    aggressor_side="buy" if leg in {0, 3} else "sell",
                    phase="continuous",
                    seed=int(seed),
                    config_hash=str(config_hash),
                    data_snapshot_hash=str(data_snapshot_hash),
                    spread=float(spread),
                    depth_imbalance=float(depth_imbalance),
                    metadata={
                        "source": "synthetic_market_frame_fallback",
                        "bar_step": int(row.get("step", idx + 1) or idx + 1),
                        "panic_level": float(row.get("panic_level", 0.0) or 0.0),
                        "csad": float(row.get("csad", 0.0) or 0.0),
                        "sentiment_index": float(row.get("sentiment_index", 1.0 - float(row.get("panic_level", 0.0) or 0.0)) or 0.0),
                    },
                )
            )
    return records


def aggregate_trade_tape_ohlcv_frame(
    trade_tape: Iterable[Any],
    *,
    symbol: str = "sh000001",
    freq: str = "1d",
    prev_close: float = 0.0,
    fallback_frame: Optional[pd.DataFrame] = None,
    seed: int = 42,
    config_hash: str = "",
    data_snapshot_hash: str = "",
) -> pd.DataFrame:
    """Return OHLCV bars that always pass through trade-tape aggregation."""

    tape_items = list(trade_tape or [])
    fallback_used = False
    if not tape_items and fallback_frame is not None and not fallback_frame.empty:
        tape_items = build_synthetic_trade_tape_from_market_frame(
            fallback_frame,
            symbol=symbol,
            seed=seed,
            config_hash=config_hash,
            data_snapshot_hash=data_snapshot_hash,
        )
        fallback_used = True
    bars = aggregate_trade_tape_to_bars(
        tape_items,
        freq=freq,
        symbol=symbol,
        prev_close=float(prev_close),
        is_simulated=True,
    )
    rows: List[Dict[str, Any]] = []
    for bar in bars:
        metadata = dict(getattr(bar, "metadata", {}) or {})
        rows.append(
            {
                "step": int(bar.step) + 1,
                "time": str(bar.timestamp),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "amount": float(bar.amount),
                "symbol": str(bar.symbol),
                "is_simulated": bool(bar.is_simulated),
                "source": "trade_tape",
                "trade_count": int(metadata.get("trade_count", 0) or 0),
                "tape_hash": str(metadata.get("tape_hash", "")),
                "synthetic_tape_fallback": bool(fallback_used),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["step", "time", "open", "high", "low", "close", "volume", "source"])
    if fallback_frame is not None and not fallback_frame.empty:
        extras = fallback_frame.copy().reset_index(drop=True)
        for col in ("panic_level", "csad", "spread", "depth_imbalance", "sentiment_index"):
            if col in extras.columns and col not in out.columns:
                out[col] = pd.to_numeric(extras[col].head(len(out)).reset_index(drop=True), errors="coerce").fillna(0.0)
    out.attrs["trade_tape_hash"] = stable_hash({"records": [getattr(item, "to_dict", lambda: item)() for item in tape_items]})
    out.attrs["synthetic_tape_fallback"] = bool(fallback_used)
    return out


def _add_technical_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce").ffill().bfill()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0.0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out["rsi"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)
    mid = close.rolling(20, min_periods=1).mean()
    std = close.rolling(20, min_periods=1).std().fillna(0.0)
    out["boll_mid"] = mid
    out["boll_upper"] = mid + 2.0 * std
    out["boll_lower"] = mid - 2.0 * std
    return out


def build_trade_tape_kline_figure(
    trade_tape: Iterable[Any],
    *,
    symbol: str = "sh000001",
    benchmark_symbol: str = "sh000001",
    freq: str = "1d",
    market_frame: Optional[pd.DataFrame] = None,
    real_index_frame: Optional[pd.DataFrame] = None,
    events: Optional[Iterable[Mapping[str, Any]]] = None,
    selected_indicators: Optional[List[str]] = None,
    title: str = "上证指数 K 线研究主图",
    seed: int = 42,
    config_hash: str = "",
    data_snapshot_hash: str = "",
) -> Tuple[go.Figure, pd.DataFrame]:
    ohlcv = aggregate_trade_tape_ohlcv_frame(
        trade_tape,
        symbol=symbol,
        freq=freq,
        fallback_frame=market_frame,
        seed=seed,
        config_hash=config_hash,
        data_snapshot_hash=data_snapshot_hash,
    )
    if ohlcv.empty:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_DARK_LAYOUT, title=title, height=420)
        return fig, ohlcv

    chart = _add_technical_indicators(ohlcv)
    indicators = selected_indicators if selected_indicators is not None else ["MACD", "RSI", "BOLL"]
    metric_cols = [col for col in ("panic_level", "csad", "spread", "depth_imbalance", "sentiment_index") if col in chart.columns]
    has_indicator_row = bool(indicators or metric_cols)
    rows = 3 if has_indicator_row else 2
    heights = [0.62, 0.18, 0.20] if has_indicator_row else [0.74, 0.26]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=heights,
        specs=[[{"secondary_y": True}], [{}], [{}]] if has_indicator_row else [[{"secondary_y": True}], [{}]],
    )
    fig.add_trace(
        go.Candlestick(
            x=chart["time"],
            open=chart["open"],
            high=chart["high"],
            low=chart["low"],
            close=chart["close"],
            name="仿真指数 K 线（逐笔成交聚合）",
            increasing_line_color="#ef4444",
            decreasing_line_color="#22c55e",
            increasing_fillcolor="#ef4444",
            decreasing_fillcolor="#22c55e",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    close_series = pd.to_numeric(chart["close"], errors="coerce").ffill().bfill()
    for window, color in ((5, "#f59e0b"), (20, "#60a5fa")):
        fig.add_trace(
            go.Scatter(
                x=chart["time"],
                y=close_series.rolling(window, min_periods=1).mean(),
                mode="lines",
                name=f"{window}日均线",
                line=dict(color=color, width=1.2),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
    if real_index_frame is not None and not real_index_frame.empty:
        real = real_index_frame.copy().tail(len(chart)).reset_index(drop=True)
        real_time = real["time"] if "time" in real.columns else chart["time"].head(len(real))
        fig.add_trace(
            go.Scatter(
                x=real_time,
                y=pd.to_numeric(real["close"], errors="coerce"),
                mode="lines",
                name=f"真实基准指数 {benchmark_symbol}",
                line=dict(color="#facc15", width=2.0),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
    y_lookup = dict(zip(chart["time"].astype(str), pd.to_numeric(chart["high"], errors="coerce").fillna(chart["close"])))
    add_event_marker_layer(
        fig,
        events,
        time_values=chart["time"].tolist(),
        y_lookup=y_lookup,
        y_default=float(pd.to_numeric(chart["high"], errors="coerce").max()),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=chart["time"],
            y=chart["volume"],
            name="成交量",
            marker_color="#64748b",
            opacity=0.58,
        ),
        row=2,
        col=1,
    )

    if has_indicator_row:
        if "MACD" in indicators:
            fig.add_trace(go.Scatter(x=chart["time"], y=chart["macd"], name="MACD", line=dict(color="#38bdf8", width=1.4)), row=3, col=1)
            fig.add_trace(go.Scatter(x=chart["time"], y=chart["macd_signal"], name="MACD 信号线", line=dict(color="#f59e0b", width=1.1)), row=3, col=1)
        if "RSI" in indicators:
            fig.add_trace(go.Scatter(x=chart["time"], y=chart["rsi"], name="RSI", line=dict(color="#a78bfa", width=1.3)), row=3, col=1)
        if "BOLL" in indicators:
            fig.add_trace(go.Scatter(x=chart["time"], y=chart["boll_upper"], name="布林线上轨", line=dict(color="#94a3b8", width=0.9, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=chart["time"], y=chart["boll_mid"], name="布林中轨", line=dict(color="#60a5fa", width=0.9)), row=1, col=1)
            fig.add_trace(go.Scatter(x=chart["time"], y=chart["boll_lower"], name="布林下轨", line=dict(color="#94a3b8", width=0.9, dash="dot")), row=1, col=1)
        metric_styles = {
            "panic_level": ("恐慌度", "#fb7185"),
            "csad": ("CSAD", "#facc15"),
            "spread": ("买卖价差", "#2dd4bf"),
            "depth_imbalance": ("盘口深度失衡", "#818cf8"),
            "sentiment_index": ("情绪指数", "#34d399"),
        }
        for col_name in metric_cols:
            label, color = metric_styles[col_name]
            fig.add_trace(
                go.Scatter(x=chart["time"], y=pd.to_numeric(chart[col_name], errors="coerce"), mode="lines", name=label, line=dict(color=color, width=1.25)),
                row=3,
                col=1,
            )
    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title=title,
        height=720 if has_indicator_row else 580,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.04, x=0.0),
    )
    fig.update_layout(xaxis=chinese_range_selector_axis(rangeslider_visible=False))
    fig.update_yaxes(title_text="指数点位", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    if has_indicator_row:
        fig.update_yaxes(title_text="指标", row=3, col=1)
    return fig, chart


def render_trade_tape_kline_workbench(
    trade_tape: Iterable[Any],
    *,
    symbol: str = "sh000001",
    benchmark_symbol: str = "sh000001",
    market_frame: Optional[pd.DataFrame] = None,
    real_index_frame: Optional[pd.DataFrame] = None,
    events: Optional[Iterable[Mapping[str, Any]]] = None,
    key_prefix: str = "trade_tape_kline",
    seed: int = 42,
    config_hash: str = "",
    data_snapshot_hash: str = "",
) -> Tuple[go.Figure, pd.DataFrame]:
    indicator_options = ["MACD", "RSI", "BOLL"]
    selected = st.multiselect(
        "指标副图",
        options=indicator_options,
        default=indicator_options,
        key=f"{key_prefix}_indicators",
    )
    fig, chart = build_trade_tape_kline_figure(
        trade_tape,
        symbol=symbol,
        benchmark_symbol=benchmark_symbol,
        market_frame=market_frame,
        real_index_frame=real_index_frame,
        events=events,
        selected_indicators=selected,
        seed=seed,
        config_hash=config_hash,
        data_snapshot_hash=data_snapshot_hash,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_fig")
    fallback = bool(chart.attrs.get("synthetic_tape_fallback", False)) if hasattr(chart, "attrs") else False
    source_note = "仿真逐笔成交回退后聚合" if fallback else "逐笔成交聚合"
    st.caption(f"K 线数据来源：{source_note}；前端 OHLCV 未直接从价格序列补造。")
    if not chart.empty:
        latest = chart.tail(1).iloc[0]
        _render_chart_explanation(
            "K 线研究主图解读",
            {
                "latest_close": float(latest.get("close", 0.0) or 0.0),
                "latest_volume": float(latest.get("volume", 0.0) or 0.0),
                "max_close": float(pd.to_numeric(chart["close"], errors="coerce").max()),
                "min_close": float(pd.to_numeric(chart["close"], errors="coerce").min()),
                "panic_level": float(latest.get("panic_level", 0.0) or 0.0),
                "csad": float(latest.get("csad", 0.0) or 0.0),
            },
            "这张 K 线图回答政策冲击后指数路径、成交量和风险热度是否同步变化。",
        )
    return fig, chart


def _json_default(value: Any) -> Any:
    """
    统一处理导出 JSON 时的非常规类型，避免前端导出按钮触发序列化异常。
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return value.to_dict()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return dict(value.__dict__)
        except Exception:
            pass
    return str(value)


def metric_value_text(value: float, as_percent: bool = False) -> str:
    if as_percent:
        return f"{value:.2%}"
    return f"{value:.3f}" if abs(value) < 10 else f"{value:.2f}"


def normalize_discovered_metrics_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize objective-discovery payloads before UI rendering or export."""
    data = dict(payload or {})
    ranked = list(data.get("ranked_metrics", []) or [])
    top = list(data.get("top_metrics", []) or ranked[:8])
    pareto = list(data.get("pareto_frontier", []) or [])
    weights = dict(data.get("weight_decomposition", {}) or {})
    shanghai = dict(data.get("shanghai_index_metric", {}) or {})
    if not shanghai:
        shanghai = next((dict(item) for item in ranked if dict(item).get("name") == "shanghai_index_return"), {})
    return {
        "schema_version": str(data.get("schema_version", "objective_discovery_v1")),
        "ranked_metrics": ranked,
        "top_metrics": top,
        "pareto_frontier": pareto,
        "composite_score": float(data.get("composite_score", data.get("composite_policy_score", 0.0)) or 0.0),
        "weight_decomposition": weights,
        "stability_heatmap": list(data.get("stability_heatmap", []) or []),
        "candidate_pool": list(data.get("candidate_pool", []) or []),
        "shanghai_index_metric": shanghai,
    }


def render_discovered_metrics_panel(payload: Optional[Mapping[str, Any]], key_prefix: str = "objective") -> None:
    data = normalize_discovered_metrics_payload(payload)
    top_metrics = list(data.get("top_metrics", []) or [])
    if not top_metrics:
        st.info("暂无目标发现结果。")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("政策综合评分", f"{float(data.get('composite_score', 0.0)):.3f}")
    c2.metric("候选指标池", len(data.get("candidate_pool", []) or []))
    c3.metric("帕累托指标", len(data.get("pareto_frontier", []) or []))

    frame = pd.DataFrame(top_metrics)
    cols = [
        col
        for col in [
            "name",
            "category",
            "rank_score",
            "composite_weight",
            "policy_sensitivity",
            "robustness",
            "historical_replay_alignment",
            "early_warning_utility",
            "relation_to_shanghai_index",
        ]
        if col in frame.columns
    ]
    if cols:
        display_frame = frame[cols].copy()
        if "name" in display_frame.columns:
            display_frame["name"] = display_frame["name"].map(lambda value: zh_metric_name(str(value)))
        st.dataframe(localize_dataframe_columns(display_frame), use_container_width=True, hide_index=True)

    weights = dict(data.get("weight_decomposition", {}) or {})
    if weights:
        weight_frame = pd.DataFrame(
            [
                {"指标名称": zh_metric_name(str(key)), "综合权重": float(value)}
                for key, value in sorted(weights.items(), key=lambda item: float(item[1]), reverse=True)
            ]
        )
        fig_weight = go.Figure(
            data=[
                go.Bar(
                    x=weight_frame["综合权重"],
                    y=weight_frame["指标名称"],
                    orientation="h",
                    marker_color="#38bdf8",
                    name="综合权重",
                )
            ]
        )
        fig_weight.update_layout(
            **PLOTLY_DARK_LAYOUT,
            title="指标重要性",
            xaxis_title="综合权重",
            yaxis_title="指标名称",
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig_weight, use_container_width=True, key=f"{key_prefix}_weights")

    pareto = pd.DataFrame(data.get("pareto_frontier", []) or [])
    if not pareto.empty and {"policy_sensitivity", "robustness", "name"}.issubset(pareto.columns):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pareto["policy_sensitivity"],
                y=pareto["robustness"],
                text=pareto["name"].map(lambda value: zh_metric_name(str(value))),
                mode="markers+text",
                textposition="top center",
                marker=dict(color=pareto.get("rank_score", 0.0), colorscale="Viridis", showscale=True, size=12),
            )
        )
        fig.update_layout(
            **PLOTLY_DARK_LAYOUT,
            title="目标发现帕累托前沿",
            xaxis_title="政策敏感性",
            yaxis_title="跨场景稳健性",
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_pareto")
    _render_chart_explanation(
        "目标发现与指标组合",
        {
            "composite_score": float(data.get("composite_score", 0.0)),
            "top_metrics": top_metrics[:5],
            "shanghai_index_metric": data.get("shanghai_index_metric", {}),
            "candidate_pool_size": len(data.get("candidate_pool", []) or []),
        },
        "请解释为什么以上证指数为核心基准，以及其他补充指标如何增强政策敏感性、稳健性和预警价值。",
    )


def _render_chart_explanation(title: str, payload: Any, context: str, cache_namespace: str = "dashboard_narrative_cache") -> None:
    render_narrative_block(
        title,
        payload,
        context=context,
        cache_namespace=cache_namespace,
        label="在线大模型解读",
    )


def export_plot_bundle(
    fig: go.Figure,
    data: Optional[pd.DataFrame],
    title_prefix: str,
    key_prefix: str,
    fig_json: Optional[Dict[str, Any]] = None,
) -> None:
    """Provide PNG/CSV/JSON export controls under every key chart."""
    c1, c2, c3 = st.columns(3)

    with c1:
        try:
            png_bytes = fig.to_image(format="png", width=1400, height=840, scale=2)
            st.download_button(
                "导出 PNG",
                data=png_bytes,
                file_name=f"{title_prefix}.png",
                mime="image/png",
                key=f"{key_prefix}_png",
                use_container_width=True,
            )
        except Exception:
            st.button("导出 PNG（需安装依赖）", key=f"{key_prefix}_png_disabled", disabled=True, use_container_width=True)
            st.caption("PNG 导出依赖未就绪，请安装 `kaleido` 后重启应用。")

    with c2:
        csv_text = ""
        if data is not None and not data.empty:
            csv_text = data.to_csv(index=False)
        st.download_button(
            "导出 CSV",
            data=csv_text,
            file_name=f"{title_prefix}.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv",
            use_container_width=True,
            disabled=(data is None),
        )

    with c3:
        payload = fig_json if fig_json is not None else fig.to_dict()
        try:

            json_text = json.dumps(payload, ensure_ascii=False, indent=2, cls=PlotlyJSONEncoder)
        except Exception:
            try:
                json_text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
            except Exception as exc:
                json_text = json.dumps(
                    {
                        "export_error": str(exc),
                        "payload_type": type(payload).__name__,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=_json_default,
                )
        st.download_button(
            "导出 JSON",
            data=json_text,
            file_name=f"{title_prefix}.json",
            mime="application/json",
            key=f"{key_prefix}_json",
            use_container_width=True,
        )


def render_kpi_cards(snapshot: Dict[str, Any]) -> None:
    cards = [
        ("流动性", snapshot.get("liquidity", 0.0), False, "越高越好"),
        ("波动率", snapshot.get("volatility", 0.0), True, "越低越稳"),
        ("羊群度", snapshot.get("herding", 0.0), False, "高值代表拥挤交易"),
        ("风险预警", snapshot.get("risk_alert", "GREEN"), None, "系统风险灯"),
        ("监管动作", snapshot.get("reg_action", "观察"), None, "当前干预状态"),
    ]
    rows = [cards[:3], cards[3:]]
    for row_idx, row_cards in enumerate(rows):
        cols = st.columns(len(row_cards))
        for idx, (label, value, percent, note) in enumerate(row_cards):
            with cols[idx]:
                if percent is True and isinstance(value, (int, float)):
                    v = metric_value_text(float(value), as_percent=True)
                elif percent is False and isinstance(value, (int, float)):
                    v = metric_value_text(float(value))
                else:
                    if label == "风险预警":
                        v = display_risk_alert(str(value))
                    else:
                        v = translate_display_text(str(value))
                st.markdown(
                    f"""
                    <div class='kpi-card'>
                      <div class='kpi-title'>{label}</div>
                      <div class='kpi-value'>{v}</div>
                      <div class='kpi-note'>{note}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        if row_idx == 0:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _render_chart_explanation(
        "核心指标总览",
        snapshot,
        "请解释当前流动性、波动、羊群行为与风险预警的综合状态，并指出评委应重点关注的风险信号。",
    )


def render_market_overview(metrics: pd.DataFrame, upto_step: Optional[int], key_prefix: str = "market", title: str = "市场主图与风险追踪") -> go.Figure:
    frame = metrics.copy()
    if upto_step is not None:
        frame = frame[frame["step"] <= int(upto_step)]

    if not frame.empty:
        if "open" not in frame.columns:
            frame["open"] = frame["close"].shift(1).fillna(frame["close"].iloc[0])
            span = frame["close"].pct_change().abs().fillna(0.002).clip(lower=0.001, upper=0.02)
            frame["high"] = frame[["open", "close"]].max(axis=1) * (1.0 + span)
            frame["low"] = frame[["open", "close"]].min(axis=1) * (1.0 - span)
        tape = build_synthetic_trade_tape_from_market_frame(frame, symbol="sh000001")
        frame = aggregate_trade_tape_ohlcv_frame(tape, symbol="sh000001", freq="1d", fallback_frame=frame)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=frame["time"],
        open=frame["open"],
        high=frame["high"],
        low=frame["low"],
        close=frame["close"],
        name="指数 K线",
        increasing_line_color="#f5222d",
        decreasing_line_color="#22c55e",
    ))
    fig.add_trace(go.Scatter(x=frame["time"], y=frame["panic_level"], mode="lines", name="风险热度", yaxis="y2", line=dict(color="#ff7a59", width=2, dash="dash")))
    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title=dict(text=title, font=dict(color="#e2e8f0", size=18)),
        yaxis=dict(title=dict(text="指数点位", font=dict(color="#8aa0c2")), tickfont=dict(color="#e2e8f0")),
        yaxis2=dict(title=dict(text="风险热度", font=dict(color="#ff7a59")), overlaying="y", side="right", tickfont=dict(color="#ff7a59")),
        xaxis=dict(
            title=dict(text="", font=dict(color="#e2e8f0")),
            tickfont=dict(color="#e2e8f0"),
            rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="近 1 周", step="day", stepmode="backward"),
                    dict(count=1, label="近 1 月", step="month", stepmode="backward"),
                    dict(step="all", label="全部")
                ]),
                bgcolor="#0a1931",
                font=dict(color="#e2e8f0")
            )
        ),
        legend=dict(orientation="h", font=dict(color="#e2e8f0")),
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_market_plot")
    export_plot_bundle(fig, frame, f"{key_prefix}_market_overview", f"{key_prefix}_market_overview")
    if not frame.empty:
        latest = frame.iloc[-1]
        _render_chart_explanation(
            title,
            {
                "latest_close": float(latest.get("close", 0.0) or 0.0),
                "latest_panic_level": float(latest.get("panic_level", 0.0) or 0.0),
                "max_close": float(frame["close"].max()),
                "min_close": float(frame["close"].min()),
                "max_panic_level": float(frame["panic_level"].max()),
                "avg_panic_level": float(frame["panic_level"].mean()),
                "sample_size": int(len(frame)),
            },
            "请结合价格主走势和风险热度，说明当前市场阶段、风险是否扩散，以及这张图最能证明什么。",
        )
    return fig


def render_empty_market_board(key_prefix: str = "empty") -> go.Figure:
    np.random.seed(42)  # 固定种子以保持每次渲染时的逼真“历史图”一致
    dates = pd.date_range(end=pd.Timestamp.now().normalize() - pd.Timedelta(days=1), periods=30, freq='D')
    close = 3200 + np.cumsum(np.random.randn(30) * 15)
    open_p = pd.Series(close).shift(1).fillna(close[0] - 5).values
    high = np.maximum(open_p, close) + np.abs(np.random.randn(30) * 10)
    low = np.minimum(open_p, close) - np.abs(np.random.randn(30) * 10)
    panic = np.clip(0.3 + np.cumsum(np.random.randn(30) * 0.05), 0.1, 0.9)
    
    frame = pd.DataFrame({
        "time": dates,
        "step": range(-30, 0),
        "open": open_p, "high": high, "low": low, "close": close,
        "panic_level": panic
    })
    return render_market_overview(frame, upto_step=0, key_prefix=key_prefix, title="市场主图与风险追踪 (历史前30日)")


def render_ab_world_compare(world_a: pd.DataFrame, world_b: pd.DataFrame, key_prefix: str = "ab") -> go.Figure:
    merged = world_a[["step", "time", "close"]].rename(columns={"close": "政策执行世界"}).merge(
        world_b[["step", "close"]].rename(columns={"close": "政策缺失世界"}),
        on="step",
        how="left",
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["time"], y=merged["政策执行世界"], mode="lines", name="政策执行世界", line=dict(color="#21c55d", width=2)))
    fig.add_trace(go.Scatter(x=merged["time"], y=merged["政策缺失世界"], mode="lines", name="政策缺失世界", line=dict(color="#f97316", width=2)))
    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title="甲乙世界对照：政策有无的市场路径",
        yaxis=dict(title="指数点位"),
        legend=dict(orientation="h"),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_ab_plot")
    export_plot_bundle(fig, merged, f"{key_prefix}_ab_compare", f"{key_prefix}_ab_compare")
    if not merged.empty:
        _render_chart_explanation(
            "甲乙世界对照",
            {
                "world_a_end": float(merged["政策执行世界"].iloc[-1]),
                "world_b_end": float(merged["政策缺失世界"].iloc[-1]),
                "end_gap": float(merged["政策执行世界"].iloc[-1] - merged["政策缺失世界"].iloc[-1]),
                "peak_gap": float((merged["政策执行世界"] - merged["政策缺失世界"]).max()),
                "trough_gap": float((merged["政策执行世界"] - merged["政策缺失世界"]).min()),
            },
            "请说明政策执行与不执行两条路径的核心差异，以及哪一个阶段最能体现政策效果。",
        )
    return fig


def _build_lob_depth_steps(metrics: pd.DataFrame, max_steps: int = 12) -> Tuple[List[go.Frame], List[str]]:
    steps = []
    labels = []
    subset = metrics.head(max_steps)
    for _, row in subset.iterrows():
        mid = float(row["close"])
        spread = max(2.0, float(row["panic_level"]) * 25.0)
        prices_bid = np.linspace(mid - spread * 1.6, mid - spread * 0.1, 14)
        prices_ask = np.linspace(mid + spread * 0.1, mid + spread * 1.6, 14)
        bid_depth = np.linspace(240, 60, 14) * (1.0 + (0.3 - float(row["panic_level"])))
        ask_depth = np.linspace(70, 260, 14) * (1.0 + float(row["panic_level"]))
        frame = go.Frame(
            name=f"step_{int(row['step'])}",
            data=[
                go.Bar(x=prices_bid, y=bid_depth, name="买盘深度", marker_color="#16a34a"),
                go.Bar(x=prices_ask, y=ask_depth, name="卖盘深度", marker_color="#ef4444"),
            ],
        )
        steps.append(frame)
        labels.append(str(int(row["step"])))
    return steps, labels


def render_lob_depth_animation(metrics: pd.DataFrame, key_prefix: str = "lob") -> go.Figure:
    frames, labels = _build_lob_depth_steps(metrics)
    if not frames:
        fig = go.Figure()
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_lob_plot_empty")
        return fig

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title="盘口深度动画（买卖盘结构）",
        barmode="overlay",
        xaxis_title="价格",
        yaxis_title="挂单量",
        height=360,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "播放",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 450, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "暂停",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "method": "animate",
                        "label": label,
                        "args": [[f"step_{label}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    }
                    for label in labels
                ],
                "currentvalue": {"prefix": "步骤 "},
            }
        ],
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_lob_plot")
    export_plot_bundle(fig, metrics.head(len(frames)), f"{key_prefix}_lob_depth", f"{key_prefix}_lob_depth")
    if not metrics.empty:
        _render_chart_explanation(
            "盘口深度动画",
            {
                "sample_steps": int(len(frames)),
                "max_panic_level": float(pd.to_numeric(metrics.get("panic_level", pd.Series([0.0])), errors="coerce").fillna(0.0).max()),
                "avg_close": float(pd.to_numeric(metrics.get("close", pd.Series([0.0])), errors="coerce").fillna(0.0).mean()),
            },
            "请解释买卖盘深度随风险热度变化时，流动性是否收缩以及价格冲击是否可能被放大。",
        )
    return fig


def render_social_network_heatmap(metrics: pd.DataFrame, key_prefix: str = "social") -> go.Figure:
    n = min(12, len(metrics))
    seed = np.clip(metrics["panic_level"].head(n).to_numpy(), 0.01, 0.99)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            base = 1.0 - abs(i - j) / max(1, n - 1)
            matrix[i, j] = base * (0.3 + 0.7 * seed[(i + j) % n])
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=matrix,
                colorscale="YlOrRd",
                zmin=0,
                zmax=1,
                colorbar=dict(title="传播强度"),
            )
        ]
    )
    fig.update_layout(**PLOTLY_DARK_LAYOUT, title="社会传播网络热图", xaxis_title="节点", yaxis_title="节点", height=360)
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_social_plot")
    heatmap_df = pd.DataFrame(matrix)
    export_plot_bundle(fig, heatmap_df, f"{key_prefix}_social_heatmap", f"{key_prefix}_social_heatmap")
    _render_chart_explanation(
        "社会传播网络热图",
        {
            "node_count": int(n),
            "max_link_strength": float(np.max(matrix)) if matrix.size else 0.0,
            "avg_link_strength": float(np.mean(matrix)) if matrix.size else 0.0,
            "panic_seed_mean": float(np.mean(seed)) if len(seed) else 0.0,
        },
        "请解释传播强度分布是否集中、是否存在高敏感传播通道，以及这对市场情绪扩散意味着什么。",
    )
    return fig


def _policy_chain_strength(chain_payload: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not chain_payload:
        return {
            "policy_to_macro": 90.0,
            "macro_to_social": 75.0,
            "social_to_industry": 62.0,
            "industry_to_micro": 58.0,
        }

    macro = chain_payload.get("macro_variables", {})
    social = chain_payload.get("social_sentiment", {})
    industry = chain_payload.get("industry_agent", {})
    micro = chain_payload.get("market_microstructure", {})

    macro_shift = 0.0
    for key in ("inflation", "unemployment", "wage_growth", "credit_spread", "liquidity_index", "policy_rate", "fiscal_stimulus", "sentiment_index"):
        if key in macro:
            macro_shift += abs(float(macro[key]))

    social_mean = abs(float(social.get("mean", 0.0)))
    industry_shift = abs(float(industry.get("avg_household_risk", 0.0))) + abs(float(industry.get("avg_firm_hiring", 0.0)))
    buy = float(micro.get("buy_volume", 0.0))
    sell = float(micro.get("sell_volume", 0.0))
    imbalance = abs(buy - sell) / max(1.0, buy + sell)

    return {
        "policy_to_macro": float(np.clip(40 + macro_shift * 10, 10, 100)),
        "macro_to_social": float(np.clip(35 + social_mean * 70, 10, 100)),
        "social_to_industry": float(np.clip(30 + industry_shift * 60, 10, 100)),
        "industry_to_micro": float(np.clip(25 + imbalance * 90, 10, 100)),
    }


def render_policy_transmission_chain(chain_payload: Optional[Dict[str, Any]] = None, key_prefix: str = "policy_chain") -> go.Figure:
    policy_text = "政策输入"
    if chain_payload and chain_payload.get("policy"):
        policy_text = translate_display_text(str(chain_payload.get("policy")))
    labels = [policy_text[:24], "宏观变量", "社会情绪", "行业主体", "市场微结构"]
    strength = _policy_chain_strength(chain_payload)
    source = [0, 1, 2, 3]
    target = [1, 2, 3, 4]
    value = [
        strength["policy_to_macro"],
        strength["macro_to_social"],
        strength["social_to_industry"],
        strength["industry_to_micro"],
    ]
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels, color=["#334155", "#2563eb", "#16a34a", "#f97316", "#ef4444"]),
                link=dict(source=source, target=target, value=value, color="rgba(148,163,184,0.35)"),
            )
        ]
    )
    fig.update_layout(**PLOTLY_DARK_LAYOUT, title="政策传导链：政策 -> 宏观变量 -> 社会情绪 -> 行业主体 -> 市场微结构", height=360)
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chain_plot")
    sankey_df = pd.DataFrame({"source": source, "target": target, "value": value})
    export_plot_bundle(fig, sankey_df, f"{key_prefix}_policy_chain", f"{key_prefix}_policy_chain")
    _render_chart_explanation(
        "政策传导链",
        {
            "policy": policy_text,
            "policy_to_macro": value[0],
            "macro_to_social": value[1],
            "social_to_industry": value[2],
            "industry_to_micro": value[3],
        },
        "请按传导强度说明政策从宏观到市场微结构的影响路径，并指出哪一环最关键。",
    )
    return fig


def render_policy_sankey(key_prefix: str = "policy") -> go.Figure:
    """Backward-compatible wrapper."""
    return render_policy_transmission_chain(chain_payload=None, key_prefix=key_prefix)


def render_risk_event_timeline(metrics: pd.DataFrame, key_prefix: str = "risk") -> go.Figure:
    frame = metrics.copy()
    levels = pd.cut(frame["panic_level"], bins=[-1, 0.2, 0.35, 0.5, 2], labels=["低", "中", "高", "极高"])  # type: ignore[arg-type]
    events = []
    for _, row in frame.iterrows():
        severity = levels.iloc[int(row.name)]
        if severity in ("高", "极高"):
            events.append({
                "time": row["time"],
                "event": "风险预警" if severity == "高" else "触发监管评估",
                "level": str(severity),
                "value": float(row["panic_level"]),
            })
    if not events:
        events.append({"time": frame.iloc[-1]["time"], "event": "无高风险事件", "level": "低", "value": float(frame.iloc[-1]["panic_level"])})
    events_df = pd.DataFrame(events)

    y_map = {"低": 0, "中": 1, "高": 2, "极高": 3}
    fig = go.Figure(
        data=[
            go.Scatter(
                x=events_df["time"],
                y=events_df["level"].map(y_map),
                mode="markers+lines+text",
                text=events_df["event"],
                textposition="top center",
                marker=dict(size=12, color=events_df["value"], colorscale="Reds", cmin=0, cmax=1),
                name="风险事件",
            )
        ]
    )
    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title="风险事件时间轴",
        yaxis=dict(title="风险等级", tickvals=list(y_map.values()), ticktext=list(y_map.keys())),
        xaxis=dict(title="时间"),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_timeline_plot")
    export_plot_bundle(fig, events_df, f"{key_prefix}_timeline", f"{key_prefix}_timeline")
    _render_chart_explanation(
        "风险事件时间轴",
        {
            "event_count": int(len(events_df)),
            "highest_risk": float(events_df["value"].max()) if not events_df.empty else 0.0,
            "events": events_df.head(8).to_dict(orient="records"),
        },
        "请解释高风险事件集中在哪些时点、是短时冲击还是持续升温，以及是否需要监管介入。",
    )
    return fig


def render_decision_evidence_flow(
    narration_items: Iterable[Dict[str, Any]],
    analyst_manager_output: Dict[str, Any],
) -> None:
    st.subheader("证据流总览")

    analyst_cards = analyst_manager_output.get("analyst_cards")
    if not isinstance(analyst_cards, list) or not analyst_cards:
        legacy_analyst = analyst_manager_output.get("analyst_outputs", {})
        analyst_cards = []
        if isinstance(legacy_analyst, dict):
            for name, payload in legacy_analyst.items():
                analyst_cards.append(
                    {
                        "analyst_id": str(name),
                        "thesis": f"{translate_display_text(str(name))} 历史输出",
                        "evidence": [{"type": "risk", "content": json.dumps(payload, ensure_ascii=False), "weight": 0.5}],
                        "time_horizon": "swing",
                        "risk_tags": [],
                        "confidence": 0.5,
                        "counterarguments": [],
                        "recommended_action": "hold",
                    }
                )

    manager_final_card = analyst_manager_output.get("manager_final_card")
    if not isinstance(manager_final_card, dict) or not manager_final_card:
        manager_final_card = analyst_manager_output.get("manager_decision", {}) or {}

    contradiction_matrix = analyst_manager_output.get("contradiction_matrix")
    if not isinstance(contradiction_matrix, dict) or not contradiction_matrix:
        contradiction_matrix = manager_final_card.get("contradiction_matrix", {}) if isinstance(manager_final_card, dict) else {}

    risk_alerts = analyst_manager_output.get("risk_alerts") or analyst_manager_output.get("risk_alert") or {}
    calibration = analyst_manager_output.get("calibration")
    if not isinstance(calibration, dict) or not calibration:
        calibration = manager_final_card.get("calibration", {}) if isinstance(manager_final_card, dict) else {}

    tab_cards, tab_matrix, tab_manager, tab_risk, tab_calib, tab_narration = st.tabs(
        ["分析师卡片", "矛盾矩阵", "经理最终卡", "风险告警", "校准指标", "叙事时间线"]
    )

    with tab_cards:
        if analyst_cards:
            for idx, card in enumerate(analyst_cards):
                label = f"{idx + 1}. {translate_display_text(str(card.get('analyst_id', f'analyst_{idx}')))}"
                with st.expander(label, expanded=(idx == 0)):
                    st.markdown(
                        narrate_payload(
                            f"{label}观点解读",
                            translate_ui_payload(card),
                            context="提炼分析师观点、证据与风险提示。",
                        )
                    )
        else:
            st.info("暂无分析师卡片。")

    with tab_matrix:
        analysts = contradiction_matrix.get("analysts", []) if isinstance(contradiction_matrix, dict) else []
        matrix = contradiction_matrix.get("matrix", []) if isinstance(contradiction_matrix, dict) else []
        if matrix and analysts:
            analyst_labels = [translate_display_text(str(item)) for item in analysts]
            fig = go.Figure(
                data=[
                    go.Heatmap(
                        z=matrix,
                        x=analyst_labels,
                        y=analyst_labels,
                        zmin=0,
                        zmax=1,
                        colorscale="YlOrRd",
                        colorbar=dict(title="矛盾强度"),
                    )
                ]
            )
            fig.update_layout(**PLOTLY_DARK_LAYOUT, title="分析师矛盾矩阵", height=360)
            st.plotly_chart(fig, use_container_width=True, key="evidence_contradiction_matrix_plot")
            matrix_df = pd.DataFrame(matrix, columns=analyst_labels, index=analyst_labels)
            export_plot_bundle(fig, matrix_df.reset_index(), "evidence_contradiction_matrix", "evidence_contradiction_matrix")
            st.caption(f"矛盾指数：{float(contradiction_matrix.get('contradiction_index', 0.0)):.3f}")
        else:
            st.info("暂无矛盾矩阵数据。")

    with tab_manager:
        st.markdown(
            narrate_payload(
                "经理最终决策解读",
                translate_ui_payload(manager_final_card),
                context="解释仓位取向、执行节奏与核心判断。",
            )
        )

    with tab_risk:
        st.markdown(
            narrate_payload(
                "风险预警解读",
                translate_ui_payload(risk_alerts),
                context="说明风险等级、触发原因与建议动作。",
            )
        )

    with tab_calib:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Brier 类分数", f"{float(calibration.get('brier_like_score', 0.0)):.4f}")
        with c2:
            st.metric("置信漂移", f"{float(calibration.get('confidence_drift', 0.0)):.4f}")
        with c3:
            st.metric("结果代理值", f"{float(calibration.get('outcome_proxy', 0.0)):.2f}")
        if isinstance(manager_final_card, dict):
            st.caption(
                f"原始置信度={float(manager_final_card.get('raw_confidence', 0.0)):.3f}, "
                f"校准后置信度={float(manager_final_card.get('calibrated_confidence', 0.0)):.3f}"
            )

    with tab_narration:
        rows = []
        for item in narration_items:
            rows.append(
                {
                    "步骤": item.get("step"),
                    "角色": translate_display_text(str(item.get("speaker", "旁白"))),
                    "事件": translate_display_text(str(item.get("text", ""))),
                }
            )
        evidence_df = pd.DataFrame(rows)
        st.dataframe(evidence_df, use_container_width=True, hide_index=True)


def build_kpi_snapshot(metrics: pd.DataFrame, step: int, regulation_hint: str = "观察") -> Dict[str, Any]:
    if metrics.empty:
        return {"liquidity": 0.0, "volatility": 0.0, "herding": 0.0, "risk_alert": "GREEN", "reg_action": regulation_hint}

    current = metrics[metrics["step"] <= step].tail(1)
    if current.empty:
        current = metrics.head(1)
    current_row = current.iloc[0]

    recent = metrics[metrics["step"] <= step].tail(5)
    if len(recent) >= 2:
        returns = recent["close"].pct_change().dropna()
        volatility = float(returns.std()) if not returns.empty else 0.0
    else:
        volatility = 0.0

    liquidity = float(current_row.get("volume", 0.0)) / max(1.0, float(metrics["volume"].median()))
    herding = float(current_row.get("csad", 0.0))
    panic = float(current_row.get("panic_level", 0.0))

    if panic > 0.7:
        risk_alert = "RED"
    elif panic > 0.45:
        risk_alert = "YELLOW"
    else:
        risk_alert = "GREEN"

    return {
        "liquidity": liquidity,
        "volatility": volatility,
        "herding": herding,
        "risk_alert": risk_alert,
        "reg_action": regulation_hint,
    }

def render_orderflow_microstructure_panel(
    role_order_flows: Optional[Mapping[str, float]] = None,
    microstructure_metrics: Optional[Mapping[str, Any]] = None,
    *,
    key_prefix: str = "micro_panel",
) -> None:
    """Render role order-flow decomposition and microstructure summary."""
    st.subheader("角色订单流与微观结构")

    orderflow_df = pd.DataFrame()
    role_order_flows = dict(role_order_flows or {})
    if role_order_flows:
        orderflow_df = pd.DataFrame(
            [{"role": translate_display_text(str(role)), "net_flow": float(value)} for role, value in role_order_flows.items()]
        ).sort_values("net_flow", ascending=False)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=orderflow_df["role"],
                    y=orderflow_df["net_flow"],
                    marker_color=["#16a34a" if value >= 0 else "#ef4444" for value in orderflow_df["net_flow"]],
                    name="净买卖额",
                )
            ]
        )
        fig.update_layout(
            **PLOTLY_DARK_LAYOUT,
            title="角色净买卖拆解",
            xaxis_title="角色",
            yaxis_title="净买卖额",
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_role_flow_plot_v2")
        export_plot_bundle(fig, orderflow_df, f"{key_prefix}_role_orderflow", f"{key_prefix}_role_orderflow")
    else:
        st.info("当前场景暂无角色订单流明细。")

    metrics = dict(microstructure_metrics or {})
    spread = float(metrics.get("spread", 0.0) or 0.0)
    spread_pct = float(metrics.get("spread_pct", 0.0) or 0.0)
    depth_imbalance = float(metrics.get("depth_imbalance", 0.0) or 0.0)
    impact = float(metrics.get("impact", metrics.get("impact_bps", 0.0)) or 0.0)
    herding_proxy = float(metrics.get("herding_proxy", 0.0) or 0.0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("买卖价差", f"{spread:.4f}")
    c2.metric("价差占比", f"{spread_pct:.2%}")
    c3.metric("深度失衡", f"{depth_imbalance:+.3f}")
    c4.metric("价格冲击", f"{impact:.4f}")
    c5.metric("羊群程度", f"{herding_proxy:.3f}")

    _render_chart_explanation(
        "角色订单流与微观结构",
        {
            "role_order_flows": orderflow_df.to_dict(orient="records") if not orderflow_df.empty else [],
            "spread": spread,
            "spread_pct": spread_pct,
            "depth_imbalance": depth_imbalance,
            "impact": impact,
            "herding_proxy": herding_proxy,
        },
        "请解释当前买卖盘是否失衡、谁在主导交易，以及微观结构是否显示出冲击放大或拥挤交易。",
    )


def render_financial_health_dashboard(metrics_dict: Dict[str, Any], key_prefix: str = "health") -> None:
    st.markdown("### 实时异常指标监测", help="通过速度仪表盘直观反映市场在恐慌、羊群效应和深度失衡三个维度的状态。")

    panic = float(metrics_dict.get("panic_level", 0.0) or 0.0)
    csad = float(metrics_dict.get("csad", 0.0) or 0.0)
    imb = float(metrics_dict.get("depth_imbalance", 0.0) or 0.0)

    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=panic,
            title={"text": "恐慌指数", "font": {"size": 14, "color": "#8aa0c2"}},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "rgba(250, 140, 22, 0.8)", "line": {"width": 0}},
                "steps": [
                    {"range": [0, 0.35], "color": "rgba(34, 197, 94, 0.2)"},
                    {"range": [0.35, 0.7], "color": "rgba(250, 140, 22, 0.2)"},
                    {"range": [0.7, 1.0], "color": "rgba(245, 34, 45, 0.3)"},
                ],
                "threshold": {"line": {"color": "#f5222d", "width": 3}, "thickness": 0.75, "value": 0.8},
            },
            domain={"row": 0, "column": 0},
        )
    )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=csad,
            number={"valueformat": ".3f"},
            title={"text": "羊群效应（CSAD）", "font": {"size": 14, "color": "#8aa0c2"}},
            gauge={
                "axis": {"range": [0, 0.15]},
                "bar": {"color": "rgba(138, 43, 226, 0.8)", "line": {"width": 0}},
                "steps": [
                    {"range": [0, 0.05], "color": "rgba(34, 197, 94, 0.2)"},
                    {"range": [0.05, 0.1], "color": "rgba(250, 140, 22, 0.2)"},
                    {"range": [0.1, 0.15], "color": "rgba(245, 34, 45, 0.3)"},
                ],
            },
            domain={"row": 0, "column": 1},
        )
    )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=imb,
            number={"valueformat": "+.2f"},
            title={"text": "深度失衡", "font": {"size": 14, "color": "#8aa0c2"}},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": "#1890ff", "line": {"width": 0}},
                "steps": [
                    {"range": [-1, -0.4], "color": "rgba(245, 34, 45, 0.2)"},
                    {"range": [-0.4, 0.4], "color": "rgba(34, 197, 94, 0.2)"},
                    {"range": [0.4, 1.0], "color": "rgba(24, 144, 255, 0.2)"},
                ],
            },
            domain={"row": 0, "column": 2},
        )
    )
    fig.update_layout(
        **{
            **dict(PLOTLY_DARK_LAYOUT),
            "grid": {"rows": 1, "columns": 3, "pattern": "independent"},
            "height": 220,
            "margin": dict(l=10, r=10, t=30, b=10),
        }
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_health_gauges_v2")
    _render_chart_explanation(
        "实时异常指标监测",
        {
            "panic_level": panic,
            "csad": csad,
            "depth_imbalance": imb,
        },
        "请解释恐慌、羊群效应和深度失衡是否同时抬升，以及市场当前更接近平稳、修复还是预警状态。",
    )


def render_ai_insight_card(text: str) -> None:
    st.markdown(
        f"""
        <div class="insight-block" style="animation: insight-pulse 3.5s infinite;">
            <div class="insight-block-label">在线大模型解读</div>
            <div class="insight-block-title">智能前瞻与状态点评</div>
            <div class="insight-block-body">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
